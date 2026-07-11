import vinext from "vinext";
import { existsSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";
import { defineConfig, loadEnv, type Plugin } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";

const SITE_CREATOR_PLACEHOLDER_DATABASE_ID =
  "00000000-0000-4000-8000-000000000000";

const { d1, r2 } = hostingConfig;

const DEFAULT_SCRIPT_PATH = resolve(homedir(), "Desktop", "ufund-agent", "app.py");
const DEFAULT_FRAMEWORK_PYTHON =
  "/Library/Frameworks/Python.framework/Versions/3.14/bin/python3";

function sendJson(res: ServerResponse, status: number, value: unknown) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "no-store");
  res.end(JSON.stringify(value));
}

async function readJsonBody(req: IncomingMessage) {
  const chunks: Buffer[] = [];
  let size = 0;

  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    size += buffer.length;
    if (size > 64_000) throw new Error("Request is too large.");
    chunks.push(buffer);
  }

  return JSON.parse(Buffer.concat(chunks).toString("utf8"));
}

function extractAnswer(output: string) {
  const text = output.trim();
  if (!text) return null;

  const lines = text.split("\n");
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    try {
      const value = JSON.parse(lines[index]);
      if (value && typeof value.answer === "string") return value.answer.trim();
    } catch {
      // Plain-text output is also supported.
    }
  }

  const markedAnswer = text.match(/(?:^|\n)ANSWER:\s*\n?([\s\S]*)$/i);
  if (markedAnswer?.[1]) return markedAnswer[1].trim();

  // app.py currently prints a retrieved-document count after its answer.
  // Keep that diagnostic out of the user-facing chat response.
  const withoutTrailingCount = text.match(/^([\s\S]*\S)\n\d+\s*$/);
  return withoutTrailingCount?.[1]?.trim() || text;
}

function localScriptBridge(runtimeEnv: Record<string, string | undefined>): Plugin {
  return {
    name: "ufund-local-script-bridge",
    enforce: "pre",
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url || "/", "http://localhost").pathname;
        const scriptPath =
          runtimeEnv.UFUND_SCRIPT_PATH || process.env.UFUND_SCRIPT_PATH || DEFAULT_SCRIPT_PATH;

        if (pathname === "/api/health" && req.method === "GET") {
          sendJson(res, 200, {
            ok: true,
            scriptFound: existsSync(scriptPath),
            scriptName: scriptPath.split("/").pop(),
          });
          return;
        }

        if (pathname !== "/api/chat") {
          next();
          return;
        }

        if (req.method !== "POST") {
          sendJson(res, 405, { error: "Only POST requests are supported." });
          return;
        }

        if (!existsSync(scriptPath)) {
          sendJson(res, 500, {
            error: "The configured script could not be found. Check UFUND_SCRIPT_PATH.",
          });
          return;
        }

        let message = "";
        try {
          const body = await readJsonBody(req);
          message = typeof body?.message === "string" ? body.message.trim() : "";
        } catch {
          sendJson(res, 400, { error: "The request body must be valid JSON." });
          return;
        }

        if (!message) {
          sendJson(res, 400, { error: "Please enter a question." });
          return;
        }

        const pythonBin =
          runtimeEnv.UFUND_PYTHON_BIN ||
          process.env.UFUND_PYTHON_BIN ||
          (existsSync(DEFAULT_FRAMEWORK_PYTHON) ? DEFAULT_FRAMEWORK_PYTHON : "python3");
        const child = spawn(pythonBin, [scriptPath, "--prompt", message], {
          cwd: dirname(scriptPath),
          env: process.env,
          stdio: ["ignore", "pipe", "pipe"],
        });
        const stdout: Buffer[] = [];
        const stderr: Buffer[] = [];
        let outputSize = 0;
        let didTimeOut = false;

        const timeout = setTimeout(() => {
          didTimeOut = true;
          child.kill("SIGTERM");
        }, 10 * 60 * 1000);

        child.stdout.on("data", (chunk: Buffer) => {
          outputSize += chunk.length;
          if (outputSize <= 2_000_000) stdout.push(chunk);
        });
        child.stderr.on("data", (chunk: Buffer) => {
          outputSize += chunk.length;
          if (outputSize <= 2_000_000) stderr.push(chunk);
        });

        child.on("error", (error) => {
          clearTimeout(timeout);
          sendJson(res, 500, { error: `The script could not start: ${error.message}` });
        });

        child.on("close", (code) => {
          clearTimeout(timeout);
          if (res.writableEnded) return;

          const standardOutput = Buffer.concat(stdout).toString("utf8");
          const errorOutput = Buffer.concat(stderr).toString("utf8").trim();

          if (didTimeOut) {
            sendJson(res, 504, {
              error: "The script took longer than 10 minutes and was stopped.",
            });
            return;
          }

          if (code !== 0) {
            const lastErrorLine = errorOutput.split("\n").filter(Boolean).pop();
            sendJson(res, 500, {
              error: lastErrorLine
                ? `The script stopped with an error: ${lastErrorLine}`
                : `The script stopped with exit code ${code}.`,
            });
            return;
          }

          const answer = extractAnswer(standardOutput);
          sendJson(res, 200, {
            answer:
              answer ||
              "The script connection is working, but app.py did not return an answer yet. You can add its response output when you’re ready.",
            source: answer ? "script" : "placeholder",
          });
        });
      });
    },
  };
}

// macOS Seatbelt blocks FSEvents, so Codex previews need polling for HMR.
const isCodexSeatbeltSandbox = process.env.CODEX_SANDBOX === "seatbelt";

const localBindingConfig = {
  main: "./worker/index.ts",
  compatibility_flags: ["nodejs_compat"],
  d1_databases: d1
    ? [
        {
          binding: d1,
          database_name: "site-creator-d1",
          database_id: SITE_CREATOR_PLACEHOLDER_DATABASE_ID,
        },
      ]
    : [],
  r2_buckets: r2
    ? [
        {
          binding: r2,
          bucket_name: "site-creator-r2",
        },
      ]
    : [],
};

export default defineConfig(async ({ mode }) => {
  const runtimeEnv: Record<string, string | undefined> = {
    ...loadEnv(mode, process.cwd(), ""),
    ...process.env,
  };
  // Keep Wrangler and Miniflare state project-local. These are non-secret tool
  // settings; application environment belongs in ignored `.env*` files.
  process.env.WRANGLER_WRITE_LOGS ??= "false";
  process.env.WRANGLER_LOG_PATH ??= ".wrangler/logs";
  process.env.MINIFLARE_REGISTRY_PATH ??= ".wrangler/registry";

  // Wrangler snapshots its log path while the Cloudflare plugin is imported.
  const { cloudflare } = await import("@cloudflare/vite-plugin");

  return {
    server: isCodexSeatbeltSandbox
      ? { watch: { useFsEvents: false, usePolling: true } }
      : undefined,
    plugins: [
      localScriptBridge(runtimeEnv),
      vinext(),
      sites(),
      cloudflare({
        viteEnvironment: { name: "rsc", childEnvironments: ["ssr"] },
        config: localBindingConfig,
      }),
    ],
  };
});
