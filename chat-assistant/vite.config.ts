import vinext from "vinext";
import { existsSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import { homedir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawn } from "node:child_process";
import { defineConfig, loadEnv, type Plugin } from "vite";
import hostingConfig from "./.openai/hosting.json";
import { sites } from "./build/sites-vite-plugin";
import { parseScriptOutput } from "./build/script-result.mjs";

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

type ScriptProcessResult = {
  code: number | null;
  stdout: string;
  stderr: string;
  timedOut: boolean;
};

function boundedNumber(
  value: unknown,
  fallback: number,
  minimum: number,
  maximum: number,
) {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed)
    ? Math.min(maximum, Math.max(minimum, parsed))
    : fallback;
}

function buildChatArguments(message: string, options: Record<string, unknown>) {
  const llmProvider = options.llmProvider === "openai" ? "openai" : "local";
  const openaiModel =
    typeof options.openaiModel === "string" && options.openaiModel.trim()
      ? options.openaiModel.trim().slice(0, 100)
      : "gpt-3.5-turbo";
  const maxTokens = Math.round(boundedNumber(options.maxTokens, 512, 64, 4096));
  const temperature = boundedNumber(options.temperature, 0, 0, 2);
  const nCtx = Math.round(boundedNumber(options.nCtx, 2048, 512, 32768));
  const maxSteps = Math.round(boundedNumber(options.maxSteps, 5, 1, 10));

  const args = [
    "--prompt",
    message,
    "--llm-provider",
    llmProvider,
    "--max-tokens",
    String(maxTokens),
    "--temperature",
    String(temperature),
    "--n-ctx",
    String(nCtx),
    "--max-steps",
    String(maxSteps),
  ];

  if (llmProvider === "openai") {
    args.push("--openai-model", openaiModel);
  }
  if (options.trace === true) {
    args.push("--trace");
  }

  return args;
}

function runScript(
  pythonBin: string,
  scriptPath: string,
  args: string[],
  timeoutMs: number,
): Promise<ScriptProcessResult> {
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(pythonBin, [scriptPath, ...args], {
      cwd: dirname(scriptPath),
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    let outputSize = 0;
    let timedOut = false;

    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      outputSize += chunk.length;
      if (outputSize <= 2_000_000) stdout.push(chunk);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      outputSize += chunk.length;
      if (outputSize <= 2_000_000) stderr.push(chunk);
    });
    child.once("error", (error) => {
      clearTimeout(timeout);
      rejectRun(error);
    });
    child.once("close", (code) => {
      clearTimeout(timeout);
      resolveRun({
        code,
        stdout: Buffer.concat(stdout).toString("utf8"),
        stderr: Buffer.concat(stderr).toString("utf8").trim(),
        timedOut,
      });
    });
  });
}

function processError(result: ScriptProcessResult) {
  const lastErrorLine = result.stderr.split("\n").filter(Boolean).pop();
  return lastErrorLine
    ? `The script stopped with an error: ${lastErrorLine}`
    : `The script stopped with exit code ${result.code}.`;
}

function localScriptBridge(runtimeEnv: Record<string, string | undefined>): Plugin {
  return {
    name: "ufund-local-script-bridge",
    enforce: "pre",
    configureServer(server) {
      let reindexRunning = false;

      server.middlewares.use(async (req, res, next) => {
        const pathname = new URL(req.url || "/", "http://localhost").pathname;
        const scriptPath =
          runtimeEnv.UFUND_SCRIPT_PATH || process.env.UFUND_SCRIPT_PATH || DEFAULT_SCRIPT_PATH;
        const pythonBin =
          runtimeEnv.UFUND_PYTHON_BIN ||
          process.env.UFUND_PYTHON_BIN ||
          (existsSync(DEFAULT_FRAMEWORK_PYTHON) ? DEFAULT_FRAMEWORK_PYTHON : "python3");

        if (pathname === "/api/health" && req.method === "GET") {
          sendJson(res, 200, {
            ok: true,
            scriptFound: existsSync(scriptPath),
            indexFound: existsSync(resolve(dirname(scriptPath), "chroma_db")),
            reindexRunning,
            scriptName: scriptPath.split("/").pop(),
          });
          return;
        }

        if (pathname !== "/api/chat" && pathname !== "/api/reindex") {
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

        let body: Record<string, unknown> = {};
        try {
          const parsedBody = await readJsonBody(req);
          body = parsedBody && typeof parsedBody === "object" && !Array.isArray(parsedBody)
            ? parsedBody
            : {};
        } catch {
          sendJson(res, 400, { error: "The request body must be valid JSON." });
          return;
        }

        if (pathname === "/api/reindex") {
          if (reindexRunning) {
            sendJson(res, 409, { error: "A reindex job is already running." });
            return;
          }

          reindexRunning = true;
          const args = ["--reindex"];
          if (body.exportDrive === true) args.push("--export-drive");

          try {
            const result = await runScript(
              pythonBin,
              scriptPath,
              args,
              30 * 60 * 1000,
            );

            if (result.timedOut) {
              sendJson(res, 504, {
                error: "Reindexing took longer than 30 minutes and was stopped.",
              });
            } else if (result.code !== 0) {
              sendJson(res, 500, { error: processError(result) });
            } else {
              sendJson(res, 200, {
                ok: true,
                message: body.exportDrive === true
                  ? "Drive files were refreshed and the document index was rebuilt."
                  : "The document index was rebuilt from the current exported files.",
              });
            }
          } catch (error) {
            sendJson(res, 500, {
              error: `The reindex job could not start: ${error instanceof Error ? error.message : "Unknown error"}`,
            });
          } finally {
            reindexRunning = false;
          }
          return;
        }

        if (reindexRunning) {
          sendJson(res, 409, {
            error: "The document index is being rebuilt. Please wait for it to finish.",
          });
          return;
        }

        const message = typeof body.message === "string" ? body.message.trim() : "";
        if (!message) {
          sendJson(res, 400, { error: "Please enter a question." });
          return;
        }

        const options =
          body.options && typeof body.options === "object"
            ? body.options as Record<string, unknown>
            : {};

        try {
          const result = await runScript(
            pythonBin,
            scriptPath,
            buildChatArguments(message, options),
            10 * 60 * 1000,
          );

          if (result.timedOut) {
            sendJson(res, 504, {
              error: "The script took longer than 10 minutes and was stopped.",
            });
            return;
          }
          if (result.code !== 0) {
            sendJson(res, 500, { error: processError(result) });
            return;
          }

          const scriptResult = parseScriptOutput(result.stdout);
          sendJson(res, 200, {
            ...scriptResult,
            answer:
              scriptResult.answer ||
              "The script connection is working, but app.py did not return an answer yet.",
            source: scriptResult.answer ? "script" : "placeholder",
          });
        } catch (error) {
          sendJson(res, 500, {
            error: `The script could not start: ${error instanceof Error ? error.message : "Unknown error"}`,
          });
        }
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
