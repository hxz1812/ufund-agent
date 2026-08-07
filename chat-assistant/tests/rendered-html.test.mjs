import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the UFund chat interface", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>UFund Knowledge Assistant<\/title>/i);
  assert.match(html, /UFund Assistant/);
  assert.match(html, /Ask about your UFund documents/);
  assert.match(html, /New conversation/);
  assert.doesNotMatch(html, /codex-preview|taking shape|loading skeleton/i);
});

test("keeps the Python script behind an independent-message bridge", async () => {
  const [page, layout, viteConfig, packageJson, parserSource] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../vite.config.ts", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../build/script-result.mjs", import.meta.url), "utf8"),
  ]);

  assert.match(page, /body:\s*JSON\.stringify\(\{ message: question, options: runSettings \}\)/);
  assert.match(page, /independent questions/);
  assert.match(viteConfig, /"Desktop", "ufund-agent", "app\.py"/);
  assert.match(viteConfig, /buildChatArguments\(message, options\)/);
  assert.match(viteConfig, /"--llm-provider"/);
  assert.match(viteConfig, /"--openai-model"/);
  assert.match(viteConfig, /"--max-tokens"/);
  assert.match(viteConfig, /"--temperature"/);
  assert.match(viteConfig, /"--n-ctx"/);
  assert.match(viteConfig, /"--max-steps"/);
  assert.match(viteConfig, /"--trace"/);
  assert.match(viteConfig, /pathname === "\/api\/reindex"/);
  assert.match(viteConfig, /"--reindex"/);
  assert.match(viteConfig, /"--export-drive"/);
  assert.match(viteConfig, /process\.env\.UFUND_SCRIPT_PATH/);
  assert.match(viteConfig, /process\.env\.UFUND_PYTHON_BIN/);
  assert.match(viteConfig, /indexFound/);
  assert.match(viteConfig, /parseScriptOutput/);
  assert.match(page, /documentsUsed/);
  assert.match(page, /source-card/);
  assert.match(page, /Run settings/);
  assert.match(page, /Rebuild document index/);
  assert.match(parserSource, /documents_used/);
  assert.match(layout, /title: "UFund Knowledge Assistant"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});

test("parses the updated app.py structured result", async () => {
  const { parseScriptOutput } = await import("../build/script-result.mjs");
  const result = parseScriptOutput(JSON.stringify({
    answer: "The investment memo supports this claim [2].",
    documents_used: [{
      evidence_id: 2,
      file_name: "Investment memo.pdf",
      source_system: "google_drive",
      relative_path: "Deals/Investment memo.pdf",
      file_type: ".pdf",
      sheet_name: null,
      row_start: null,
      row_end: null,
    }],
    steps: [{ action: "tool_use" }, { action: "final_answer" }],
  }, null, 2));

  assert.equal(result.answer, "The investment memo supports this claim [2].");
  assert.equal(result.documentsUsed.length, 1);
  assert.equal(result.documentsUsed[0].evidenceId, 2);
  assert.equal(result.documentsUsed[0].fileName, "Investment memo.pdf");
  assert.equal(result.stepCount, 2);
});
