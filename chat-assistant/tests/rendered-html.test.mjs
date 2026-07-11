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
  const [page, layout, viteConfig, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../vite.config.ts", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /body:\s*JSON\.stringify\(\{ message: question \}\)/);
  assert.match(page, /Each question runs independently/);
  assert.match(viteConfig, /"Desktop", "ufund-agent", "app\.py"/);
  assert.match(viteConfig, /\[scriptPath, "--prompt", message\]/);
  assert.match(viteConfig, /process\.env\.UFUND_SCRIPT_PATH/);
  assert.match(viteConfig, /process\.env\.UFUND_PYTHON_BIN/);
  assert.match(layout, /title: "UFund Knowledge Assistant"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
});
