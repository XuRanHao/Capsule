import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

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

test("server-renders the Capsule search workspace", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Capsule · 多模态记忆检索<\/title>/i);
  assert.match(html, /CAPSULE/);
  assert.match(html, /搜到你记得的/);
  assert.match(html, /多模态记忆检索/);
  assert.match(html, /演示数据/);
  assert.match(html, /关联段落/);
  assert.match(html, /Search Capsule/);
  assert.match(html, /QUERY PLAN/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("removes all disposable starter-preview references", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /\/api\/v1\/search/);
  assert.match(page, /source_contexts/);
  assert.match(page, /Weighted RRF/);
  assert.match(page, /normalized_weighted_similarity/);
  assert.match(page, /\/api\/v1\/search-capsules/);
  assert.match(page, /\/api\/v1\/query-images/);
  assert.match(layout, /Capsule · 多模态记忆检索/);
  assert.doesNotMatch(
    `${page}\n${layout}\n${packageJson}`,
    /codex-preview|react-loading-skeleton|_sites-preview/i,
  );

  await assert.rejects(
    access(new URL("../app/_sites-preview", templateRoot)),
  );
});
