import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const templateRoot = new URL("../", import.meta.url);

async function render(pathname = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set(
    "test",
    `${process.pid}-${Date.now()}-${pathname.replaceAll("/", "-")}`,
  );
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(new URL(pathname, "http://localhost/"), {
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
  assert.match(html, /<title>Capsule · 个人多模态素材工作台<\/title>/i);
  assert.match(html, /CAPSULE/);
  assert.match(html, /搜到你记得的/);
  assert.match(html, /个人多模态素材库/);
  assert.match(html, /实时服务/);
  assert.match(html, /默认检索原始内容/);
  assert.match(html, /检索维度/);
  assert.match(html, /原始内容/);
  assert.match(html, /目标素材类型/);
  assert.match(html, /Markdown 段落/);
  assert.match(html, /纯文本块/);
  assert.match(html, /系统默认等权/);
  assert.match(html, /检索文字明确表达倾向时自动解析权重/);
  assert.match(html, /图片 \/ 视频本身不参与权重解析/);
  assert.doesNotMatch(html, /精搜模式|普通模式|precision_mode/);
  const targetTypeInputs = [
    ...html.matchAll(/<input[^>]*name="target_asset_types"[^>]*>/g),
  ].map((match) => match[0]);
  assert.equal(targetTypeInputs.length, 4);
  const dimensionInputs = [
    ...html.matchAll(/<input[^>]*name="embedding_types"[^>]*>/g),
  ].map((match) => match[0]);
  assert.equal(dimensionInputs.length, 12);
  assert.match(
    dimensionInputs.find((input) => /value="native_multimodal"/.test(input)) ?? "",
    /checked=""/,
  );
  assert.match(html, /开始检索/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
});

test("server-renders every POC workspace page", async () => {
  const routes = [
    ["/import", /把散落的素材/],
    ["/tasks", /每一步，都看得见/],
    ["/assets", /所有素材，都有语义/],
    ["/assets/asset_twilight_01", /ASSET DETAIL/],
    ["/clusters", /从相似中，看见结构/],
    ["/search", /搜到你记得的/],
    ["/capsules", /把一次发现，变成可复用的入口/],
  ];

  for (const [pathname, expectedCopy] of routes) {
    const response = await render(pathname);
    assert.equal(response.status, 200, `${pathname} should render`);
    const html = await response.text();
    assert.match(html, /CAPSULE/);
    assert.match(html, expectedCopy);
    assert.match(html, /导入/);
    assert.match(html, /处理任务/);
    assert.match(html, /Assets/);
    assert.match(html, /Cluster/);
    assert.match(html, /搜索/);
    assert.match(html, /Capsule/);
    assert.doesNotMatch(html, /codex-preview|Your site is taking shape/i);
  }
});

test("cluster selector renders exactly the ten Feature dimensions", async () => {
  const response = await render("/clusters");
  assert.equal(response.status, 200);
  const html = await response.text();
  const optionValues = [...html.matchAll(/<option[^>]*\bvalue="([^"]+)"/g)]
    .map((match) => match[1])
    .filter(Boolean);

  assert.deepEqual(optionValues, [
    "subject_content",
    "scene_theme",
    "visual_style",
    "color_composition",
    "mood_atmosphere",
    "character_state_or_psychology",
    "asset_usage",
    "target_audience",
    "provenance",
    "rights_version_authorship",
  ]);
});

test("cluster workspace keeps history and exposes current resident controls", async () => {
  const response = await render("/clusters");
  assert.equal(response.status, 200);
  const html = await response.text();

  assert.match(html, /当前聚类/);
  assert.match(html, /Cluster Run/);
  assert.match(html, /正在读取当前簇/);
});

test("removes all disposable starter-preview references", async () => {
  const [page, layout, packageJson, shell, importPage, tasksPage, assetsPage, detailPage, clustersPage, capsulesPage] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
    readFile(new URL("../app/components/DemoShell.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/import/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/tasks/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/assets/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/assets/[id]/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/clusters/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/capsules/page.tsx", import.meta.url), "utf8"),
  ]);

  assert.match(page, /\/api\/v1\/search/);
  assert.match(page, /source_contexts/);
  assert.match(page, /Weighted RRF/);
  assert.match(page, /normalized_weighted_similarity/);
  assert.doesNotMatch(page, /precisionMode|precision_mode|精搜模式|普通模式/);
  assert.match(page, /\/api\/v1\/search-capsules/);
  assert.match(page, /\/api\/v1\/query-images/);
  assert.match(page, /cluster_run_id=/);
  assert.match(page, /cluster_capsule_id=/);
  assert.match(page, /target="_blank"/);
  assert.match(page, /rel="noopener noreferrer"/);
  assert.match(page, /ProductTopbar/);
  assert.match(shell, /ProductTopbar/);
  assert.match(shell, /product-nav-compact/);
  assert.doesNotMatch(shell, /demo-sidebar/);
  assert.match(importPage, /选择文件夹/);
  assert.match(tasksPage, /失败记录/);
  assert.match(assetsPage, /source_file/);
  assert.match(detailPage, /source_contexts/);
  assert.match(detailPage, /asset-video-player/);
  assert.match(detailPage, /asset\.content_url/);
  assert.match(clustersPage, /embedding/);
  assert.match(clustersPage, /URLSearchParams/);
  assert.match(clustersPage, /deepLinkTargetRef/);
  assert.match(clustersPage, /clusterDetailRef/);
  assert.match(clustersPage, /\/api\/v1\/clusters\?/);
  assert.match(clustersPage, /members:attach/);
  assert.match(clustersPage, /members:detach/);
  assert.match(clustersPage, /开放常驻/);
  assert.match(clustersPage, /人工常驻/);
  assert.match(capsulesPage, /Search Capsule/);
  assert.match(layout, /Capsule · 个人多模态素材工作台/);
  assert.doesNotMatch(
    `${page}\n${layout}\n${packageJson}\n${shell}\n${importPage}\n${tasksPage}\n${assetsPage}\n${detailPage}\n${clustersPage}\n${capsulesPage}`,
    /codex-preview|react-loading-skeleton|_sites-preview/i,
  );

  await assert.rejects(
    access(new URL("../app/_sites-preview", templateRoot)),
  );
});
