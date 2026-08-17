import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the ChatGarment function home", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<html[^>]+lang="zh-CN"/i);
  assert.match(html, /<title>AI 服装制版与三维展示<\/title>/i);
  assert.match(html, /从一张服装图/);
  assert.match(html, /选择要使用的功能/);
  assert.match(html, /href="\/workflow"/);
  assert.doesNotMatch(html, /ChatGarment Reproduction Lab<\/strong>/);
});

test("server-renders the result atlas on its own route", async () => {
  const response = await render("/results");
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /十组官方示例，一页完成核对/);
  assert.match(html, /示例 10/);
});

test("keeps the result inventory and downloadable outputs wired", async () => {
  const [page, layout, packageJson] = await Promise.all([
    readFile(new URL("../app/lab.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.equal((page.match(/id: "valid_garment_/g) ?? []).length, 10);
  assert.match(page, /_render_front\.png/);
  assert.match(page, /_render_back\.png/);
  assert.match(page, /_sim\.obj/);
  assert.match(page, /_specification\.json/);
  assert.match(layout, /lang="zh-CN"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.doesNotMatch(page, /_sites-preview|SkeletonPreview/);
});
