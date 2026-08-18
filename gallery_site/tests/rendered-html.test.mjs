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
  assert.doesNotMatch(html, /href="\/body-customizer"/);
  assert.doesNotMatch(html, /href="\/body-pattern"/);
  assert.doesNotMatch(html, /方案 A|方案 B/);
  assert.doesNotMatch(html, /ChatGarment Reproduction Lab<\/strong>/);
});

test("keeps hidden body tools out of their former public routes", async () => {
  for (const path of ["/body-customizer", "/body-pattern"]) {
    const response = await render(path);
    assert.equal(response.status, 200);
    const html = await response.text();
    assert.match(html, /从一张服装图/);
    assert.doesNotMatch(html, /用人体尺寸和语义描述生成定制体型/);
    assert.doesNotMatch(html, /用少量量体数据补全制版人体尺寸/);
  }
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
  assert.match(page, /useState<GarmentMode>\("mens_suit"\)/);
  assert.match(page, /useState<BodyGender>\("male"\)/);
  assert.match(
    page,
    /const \[measurements, setMeasurements\] = useState\(\{\s*height_cm: "180",\s*weight_kg: "75",\s*chest_cm: "100",\s*waist_cm: "84",\s*hips_cm: "98",\s*\}\);/s,
  );
  assert.match(layout, /lang="zh-CN"/);
  assert.doesNotMatch(packageJson, /react-loading-skeleton/);
  assert.doesNotMatch(page, /_sites-preview|SkeletonPreview/);
});
