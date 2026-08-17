# Web 应用

统一管理服装任务、人体尺寸输入、执行进度、中间模型数据和最终产物。前端通过 `gateway.mjs` 访问人体服务与任务服务。

## 本地启动

需要 Node.js 22.13 或更高版本，推荐使用 pnpm：

```bash
pnpm install
pnpm dev
```

终端会打印访问地址，默认从 `http://localhost:3000` 开始选择可用端口。

生产模式：

```bash
pnpm build
pnpm start -- --host 0.0.0.0 --port 3000
```

## 验证

```bash
pnpm test
```

测试会执行生产构建，并检查主要页面、示例清单和结果下载链接。

## 目录

- `app/page.tsx`：功能入口与示例索引。
- `app/workflow/`：图片、人体参数和任务提交页面。
- `app/results/`：任务进度、产物预览与下载页面。
- `app/globals.css`：响应式页面样式。
- `public/cases`：按案例组织的内置示例。
- `tests/rendered-html.test.mjs`：服务端渲染与结果清单测试。

每项三维结果位于：

```text
public/cases/<case>/valid_garment_<type>/valid_garment_<type>/
```

其中包括模拟后的 `*_sim.obj`、正反面 `*_render_*.png`、材质、纹理和 `sim_props.yaml`。

每个案例的正式动态视频位于：

```text
public/cases/<case>/dynamic/contourcraft.mp4
```

网页会同时标注帧数、帧率、碰撞峰值、末帧碰撞数和零碰撞帧数。
