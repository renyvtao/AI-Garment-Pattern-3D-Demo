# Asset and Data Policy

本仓库用于项目重建和后续研发。以下资产因文件体积、数据授权或第三方许可证限制，不进入 Git 历史。

| 资产 | 是否包含 | 获取方式或处理方式 |
| --- | --- | --- |
| 项目自研 Python/TypeScript/Shell/PowerShell 代码 | 是 | 直接包含在仓库中 |
| README 西装输入、板片和静态渲染示例 | 是 | 位于 `docs/assets/mens-suit/` |
| Web 应用内置示例 | 是 | 位于 `gallery_site/public/cases/` |
| ChatGarment 官方 checkpoint | 否 | 按 ChatGarment 官方说明申请或下载 |
| LLaVA / CLIP 基础模型 | 否 | 从模型发布方获取，并遵循原许可证 |
| ContourCraft-CG checkpoint | 否 | 从 ContourCraft 官方发布渠道获取 |
| SMPL-X `.pkl` 人体模型 | 否 | 在 SMPL-X 官网接受许可后下载，禁止随仓库二次分发 |
| 男西装 LoRA 权重 | 否 | 作为部署资产单独保存，不进入公开仓库 |
| 西装训练/验证原图、划分 JSON 与完整标注 | 是 | 已获公开授权，位于 `suit_finetune/prepared_data/` |
| K62 三维交接包、`mean_all` 人体与运行覆盖文件 | 是 | 已获公开授权，解包后位于 `incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816/` |
| 服务器地址、端口、密码和本地连接配置 | 否 | 仅提供 `server_connection.example.json` 模板 |
| `node_modules`、构建目录和运行缓存 | 否 | 通过包管理器或运行流程重新生成 |
| 大型 ZIP/TAR 结果包 | 否 | 通过网页任务结果页按需生成与下载 |

## 发布检查

1. 本地连接配置、`.env`、密钥和密码不得被跟踪；
2. 不包含模型权重、SMPL-X 文件或其他受限资产；
3. 西装数据已获公开授权；新增或替换数据时仍需重新确认公开权限；
4. 单文件不超过 GitHub 100 MB 限制；
5. 示例图片和其他媒体必须拥有公开展示权限。
