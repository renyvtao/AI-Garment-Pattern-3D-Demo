# 项目重建与运行指南

本文档说明如何在全新的 Ubuntu 22.04 GPU 环境中安装项目、获取官方依赖并启动完整应用。

## 1. 仓库与外部资产

仓库直接包含：

- 项目集成代码与展示网页；
- 固定上游 Git revision 的项目清单；
- 579/72/72 个基础西装款式划分，以及由其扩展得到的 1742/219/216 张训练、验证、测试图片与标注；
- K62 三维装配包、`mean_all` 人体和运行覆盖文件；
- README 使用的西装静态展示素材。

需要从官方渠道单独获取：

- ChatGarment 官方约 15 GB checkpoint；
- LLaVA 1.5 7B 与 CLIP vision tower；
- ContourCraft-CG 官方 checkpoint 和配套数据；
- 需要接受单独许可的 SMPL-X v1.1 人体模型；
- Blender 3.6.14 Linux 发行包。

这些资产需要由使用者从发布方获取，因为 GitHub 文件限制或第三方许可证禁止重新分发。

## 2. 推荐硬件

- Ubuntu 22.04；
- RTX 4090 24 GB 或更高；
- 数据盘至少 100 GB，建议 150 GB；
- CUDA Toolkit 11.8；
- Python 3.10.8；
- Node.js 22.13+。

完整通路建议使用 24 GB 或更高显存，并预留至少 150 GB 磁盘空间用于模型、构建缓存和任务输出。

## 3. 克隆项目与上游源码

```bash
git clone https://github.com/renyvtao/AI-Garment-Pattern-3D-Demo.git
cd AI-Garment-Pattern-3D-Demo

python3 scripts/bootstrap_sources.py \
  --apply-overrides \
  --install-k62-overlay
```

脚本根据 `PROJECT_MANIFEST.json` 和 `dynamic3d/SOURCE_LOCK.json` checkout 固定 commit。若已有 checkout 包含未提交修改，脚本会停止，避免覆盖开发工作。

只检查源码版本：

```bash
python3 scripts/bootstrap_sources.py --check-only
```

## 4. ChatGarment 环境

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-app.lock.txt
python -m pip install flash-attn==2.5.9.post1 --no-build-isolation
```

随后按 ChatGarment 固定 revision 中的安装说明编译其额外 CUDA 扩展。`requirements-app.lock.txt` 锁定了本项目服务和男西装 LoRA 使用的核心 Python 包；上游仓库特有依赖仍以上游固定 revision 为准。

## 5. 基础模型与官方任务 checkpoint

将模型放到以下精确位置：

```text
models/llava-v1.5-7b-4481d270/
cache/huggingface/hub/models--openai--clip-vit-large-patch14-336/
ChatGarment/checkpoints/try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final/pytorch_model.bin
```

固定 revision 与 checkpoint 的大小、SHA-256 位于 `PROJECT_MANIFEST.json`。官方 checkpoint 校验值为：

```text
bytes  = 14987210682
sha256 = 3d6ca6dc52d4400d5603ac0dcb163aeb82d1021d99c90686253f6cc4a72b8a3a
```

不要使用来源不明的第三方 checkpoint 替换。

## 6. ContourCraft、SMPL-X 与 Blender

按 `dynamic3d/OFFICIAL_ASSETS.md` 获取官方压缩包和 SMPL-X v1.1 文件，然后执行：

```bash
python dynamic3d/scripts/install_official_assets.py \
  --inbox dynamic3d/assets/inbox \
  --data-root dynamic3d/assets/ccraft_data \
  --chatgarment-cg-root dynamic3d/src/ContourCraft-CG
```

最终至少应存在：

```text
dynamic3d/assets/ccraft_data/trained_models/contourcraft.pth
dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_MALE.pkl
dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_FEMALE.pkl
dynamic3d/assets/ccraft_data/aux_data/body_models/smplx/SMPLX_NEUTRAL.pkl
dynamic3d/blender-3.6.14-linux-x64/blender
```

ContourCraft 的 CUDA 扩展请在独立环境 `dynamic3d/envs/ccraft` 中按固定源码安装。仓库中的 `dynamic3d/scripts/install_no_gpu_extensions.sh` 可先完成无 GPU 阶段，GPU 扩展必须在目标 CUDA 环境中编译。

## 7. 西装数据与 LoRA 微调

公开数据已位于：

```text
suit_finetune/prepared_data/images/
suit_finetune/prepared_data/train.json
suit_finetune/prepared_data/validation.json
suit_finetune/prepared_data/test.json
suit_finetune/prepared_data/manifest.json
```

训练：

```bash
export AI_GARMENT_PROJECT_ROOT="$PWD"
export SUIT_DATA_ROOT="$PWD/suit_finetune/prepared_data"
bash suit_finetune/run_suit_poc_train.sh
```

期望输出：

```text
ChatGarment/runs/suit_poc_lora_v1/suit_lora_state.bin
```

LoRA rank、训练步数、梯度累积和学习率均可通过训练脚本中的环境变量调整。

## 8. 安装自检

公开仓库内容检查，不要求第三方权重：

```bash
python scripts/project_preflight.py --source-only
```

所有模型与运行时安装后执行完整检查：

```bash
python scripts/project_preflight.py --hash-large-assets
```

继续执行单元测试：

```bash
(
  cd pipeline
  python -m unittest test_app_service.py -v
)
python -m unittest dynamic3d/body_customization/tests/test_gbt1335_adapter.py -v

cd gallery_site
corepack enable
pnpm install --frozen-lockfile
pnpm test
```

## 9. 启动服务

```bash
cd /path/to/AI-Garment-Pattern-3D-Demo
bash deployment/start_services.sh
```

健康检查：

```bash
curl http://127.0.0.1:3000/api/body/health
curl http://127.0.0.1:3000/api/jobs/health
curl -I http://127.0.0.1:3000/
```

远程服务器从 Windows 访问时，复制 `deployment/server_connection.example.json` 为 `server_connection.local.json`，填写 SSH 信息并运行 `deployment/run_ssh_tunnel.ps1`。

## 10. 服务管理

```bash
bash deployment/status_services.sh
bash deployment/stop_services.sh
```

运行日志写入 `deployment/logs/`，PID 文件写入 `deployment/runtime/`，用户任务与产物写入 `app_data/`。这些目录均为运行时数据，不应提交到 Git。
