# ContourCraft 官方资产接入说明

动态推理需要以下官方或受许可文件。这些文件不随仓库分发。

## 1. 官方文件

| 文件 | 官方来源 | 作用 | 建议上传位置 |
| --- | --- | --- | --- |
| ChatGarment-CG `assets.zip` | <https://drive.google.com/file/d/1QXezA3J6uXqWHGATmcw3jaYxRXY2Ctte/view?usp=sharing> | 论文示例动作、示例服装及 `registered_params.pkl` 等配套数据 | `assets/inbox/chatgarment-cg-assets.zip` |
| ContourCraft data archive | <https://drive.google.com/file/d/1NfxAeaC2va8TWMjiO_gbAcVPnZ8BYFPD/view?usp=sharing> | 官方 `contourcraft.pth` 和 ContourCraft 辅助数据 | `assets/inbox/contourcraft-data.zip` |
| SMPL-X v1.1 neutral model | <https://smpl-x.is.tue.mpg.de/> | 人体网格生成；需账号登录并接受许可证 | `assets/inbox/SMPLX_NEUTRAL.pkl` |

SMPL-X 要求登录并接受许可证。请从表中的官方渠道取得资产，不要使用来源不明的第三方权重。

## 2. 目录结构

```text
dynamic3d/assets/
├── inbox/                         # 把下载的原文件放到这里
└── ccraft_data/
    ├── trained_models/
    │   └── contourcraft.pth
    ├── aux_data/body_models/smplx/
    │   └── SMPLX_NEUTRAL.pkl
    ├── motions/
    │   └── *.npz
    └── rest_pose/
        └── registered_params.pkl
```

不要手工猜测压缩包内部路径。上传后运行接入脚本，脚本会采用防路径穿越的方式解压、扫描并复制目标文件：

```bash
cd dynamic3d
python scripts/install_official_assets.py \
  --inbox assets/inbox \
  --data-root assets/ccraft_data \
  --chatgarment-cg-root src/ContourCraft-CG
```

随后重新预检：

```bash
source envs/ccraft/bin/activate
export HOOD_PROJECT="$PWD/src/ContourCraft-CG"
export HOOD_DATA="$PWD/assets/ccraft_data"
python scripts/preflight_contourcraft.py \
  --project-root "$HOOD_PROJECT" \
  --data-root "$HOOD_DATA" \
  --manifest prepared_inputs/dynamic_manifest.json \
  --output preflight_report_gpu.json
```

报告中的 `contourcraft_inference_ready` 为 `true` 时，运行环境已具备动态推理所需的依赖与资产。

## 3. 磁盘要求

完整源码、运行环境、模型和中间网格需要较大空间。建议准备至少 150 GB 可用磁盘，并将运行结果目录纳入定期清理策略。
