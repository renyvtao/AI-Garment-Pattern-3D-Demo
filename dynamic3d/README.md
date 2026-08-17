# 动态三维模块

ContourCraft 权重、SMPL-X 和动作资源的目录约定及安装方式见
[`OFFICIAL_ASSETS.md`](OFFICIAL_ASSETS.md)。

该目录用于在现有 GarmentCodeRC + Warp 静态缝合结果之后接入
ContourCraft-CG，最终输出由 SMPL-X 动作驱动的服装动态网格和视频。

## 依赖

- PyTorch 2.1.2 + CUDA 11.8；
- PyTorch3D、PyTorch Geometric、Warp、cuDF、cuGraph；
- ContourCraft-CG、ContourCraft、CCCollisions 固定源码；
- ContourCraft checkpoint、SMPL-X v1.1、动作和注册人体参数；
- Blender 3.6 LTS，用于离屏渲染。

上游源码版本位于 `SOURCE_LOCK.json`。CUDA 扩展可通过 `scripts/install_no_gpu_extensions.sh` 在目标 GPU 环境中编译。

## 准备动态输入

```bash
python scripts/prepare_dynamic_inputs.py \
  --vis-root /path/to/example_imgs_img_recon/vis_new \
  --output-root /path/to/dynamic3d/prepared_inputs
```

脚本只使用 CPU，输出：

```text
prepared_inputs/
├─ <case-id>/
│  └─ combined_garment_meters.obj
└─ dynamic_manifest.json
```

## 环境与资源预检

```bash
python scripts/preflight_contourcraft.py \
  --project-root /path/to/ContourCraft-CG \
  --data-root /path/to/ccraft_data \
  --manifest /path/to/dynamic_manifest.json \
  --output /path/to/preflight_report.json
```

只有下列条件全部满足，才能启动 ContourCraft 动态推理：

- PyTorch、PyTorch3D、PyTorch Geometric、SMPL-X、Warp 可导入；
- CCCollision CUDA 扩展可导入；
- `trained_models/contourcraft.pth` 存在；
- `SMPLX_NEUTRAL.pkl` 存在；
- 至少一条动作序列存在；
- `torch.cuda.is_available()` 为 `true`。

## 单案例动态推理入口

无 GPU 时先使用 `--dry-run` 检查路径：

```bash
python scripts/run_dynamic_case.py \
  --project-root /path/to/ContourCraft-CG \
  --hood-data /path/to/ccraft_data \
  --manifest /path/to/dynamic_manifest.json \
  --case-id <case-id> \
  --checkpoint /path/to/contourcraft.pth \
  --smplx-model /path/to/SMPLX_NEUTRAL.pkl \
  --motion /path/to/motion.npz \
  --rest-body-params /path/to/rest_body_params.pkl \
  --output-dir /path/to/output/<case-id> \
  --dry-run
```

在已配置 GPU 依赖的环境中去掉 `--dry-run`。成功时输出压缩的
`contourcraft_sequence.npz`，其中包含服装逐帧顶点、三角面、人体序列、
固定点和碰撞指标。

## 批量执行并接入网页

```bash
python scripts/run_dynamic_batch.py \
  --project-root /path/to/ContourCraft-CG \
  --hood-data /path/to/ccraft_data \
  --manifest /path/to/dynamic_manifest.json \
  --checkpoint /path/to/contourcraft.pth \
  --smplx-model /path/to/SMPLX_NEUTRAL.pkl \
  --motion /path/to/motion.npz \
  --rest-body-params /path/to/registered_params.pkl \
  --output-root /path/to/dynamic_outputs \
  --blender /path/to/blender \
  --gallery-public /path/to/gallery_site/public \
  --skip-existing
```

批处理默认串行执行，避免多个高分辨率服装同时占用显存。每个成功视频都会
被复制到网页约定路径，刷新页面即可看到，无需再手工分文件夹查看。

## 提取动态网格帧

```bash
python scripts/extract_dynamic_meshes.py \
  --sequence /path/to/contourcraft_sequence.npz \
  --output-dir /path/to/mesh_frames \
  --stride 5 \
  --include-body
```

逐帧 OBJ 仅用于渲染中间过程，渲染完成后应保留 NPZ 和 MP4，避免耗尽磁盘。

## 渲染动态视频

Blender 3.6 LTS 到位后执行：

```bash
blender -b \
  --python scripts/render_dynamic_sequence.py -- \
  --sequence /path/to/contourcraft_sequence.npz \
  --output /path/to/contourcraft.mp4 \
  --fps 30 \
  --stride 1
```

渲染脚本在每一帧原位更新同一个服装和人体网格，不会为数百帧创建数百个
Blender 对象。网页约定的最终视频路径为：

```text
gallery_site/public/cases/<valid_garment_case_id>/dynamic/contourcraft.mp4
```

## 受许可资产

- SMPL-X v1.1 模型；
- 官方 ContourCraft 数据包和 `contourcraft.pth`；
- 若严格对齐论文，需准备 BEDLAM 的 SMPL-X 动作。

其中 `SMPLX_NEUTRAL.pkl` 和 BEDLAM/AMASS 动作都受各自许可约束，不能由仓库匿名下载。`registered_params.pkl` 需要把 SMPL-X 静止人体对齐到 GarmentCode 使用的 A-pose 坐标系；动作 NPZ、人体注册参数和服装固定点必须配套使用。
