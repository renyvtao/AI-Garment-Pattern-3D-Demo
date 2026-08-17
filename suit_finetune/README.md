# 男西装训练与三维适配

该目录包含男西装数据集、LoRA 训练与推理、输出标准化和 K62 三维装配工具。西装 LoRA 以 ChatGarment 基础任务权重为起点，只保存可训练的 LoRA 参数，不覆盖基础 checkpoint。

当前参数约定：

- 用户在网页中明确选择“男西装”；
- `button_count` 夹到 `[1, 2]`；
- `body_panel_layout` 固定为 `six_panel`；
- `front_lower_edge_style` 固定为 `curved`；
- `button_spacing_cm` 固定为 `9.0`；
- 其他工程字段由 `MensSuitJacketCleanFinal` 标准 YAML 补齐；
- 训练只更新官方结构中的 LoRA 参数。

数据集按基础西装 ID 分组后进行 80/10/10 划分，避免同一款的增强图同时
进入训练集和验证集。

## 上衣与下装分支

男西装任务现采用串行双分支，不把西装 LoRA 强行用于下装：

```text
同一批输入图片
├─ 西装上衣：官方基础权重 + 西装 LoRA → 西装参数与 K62 上衣
└─ 下装：官方基础权重（不加载西装 LoRA）→ lower 参数、板片与独立静态预览
```

官方分支生成的上装结果会被丢弃，只保留 `lower`。若图片没有可识别下装，
任务记录显示“官方下装 · 未检测到”，但不会阻止西装上衣继续完成。检测到
下装时，制版和静态仿真统一使用 K62 的 `mean_all` 基准人体；网页同时展示
上衣、下装和上下装同穿的静态正反面结果。动态阶段将下装网格与 K62 上衣
合并为同一个 ContourCraft 输入，因此最终视频同时包含上装和下装。

## K62 三维装配

K62 包位于 `incoming/K62_SUIT_3D_HANDOFF_MOTION_READY_V2_20260816/`。运行 `scripts/bootstrap_sources.py --install-k62-overlay` 后，覆盖文件会安装到隔离的 `GarmentCodeRC_K62_3D/`，不会修改基础 `GarmentCodeRC/` checkout。

固定 K62 规格包含 11 个实体面片、66 条基础缝合和三维摆放。`build_suit_3d_spec.py` 保留这套拓扑、标签和摆放，根据 LoRA 输出的六类面片尺寸缩放对应 K62 面片，并按 `predicted.button_count` 连接一个或两个虚拟扣位。下装使用同一 `mean_all` 基准人体进行制版和静态仿真，组合渲染阶段再把上下装同时载入场景。

## 数据集

```text
prepared_data/
├── images/
├── train.json
├── validation.json
├── test.json
└── manifest.json
```

数据按基础款 ID 分组划分，再把同一基础款的增强图片放入同一数据子集。`manifest.json` 记录样本数、基础款数和字段统计。

## 训练

在项目根目录执行：

```bash
export AI_GARMENT_PROJECT_ROOT="$PWD"
export SUIT_DATA_ROOT="$PWD/suit_finetune/prepared_data"
bash suit_finetune/run_suit_poc_train.sh
```

默认输出目录为 `ChatGarment/checkpoints/suit_poc_lora_v1/`。可通过 `SUIT_EPOCHS`、`SUIT_STEPS_PER_EPOCH`、`SUIT_GRAD_ACCUMULATION_STEPS` 和 `SUIT_LEARNING_RATE` 调整训练参数。

## 代码入口

- `prepare_suit_poc_dataset.py`：把授权数据转换为训练、验证和测试 JSON；
- `train_garmentcode_outfit_suit_poc.py`：西装 LoRA 训练器；
- `run_suit_poc_inference.py`：加载基础 checkpoint 与西装 LoRA 进行图片推理；
- `suit_output_adapter.py`：把模型文本转换为标准西装设计 YAML；
- `build_suit_3d_spec.py`：把参数化西装输出映射到 K62 三维装配规格；
- `inspect_spec_topology.py`：检查面片、边界与缝合拓扑。

完整应用中的输出由 `pipeline/app_service.py` 组织到 `app_data/runs/<job-id>/`，并生成任务清单与可下载结果包。
