# 男西装 LoRA 评测

## ChatGarment 原论文指标与数值

[ChatGarment 原论文](https://arxiv.org/abs/2412.17811)的图像服装重建实验使用
CLoSE 与 Dress4D：CLoSE 取 145 个带准确 SMPL-X 拟合的扫描，Dress4D 取 4 套
宽松服装、36 张渲染图。预测服装先通过 Linear Blend Skinning 放到真值人体和
姿态，再进行下游几何评测。

| 指标 | 定义 | 评测意义 |
|---|---|---|
| mean Chamfer Distance（CD，越低越好） | 计算预测服装表面点到真值表面最近点的双向平均距离 | 衡量最终三维服装几何与真值的接近程度；会同时受到模型预测、参数化解码、人体对齐和三维处理影响。 |
| mean F-Score（越高越好） | 在论文阈值 `τ = 0.001` 下，由表面点精确率和召回率计算调和平均 | 衡量预测表面在容差范围内的覆盖度和准确度，比单一平均距离更能反映缺失或多余区域。 |
| stitching failure rate（越低越好） | 预测板片无法形成有效缝合服装的样本比例 | 衡量输出拓扑和缝合关系的有效性；它属于端到端制版通路指标，不是 JSON 字段值准确率。 |
| CLIP Score（越高越好） | 计算文本提示与生成服装渲染图的 CLIP 语义相似度 | 衡量文本生成服装是否符合提示；结果依赖渲染图，因此不作为本次纯模型评测。 |

论文中 ChatGarment 的原始结果如下：

| 任务与测试集 | CD | F-Score | 缝合失败率 / CLIP Score |
|---|---:|---:|---:|
| Dress4D 图像重建，Target-Pose | 3.12 | 0.75 | 0% |
| Dress4D 图像重建，A-Pose | 3.06 | 0.78 | 0% |
| CLoSE 图像重建 | 2.94 | 0.790 | 0% |
| 135 对服装编辑样本 | 2.51 | 0.893 | 论文表中未报告失败率 |
| 150 条文本生成提示 | 不适用 | 不适用 | CLIP Score 23.7 |

论文没有报告 JSON 解析率、字段完整率、参数 MAE/RMSE、字段准确率、平衡准确率
或逐类别召回率。因此本项目不能把论文 CD/F-Score 与西装六字段准确率换算成同一
排名。纯模型结论仍以六字段留出集 A/B 为主；渲染后 CLIP Score 不纳入本次图片
识别任务。论文同名的缝合失败率已经单独执行，CD/F-Score 则因缺少西装三维真值
暂时不能计算。

## 西装 LoRA 留出集 A/B

当前西装数据按基础款分组后划分，测试集包含 72 个基础款、216 张图片。脚本在
同一测试集、同一提示词和解码参数下分别运行：

1. ChatGarment 官方基础任务权重；
2. 官方基础任务权重 + 男西装 LoRA。

A/B 默认统一使用 `max_new_tokens=1024`，避免官方基础模型输出完整通用
GarmentCode 配置时因 256-token 截断而被错误计为解析失败；可通过
`SUIT_EVAL_MAX_NEW_TOKENS` 覆盖。

评估六个真正参与 LoRA 监督的字段：

- 连续：`garment_length_ratio`、`waist_ease_cm`；
- 离散：`lapel_style`、`button_count`、`small_pocket_enabled`、
  `large_pockets_enabled`。

连续字段同时报告 MAE、RMSE 和容差命中率。默认容差为长度比例 `0.01`、腰围
松量 `0.5 cm`。离散字段采用严格准确率；另外报告解析成功率、原始字段覆盖率、
六字段宏平均、平衡准确率、各类别召回率、多数类常量基线、全字段同时通过率、
参数修正规则触发率和二维板片生成成功率。由于当前测试标签不均衡，判断是否真正
学到图像条件时，应优先看平衡准确率及相对多数类基线的提升。

### 本次指标定义

| 指标 | 计算方式 | 主要回答的问题 |
|---|---|---|
| 生成成功率 | 产生非空模型输出的样本数 / 全部样本数 | 推理是否稳定完成？ |
| 目标结构解析成功率 | 能解析成西装目标结构的样本数 / 全部样本数 | 输出协议是否符合适配器要求？ |
| 六字段完整率 | 六字段全部存在且类型合法的样本数 / 全部样本数 | 输出结构是否完整？ |
| 连续字段容差命中率 | 绝对误差不超过指定容差的样本比例 | 数值是否精确到当前制版规则可接受范围？ |
| 离散字段准确率 | 预测值严格等于真值的样本比例 | 驳领、纽扣和口袋属性是否识别正确？ |
| 常规六字段宏平均 | 六个字段各自通过率的算术平均 | 总体字段表现如何？标签不均衡时会偏乐观。 |
| 平衡六字段宏平均 | 每字段各取值召回率等权平均，再对六字段平均 | 是否同时识别多数类和少数类？这是当前主要识别指标。 |
| 全字段同时通过率 | 每张图六字段全部命中的比例 | 有多少图片可以不修正参数直接进入下游？ |
| 多数类常量基线 | 每字段固定预测测试集最常见值 | 模型是否真的利用图像，而非只复现数据分布？ |
| MAE / RMSE | 连续字段绝对误差均值 / 平方误差均方根 | 平均误差多大，是否存在少数严重错误？ |
| 类别召回率 | 某真实取值被正确识别的比例 | 哪些少数类别被模型忽略？ |
| 板片生成成功率 | 经参数适配后成功生成二维板片的比例 | 输出能否被工程通路消费？不判断板片是否与真值一致。 |

服务器一键执行：

```bash
cd /path/to/AI_Garment_Technology_Showcase
export AI_GARMENT_PROJECT_ROOT="$PWD"
bash suit_finetune/run_suit_evaluation.sh
```

默认使用全部 216 张测试图片。快速冒烟测试可设置：

```bash
SUIT_EVAL_LIMIT=12 SUIT_EVAL_SKIP_PATTERN=1 \
  bash suit_finetune/run_suit_evaluation.sh
```

结果位于 `evaluation/suit_lora/report/`：

- `evaluation_report.md`：可直接汇报的表格；
- `evaluation_summary.json`：结构化汇总；
- `evaluation_cases.csv`：逐图片误差与命中情况。

## 论文同名缝合失败率

本次把 216 个 LoRA 预测逐例经过参数适配、K62 网格构建、Warp 缝合/垂坠和最终
服装网格导出。216 个样本全部完成，缝合失败率为 `0 / 216 = 0.00%`。渲染图和
视频只作为现有流程的完成性产物，不参与计分。

```bash
python suit_finetune/run_suit_stitching_evaluation.py \
  --project-root "$PWD" \
  --inference-root evaluation/suit_lora/suit_lora \
  --output-dir evaluation/suit_lora/paper_aligned_stitching \
  --run-simulation --skip-completed
```

这与论文是同名、同方向的有效性指标，但不是同测试集上的严格横向排名：论文使用
Dress4D/CLoSE，本项目使用西装留出集；本项目还把预测参数迁移到固定的 K62 11 片、
66 条基础缝合拓扑，因此该指标对尺寸或类别预测错误不敏感。

## CD 与 F-Score 还缺什么

当前测试集只有图片和六字段标签，没有逐样本三维服装真值。要得到可信的西装
CD/F-Score，至少还需要：

1. 每张测试图片对应的真实西装三维表面网格或高质量扫描；
2. 对应人体的准确 SMPL-X 注册，以及 Target-Pose/A-Pose 对齐结果；
3. 预测与真值一致的坐标系、长度单位和服装表面分割；
4. 固定表面采样数、随机种子和聚合方式。

论文给出了双向 CD、F-Score 和平方距离阈值 `τ=0.001`，但公开 ChatGarment 仓库
没有提供可直接运行的完整基准数据与端到端评测入口。仓库中的
`evaluate_paper_mesh_metrics.py` 已实现公式；它不会自动缩放或配准，以免产生
看似合理但不可比较的结果。脚本默认优先使用 PyTorch3D 的
`sample_points_from_meshes` 与 `chamfer_distance`；没有 PyTorch3D 时才回退到
等价的面积加权三角形采样和 SciPy 双向最近邻。

论文正文和公开仓库未给出完整的表面采样点数配置；命令中的 10,000 点是本项目
固定的可复验设置，不宣称是作者未公开的精确采样数。若后续获得官方评测配置，
应同时固定该数值后再做横向比较。

与项目的 PyTorch 2.1.2 + CUDA 11.8 环境匹配的安装方式：

```bash
python -m pip install fvcore iopath
python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"
```

```bash
python suit_finetune/evaluate_paper_mesh_metrics.py \
  --manifest mesh_pairs.json \
  --output-dir evaluation/suit_lora/mesh_metrics \
  --sample-count 10000 \
  --fscore-threshold 0.001 \
  --backend pytorch3d
```

完整结论见 [`RESULTS.md`](RESULTS.md)，机器可读摘要见
[`results/suit_lora_v1_summary.json`](results/suit_lora_v1_summary.json)。
