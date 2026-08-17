# ChatGarment 项目集成接口

`ChatGarment/llava/garment_utils_v2.py` 提供任务级人体尺寸入口：

- 若环境变量 `CHATGARMENT_BODY_MEASUREMENT_PATH` 指向有效 YAML，`try_generate_garments()` 使用该任务人体数据；
- 未设置环境变量时，使用 `assets/bodies/mean_all.yaml`。

任务服务在每个任务中生成 `outputs/body_measurements/garmentcode_body.yaml`，然后以绝对路径设置该环境变量。因此用户身高、胸围、腰围及输入或估算的臀围会参与 GarmentCode 板片解码。

静态仿真包装器使用任务目录内的 `body_measurements.yaml` 保存和传递尺寸，碰撞网格使用标准人体 OBJ。定制 SMPL-X 人体用于动态 ContourCraft 仿真。数据关系如下：

- 2D 板片参数：任务 26 字段人体 YAML；
- 静态缝合与结果中的测量数据：任务 26 字段人体 YAML；
- 静态碰撞网格：标准人体 OBJ；
- 动态碰撞人体：本次选择的预设或定制 SMPL-X。

运行 `python3 scripts/bootstrap_sources.py --apply-overrides` 会将该文件安装到 `ChatGarment/llava/garment_utils_v2.py`。
