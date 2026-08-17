# K62 人物运动接口

该目录定义 K62 服装与逐帧人体网格之间的数据接口。动态服装由项目的 ContourCraft-CG 模块处理，K62 资产负责提供版片拓扑和静态初始垂坠网格。

## 人体序列约定

输入目录按帧保存人体 OBJ：

```text
body_000000.obj
body_000001.obj
body_000002.obj
...
```

所有帧必须满足：

- 顶点数量、顶点顺序和三角面拓扑不变；
- 单位和坐标系与服装输入一致；
- 初始帧与 GarmentCode 使用的人体注册姿态对齐；
- 不包含 NaN 或无穷值。

校验命令：

```bash
python 07_MOTION_BRIDGE/validate_body_mesh_sequence.py \
  --input-dir /path/to/body-sequence
```

## 数据流

```text
K62 规格与初始垂坠网格
+ 固定拓扑人体序列
-> ContourCraft-CG 输入准备
-> 动态服装顶点序列
-> Blender 渲染
```

`experimental_dynamic_body_hook.py` 提供 Warp moving-collider 的研究入口。若沿该方向扩展，需要同时更新 whole-body collider 与 cloth-reference 分区网格，并保持人体拓扑恒定。

## 上游参考

- ChatGarment: https://github.com/biansy000/ChatGarment
- ContourCraft: https://github.com/dolorousrtur/ContourCraft
