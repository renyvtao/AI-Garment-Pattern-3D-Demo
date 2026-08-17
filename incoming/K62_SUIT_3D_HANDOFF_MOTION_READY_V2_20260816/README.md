# K62 三维装配资产

该目录提供男西装 K62 三维装配所需的 Golden 规格、标准人体、GarmentCodeRC 运行覆盖文件、材质和动作接口。

项目级安装方式：

```bash
python3 scripts/bootstrap_sources.py --install-k62-overlay
```

安装脚本会从基础 `GarmentCodeRC/` 创建隔离的 `GarmentCodeRC_K62_3D/`，再写入以下内容：

- `03_RUNTIME_PATCH/` 中的网格、仿真和离屏渲染代码；
- `02_BODY/` 中的 `mean_all` 人体与分割数据。

`01_GOLDEN_BASE/` 是 `suit_finetune/build_suit_3d_spec.py` 使用的 K62 拓扑基准，`06_OFFICIAL_MATERIAL/` 提供材质和纹理，`07_MOTION_BRIDGE/` 定义动作人体序列接口。

第三方来源和许可证说明见 [THIRD_PARTY_AND_PROVENANCE.md](THIRD_PARTY_AND_PROVENANCE.md)。
