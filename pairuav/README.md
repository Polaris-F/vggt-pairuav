# pairuav

这里存放所有 PairUAV 专用代码。

按复现主链组织模块:

- `data`: PairUAV json/image 读取和确定性 pair 顺序;
- `features`: 冻结 VGGT 加载和 aggregator 特征抽取;
- `geometry`: 相对轨迹步长 D、heading/range 和 6DoF 几何标签;
- `heads`: 任务头模块,例如 6DoF 角度头和距离 MLP;
- `metrics`: 官方指标与评估;
- `train_*`: 训练入口;
- `infer_*`: 推理入口;
- `postproc_*`: 后处理入口;
- `resources/`: 小型固定资源,例如合法 相对轨迹步长 D或验证集拟合权重。

这里不修改 VGGT 源码。使用 VGGT 时,先在仓库根目录执行 `pip install -e 3rdparty/vggt`。
