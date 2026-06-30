# pairuav

这里存放所有 PairUAV 专用代码。

当前按复现主链组织模块:

- `data.py`: PairUAV json/image 读取和确定性 pair 顺序;
- `cache.py`: 冻结特征 cache 的软链接和小样本切片工具;
- `features.py`: 冻结 VGGT 加载、aggregator 特征抽取和 cache 写入;
- `geometry.py`: 公开螺旋采集轨迹上的相对步长坐标、heading/range 投影和 6DoF 辅助标签;
- `index.py`: 固定训练/验证 split index,并校验 json 目录或特征 cache 的 pair 顺序;
- `metrics.py`: 官方指标与本地评估辅助;
- `heads.py`: 6DoF 角度头和独立距离 MLP;
- `train_angle.py`: S0 6DoF 角度头训练入口;
- `train_range.py`: C_rel_rich 独立距离头训练入口;
- `eval_val.py`: 验证 cache 上的角度头 + 距离头组合评估;
- `infer_test.py`: 连续 test 推理入口;
- `postproc_maphard.py`: MAP-hard 后处理入口;
- `resources/`: 小型固定资源,例如轨迹采样步长表和验证集拟合权重。

这里不修改 VGGT 源码。使用 VGGT 时,先在仓库根目录执行 `pip install -e 3rdparty/vggt`。
