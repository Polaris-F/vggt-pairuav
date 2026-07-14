# pairuav

这里存放所有 PairUAV 专用代码。

当前按复现主链组织模块:

- `data.py`: PairUAV json/image 读取和确定性 pair 顺序;
- `cache.py`: 冻结特征 cache 的软链接和小样本切片工具;
- `features.py`: 冻结 VGGT 加载、aggregator 特征抽取和 cache 写入;不自动覆盖已有或不完整 cache;
- `geometry.py`: 公开螺旋采集轨迹上的相对步长坐标、heading/range 投影和 6DoF 辅助标签;
- `index.py`: 固定训练/验证 split index,并校验 json 目录或特征 cache 的 pair 顺序;
- `metrics.py`: 官方指标与本地评估辅助;
- `heads.py`: 6DoF 角度头和独立距离 MLP;
- `head_io.py`:统一读取任务头 config/result 并加载 checkpoint,供验证、原图推理和 cache 推理复用;
- `train_angle.py`: S0 6DoF 角度头训练入口;
- `train_range.py`: 独立距离头训练入口;必须显式选择配置,逐 batch 构造 pair 输入并自动选择
  fp16 特征存放设备,避免大数据量时常驻展开特征;
- `eval_val.py`: 验证 cache 上的角度头 + 距离头组合评估;
- `infer_test.py`: 连续 test 推理入口;
- `infer_cache.py`:直接在已归档的冻结 pair feature cache 上运行一个或两个任务头;
- `gate_merge.py`: 历史双距离头提交结构的门控合成,默认要求分片完整覆盖;
- `postproc_maphard.py`: MAP-hard 后处理入口;
- `reproducibility.py`: 统一 seed、数值精度和确定性设置;
- `resources/`: 小型固定资源,例如轨迹采样步长表和验证集拟合权重。

这里不修改 VGGT 源码。使用 VGGT 时,先在仓库根目录执行 `pip install -e 3rdparty/vggt`。
核心自检使用 `python -m unittest discover -s tests -v`。
