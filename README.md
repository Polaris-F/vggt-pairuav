# vggt-pairuav

这是 ACMMM 2026 UAV Workshop：PairUAV 任务 的 实现代码，主要基于 VGGT 结合低维流形假设来解决无人机绕目标螺旋下降逼近等条件下的先验嵌入建模。

## 目录结构

```text
.
├── 3rdparty/
│   └── vggt/        # 官方 VGGT submodule,固定在 a288dd0
├── pairuav/         # PairUAV 专用 Python 包
├── configs/         # submission / LaMP 两组权威配置
├── data_index/      # 固定训练/验证 split 的相对路径清单
├── scripts/         # 两条一键复现链路
├── artifacts/       # 网盘大文件的本地落盘约定
├── docs/            # 设计说明与实验记录
├── REPRODUCE.md
└── pyproject.toml
```

VGGT 源码保存在 `3rdparty/vggt` 中,原则上保持不修改。所有 PairUAV 相关逻辑都应放在 `pairuav/` 下。

`configs/paths.env` 中的 `VGGT_WEIGHT` 必须指向本地下载的官方 VGGT 模型权重文件。
官方权重下载页面：
https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt

训练、特征抽取和推理入口默认固定随机种子 `2026` 并记录复现设置。稳定性说明见
[`docs/reproducibility.md`](docs/reproducibility.md)。

## 复现口径

本项目明确区分两件事:

1. **复现比赛提交**:历史训练没有保存随机种子和 RNG 状态,因此精确复现依赖发布的
   三个任务头 checkpoint、完整 SHA256 清单和固定后处理。官方隐藏测试成绩为 `0.002402`。
2. **从头确定性重训论文方法**:当前代码默认 seed `2026`,角度头使用全 FP32 matmul,距离头使用
   TF32 `high`。论文方法使用一个 `[a,b] + rel_smooth` 距离头;在固定验证集上连续输出为
   `0.005907 +/- 0.000070`,加 MAP-hard 为 `0.003134 +/- 0.000119`(`n=3`)。

双距离头 + 80 m 门控入口仅为复现历史提交结构保留;多种子评测显示它相对单 C 头没有
显著增益。

## 目录职责

- `3rdparty/`: 第三方代码区。目前只包含官方 VGGT submodule,视为只读依赖。
- `pairuav/`: 我方实现区。数据读取、特征缓存、几何标签、任务头、指标、训练、推理和后处理都放这里。
- `configs/`: 可复现实验配置和路径模板。公开配置描述方法参数,实际路径由环境变量或命令行参数提供。
- `data_index/`: 固定训练/验证数据清单,用于重建 split 和校验特征 cache 顺序。
- `docs/`: 设计说明、实验记录和结果追踪等非执行内容放这里。
- `REPRODUCE.md`: 最终命令级复现流程。这里应保持简洁、可执行;细节和理由放到 `docs/`。

## 两条一键流程

比赛提交复现使用下载后的三个任务头和官方 test pair 的冻结特征 cache,不重复运行 VGGT:

```bash
bash scripts/reproduce_submission.sh
```

论文最终 LaMP 从固定 index 开始训练。默认使用 `32,768` 个训练 pair 和 `2,048` 个验证 pair;
传入其他 index 即可更换数据规模:

```bash
cp configs/paths.example.env configs/paths.env
# 编辑 configs/paths.env 后:
PAIRUAV_ENV_FILE=configs/paths.env bash scripts/train_lamp.sh
# bash scripts/train_lamp.sh /path/to/train_index.txt /path/to/val_index.txt
```

已有冻结特征时可设置 `PAIRUAV_TRAIN_CACHE` 和 `PAIRUAV_VAL_CACHE`,脚本会先按 index 校验顺序并跳过 VGGT。

## 环境准备

```bash
git submodule update --init --recursive
pip install -e 3rdparty/vggt
pip install -e .
```

## 结果(Codabench 隐藏测试集,官方相对误差口径,越低越好)

| 输出 | final | Codabench 提交 ID |
| --- | ---: | --- |
| 连续输出(冻结 VGGT + 角度头 + 距离头) | 0.009292 | 811088 |
| 连续 + MAP-hard 后处理 | 0.002517 | 811089 |
| 双距离头门控(gate)连续 | 0.009135 | 822840 |
| gate + MAP-hard 后处理 | **0.002402** | 822841 |

复现命令、已知验证集指标和字节级复现说明见 [REPRODUCE.md](REPRODUCE.md)。

## 自检

```bash
python -m unittest discover -s tests -v
python -m pairuav.index verify-manifest \
  --manifest data_index/manifest.json \
  --index-root data_index
```

完整 checkpoint、test feature cache 和历史提交文件体积较大,作为独立 release bundle 发布,不直接纳入
Git 历史。下载后的目录结构见 [artifacts/README.md](artifacts/README.md)。
