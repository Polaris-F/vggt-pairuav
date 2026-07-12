# vggt-pairuav

这是 ACMMM 2026 UAV Workshop：PairUAV 任务 的 实现代码，主要基于 VGGT 结合低维流形假设来解决无人机绕目标螺旋下降逼近等条件下的先验嵌入建模。

## 目录结构

```text
.
├── 3rdparty/
│   └── vggt/        # 官方 VGGT submodule,固定在 a288dd0
├── pairuav/         # PairUAV 专用 Python 包
├── configs/         # 配置与路径模板
├── data_index/      # 固定训练/验证 split 的相对路径清单
├── docs/            # 设计说明与实验记录
├── REPRODUCE.md
└── pyproject.toml
```

VGGT 源码保存在 `3rdparty/vggt` 中,原则上保持不修改。所有 PairUAV 相关逻辑都应放在 `pairuav/` 下。

训练、特征抽取和推理入口默认固定随机种子 `20260712` 并记录复现设置。稳定性说明见
[`docs/reproducibility.md`](docs/reproducibility.md)。

## 目录职责

- `3rdparty/`: 第三方代码区。目前只包含官方 VGGT submodule,视为只读依赖。
- `pairuav/`: 我方实现区。数据读取、特征缓存、几何标签、任务头、指标、训练、推理和后处理都放这里。
- `configs/`: 可复现实验配置和路径模板。公开配置描述方法参数,实际路径由环境变量或命令行参数提供。
- `data_index/`: 固定训练/验证数据清单,用于重建 split 和校验特征 cache 顺序。
- `docs/`: 设计说明、实验记录和结果追踪等非执行内容放这里。
- `REPRODUCE.md`: 最终命令级复现流程。这里应保持简洁、可执行;细节和理由放到 `docs/`。

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

复现命令见 [REPRODUCE.md](REPRODUCE.md);已知验证集指标与字节级复现说明见其 §11。
