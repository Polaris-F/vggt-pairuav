# configs

这里存放可复用的配置文件和路径模板。

配置约定:

- 公开配置只描述方法参数和所需路径变量;
- 数据集路径、权重路径和输出目录由环境变量或命令行参数提供;
- cache、预测结果、训练输出等运行产物默认写入 git 忽略目录;
- 配置文件应能被 `REPRODUCE.md` 中的命令直接引用。
- 训练入口必须显式传入 `--config`;Python dataclass 中的默认值不是实验配方来源。

权威配置分为两组:

- `paths.example.env`: 所需环境变量清单;
- `submission/`:比赛提交 `822841` 的原始三个头配方,即 S0 + C/B + 80 m gate + MAP-hard;
- `lamp/`:论文最终方法,S0 6-DoF pose head + 单个 `[a,b] + rel_smooth` range head;
- `smoke_angle.json`: 小样本角度头冒烟配置;
- `smoke_range.json`:旧 rich 距离头冒烟配置;
- `smoke_range_ab.json`:论文 `[a,b]` 距离头冒烟配置。

根目录的 `angle_s0.json`、`range_c_rel_rich.json` 和 `range_b_mse_ab.json` 仅为旧命令兼容保留。新的
复现脚本只引用分组目录,避免把历史 C 头、稳定化 C 头和论文最终 `[a,b]` 头混为同一个系统。

所有 PyTorch 入口默认使用固定随机种子 `2026`,也可通过 `--seed` 显式覆盖。训练产物的
`result.json` 会记录 seed、确定性算法和数值精度设置。历史提交训练没有固定 seed,因此
`submission/` 配方用于解释归档权重结构,不构成可字节级重训权重的承诺。

距离训练逐 batch 构造 pair 输入,不会把整份 cache 预展开为 16,384 维。`--feature-device auto`
在显存足够且保留 2 GiB 余量时把 fp16 特征常驻 GPU,否则保留在 CPU;两种路径都只使用单进程、
单卡训练。已有非空 run 目录会被拒绝,避免不同配方相互覆盖。
