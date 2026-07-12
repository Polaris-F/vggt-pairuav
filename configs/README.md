# configs

这里存放可复用的配置文件和路径模板。

配置约定:

- 公开配置只描述方法参数和所需路径变量;
- 数据集路径、权重路径和输出目录由环境变量或命令行参数提供;
- cache、预测结果、训练输出等运行产物默认写入 git 忽略目录;
- 配置文件应能被 `REPRODUCE.md` 中的命令直接引用。

当前文件:

- `paths.example.env`: 所需环境变量清单;
- `angle_s0.json`: S0 角度头配置;
- `range_c_rel_rich.json`:稳定 C 距离头配置(`lr=1e-3,epochs=240,batch=512`);
- `range_c_rel_rich_legacy.json`:历史提交的旧优化预算,仅用于审计;
- `range_b_mse_ab.json`: B 距离头配置;
- `smoke_angle.json`: 小样本角度头冒烟配置;
- `smoke_range.json`: 小样本距离头冒烟配置。

所有 PyTorch 入口默认使用固定随机种子 `2026`,也可通过 `--seed` 显式覆盖。训练产物的
`result.json` 会记录 seed、确定性算法和数值精度设置。
