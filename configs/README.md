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
- `range_c_rel_rich.json`: 独立距离头配置;
- `smoke_angle.json`: 小样本角度头冒烟配置;
- `smoke_range.json`: 小样本距离头冒烟配置。
