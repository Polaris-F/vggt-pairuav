# Reproducibility

PairUAV 的 PyTorch 入口统一使用 `pairuav.reproducibility.seed_everything`。默认 seed 为
`2026`,命令行可用 `--seed` 覆盖。默认开启以下设置:

- Python、NumPy、CPU/CUDA PyTorch RNG 固定;
- DataLoader shuffle 和 worker 使用显式 generator/worker seed;
- `torch.set_float32_matmul_precision("high")`;
- cuDNN benchmark 关闭、deterministic 开启;
- PyTorch deterministic algorithms 以 `warn_only=True` 请求。

训练产物的 `result.json` 和角度头的 `summary.json` 记录完整复现设置。固定 seed 主要保证同一
软件栈和同类 GPU 上可重放;跨 CUDA、PyTorch 或 GPU 架构仍可能出现浮点舍入差异。

## C distance head

历史 C 配方 `lr=2e-3,epochs=120,batch=512` 对初始化敏感,且验证集最佳点贴近训练终点。
公开默认配置已改为 `lr=1e-3,epochs=240,batch=512`,模型结构、输入和 `rel_smooth` loss 不变。
旧参数保存在 `configs/range_c_rel_rich_legacy.json`,仅用于解释历史归档权重。

论文实验应显式记录 seed。完整数据尺度曲线先使用同一 seed 比较趋势,随后在关键尺度运行多个
seed 并报告 mean、sample standard deviation 和 worst run。
