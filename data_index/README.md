# data_index

这里固定本实现使用的训练/验证 split。每个 `.txt` 文件一行一个 pair json 的相对路径,路径相对于 PairUAV 训练 json 根目录。

当前固定 split:

- `train_balanced_32768.txt`: 训练用 32,768 个 pair;
- `val_quick_2048.txt`: 验证用 2,048 个 pair。

这些 index 同时用于:

- 从完整 PairUAV train json 目录重建训练/验证 split;
- 校验已有 VGGT 特征 cache 的 `json_paths.json` 顺序;
- 保证后续训练和验证指标使用同一批、同一顺序的数据。

常用命令:

```bash
python -m pairuav.index verify-json \
  --index data_index/train_balanced_32768.txt \
  --json-dir "$PAIRUAV_TRAIN_JSON"

python -m pairuav.index materialize \
  --index data_index/train_balanced_32768.txt \
  --source-json-dir "$PAIRUAV_ALL_TRAIN_JSON" \
  --out-json-dir "$PAIRUAV_WORKSPACE/splits/train_balanced_32768"

python -m pairuav.index verify-cache \
  --index data_index/train_balanced_32768.txt \
  --cache-dir "$PAIRUAV_CACHE_ROOT/train_nfull_s518"
```

`manifest.json` 记录每个 index 的样本数、sha256 和首尾样本,用于快速确认版本。
