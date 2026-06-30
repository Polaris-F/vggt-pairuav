# 复现指南

本文档是 PairUAV-on-VGGT 实现的命令级复现入口。

## 1. 环境准备

```bash
git submodule update --init --recursive
pip install -e 3rdparty/vggt
pip install -e .
```

路径变量可参考 `configs/paths.example.env` 设置。

## 2. 固定训练/验证 split

```bash
python -m pairuav.index materialize \
  --index data_index/train_balanced_32768.txt \
  --source-json-dir "$PAIRUAV_ALL_TRAIN_JSON" \
  --out-json-dir "$PAIRUAV_TRAIN_JSON"

python -m pairuav.index materialize \
  --index data_index/val_quick_2048.txt \
  --source-json-dir "$PAIRUAV_ALL_TRAIN_JSON" \
  --out-json-dir "$PAIRUAV_VAL_JSON"
```

如果直接复用已抽取的特征 cache,先校验 cache 顺序:

```bash
python -m pairuav.index verify-cache \
  --index data_index/train_balanced_32768.txt \
  --cache-dir "$PAIRUAV_CACHE_ROOT/train_nfull_s518"

python -m pairuav.index verify-cache \
  --index data_index/val_quick_2048.txt \
  --cache-dir "$PAIRUAV_CACHE_ROOT/val_nfull_s518"
```

## 3. 抽取冻结 VGGT 特征

```bash
python -m pairuav.features \
  --train-json-dir "$PAIRUAV_TRAIN_JSON" \
  --val-json-dir "$PAIRUAV_VAL_JSON" \
  --image-dir "$PAIRUAV_TRAIN_IMAGES" \
  --vggt-weight "$VGGT_WEIGHT" \
  --cache-root "$PAIRUAV_CACHE_ROOT" \
  --image-size 518
```

该步骤只调用冻结 VGGT aggregator,并写出 `features.npy`、`heading.npy`、`range.npy`、`json_paths.json` 和 `meta.json`。

## 4. 生成 6DoF 辅助标签

```bash
python -m pairuav.geometry \
  --cache-dir "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --out "$PAIRUAV_RUN_ROOT/geometry_labels_train.npz"

python -m pairuav.geometry \
  --cache-dir "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --out "$PAIRUAV_RUN_ROOT/geometry_labels_val.npz"
```

本实现依据 University-1652 公开的螺旋采集过程,定义一个规范的 object-centric 轨迹参数化;给定配对帧索引,即可闭式生成用于训练的 6DoF 辅助标签。

## 5. 训练 S0 角度头

```bash
python -m pairuav.train_angle \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --train-geom "$PAIRUAV_RUN_ROOT/geometry_labels_train.npz" \
  --val-geom "$PAIRUAV_RUN_ROOT/geometry_labels_val.npz" \
  --config configs/angle_s0.json \
  --run-root "$PAIRUAV_RUN_ROOT/angle"
```

最终连续预测使用该头的旋转 yaw 作为 heading。

## 6. 训练独立距离头

```bash
python -m pairuav.train_range \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --config configs/range_c_rel_rich.json \
  --output-dir "$PAIRUAV_RUN_ROOT/range"
```

距离头直接从冻结 VGGT pair feature 回归 range,以适配官方距离相对误差口径。
该训练入口用验证集 heading 标签作为占位来隔离距离误差,模型选择以 `distance_rel_error` 为准。

## 7. 连续 test 推理

在验证集上组合评估已有 checkpoint:

```bash
python -m pairuav.eval_val \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --val-geom "$PAIRUAV_RUN_ROOT/geometry_labels_val.npz" \
  --angle-run-dir "$PAIRUAV_RUN_ROOT/angle_YYYYMMDD_HHMMSS/S0_rich_noc" \
  --angle-ckpt "$PAIRUAV_RUN_ROOT/angle_YYYYMMDD_HHMMSS/S0_rich_noc/head_best_angle.pt" \
  --range-run-dir "$PAIRUAV_RUN_ROOT/range/C_rel_rich" \
  --range-ckpt "$PAIRUAV_RUN_ROOT/range/C_rel_rich/range_head_best_distance.pt" \
  --out "$PAIRUAV_RUN_ROOT/val_combo_eval.json"
```

```bash
python -m pairuav.infer_test \
  --test-json-dir "$PAIRUAV_TEST_JSON" \
  --test-image-dir "$PAIRUAV_TEST_IMAGES" \
  --vggt-weight "$VGGT_WEIGHT" \
  --angle-run-root "$PAIRUAV_RUN_ROOT/angle_YYYYMMDD_HHMMSS" \
  --angle-name S0_rich_noc \
  --angle-ckpt head_best_angle.pt \
  --range-run-dir "$PAIRUAV_RUN_ROOT/range/C_rel_rich" \
  --range-ckpt "$PAIRUAV_RUN_ROOT/range/C_rel_rich/range_head_best_distance.pt" \
  --out "$PAIRUAV_RUN_ROOT/result_continuous.txt" \
  --pairs-cache "$PAIRUAV_RUN_ROOT/test_pairs_ordered.txt"
```

## 8. 可选 MAP-hard 后处理

```bash
python -m pairuav.postproc_maphard \
  --pred "$PAIRUAV_RUN_ROOT/result_continuous.txt" \
  --out-dir "$PAIRUAV_RUN_ROOT/maphard"
```

该步骤是纯 NumPy 后处理,读取随包资源中的轨迹采样步长表和验证集拟合权重。

## 9. 小样本冒烟

如果已有完整特征 cache,可以先切出小样本验证代码链路:

```bash
python -m pairuav.cache slice \
  --cache-dir "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --out-dir "$PAIRUAV_CACHE_ROOT/train_smoke_n512_s518" \
  --limit 512

python -m pairuav.cache slice \
  --cache-dir "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --out-dir "$PAIRUAV_CACHE_ROOT/val_smoke_n128_s518" \
  --limit 128

python -m pairuav.geometry \
  --cache-dir "$PAIRUAV_CACHE_ROOT/train_smoke_n512_s518" \
  --out "$PAIRUAV_RUN_ROOT/geometry_labels_train_smoke.npz"

python -m pairuav.geometry \
  --cache-dir "$PAIRUAV_CACHE_ROOT/val_smoke_n128_s518" \
  --out "$PAIRUAV_RUN_ROOT/geometry_labels_val_smoke.npz"

python -m pairuav.train_angle \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_smoke_n512_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_smoke_n128_s518" \
  --train-geom "$PAIRUAV_RUN_ROOT/geometry_labels_train_smoke.npz" \
  --val-geom "$PAIRUAV_RUN_ROOT/geometry_labels_val_smoke.npz" \
  --config configs/smoke_angle.json \
  --run-root "$PAIRUAV_RUN_ROOT/smoke_angle"

python -m pairuav.train_range \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_smoke_n512_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_smoke_n128_s518" \
  --config configs/smoke_range.json \
  --output-dir "$PAIRUAV_RUN_ROOT/smoke_range"
```
