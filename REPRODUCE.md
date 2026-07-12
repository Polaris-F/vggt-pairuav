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
  --image-size 518 \
  --seed 2026
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
  --run-root "$PAIRUAV_RUN_ROOT/angle" \
  --seed 2026
```

最终连续预测使用该头的旋转 yaw 作为 heading。

## 6. 训练独立距离头

```bash
python -m pairuav.train_range \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --config configs/range_c_rel_rich.json \
  --output-dir "$PAIRUAV_RUN_ROOT/range" \
  --seed 2026
```

距离头直接从冻结 VGGT pair feature 回归 range,以适配官方距离相对误差口径。
该训练入口用验证集 heading 标签作为占位来隔离距离误差,模型选择以 `distance_rel_error` 为准。
公开的 C 配方使用 `lr=1e-3,epochs=240,batch=512`;历史提交对应的旧优化预算保存在
`configs/range_c_rel_rich_legacy.json`,但原始 seed/RNG 状态未保存,不能保证从头重现归档权重。

## 7. 连续 test 推理

在验证集上组合评估已有 checkpoint:

```bash
python -m pairuav.eval_val \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --val-geom "$PAIRUAV_RUN_ROOT/geometry_labels_val.npz" \
  --angle-ckpt "$PAIRUAV_RUN_ROOT/angle_YYYYMMDD_HHMMSS/S0_rich_noc/head_best_angle.pt" \
  --range-ckpt "$PAIRUAV_RUN_ROOT/range/C_rel_rich/range_head_best_distance.pt" \
  --range2-ckpt "$PAIRUAV_RUN_ROOT/range/B_mse_ab/range_head_best_distance.pt" \
  --out "$PAIRUAV_RUN_ROOT/val_combo_eval.json" \
  --seed 2026
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
  --pairs-cache "$PAIRUAV_RUN_ROOT/test_pairs_ordered.txt" \
  --seed 2026
```

## 8. 可选 MAP-hard 后处理

```bash
python -m pairuav.postproc_maphard \
  --pred "$PAIRUAV_RUN_ROOT/result_continuous.txt" \
  --out-dir "$PAIRUAV_RUN_ROOT/maphard"
```

该步骤是纯 NumPy 后处理,读取随包资源中的轨迹采样步长表和验证集拟合权重。

## 9. 双距离头门控(gate)

验证集分距离段消融显示:`B_mse_ab` 距离头(输入 `[a,b]`、MSE loss)仅在大距离段优于默认的
`C_rel_rich`(相对 smooth-L1),两头的交叉点约在 |range| = 80 m。门控方案以连续输出为底,
仅对基线距离 |range| > 80 m 的行换用 `B_mse_ab` 的距离预测,其余行保持不变;之后可照常接
MAP-hard。该步骤同样是纯 NumPy 后处理,不重新前向、不读测试标签。

训练第二距离头(与 §6 同一份 cache):

```bash
python -m pairuav.train_range \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_nfull_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_nfull_s518" \
  --config configs/range_b_mse_ab.json \
  --output-dir "$PAIRUAV_RUN_ROOT/range" \
  --seed 2026
```

双距离头 test 推理(在 §7 命令上追加 `--range2-*`,输出三列 `heading range range2`;
`--start/--end` 可按行区间做多卡/多机分片):

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
  --range2-run-dir "$PAIRUAV_RUN_ROOT/range/B_mse_ab" \
  --range2-ckpt "$PAIRUAV_RUN_ROOT/range/B_mse_ab/range_head_best_distance.pt" \
  --out "$PAIRUAV_RUN_ROOT/result_dual.txt" \
  --pairs-cache "$PAIRUAV_RUN_ROOT/test_pairs_ordered.txt" \
  --seed 2026
```

门控合成与 MAP-hard(分片时传多个 `--pred file:start`,start 为分片行起点):

```bash
python -m pairuav.gate_merge \
  --base-result "$PAIRUAV_RUN_ROOT/result_continuous.txt" \
  --pred "$PAIRUAV_RUN_ROOT/result_dual.txt:0" \
  --threshold 80 \
  --out "$PAIRUAV_RUN_ROOT/result_gate.txt" \
  --zip "$PAIRUAV_RUN_ROOT/result_gate.zip"

python -m pairuav.postproc_maphard \
  --pred "$PAIRUAV_RUN_ROOT/result_gate.txt" \
  --out-dir "$PAIRUAV_RUN_ROOT/gate_maphard"
```

## 10. 小样本冒烟

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
  --run-root "$PAIRUAV_RUN_ROOT/smoke_angle" \
  --seed 2026

python -m pairuav.train_range \
  --train-cache "$PAIRUAV_CACHE_ROOT/train_smoke_n512_s518" \
  --val-cache "$PAIRUAV_CACHE_ROOT/val_smoke_n128_s518" \
  --config configs/smoke_range.json \
  --output-dir "$PAIRUAV_RUN_ROOT/smoke_range" \
  --seed 2026
```

## 11. 已知指标

指标均为官方相对误差口径,数值越低越好。验证集结果使用 `data_index/val_quick_2048.txt` 固定的 2,048 个 pair;Codabench 结果来自对应 test 提交,二者数据口径不同,只用于复现锚定。

| 输出 | val angle_rel | val distance_rel | val final | 备注 |
| --- | ---: | ---: | ---: | --- |
| 连续输出 | 0.005263 | 0.011554 | 0.008407 | 冻结 VGGT + S0 6DoF 角度头 + C_rel_rich 距离头 |
| MAP-hard 后处理 | 0.004476 | 0.002096 | 0.003286 | 在连续输出上执行 D 空间 MAP-hard 解码;val D 命中率 91.85% |

| 输出 | Codabench angle_rel | Codabench distance_rel | Codabench final | Codabench ID |
| --- | ---: | ---: | ---: | --- |
| 连续输出 | 0.003170 | 0.015414 | 0.009292 | 811088 |
| MAP-hard 后处理 | 0.002325 | 0.002709 | 0.002517 | 811089 |

| 输出 | Codabench final | Codabench ID |
| --- | ---: | --- |
| 双距离头门控(gate)连续 | 0.009135 | 822840 |
| gate + MAP-hard 后处理 | 0.002402 | 822841 |

门控阈值 80 m 由验证集分距离段消融确定(两个距离头的交叉点)。822840 / 822841 两个归档提交已用
本仓库 `pairuav.gate_merge` + `pairuav.postproc_maphard` 对归档的分片推理结果做**字节级复现**
(`result.txt` md5 与归档一致)。
