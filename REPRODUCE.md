# Reproduction Guide

This repository exposes two separate workflows. Their claims and assets must not be mixed.

1. **Archived competition submission:** run the released task-head checkpoints on the released frozen test-pair
   feature cache, then apply the historical 80 m gate and MAP-hard decoder. This reconstructs the topology of
   Codabench submission `822841` (`0.002402`, rank 5).
2. **Paper LaMP method:** start from fixed pair indexes, train with recorded seeds, and report validation statistics.
   The historical competition training did not preserve RNG state, so deterministic retraining is a stability claim,
   not a promise to regenerate the archived checkpoint byte-for-byte.

## 1. Environment

```bash
git submodule update --init --recursive
pip install -e 3rdparty/vggt
pip install -e .
```

The VGGT submodule is fixed at `a288dd0f14786c93483e45524328726ab7b1b4ce`. Run the CPU checks before a GPU job:

```bash
python -m unittest discover -s tests -v
python -m pairuav.index verify-manifest \
  --manifest data_index/manifest.json \
  --index-root data_index
```

## 2. Authoritative Configurations

| target | angle head | range head(s) | post-processing |
|---|---|---|---|
| archived submission | `configs/submission/angle_s0.json` | `range_c_rel_rich.json` + `range_b_mse_ab.json` | 80 m gate + MAP-hard |
| paper LaMP | `configs/lamp/angle_s0.json` | `range_ab_relsmooth.json` | optional MAP-hard |

The paper range head consumes only `[a,b]`. The archived C head consumes `[a,b,a-b,a*b]`. Root-level config files
remain for compatibility with older commands, but the scripts below use only the grouped configurations.

## 3. Archived Submission, One Command

Place the downloaded release bundle under `artifacts/competition_submission/`. The complete layout is documented in
`artifacts/README.md`; the required model-level files are:

```text
artifacts/competition_submission/
├── MANIFEST.sha256
├── checkpoints/
│   ├── S0_rich_noc/head_best_angle.pt
│   ├── C_rel_rich/range_head_best_distance.pt
│   └── B_mse_ab/range_head_best_distance.pt
├── cache/test_pairs_s518/
│   ├── features.npy       # (2773116, 2, 4096), official pair order
│   └── meta.json
└── submissions/
    ├── result_continuous.zip
    └── result_maphard.zip
```

Run:

```bash
bash scripts/reproduce_submission.sh
```

The script verifies `MANIFEST.sha256`, runs the three archived heads directly on the cache, applies the historical
gate, and writes:

```text
outputs/submission_<timestamp>/
├── raw_heads.txt                 # heading, C range, B range
├── result_continuous.txt         # heading, gated range
├── result_continuous.txt.meta.json
├── maphard/result.txt
├── maphard/result.zip
├── workflow.env
└── workflow.log
```

Custom locations can be passed positionally or through `PAIRUAV_SUBMISSION_ASSETS`, `PAIRUAV_SUBMISSION_CACHE`, and
`PAIRUAV_SUBMISSION_RUN_DIR`:

```bash
bash scripts/reproduce_submission.sh /path/to/assets /path/to/test_feature_cache
```

`pairuav.infer_test` remains available as a raw-image fallback, but it reruns VGGT and is not the default release
workflow.

## 4. Paper LaMP, One Command

Set the dataset and VGGT paths. `configs/paths.example.env` is a template:

```bash
export PAIRUAV_ALL_TRAIN_JSON=/path/to/all/train/pair_json
export PAIRUAV_TRAIN_IMAGES=/path/to/train/images
export VGGT_WEIGHT=/path/to/vggt/model.pt
```

The default command uses the released 32K/2K indexes:

```bash
bash scripts/train_lamp.sh
```

Equivalently, pass other index files to change the data scale without changing code:

```bash
bash scripts/train_lamp.sh \
  /path/to/train_index.txt \
  /path/to/val_index.txt \
  /path/to/new_run_directory
```

The workflow performs, in order:

1. materialize the indexed pair JSON splits;
2. jointly encode each pair with the frozen VGGT and cache pooled features;
3. verify cache order against both indexes;
4. reconstruct the closed-form 6-DoF labels;
5. train the 6-DoF pose head with FP32 matmul precision `highest`;
6. train one `[a,b] + rel_smooth` range head with precision `high`;
7. evaluate continuous and MAP-hard outputs on the validation cache.

Outputs are self-contained under `outputs/lamp_seed<seed>_<timestamp>/`, including caches, geometry labels,
checkpoints, `metrics.json`, `workflow.env`, and `workflow.log`.

### Reuse an existing VGGT cache

To avoid rerunning VGGT, provide both caches. The script still uses the supplied indexes to verify count and pair
order before training:

```bash
export PAIRUAV_TRAIN_CACHE=/path/to/train_balanced_32768_cache
export PAIRUAV_VAL_CACHE=/path/to/val_quick_2048_cache
bash scripts/train_lamp.sh
```

Setting only one cache is rejected. A cache with a different pair order is also rejected.

## 5. Important Entry Points

- `python -m pairuav.features`: frozen joint VGGT feature extraction;
- `python -m pairuav.geometry`: closed-form 6-DoF label generation;
- `python -m pairuav.train_angle`: pose-head training;
- `python -m pairuav.train_range`: range-head training;
- `python -m pairuav.eval_val`: continuous and MAP-hard validation metrics;
- `python -m pairuav.infer_cache`: task-head inference from an existing feature cache;
- `python -m pairuav.infer_test`: raw-image test inference;
- `python -m pairuav.postproc_maphard`: pose-set decoding.

All training commands require an explicit config and refuse nonempty output directories. Existing caches and outputs
are not overwritten implicitly.

## 6. Verified Anchors

The archived hidden-test submission is `0.009135` before MAP-hard and `0.002402` after MAP-hard. Hidden labels are
not available locally.

The validation values below all use `data_index/val_quick_2048.txt`:

| system | seeds | continuous final | MAP-hard final |
|---|---:|---:|---:|
| archived submission weights, C/B gate | archived weights | `0.008060` | `0.003056` |
| historical topology, deterministic retraining | 5 | `0.006967 +/- 0.000166` | `0.003230 +/- 0.000139` |
| paper LaMP, one `[a,b]` range head | 3 | `0.005907 +/- 0.000070` | `0.003134 +/- 0.000119` |

The second range head and 80 m gate are retained solely to reproduce the competition entry. Multi-seed analysis did
not show a reliable gate benefit, and the paper method removes both.
