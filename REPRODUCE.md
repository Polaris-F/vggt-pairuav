# Reproduction Guide

This repository exposes two separate workflows. Their claims, configurations, and checkpoints must not be mixed.

1. **LaMP (challenge entry):** run the archived task-head checkpoints on the frozen test-pair feature cache, apply
   the historical 80 m gate, then apply pose-set decoding. This is Codabench submission `822841` (`0.002402`, 5th
   on the final leaderboard).
2. **LaMP (ours):** train or evaluate the paper's simplified geometry-supervised pose head and single `[a,b]`
   relative-error range head. Pose-set decoding is optional and is reported separately from the continuous method.

The historical challenge training did not preserve RNG state. Exact challenge reproduction therefore uses archived
checkpoints and SHA256 hashes; deterministic retraining demonstrates stability rather than checkpoint byte identity.

## 1. Environment

```bash
conda create -n lamp-release python=3.10 -y
conda activate lamp-release
git submodule update --init --recursive
pip install --upgrade pip
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

Obtain the image data from the official [PairUAV](https://github.com/YaxuanLi-cn/PairUAV) and
[University-1652](https://github.com/layumi/University1652-Baseline) releases. No image dataset is mirrored here.

## 2. Authoritative Configurations

| target | angle head | range head(s) | post-processing |
|---|---|---|---|
| LaMP (challenge entry) | `configs/submission/angle_s0.json` | `range_c_rel_rich.json` + `range_b_mse_ab.json` | 80 m gate + pose-set decoding |
| LaMP (ours) | `configs/lamp/angle_s0.json` | `range_ab_relsmooth.json` | optional pose-set decoding |

The paper range head consumes only `[a,b]`. The archived C head consumes `[a,b,a-b,a*b]`. Root-level config files
remain for compatibility with older commands, but the scripts below use only the grouped configurations.

## 3. Archived Submission, One Command

Extract the downloaded weight bundle under `artifacts/competition_submission/`, then place the separately distributed
frozen test-pair cache at `cache/test_pairs_s518/`. The complete layout is documented in `artifacts/README.md`; the
required files are:

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

The script verifies the weight bundle's `MANIFEST.sha256`, checks the cache row count, runs the three archived heads
directly on the cache, applies the historical gate, and writes:

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

The decoder weights in `pairuav/resources/p348_map_weights.json` are two scalar reliability weights fit on the
released validation protocol. The 211 candidate poses are enumerated from training labels and the public
University-1652 acquisition protocol. Neither component is derived from challenge test labels.

`pairuav.infer_test` remains available as a raw-image fallback, but it reruns VGGT and is not the default release
workflow. The archived submission zip can be verified without rerunning the 2.77M-pair backbone pass.

## 4. Train LaMP (ours), One Command

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

## 5. Evaluate the Released LaMP Seeds

Place `lamp_ours_3seed_valquick_v1` under `artifacts/lamp_ours_3seed/`. The bundle contains six task-head
checkpoints and a frozen validation cache:

```text
artifacts/lamp_ours_3seed/
├── MANIFEST.sha256
├── weights/
│   ├── seed2026/{angle/S0_rich_noc,range/R_ab_relsmooth}/...
│   ├── seed2027/{angle/S0_rich_noc,range/R_ab_relsmooth}/...
│   └── seed2028/{angle/S0_rich_noc,range/R_ab_relsmooth}/...
└── validation/val_quick_2048/
    ├── features.npy
    ├── heading.npy
    ├── range.npy
    ├── json_paths.json
    ├── meta.json
    └── geometry.npz
```

Run all three seeds:

```bash
bash scripts/evaluate_lamp_release.sh
```

The script checks every bundle hash and validates cache order against `data_index/val_quick_2048.txt` before
inference. Expected continuous mean +/- sample standard deviation is:

| metric | expected (`n=3`) |
|---|---:|
| heading MAE | `0.433578 +/- 0.008438 deg` |
| range MAE | `0.321625 +/- 0.022390 m` |
| official validation final | `0.005907 +/- 0.000070` |

## 6. Important Entry Points

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

## 7. Verified Anchors

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

## 8. Generalization Probe Bundles

Probe releases contain pooled frozen features, pair indexes, geometry/reference labels, and the scripts used to build
those derived files. They do not redistribute University-1652, PairUAV, Google Earth, or SUES image data.

| bundle | pairs | content |
|---|---:|---|
| `lamp_probe_on_manifold_v1` | `2048 x 3` | on-grid, off-frame, and off-grid trajectory tiers |
| `lamp_probe_off_manifold_slope_v1` | `936` | slopes `-4/-6/-10/-12`, three buildings, all within-tour pairs |
| `lamp_probe_sues200_v1` | `256` | 16 SUES-200 scenes x 16 pairs with independent COLMAP reference |

Each archive has its own `MANIFEST.sha256`. Archive-level SHA256 values are in `release/ARTIFACTS.sha256`; the
corresponding paper metrics are in Tables 3 and 4 and are mapped in `release/paper_artifact_map.json`.

## 9. Paper Table to Custody Map

Paths in the last column refer to the authors' complete experiment archive. Public users can reproduce the released
rows through the commands and bundles in the middle column.

| paper item | public command or bundle | internal custody source |
|---|---|---|
| Table 1, LaMP (challenge entry) | `scripts/reproduce_submission.sh`; bundle `lamp_challenge_entry_822841_v1` | `Eval_all/T11`, archived submission `822841` |
| Table 1, LaMP (ours) | `scripts/evaluate_lamp_release.sh`; bundle `lamp_ours_3seed_valquick_v1` | `Eval_all/T07` and `T41` |
| Table 1, official released baselines | metrics in release report | `T115_官方基线重训` |
| Table 1, other frozen-backbone baselines | custody predictions and shared evaluator | `baseline-custody-20260714` |
| Table 2, direct vs 6-DoF supervision | archived metrics | `Eval_all/T02`, `T03` |
| Table 2, relative-error range head | released LaMP weights | `Eval_all/T07`, `T41` |
| Table 2, pose-set decoding | fixed decoder resource | `Eval_all/T49` |
| Table 2, challenge gate variant | archived challenge topology | `Eval_all/T11`, `T12` |
| Table 3, trajectory tiers and discrete baseline | `lamp_probe_on_manifold_v1` | `T113_探针口径统一` / T107 scope |
| Table 4, altered slopes and SUES transfer | slope and SUES probe bundles | `T113_探针口径统一` / T111 and SUES scopes |

The same mapping is available in machine-readable form at `release/paper_artifact_map.json`.

## 10. Audit Boundary

- No image dataset or challenge test label is stored in Git history or a release bundle.
- Checkpoints, feature caches, predictions, and probe archives are covered by SHA256 manifests.
- Dataset-specific paths are supplied only through environment variables or CLI arguments.
- Existing outputs are never overwritten without an explicit flag.
- The source tree contains no credentials, proxy configuration, or machine-local `.env` file.
