# vggt-pairuav

[中文介绍](README_zh.md) · [Reproduction](REPRODUCTION.md) 

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Task](https://img.shields.io/badge/Task-PairUAV-blue?style=flat-square)

This is the implementation code for the PairUAV task in the ACMMM 2026 UAV Workshop. It is mainly based on VGGT and combines a low-dimensional manifold assumption to model prior embeddings under conditions such as UAV spiral descent and approach around a target.

## Repository Structure

```text
.
├── 3rdparty/
│   └── vggt/        # Official VGGT submodule, pinned to a288dd0
├── pairuav/         # PairUAV-specific Python package
├── configs/         # Two authoritative configuration groups: submission / LaMP
├── data_index/      # Relative-path lists for fixed training/validation splits
├── scripts/         # Two one-command reproduction workflows
├── artifacts/       # Local layout convention for large files distributed via cloud storage
├── docs/            # Design notes and experiment records
├── REPRODUCE.md
└── pyproject.toml
```

The VGGT source code is stored in `3rdparty/vggt` and should generally remain unmodified. All PairUAV-related logic should be placed under `pairuav/`.

`VGGT_WEIGHT` in `configs/paths.env` must point to the locally downloaded official VGGT model weight file.
Official weight download page:
https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt

The training, feature extraction, and inference entry points use the fixed random seed `2026` by default and record the reproducibility settings. See
[`docs/reproducibility.md`](docs/reproducibility.md) for stability notes.

## Reproduction Scope

This project explicitly distinguishes between two objectives:

1. **Reproducing the competition submission**: The historical training did not preserve the random seed or RNG state, so exact reproduction depends on the released three task-head checkpoints, the complete SHA256 manifest, and fixed post-processing. The official hidden-test score is `0.002402`.
2. **Deterministically retraining the paper method from scratch**: The current code uses seed `2026` by default. The angle head uses full FP32 matmul, while the range head uses TF32 `high`. The paper method uses one `[a,b] + rel_smooth` range head; on the fixed validation set, the continuous output is `0.005907 +/- 0.000070`, and the result with MAP-hard is `0.003134 +/- 0.000119` (`n=3`).

The dual-range-head + 80 m gating entry point is retained only to reproduce the historical submission structure; multi-seed evaluation shows that it provides no significant gain over the single C head.

## Directory Responsibilities

- `3rdparty/`: Third-party code. It currently contains only the official VGGT submodule and is treated as a read-only dependency.
- `pairuav/`: Our implementation. Data loading, feature caching, geometry labels, task heads, metrics, training, inference, and post-processing are all placed here.
- `configs/`: Reproducible experiment configurations and path templates. Public configurations describe method parameters, while actual paths are provided through environment variables or command-line arguments.
- `data_index/`: Fixed training/validation data lists used to reconstruct splits and verify the order of feature caches.
- `docs/`: Non-executable content such as design notes, experiment records, and result tracking.
- `REPRODUCE.md`: The final command-level reproduction workflow. It should remain concise and executable; details and rationale belong in `docs/`.

## Two One-Command Workflows

Competition submission reproduction uses the downloaded three task heads and the frozen feature cache for the official test pairs, without rerunning VGGT:

```bash
bash scripts/reproduce_submission.sh
```

The final paper LaMP method starts training from a fixed index. By default, it uses `32,768` training pairs and `2,048` validation pairs;
provide different index files to change the data scale:

```bash
cp configs/paths.example.env configs/paths.env
# After editing configs/paths.env:
PAIRUAV_ENV_FILE=configs/paths.env bash scripts/train_lamp.sh
# bash scripts/train_lamp.sh /path/to/train_index.txt /path/to/val_index.txt
```

If frozen features are already available, set `PAIRUAV_TRAIN_CACHE` and `PAIRUAV_VAL_CACHE`; the script will first verify their order against the indexes and then skip VGGT.

## Environment Setup

```bash
git submodule update --init --recursive
pip install -e 3rdparty/vggt
pip install -e .
```

## Results (Codabench Hidden Test Set, Official Relative Error Metric, Lower Is Better)

| Output | final | Codabench Submission ID |
| --- | ---: | --- |
| Continuous output (frozen VGGT + angle head + range head) | 0.009292 | 811088 |
| Continuous output + MAP-hard post-processing | 0.002517 | 811089 |
| Gated continuous output with dual range heads | 0.009135 | 822840 |
| Gate + MAP-hard post-processing | **0.002402** | 822841 |

See [REPRODUCE.md](REPRODUCE.md) for reproduction commands, known validation-set metrics, and byte-level reproduction notes.

## Self-Check

```bash
python -m unittest discover -s tests -v
python -m pairuav.index verify-manifest \
  --manifest data_index/manifest.json \
  --index-root data_index
```

The complete checkpoints, test feature cache, and historical submission files are large, so they are released as a separate release bundle and are not included directly in the Git history.

Release bundle download: https://drive.google.com/drive/folders/1wXMSJjkHAnjN8C8y-MriVfGiTxgqDP3a?usp=drive_link

See [artifacts/README.md](artifacts/README.md) for the directory structure after download.
