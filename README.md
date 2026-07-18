# LaMP: Last-Meter Manifold Pose Estimation

Official implementation and review artifact for **“LaMP: Last-Meter Manifold Pose Estimation for Image-Goal
Terminal UAV Navigation”** (ACM MM 2026 UAVM Workshop, OpenReview submission #12).

LaMP jointly encodes a source-goal image pair with a frozen VGGT backbone and trains lightweight task heads using
the geometry of the public target-centric acquisition trajectory. The repository implements the paper terminology
directly: a **geometry-supervised pose head**, a **relative-error range head**, and optional **pose-set decoding**.

## Two systems, two claims

The paper distinguishes the archived challenge system from the simplified method reported as the main model.

| name in paper | range side | post-processing | reported anchor |
|---|---|---|---|
| **LaMP (challenge entry)** | two range heads, 80 m gate | pose-set decoding | Codabench `822841`, final `0.002402`, 5th place |
| **LaMP (ours)** | one `[a,b]` relative-error range head | optional pose-set decoding | `0.4336 deg` heading MAE, `0.3216 m` range MAE on `val_quick_2048` (`n=3`) |

The 80 m gate is retained only to reconstruct the submitted competition topology. Multi-seed validation did not show
a reliable gate benefit, so **LaMP (ours)** removes the second range head and gate.

## Install

Python 3.10 or newer and PyTorch 2.2 or newer are required.

```bash
git clone --recursive https://github.com/Polaris-F/vggt-pairuav.git
cd vggt-pairuav
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e 3rdparty/vggt
pip install -e .
python -m unittest discover -s tests -v
```

The VGGT submodule is pinned to `a288dd0f14786c93483e45524328726ab7b1b4ce`. PairUAV-specific code lives under
`pairuav/`; the submodule is treated as a read-only dependency.
`VGGT_WEIGHT` in `configs/paths.env` must point to the locally downloaded official VGGT checkpoint.
Official checkpoint page: https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt

## Shortest verification paths

Large assets are separate release bundles. Place them under `artifacts/` using the layouts in
[`release/README.md`](release/README.md), then verify their manifests.

### LaMP (ours), three released seeds

```bash
bash scripts/evaluate_lamp_release.sh
```

This verifies the frozen `val_quick_2048` cache order, evaluates seeds `2026/2027/2028`, and writes a mean/std
summary. Expected continuous metrics are approximately `0.43 deg` heading MAE and `0.32 m` range MAE.

### LaMP (challenge entry), submission 822841

```bash
bash scripts/reproduce_submission.sh
```

This verifies the archived task-head bundle, runs the two range heads with the historical 80 m gate, and applies the
fixed pose-set decoder. The bundle also contains the archived `result_maphard.zip` submitted as `822841`.

### Train LaMP (ours) from fixed indexes

The default indexes contain 32,768 training pairs and 2,048 validation pairs:

```bash
cp configs/paths.example.env configs/paths.env
# Fill only local dataset/checkpoint paths in configs/paths.env.
PAIRUAV_ENV_FILE=configs/paths.env bash scripts/train_lamp.sh
```

Existing frozen features can be supplied through `PAIRUAV_TRAIN_CACHE` and `PAIRUAV_VAL_CACHE`; both caches are
checked against the requested indexes before training.

## Data and protocol boundary

This repository contains pair indexes and derived geometry, not the University-1652/PairUAV image datasets. Dataset
access follows the official [PairUAV](https://github.com/YaxuanLi-cn/PairUAV) and
[University-1652](https://github.com/layumi/University1652-Baseline) project pages and their terms. The 211-value pose
set is enumerated from **training labels and the public University-1652 acquisition protocol**. Challenge test images
and labels are not used to construct the pose set or tune the decoder; Codabench supplies only the aggregate
submission score.

## Repository layout

```text
3rdparty/vggt/   pinned official VGGT submodule
pairuav/         feature, geometry, head, metric, inference, and decoding code
configs/         authoritative `submission/` and `lamp/` recipes
data_index/      fixed train/validation and probe indexes
scripts/         one-command training, evaluation, and submission workflows
release/         bundle manifests and paper-to-artifact map
artifacts/       ignored local landing area for downloaded large files
REPRODUCE.md     command-level reproduction and custody guide
```

For exact commands, expected metrics, bundle hashes, and the paper table-to-experiment map, see
[`REPRODUCE.md`](REPRODUCE.md). Reproducibility controls are documented in
[`docs/reproducibility.md`](docs/reproducibility.md).
