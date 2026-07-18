# Reproducibility

## Two Claims

- The official Codabench result is reproduced with archived checkpoints and their SHA256 manifest. The original
  training did not preserve a seed or RNG state.
- From-scratch experiments use fixed seeds beginning at `2026` and report mean plus sample standard deviation.

These are complementary claims. Deterministic retraining is not expected to regenerate the historical checkpoint
byte-for-byte.

## Randomness Controls

PyTorch entry points call `pairuav.reproducibility.seed_everything` and record:

- Python, NumPy, CPU PyTorch, and CUDA PyTorch RNG seeds;
- explicit DataLoader or GPU shuffle generators;
- cuDNN benchmark and deterministic settings;
- `CUBLAS_WORKSPACE_CONFIG`;
- PyTorch and CUDA versions;
- the selected float32 matmul precision.

`PYTHONHASHSEED` is also exported for child processes. Python reads its own hash seed at interpreter startup, so callers
requiring hash-level determinism should additionally launch commands with `PYTHONHASHSEED=2026`.

PyTorch deterministic algorithms are requested with `warn_only=True`. The released MLP-head training path is
deterministic on the tested software/GPU stack, but different CUDA, PyTorch, or GPU architectures may still change
floating-point rounding.

## Numeric Precision

Precision is intentionally entry-specific:

| entry | matmul precision | reason |
|---|---|---|
| `train_angle` | `highest` | TF32 materially regresses the 6D rotation/geodesic objective |
| `train_range` | `high` | matches the range-head training recipe |
| `features`, `infer_test`, `infer_cache`, `eval_val` | `high` | matches the archived submission inference path |

Do not replace the angle training setting with a global TF32 default.

## Released Configuration Groups

- `configs/submission/` records the archived competition topology: rich S0 angle head, rich C range head, B range
  head, an 80 m gate, and MAP-hard. The historical C recipe used `lr=2e-3` for 120 epochs.
- `configs/lamp/` records the paper method: the same 6-DoF-supervised S0 angle head and one `[a,b]` range head
  trained with `rel_smooth`, `lr=1e-3`, and 240 epochs. There is no second range head or gate.
- Root-level head configs remain only for compatibility with older commands.

Paper-method validation anchors on `val_quick_2048` (`n=3`):

| output | final |
|---|---:|
| one `[a,b]` range head, continuous | `0.005907 +/- 0.000070` |
| one `[a,b]` range head + MAP-hard | `0.003134 +/- 0.000119` |

The historical C/B gate remains available because it was part of submission `822841`; later multi-seed analysis did
not find a reliable gain over one C head, and the paper method removes it.
