# Large artifact layout

The files below are intentionally excluded from Git and are distributed separately. After downloading the release
bundle, place it under `artifacts/competition_submission/` with this layout:

```text
artifacts/competition_submission/
├── MANIFEST.sha256
├── checkpoints/
│   ├── S0_rich_noc/
│   │   └── head_best_angle.pt
│   ├── C_rel_rich/
│   │   └── range_head_best_distance.pt
│   └── B_mse_ab/
│       └── range_head_best_distance.pt
├── cache/
│   └── test_pairs_s518/
│       ├── features.npy
│       └── meta.json
└── submissions/
    ├── result_continuous.zip
    └── result_maphard.zip
```

`features.npy` must have shape `(2773116, 2, 4096)` and use the official test-pair order. The one-command historical
submission workflow reads this cache and the three archived task-head checkpoints; it does not run VGGT again.

Training caches and newly trained checkpoints belong under a run directory, not under this release bundle.
