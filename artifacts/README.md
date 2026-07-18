# Large artifact layout

Large files are intentionally excluded from Git and distributed as independently hashed bundles. Bundle names and
tracked manifest copies are listed in [`release/README.md`](../release/README.md).

## LaMP (challenge entry)

Extract `lamp_challenge_entry_822841_v1` under `artifacts/competition_submission/`. The archive contains the task
heads, decoder resources, and archived submission zips. Place the separately distributed frozen test feature cache
at the shown cache path before rerunning inference:

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
│       ├── features.npy       # separate large asset
│       └── meta.json          # separate large asset
└── submissions/
    ├── result_continuous.zip
    └── result_maphard.zip
```

`features.npy` must have shape `(2773116, 2, 4096)` and use the official test-pair order. The one-command historical
submission workflow reads this cache and the three archived task-head checkpoints; it does not run VGGT again. The
weight bundle manifest does not claim custody of a test cache added after extraction; verify that cache against the
checksum distributed with the cache asset.

## LaMP (ours), three seeds

Place `lamp_ours_3seed_valquick_v1` under `artifacts/lamp_ours_3seed/`:

```text
artifacts/lamp_ours_3seed/
├── MANIFEST.sha256
├── weights/
│   ├── seed2026/
│   │   ├── angle/S0_rich_noc/{config.json,head_best_angle.pt,result.json}
│   │   └── range/R_ab_relsmooth/{config.json,range_head_best_distance.pt,result.json}
│   ├── seed2027/...
│   └── seed2028/...
└── validation/val_quick_2048/
    ├── features.npy
    ├── heading.npy
    ├── range.npy
    ├── json_paths.json
    ├── meta.json
    └── geometry.npz
```

Run `bash scripts/evaluate_lamp_release.sh` to verify all hashes and reproduce the three-seed validation aggregate.

Training caches and newly trained checkpoints belong under a timestamped run directory, not under either release
bundle.
