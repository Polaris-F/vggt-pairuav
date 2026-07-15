# Release bundles

Large checkpoints and frozen feature caches are distributed separately from Git history. After downloading, place
each bundle under `artifacts/` and verify its `MANIFEST.sha256` before running a workflow.

| bundle | default local directory | purpose |
|---|---|---|
| `lamp_challenge_entry_822841_v1` | `artifacts/competition_submission/` | Exact archived task heads and submissions 822840/822841; test cache lands separately |
| `lamp_ours_3seed_valquick_v1` | `artifacts/lamp_ours_3seed/` | Three LaMP seeds plus the frozen `val_quick_2048` cache |
| `lamp_probe_on_manifold_v1` | optional | On-grid, off-frame, and off-grid probe caches |
| `lamp_probe_off_manifold_slope_v1` | optional | Four altered slopes in one probe cache with geometry |
| `lamp_probe_sues200_v1` | optional | SUES-200 pooled features and independent COLMAP reference |

Archive hashes are in `ARTIFACTS.sha256` and machine-readable metadata is in `bundles.json`. Tracked copies of each
archive's internal manifest live under `manifests/`. Release URLs are intentionally kept outside source files until
the review channel and repository visibility are approved by the authors.

The image datasets and the VGGT checkpoint are not redistributed. See [REPRODUCE.md](../REPRODUCE.md) for acquisition
and evaluation instructions.
