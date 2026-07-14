"""双距离头门控(gate)合成。

以基线连续结果(两列 `heading range`)为底;对基线距离 |range| > threshold 的行,
把距离替换为第二距离头的输出。该入口保留用于复现历史提交结构;多种子论文实验表明
80 m 门控相对单距离头没有显著增益。

第二头的输出由 `pairuav.infer_test`(带 `--range2-*`,三列)产生,支持两种给法:
- 全量三列文件:`--pred file:0`(即一个覆盖 [0, N) 的分片);
- 多卡/多机分片:多个 `--pred file:start`,start 为该分片在**子集内**的行起点;配 `--subset-idx`
  (npy,子集行 -> 全量行索引)时分片只覆盖该子集,不配则子集 = 全量。

默认要求分片完整覆盖指定子集,避免静默混入基线结果。只有显式传 `--allow-partial` 时,
未覆盖行才保留基线值。纯 NumPy,无前向、不读测试标签。
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双距离头门控合成(纯 NumPy)。")
    parser.add_argument("--base-result", type=Path, required=True, help="基线两列结果(heading range,全量行)")
    parser.add_argument("--pred", action="append", default=[], required=True,
                        help="三列分片,格式 file:start(start = 分片在子集内的行起点),可重复")
    parser.add_argument("--subset-idx", type=Path, default=None, help="npy:子集行 -> 全量行索引;缺省 = 全量")
    parser.add_argument("--threshold", type=float, default=80.0, help="仅 |base range| > threshold 的行换用第二头距离")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=None, help="可选:同时打包为提交 zip(内含 result.txt)")
    parser.add_argument("--allow-partial", action="store_true", help="允许分片未覆盖整个子集")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    parser.add_argument(
        "--alignment-atol",
        type=float,
        default=0.1,
        help="分片 heading/C 距离与基线对齐的最大允许偏差",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = np.loadtxt(args.base_result, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"{args.base_result} 不是两列 heading range")
    heading, range_base = base[:, 0], base[:, 1]

    if args.out.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite explicitly: {args.out}")
    if args.zip is not None and args.zip.exists() and not args.overwrite:
        raise FileExistsError(f"zip exists; pass --overwrite explicitly: {args.zip}")

    idx = np.asarray(np.load(args.subset_idx) if args.subset_idx is not None else np.arange(len(base)))
    if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise ValueError("subset index must be a one-dimensional integer array")
    idx = idx.astype(np.int64, copy=False)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("subset index contains duplicate rows")
    if len(idx) and (idx.min() < 0 or idx.max() >= len(base)):
        raise ValueError(f"subset index is outside [0, {len(base)})")
    r2_sub = np.full(len(idx), np.nan)
    assigned = np.zeros(len(idx), dtype=bool)
    for spec in args.pred:
        file_part, start_part = spec.rsplit(":", 1)
        start = int(start_part)
        path = Path(file_part)
        if not path.exists() or path.stat().st_size == 0:
            print(f"[skip] {path} 缺失或为空", flush=True)
            continue
        shard = np.loadtxt(path, dtype=np.float64)
        if shard.ndim == 1:
            shard = shard[None, :]
        if shard.shape[1] < 3:
            raise ValueError(f"{path} 不是三列 heading range range2")
        end = start + len(shard)
        if start < 0 or end > len(idx):
            raise ValueError(f"shard {path} covers [{start}, {end}), outside subset length {len(idx)}")
        if assigned[start:end].any():
            raise ValueError(f"shard {path} overlaps an earlier shard in [{start}, {end})")
        target_rows = idx[start:end]
        heading_error = np.abs((shard[:, 0] - heading[target_rows] + 180.0) % 360.0 - 180.0)
        range_error = np.abs(shard[:, 1] - range_base[target_rows])
        max_heading_error = float(heading_error.max()) if len(heading_error) else 0.0
        max_range_error = float(range_error.max()) if len(range_error) else 0.0
        if max_heading_error > args.alignment_atol or max_range_error > args.alignment_atol:
            raise ValueError(
                f"shard {path} is not aligned with the base result: "
                f"max heading diff={max_heading_error:.6f}, max range diff={max_range_error:.6f}"
            )
        r2_sub[start:end] = shard[:, 2]
        assigned[start:end] = True

    covered = int(np.isfinite(r2_sub).sum())
    if covered != len(idx) and not args.allow_partial:
        raise RuntimeError(
            f"range2 shards cover only {covered}/{len(idx)} rows; pass --allow-partial only for diagnostics"
        )
    base_sub = range_base[idx]
    use_r2 = np.isfinite(r2_sub) & (np.abs(base_sub) > args.threshold)
    range_gate = range_base.copy()
    range_gate[idx] = np.where(use_r2, r2_sub, base_sub)
    print(f"[gate] 子集 {len(idx)} 行,分片覆盖 {covered} ({covered / max(len(idx), 1) * 100:.1f}%),"
          f"threshold={args.threshold}m 实际替换 {int(use_r2.sum())} 行", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.writelines(f"{h:.6f} {r:.6f}\n" for h, r in zip(heading, range_gate))
    print(f"[out] {args.out} ({len(heading)} lines)", flush=True)
    if args.zip is not None:
        args.zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname="result.txt")
        print(f"[zip] {args.zip}", flush=True)


if __name__ == "__main__":
    main()
