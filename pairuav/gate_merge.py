"""双距离头门控(gate)合成。

以基线连续结果(两列 `heading range`,来自 relL1 距离头)为底;对基线距离 |range| > threshold 的行,
把距离替换为第二距离头(MSE+ab)的输出。阈值取 80 m,来自验证集分距离段消融:MSE+ab 头仅在
大距离段优于 relL1 头,80 m 即两头的交叉点。

第二头的输出由 `pairuav.infer_test`(带 `--range2-*`,三列)产生,支持两种给法:
- 全量三列文件:`--pred file:0`(即一个覆盖 [0, N) 的分片);
- 多卡/多机分片:多个 `--pred file:start`,start 为该分片在**子集内**的行起点;配 `--subset-idx`
  (npy,子集行 -> 全量行索引)时分片只覆盖该子集,不配则子集 = 全量。

未被任何分片覆盖、或 |range| <= threshold 的行保留基线值,因此结果恒不劣于基线。
纯 NumPy,无前向、不读测试标签。
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = np.loadtxt(args.base_result, dtype=np.float64)
    if base.ndim != 2 or base.shape[1] != 2:
        raise ValueError(f"{args.base_result} 不是两列 heading range")
    heading, range_base = base[:, 0], base[:, 1]

    idx = np.load(args.subset_idx) if args.subset_idx is not None else np.arange(len(base))
    r2_sub = np.full(len(idx), np.nan)
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
        end = min(start + len(shard), len(idx))
        r2_sub[start:end] = shard[: end - start, 2]

    covered = int(np.isfinite(r2_sub).sum())
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
        with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.out, arcname="result.txt")
        print(f"[zip] {args.zip}", flush=True)


if __name__ == "__main__":
    main()
