"""Run PairUAV task heads directly on a frozen pair-feature cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch

from .features import file_sha256, write_json
from .head_io import load_range_head, load_sixdof_head
from .heads import heading_from_rot_torch, make_pair_input
from .metrics import wrap180
from .reproducibility import DEFAULT_SEED, seed_everything


def load_feature_cache(cache_dir: Path, expected_rows: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = Path(cache_dir)
    features_path = cache_dir / "features.npy"
    if not features_path.is_file():
        raise FileNotFoundError(features_path)
    features = np.load(features_path, mmap_mode="r")
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError(f"features.npy must have shape (N, 2, C), got {features.shape}")
    if features.dtype not in (np.float16, np.float32):
        raise ValueError(f"features.npy must use float16 or float32, got {features.dtype}")
    if expected_rows and features.shape[0] != expected_rows:
        raise ValueError(f"feature rows {features.shape[0]} != expected {expected_rows}")

    meta_path = cache_dir / "meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{meta_path} must contain a JSON object")
        meta = loaded
        if meta.get("samples") is not None and int(meta["samples"]) != features.shape[0]:
            raise ValueError(f"meta.samples {meta['samples']} != feature rows {features.shape[0]}")
        if meta.get("pooled_dim") is not None and int(meta["pooled_dim"]) != features.shape[2]:
            raise ValueError(f"meta.pooled_dim {meta['pooled_dim']} != feature dim {features.shape[2]}")
    return features, meta


def _ensure_outputs_available(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"outputs exist; pass --overwrite explicitly: {existing}")
    stale_temps = [path.with_suffix(path.suffix + ".tmp") for path in paths]
    stale_temps = [path for path in stale_temps if path.exists()]
    if stale_temps:
        raise FileExistsError(f"stale temporary outputs exist; move them aside before retrying: {stale_temps}")


def _write_rows(handle: TextIO, columns: tuple[np.ndarray, ...]) -> None:
    if len(columns) == 2:
        handle.writelines(f"{a:.6f} {b:.6f}\n" for a, b in zip(*columns))
        return
    if len(columns) == 3:
        handle.writelines(f"{a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in zip(*columns))
        return
    raise ValueError(f"unsupported output column count: {len(columns)}")


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    reproducibility = seed_everything(
        args.seed,
        deterministic=args.deterministic,
        matmul_precision=args.matmul_precision,
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    features, cache_meta = load_feature_cache(args.feature_cache, args.expected_rows)
    feat_dim = int(features.shape[2])

    angle, angle_cfg, angle_config_path = load_sixdof_head(
        args.angle_ckpt,
        feat_dim,
        device,
        config=args.angle_config,
    )
    range_head, range_cfg, range_config_path = load_range_head(
        args.range_ckpt,
        feat_dim,
        device,
        config=args.range_config,
    )
    range2_head = None
    range2_cfg: dict[str, Any] | None = None
    range2_config_path: Path | None = None
    if args.range2_ckpt is not None:
        range2_head, range2_cfg, range2_config_path = load_range_head(
            args.range2_ckpt,
            feat_dim,
            device,
            config=args.range2_config,
        )

    output_paths = [Path(args.out), Path(args.out).with_suffix(Path(args.out).suffix + ".meta.json")]
    if args.raw_heads_out is not None:
        output_paths.append(Path(args.raw_heads_out))
    _ensure_outputs_available(output_paths, args.overwrite)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.raw_heads_out is not None:
        Path(args.raw_heads_out).parent.mkdir(parents=True, exist_ok=True)

    out_tmp = Path(args.out).with_suffix(Path(args.out).suffix + ".tmp")
    raw_tmp = (
        Path(args.raw_heads_out).with_suffix(Path(args.raw_heads_out).suffix + ".tmp")
        if args.raw_heads_out is not None
        else None
    )
    switched_rows = 0
    processed = 0
    raw_handle: TextIO | None = None
    try:
        with out_tmp.open("w", encoding="utf-8") as out_handle, torch.inference_mode():
            if raw_tmp is not None:
                raw_handle = raw_tmp.open("w", encoding="utf-8")
            for start in range(0, features.shape[0], args.batch_size):
                end = min(start + args.batch_size, features.shape[0])
                batch_np = np.asarray(features[start:end], dtype=np.float32)
                batch = torch.from_numpy(batch_np).to(device, non_blocking=True)
                rot, _trans = angle(batch)
                heading = wrap180(heading_from_rot_torch(rot).cpu().numpy().astype(np.float64))
                range_close = (
                    range_head(make_pair_input(batch, str(range_cfg["input_mode"])))
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                if range2_head is None:
                    selected_range = range_close
                    if raw_handle is not None:
                        _write_rows(raw_handle, (heading, range_close))
                else:
                    assert range2_cfg is not None and args.gate_threshold is not None
                    range_far = (
                        range2_head(make_pair_input(batch, str(range2_cfg["input_mode"])))
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    use_far = np.abs(range_close) > args.gate_threshold
                    switched_rows += int(use_far.sum())
                    selected_range = np.where(use_far, range_far, range_close)
                    if raw_handle is not None:
                        _write_rows(raw_handle, (heading, range_close, range_far))
                _write_rows(out_handle, (heading, selected_range))
                processed = end
                if start == 0 or end == features.shape[0] or (start // args.batch_size) % 100 == 0:
                    print(f"[infer-cache] {end}/{features.shape[0]}", flush=True)
    finally:
        if raw_handle is not None:
            raw_handle.close()

    if processed != features.shape[0]:
        raise RuntimeError(f"inference stopped at {processed}/{features.shape[0]} rows")
    out_tmp.replace(args.out)
    if raw_tmp is not None and args.raw_heads_out is not None:
        raw_tmp.replace(args.raw_heads_out)

    def record(path: Path) -> dict[str, Any]:
        return {"path": str(Path(path).resolve()), "sha256": file_sha256(path)}

    meta = {
        "feature_cache": {
            "path": str(Path(args.feature_cache).resolve()),
            "features_shape": list(features.shape),
            "features_dtype": str(features.dtype),
            "features_bytes": int((Path(args.feature_cache) / "features.npy").stat().st_size),
            "meta": cache_meta,
        },
        "rows": int(features.shape[0]),
        "device": str(device),
        "reproducibility": reproducibility,
        "angle": {
            "checkpoint": record(args.angle_ckpt),
            "config": record(angle_config_path),
            "name": angle_cfg.get("name"),
        },
        "range": {
            "checkpoint": record(args.range_ckpt),
            "config": record(range_config_path),
            "name": range_cfg.get("name"),
        },
        "range2": (
            {
                "checkpoint": record(args.range2_ckpt),
                "config": record(range2_config_path),
                "name": range2_cfg.get("name") if range2_cfg else None,
            }
            if args.range2_ckpt is not None and range2_config_path is not None
            else None
        ),
        "gate_threshold_m": args.gate_threshold,
        "gate_switched_rows": switched_rows if range2_head is not None else None,
        "output": record(args.out),
        "raw_heads_output": record(args.raw_heads_out) if args.raw_heads_out is not None else None,
    }
    write_json(Path(args.out).with_suffix(Path(args.out).suffix + ".meta.json"), meta)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run archived or retrained PairUAV heads on a feature cache.")
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--angle-ckpt", type=Path, required=True)
    parser.add_argument("--angle-config", type=Path, default=None)
    parser.add_argument("--range-ckpt", type=Path, required=True)
    parser.add_argument("--range-config", type=Path, default=None)
    parser.add_argument("--range2-ckpt", type=Path, default=None)
    parser.add_argument("--range2-config", type=Path, default=None)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True, help="two-column heading/range output")
    parser.add_argument("--raw-heads-out", type=Path, default=None, help="optional ungated two/three-column audit output")
    parser.add_argument("--expected-rows", type=int, default=0, help="0 accepts the cache row count")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch algorithms (default: enabled)",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.expected_rows < 0:
        parser.error("--batch-size must be positive and --expected-rows must be non-negative")
    paired = (args.range2_ckpt is None) == (args.range2_config is None)
    if not paired:
        parser.error("--range2-ckpt and --range2-config must be supplied together")
    if (args.range2_ckpt is None) != (args.gate_threshold is None):
        parser.error("--gate-threshold is required exactly when a second range head is supplied")
    return args


def main() -> None:
    args = parse_args()
    meta = run_inference(args)
    print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
