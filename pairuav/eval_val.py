"""在验证 cache 上评估 6DoF 角度头与独立距离头的组合。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import geometry as G
from .features import write_json
from .heads import RangeMLP, SixDofHead, heading_from_rot_torch, make_pair_input
from .metrics import compute_metrics_np


def load_angle_head(run_dir: Path, checkpoint: Path, feat_dim: int, device: torch.device) -> SixDofHead:
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    model = SixDofHead(
        feat_dim=feat_dim,
        hidden_dim=int(cfg["hidden_dim"]),
        input_mode=str(cfg["input_mode"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval()


def load_range_head(run_dir: Path, checkpoint: Path, feat_dim: int, device: torch.device) -> tuple[RangeMLP, dict[str, Any]]:
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if result_path.exists():
        cfg = json.loads(result_path.read_text(encoding="utf-8"))["config"]
    else:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    mult = {"ab": 2, "ab_diff_prod": 4, "diff_prod": 2}[str(cfg["input_mode"])]
    model = RangeMLP(
        in_dim=feat_dim * mult,
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
        range_limit=132.0,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval(), cfg


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    features = np.load(Path(args.val_cache) / "features.npy", mmap_mode="r")
    feat_dim = int(features.shape[-1])
    labels = np.load(args.val_geom)
    gt_h = labels["heading"].astype(np.float64)
    gt_r = labels["range"].astype(np.float64)

    angle_head = load_angle_head(args.angle_run_dir, args.angle_ckpt, feat_dim, device)
    range_head, range_cfg = load_range_head(args.range_run_dir, args.range_ckpt, feat_dim, device)
    range_mode = str(range_cfg["input_mode"])

    pred_h: list[np.ndarray] = []
    pred_r_6dof: list[np.ndarray] = []
    pred_r_combo: list[np.ndarray] = []
    for start in range(0, features.shape[0], args.batch_size):
        feat = torch.from_numpy(np.asarray(features[start: start + args.batch_size], dtype=np.float32)).to(device)
        rot, trans = angle_head(feat)
        pred_h.append(heading_from_rot_torch(rot).detach().cpu().numpy().astype(np.float64))
        pred_r_6dof.append((trans[:, 2] / G.COS45).detach().cpu().numpy().astype(np.float64))
        pred_r_combo.append(range_head(make_pair_input(feat, range_mode)).detach().cpu().numpy().astype(np.float64))

    ph = np.concatenate(pred_h)
    pr_6dof = np.concatenate(pred_r_6dof)
    pr_combo = np.concatenate(pred_r_combo)
    return {
        "val_cache": str(args.val_cache),
        "val_geom": str(args.val_geom),
        "angle_run_dir": str(args.angle_run_dir),
        "angle_ckpt": str(args.angle_ckpt),
        "range_run_dir": str(args.range_run_dir),
        "range_ckpt": str(args.range_ckpt),
        "sixdof_projected_range": compute_metrics_np(ph, pr_6dof, gt_h, gt_r),
        "sixdof_angle_plus_range": compute_metrics_np(ph, pr_combo, gt_h, gt_r),
        "angle_mae_deg": float(np.abs(G.wrap180(ph - gt_h)).mean()),
        "combo_range_mae_m": float(np.abs(pr_combo - gt_r).mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在验证 cache 上评估角度头与距离头组合。")
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--val-geom", type=Path, required=True)
    parser.add_argument("--angle-run-dir", type=Path, required=True)
    parser.add_argument("--angle-ckpt", type=Path, required=True)
    parser.add_argument("--range-run-dir", type=Path, required=True)
    parser.add_argument("--range-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    if args.out is not None:
        write_json(args.out, result)


if __name__ == "__main__":
    main()
