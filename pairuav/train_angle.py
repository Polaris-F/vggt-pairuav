"""训练 S0 6DoF 角度头。"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from . import geometry as G
from .features import write_json
from .heads import (
    SixDofHead,
    circ_sincos_loss_deg,
    geodesic_loss,
    heading_from_rot_torch,
    sincos_angle_loss,
)
from .metrics import compute_metrics_np
from .reproducibility import DEFAULT_SEED, dataloader_generator, seed_everything


@dataclass
class AngleConfig:
    name: str = "S0_rich_noc"
    input_mode: str = "ab_diff_prod"
    hidden_dim: int = 1024
    depth: int = 3
    dropout: float = 0.0
    lr: float = 1.5e-3
    weight_decay: float = 5e-5
    batch_size: int = 384
    epochs: int = 90
    warmup_epochs: float = 2.0
    final_lr_frac: float = 0.05
    grad_clip: float = 1.0
    rot_weight: float = 1.0
    trans_weight: float = 1.0
    task_angle_weight: float = 0.0
    task_range_weight: float = 0.0
    coupling_weight: float = 0.0


def load_config(path: Path | None) -> AngleConfig:
    if path is None:
        return AngleConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return AngleConfig(**data)


def load_cache(cache_dir: Path, geom_path: Path, device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, np.ndarray]]:
    feat = torch.from_numpy(np.load(Path(cache_dir) / "features.npy", mmap_mode="r").astype(np.float32))
    geom_np = np.load(Path(geom_path))
    labels_np = {
        "heading": geom_np["heading"].astype(np.float32),
        "range": geom_np["range"].astype(np.float32),
        "rot": geom_np["rot"].astype(np.float32),
        "trans": geom_np["trans_world"].astype(np.float32),
    }
    labels_t = {key: torch.from_numpy(value).to(device) for key, value in labels_np.items()}
    return feat, labels_t, labels_np


@torch.no_grad()
def evaluate(model: SixDofHead, feat: torch.Tensor, labels_np: dict[str, np.ndarray], device: torch.device, batch_size: int) -> dict[str, Any]:
    model.eval()
    pred_h, pred_r, pred_t = [], [], []
    for start in range(0, len(feat), batch_size):
        x = feat[start: start + batch_size].to(device, non_blocking=True)
        rot, trans = model(x)
        pred_h.append(heading_from_rot_torch(rot).detach().cpu().numpy().astype(np.float64))
        pred_r.append((trans[:, 2] / G.COS45).detach().cpu().numpy().astype(np.float64))
        pred_t.append(trans.detach().cpu().numpy().astype(np.float64))
    ph = np.concatenate(pred_h)
    pr = np.concatenate(pred_r)
    pt = np.concatenate(pred_t)
    out = compute_metrics_np(ph, pr, labels_np["heading"], labels_np["range"])
    out["angle_mae_deg"] = float(np.abs(G.wrap180(ph - labels_np["heading"])).mean())
    out["range_mae_m"] = float(np.abs(pr - labels_np["range"]).mean())
    out["pred_range_min"] = float(pr.min())
    out["pred_range_max"] = float(pr.max())
    out["pred_range_mean"] = float(pr.mean())
    out["trans_mae_xyz_m"] = np.abs(pt - labels_np["trans"].astype(np.float64)).mean(axis=0).tolist()
    out["trans_mae_m"] = float(np.abs(pt - labels_np["trans"].astype(np.float64)).mean())
    return out


def train_one(
    cfg: AngleConfig,
    args: argparse.Namespace,
    run_root: Path,
    device: torch.device,
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    run_dir = run_root / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(cfg))

    train_feat, train_t, _train_np = load_cache(args.train_cache, args.train_geom, device)
    val_feat, _val_t, val_np = load_cache(args.val_cache, args.val_geom, device)
    model = SixDofHead(
        feat_dim=int(train_feat.shape[-1]),
        hidden_dim=cfg.hidden_dim,
        input_mode=cfg.input_mode,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(len(train_feat) / cfg.batch_size)
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = max(1, int(cfg.warmup_epochs * steps_per_epoch))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    loader = DataLoader(
        TensorDataset(torch.arange(len(train_feat), dtype=torch.long)),
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=dataloader_generator(args.seed),
    )
    initial = evaluate(model, val_feat, val_np, device, args.eval_batch_size)
    best = {"epoch": -1, "val": initial}
    best_angle = {"epoch": -1, "val": initial}
    history: list[dict[str, Any]] = []
    start_all = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        totals = {"loss": 0.0, "rot": 0.0, "trans": 0.0, "task_angle": 0.0, "task_range": 0.0, "coupling": 0.0, "n": 0}
        epoch_start = time.time()
        for (idx,) in loader:
            idx_dev = idx.to(device)
            x = train_feat[idx].to(device, non_blocking=True)
            rot_gt = train_t["rot"][idx_dev]
            trans_gt = train_t["trans"][idx_dev]
            heading_gt = train_t["heading"][idx_dev]
            range_gt = train_t["range"][idx_dev]
            rot, trans = model(x)
            pred_h = heading_from_rot_torch(rot)
            pred_r = trans[:, 2] / G.COS45
            rot_loss = geodesic_loss(rot, rot_gt)
            trans_loss = F.smooth_l1_loss(trans / G.TRANS_SCALE, trans_gt / G.TRANS_SCALE)
            task_angle = sincos_angle_loss(pred_h, heading_gt)
            task_range = F.smooth_l1_loss(pred_r / 132.0, range_gt / 132.0)
            coupling = circ_sincos_loss_deg(pred_h, -8.0 * pred_r)
            loss = (
                cfg.rot_weight * rot_loss
                + cfg.trans_weight * trans_loss
                + cfg.task_angle_weight * task_angle
                + cfg.task_range_weight * task_range
                + cfg.coupling_weight * coupling
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            bs = int(idx.numel())
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["rot"] += float(rot_loss.detach().cpu()) * bs
            totals["trans"] += float(trans_loss.detach().cpu()) * bs
            totals["task_angle"] += float(task_angle.detach().cpu()) * bs
            totals["task_range"] += float(task_range.detach().cpu()) * bs
            totals["coupling"] += float(coupling.detach().cpu()) * bs
            totals["n"] += bs
        val = evaluate(model, val_feat, val_np, device, args.eval_batch_size)
        n = max(1, totals["n"])
        row = {
            "epoch": epoch,
            "train_loss": totals["loss"] / n,
            "train_rot": totals["rot"] / n,
            "train_trans": totals["trans"] / n,
            "train_task_angle": totals["task_angle"] / n,
            "train_task_range": totals["task_range"] / n,
            "train_coupling": totals["coupling"] / n,
            "lr": sched.get_last_lr()[0],
            "epoch_sec": time.time() - epoch_start,
            "val": val,
        }
        history.append(row)
        if val["final_score"] < best["val"]["final_score"]:
            best = {"epoch": epoch, "val": val}
            torch.save(model.state_dict(), run_dir / "head_best_final.pt")
        if val["angle_rel_error"] < best_angle["val"]["angle_rel_error"]:
            best_angle = {"epoch": epoch, "val": val}
            torch.save(model.state_dict(), run_dir / "head_best_angle.pt")
        torch.save(model.state_dict(), run_dir / "head_last.pt")
        print(
            f"[{cfg.name} ep{epoch:03d}] loss={row['train_loss']:.5f} "
            f"val final={val['final_score']:.6f} angle={val['angle_rel_error']:.6f} "
            f"dist={val['distance_rel_error']:.6f} range_mae={val['range_mae_m']:.3f}",
            flush=True,
        )
    result = {
        "config": asdict(cfg),
        "reproducibility": reproducibility,
        "run_dir": str(run_dir),
        "initial": initial,
        "best": best,
        "best_angle": best_angle,
        "history": history,
        "elapsed_sec": time.time() - start_all,
    }
    write_json(run_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 S0 6DoF 角度头。")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--train-geom", type=Path, required=True)
    parser.add_argument("--val-geom", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/angle"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch algorithms (default: enabled)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reproducibility = seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    run_root = args.run_root.with_name(args.run_root.name + "_" + time.strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True, exist_ok=True)
    result = train_one(cfg, args, run_root, device, reproducibility)
    write_json(
        run_root / "summary.json",
        {
            "run_root": str(run_root),
            "reproducibility": reproducibility,
            "result": {"name": cfg.name, "best": result["best"], "best_angle": result["best_angle"]},
        },
    )


if __name__ == "__main__":
    main()
