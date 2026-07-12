"""训练独立距离头。"""

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

from .features import write_json
from .heads import RangeMLP, make_pair_input
from .metrics import circular_angle_abs_error, compute_metrics
from .reproducibility import DEFAULT_SEED, seed_everything


@dataclass
class RangeConfig:
    name: str = "C_rel_rich"
    input_mode: str = "ab_diff_prod"
    loss: str = "rel_smooth"
    hidden_dim: int = 512
    depth: int = 2
    dropout: float = 0.0
    lr: float = 2e-3
    weight_decay: float = 5e-5
    batch_size: int = 512
    epochs: int = 120
    warmup_epochs: float = 2.0
    final_lr_frac: float = 0.05
    grad_clip: float = 1.0
    eps: float = 1.0


def load_config(path: Path | None) -> RangeConfig:
    if path is None:
        return RangeConfig()
    return RangeConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def load_cache(cache_dir: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats = torch.from_numpy(np.load(Path(cache_dir) / "features.npy", mmap_mode="r").astype(np.float16)).to(device)
    heading = torch.from_numpy(np.load(Path(cache_dir) / "heading.npy").astype(np.float32)).to(device)
    range_m = torch.from_numpy(np.load(Path(cache_dir) / "range.npy").astype(np.float32)).to(device)
    return feats, heading, range_m


def official_eval(pred_h: torch.Tensor, pred_r: torch.Tensor, gt_h: torch.Tensor, gt_r: torch.Tensor) -> dict[str, Any]:
    ph = pred_h.detach().cpu().numpy().astype(np.float64)
    pr = pred_r.detach().cpu().numpy().astype(np.float64)
    gh = gt_h.detach().cpu().numpy().astype(np.float64)
    gr = gt_r.detach().cpu().numpy().astype(np.float64)
    gt_pairs = [(float(h), float(r), "") for h, r in zip(gh, gr)]
    pred_pairs = [(float(h), float(r)) for h, r in zip(ph, pr)]
    out = compute_metrics(gt_pairs, pred_pairs)
    out["angle_mae_deg"] = float(circular_angle_abs_error(ph, gh).mean())
    out["range_mae_m"] = float(np.abs(pr - gr).mean())
    out["pred_range_min"] = float(pr.min())
    out["pred_range_max"] = float(pr.max())
    out["pred_range_mean"] = float(pr.mean())
    return out


def loss_fn(pred: torch.Tensor, gt: torch.Tensor, name: str, range_limit: float, eps: float) -> torch.Tensor:
    if name == "mse":
        return F.mse_loss(pred / range_limit, gt / range_limit)
    if name == "weighted_mse":
        weight = (range_limit / gt.abs().clamp_min(eps)).clamp(max=20.0)
        return (weight * ((pred - gt) / range_limit).pow(2)).mean()
    if name == "rel_l1":
        return ((pred - gt).abs() / gt.abs().clamp_min(eps)).mean()
    if name == "rel_smooth":
        rel = (pred - gt) / gt.abs().clamp_min(eps)
        return F.smooth_l1_loss(rel, torch.zeros_like(rel), beta=0.1)
    raise ValueError(f"unknown loss: {name}")


@torch.no_grad()
def eval_model(model: RangeMLP, x_val: torch.Tensor, val_h_pred: torch.Tensor, val_h: torch.Tensor, val_r: torch.Tensor) -> dict[str, Any]:
    model.eval()
    return official_eval(val_h_pred, model(x_val), val_h, val_r)


def train_one(
    args: argparse.Namespace,
    cfg: RangeConfig,
    data: dict[str, torch.Tensor],
    out_dir: Path,
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    x_train = data[f"x_train_{cfg.input_mode}"]
    x_val = data[f"x_val_{cfg.input_mode}"]
    train_r = data["train_r"]
    val_h_pred = data["val_h_pred"]
    val_h = data["val_h"]
    val_r = data["val_r"]
    model = RangeMLP(
        int(x_train.shape[1]),
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        range_limit=args.range_limit,
    ).to(args.device_obj)
    model.init_mean(float(train_r.mean().detach().cpu()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = int(math.ceil(x_train.shape[0] / cfg.batch_size))
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return cfg.final_lr_frac + (1.0 - cfg.final_lr_frac) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    shuffle_generator = torch.Generator(device=args.device_obj).manual_seed(args.seed)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], device=args.device_obj, generator=shuffle_generator)
        totals = {"loss": 0.0, "range_mae": 0.0, "n": 0}
        for start in range(0, x_train.shape[0], cfg.batch_size):
            ids = perm[start: start + cfg.batch_size]
            pred = model(x_train[ids])
            gt = train_r[ids]
            loss = loss_fn(pred, gt, cfg.loss, args.range_limit, cfg.eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            bs = int(ids.shape[0])
            totals["loss"] += float(loss.detach().cpu()) * bs
            totals["range_mae"] += float((pred - gt).abs().mean().detach().cpu()) * bs
            totals["n"] += bs
        n = max(1, totals["n"])
        val = eval_model(model, x_val, val_h_pred, val_h, val_r)
        row = {"epoch": epoch, "train_loss": totals["loss"] / n, "train_range_mae_m": totals["range_mae"] / n, "lr": sched.get_last_lr()[0], "val": val}
        history.append(row)
        if best is None or val["distance_rel_error"] < best["val"]["distance_rel_error"]:
            best = row
            torch.save(model.state_dict(), out_dir / "range_head_best_distance.pt")
        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch == cfg.epochs - 1:
            print(f"[{cfg.name}] ep={epoch} dist={val['distance_rel_error']:.6f} range_mae={val['range_mae_m']:.3f}", flush=True)
    assert best is not None
    torch.save(model.state_dict(), out_dir / "range_head_last.pt")
    result = {
        "config": asdict(cfg),
        "reproducibility": reproducibility,
        "history": history,
        "best": best,
        "elapsed_sec": time.time() - t0,
    }
    write_json(out_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练独立距离头。")
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/range"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--range-limit", type=float, default=132.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
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
    reproducibility = seed_everything(
        args.seed,
        deterministic=args.deterministic,
        matmul_precision="high",
    )
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    out_dir = args.output_dir / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config.json", asdict(cfg))

    train_feats, _train_h, train_r = load_cache(args.train_cache, args.device_obj)
    val_feats, val_h, val_r = load_cache(args.val_cache, args.device_obj)
    # 距离头训练只依赖 range 标签。验证时沿用 heading 标签作为占位,
    # 这样日志中的 distance_rel_error 与 range_mae 只反映距离头本身。
    val_h_pred = val_h
    write_json(
        out_dir / "eval_note.json",
        {"range_eval": "验证阶段沿用 heading 标签作为占位,训练选择标准为 distance_rel_error。"},
    )

    data = {"train_r": train_r, "val_r": val_r, "val_h": val_h, "val_h_pred": val_h_pred}
    for mode in {cfg.input_mode}:
        data[f"x_train_{mode}"] = make_pair_input(train_feats, mode)
        data[f"x_val_{mode}"] = make_pair_input(val_feats, mode)
    train_one(args, cfg, data, out_dir, reproducibility)


if __name__ == "__main__":
    main()
