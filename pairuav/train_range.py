"""训练独立距离头。"""

from __future__ import annotations

import os

# Bound host-side indexing before NumPy/PyTorch initialize their thread pools.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .features import file_sha256, write_json
from .heads import RangeMLP, make_pair_input, pair_input_dim
from .metrics import circular_angle_abs_error, compute_metrics
from .reproducibility import DEFAULT_SEED, prepare_run_dir, seed_everything


@dataclass
class RangeConfig:
    name: str = "C_rel_rich"
    input_mode: str = "ab_diff_prod"
    loss: str = "rel_smooth"
    hidden_dim: int = 512
    depth: int = 2
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 5e-5
    batch_size: int = 512
    epochs: int = 240
    warmup_epochs: float = 2.0
    final_lr_frac: float = 0.05
    grad_clip: float = 1.0
    eps: float = 1.0


def load_config(path: Path) -> RangeConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(RangeConfig)}
    missing = expected - data.keys()
    unknown = data.keys() - expected
    if missing or unknown:
        raise ValueError(f"invalid range config fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    cfg = RangeConfig(**data)
    if not cfg.name or Path(cfg.name).name != cfg.name or cfg.name in {".", ".."}:
        raise ValueError(f"unsafe config name: {cfg.name!r}")
    pair_input_dim(4096, cfg.input_mode)
    if cfg.loss not in {"mse", "weighted_mse", "rel_l1", "rel_smooth"}:
        raise ValueError(f"unknown loss: {cfg.loss}")
    if cfg.lr <= 0 or cfg.batch_size <= 0 or cfg.epochs <= 0 or cfg.hidden_dim <= 0 or cfg.depth <= 0:
        raise ValueError("lr, batch_size, epochs, hidden_dim and depth must be positive")
    if not 0.0 <= cfg.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return cfg


def feature_cache_bytes(cache_dir: Path) -> int:
    features = np.load(Path(cache_dir) / "features.npy", mmap_mode="r")
    if features.ndim != 3 or features.shape[1] != 2 or features.dtype != np.float16:
        raise ValueError(f"unexpected feature cache: shape={features.shape}, dtype={features.dtype}")
    return int(features.nbytes)


def resolve_feature_device(
    requested: str,
    model_device: torch.device,
    cache_dirs: tuple[Path, ...],
) -> torch.device:
    if requested == "cpu" or model_device.type != "cuda":
        if requested == "cuda" and model_device.type != "cuda":
            raise RuntimeError("--feature-device cuda requires CUDA")
        return torch.device("cpu")
    if requested == "cuda":
        return model_device

    required = sum(feature_cache_bytes(path) for path in cache_dirs)
    with torch.cuda.device(model_device):
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
    reserve_bytes = 2 * 1024**3
    return model_device if required + reserve_bytes <= free_bytes else torch.device("cpu")


def load_cache(cache_dir: Path, storage_device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features_np = np.load(Path(cache_dir) / "features.npy", mmap_mode="c")
    if features_np.ndim != 3 or features_np.shape[1] != 2 or features_np.dtype != np.float16:
        raise ValueError(f"unexpected feature cache: shape={features_np.shape}, dtype={features_np.dtype}")
    feats = torch.from_numpy(features_np)
    heading = torch.from_numpy(np.load(Path(cache_dir) / "heading.npy").astype(np.float32, copy=False))
    range_m = torch.from_numpy(np.load(Path(cache_dir) / "range.npy").astype(np.float32, copy=False))
    if len(feats) != len(heading) or len(feats) != len(range_m):
        raise ValueError(
            f"cache length mismatch: features={len(feats)}, heading={len(heading)}, range={len(range_m)}"
        )
    if storage_device.type != "cpu":
        feats = feats.to(storage_device)
        heading = heading.to(storage_device)
        range_m = range_m.to(storage_device)
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
def eval_model(
    model: RangeMLP,
    val_feats: torch.Tensor,
    input_mode: str,
    val_h_pred: torch.Tensor,
    val_h: torch.Tensor,
    val_r: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.eval()
    predictions: list[torch.Tensor] = []
    for start in range(0, len(val_feats), batch_size):
        features = val_feats[start: start + batch_size].to(device, non_blocking=True)
        predictions.append(model(make_pair_input(features, input_mode)).cpu())
    return official_eval(val_h_pred, torch.cat(predictions), val_h, val_r)


def train_one(
    args: argparse.Namespace,
    cfg: RangeConfig,
    data: dict[str, torch.Tensor],
    out_dir: Path,
    reproducibility: dict[str, Any],
) -> dict[str, Any]:
    train_feats = data["train_feats"]
    val_feats = data["val_feats"]
    train_r = data["train_r"]
    val_h_pred = data["val_h_pred"]
    val_h = data["val_h"]
    val_r = data["val_r"]
    model = RangeMLP(
        pair_input_dim(int(train_feats.shape[-1]), cfg.input_mode),
        hidden_dim=cfg.hidden_dim,
        depth=cfg.depth,
        dropout=cfg.dropout,
        range_limit=args.range_limit,
    ).to(args.device_obj)
    model.init_mean(float(train_r.mean().detach().cpu()))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    steps_per_epoch = int(math.ceil(train_feats.shape[0] / cfg.batch_size))
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
        perm = torch.randperm(train_feats.shape[0], device=args.device_obj, generator=shuffle_generator)
        totals = {"loss": 0.0, "range_mae": 0.0, "n": 0}
        for start in range(0, train_feats.shape[0], cfg.batch_size):
            ids = perm[start: start + cfg.batch_size]
            storage_ids = ids if train_feats.device.type == "cuda" else ids.cpu()
            features = train_feats[storage_ids].to(args.device_obj, non_blocking=True)
            gt = train_r[storage_ids].to(args.device_obj, non_blocking=True)
            pred = model(make_pair_input(features, cfg.input_mode))
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
        val = eval_model(
            model,
            val_feats,
            cfg.input_mode,
            val_h_pred,
            val_h,
            val_r,
            args.device_obj,
            args.eval_batch_size,
        )
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
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="explicit JSON recipe; implicit training defaults are disabled",
    )
    parser.add_argument("--range-limit", type=float, default=132.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="where frozen features live; auto keeps them on CUDA only when they fit with a 2 GiB reserve",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=1)
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
    if args.cpu_threads < 1:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    reproducibility = seed_everything(
        args.seed,
        deterministic=args.deterministic,
        matmul_precision="high",
    )
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    reproducibility["config_source"] = {
        "path": str(args.config.resolve()),
        "sha256": file_sha256(args.config),
    }
    args.feature_device_obj = resolve_feature_device(
        args.feature_device,
        args.device_obj,
        (args.train_cache, args.val_cache),
    )
    reproducibility["feature_device"] = str(args.feature_device_obj)
    reproducibility["cpu_threads"] = torch.get_num_threads()
    print(
        f"[storage] model={args.device_obj}, features={args.feature_device_obj}, "
        f"cpu_threads={torch.get_num_threads()}",
        flush=True,
    )
    out_dir = args.output_dir / cfg.name
    prepare_run_dir(out_dir)
    write_json(out_dir / "config.json", asdict(cfg))

    train_feats, _train_h, train_r = load_cache(args.train_cache, args.feature_device_obj)
    val_feats, val_h, val_r = load_cache(args.val_cache, args.feature_device_obj)
    # 距离头训练只依赖 range 标签。验证时沿用 heading 标签作为占位,
    # 这样日志中的 distance_rel_error 与 range_mae 只反映距离头本身。
    val_h_pred = val_h
    write_json(
        out_dir / "eval_note.json",
        {"range_eval": "验证阶段沿用 heading 标签作为占位,训练选择标准为 distance_rel_error。"},
    )

    data = {
        "train_feats": train_feats,
        "val_feats": val_feats,
        "train_r": train_r,
        "val_r": val_r,
        "val_h": val_h,
        "val_h_pred": val_h_pred,
    }
    train_one(args, cfg, data, out_dir, reproducibility)


if __name__ == "__main__":
    main()
