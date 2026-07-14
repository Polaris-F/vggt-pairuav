"""在验证 cache 上组合评测角度头、距离头和 MAP-hard 解码。

直接吃 checkpoint 文件(`.pt`),配置从同目录的 `result.json` / `config.json` 自动读取,
后处理(门控 + MAP-hard)在内部完成 —— 不再需要先落 txt 再跑 `postproc_maphard`。

用法
----
历史提交结构(三个头,= Codabench 822841 的链路):
    python -m pairuav.eval_val \
      --val-cache workspace/cache/full/val_nfull_s518 \
      --val-geom  workspace/geometry/geometry_labels_val.npz \
      --angle-ckpt  ckpt/S0_rich_noc/head_best_angle.pt \
      --range-ckpt  ckpt/C_rel_rich/range_head_best_distance.pt \
      --range2-ckpt ckpt/B_mse_ab/range_head_best_distance.pt \
      --threshold 80

确定性重训的默认单距离头:去掉 `--range2-ckpt` 即可。多种子评测表明 80 m 门控没有稳定增益。

评出的配置(每个都给全部指标列)
--------------------------------
    pose_head_only      角度头自读 range(用预测平移的 z 分量投影)—— 证明必须配独立距离头
    range_C             角度头 + 近距距离头
    range_B             角度头 + 远距距离头            (仅在给了 --range2-ckpt 时)
    gate                角度头 + 门控(|C| > T 时换 B) (仅在给了 --range2-ckpt 时)
    <上面每个连续配置> + _maphard   D 空间 MAP-hard 解码后的结果

指标:angle_rel / dist_rel / final(官方相对口径)、MAE_H(度)、MAE_R(米)、
      SR@{1,2,5,10}m 与端点 MAE(端点 = 把预测位姿施加到源相机的轨迹坐标)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import geometry as G
from .features import write_json
from .geometry import grid_from_d_values
from .head_io import load_range_head, load_sixdof_head
from .heads import heading_from_rot_torch, make_pair_input
from .metrics import compute_metrics_np, wrap180
from .postproc_maphard import default_weights_path, map_decode
from .reproducibility import DEFAULT_SEED, seed_everything

SIN45, COS45 = float(np.sin(np.deg2rad(45.0))), float(np.cos(np.deg2rad(45.0)))
SR_TAUS = (1.0, 2.0, 5.0, 10.0)


# --------------------------------------------------------------------- 指标
def _pos(alpha_deg: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """规范目标坐标系下的相机位置。"""
    a = np.deg2rad(alpha_deg)
    horiz = rho * SIN45
    return np.stack([horiz * np.sin(a), horiz * np.cos(a), rho * COS45], axis=-1)


def _metrics(pred_h, pred_r, gt_h, gt_r, alpha_a, rho_a, p_gt) -> dict[str, float]:
    m = compute_metrics_np(pred_h, pred_r, gt_h, gt_r)
    err = np.linalg.norm(_pos(alpha_a + pred_h, rho_a + pred_r) - p_gt, axis=-1)
    out = {
        "angle_rel": round(float(m["angle_rel_error"]), 6),
        "dist_rel": round(float(m["distance_rel_error"]), 6),
        "final": round(float(m["final_score"]), 6),
        "MAE_H_deg": round(float(np.abs(wrap180(pred_h - gt_h)).mean()), 4),
        "MAE_R_m": round(float(np.abs(pred_r - gt_r).mean()), 4),
        "endpoint_MAE_m": round(float(err.mean()), 4),
    }
    for tau in SR_TAUS:
        out[f"SR@{tau:g}m"] = round(float((err < tau).mean() * 100.0), 2)
    return out


# --------------------------------------------------------------------- 主流程
@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    feats = np.load(Path(args.val_cache) / "features.npy", mmap_mode="r")
    feat_dim = int(feats.shape[-1])

    lab = np.load(args.val_geom)
    gt_h = lab["heading"].astype(np.float64)
    gt_r = lab["range"].astype(np.float64)
    step_a = lab["step_a"].astype(np.float64)
    step_b = lab["step_b"].astype(np.float64)

    # 端点几何:把预测位姿施加到源相机在轨迹上的实际位置(轨迹非平移不变,必须用真实 step_a)
    alpha_a, rho_a = 4.0 * step_a, 256.0 - 0.5 * step_a
    p_gt = _pos(4.0 * step_b, 256.0 - 0.5 * step_b)
    assert np.abs(_pos(alpha_a + gt_h, rho_a + gt_r) - p_gt).max() < 1e-6, "端点定义与官方标签不自洽"

    angle, _angle_cfg, _angle_source = load_sixdof_head(args.angle_ckpt, feat_dim, device)
    head_c, cfg_c, _range_source = load_range_head(args.range_ckpt, feat_dim, device)
    mode_c = str(cfg_c["input_mode"])
    head_b, mode_b = (None, "")
    if args.range2_ckpt is not None:
        head_b, cfg_b, _range2_source = load_range_head(args.range2_ckpt, feat_dim, device)
        mode_b = str(cfg_b["input_mode"])

    ph, pr_pose, pr_c, pr_b = [], [], [], []
    for s in range(0, feats.shape[0], args.batch_size):
        x = torch.from_numpy(np.asarray(feats[s:s + args.batch_size], dtype=np.float32)).to(device)
        rot, trans = angle(x)
        ph.append(heading_from_rot_torch(rot).cpu().numpy().astype(np.float64))
        pr_pose.append((trans[:, 2] / G.COS45).cpu().numpy().astype(np.float64))
        pr_c.append(head_c(make_pair_input(x, mode_c)).cpu().numpy().astype(np.float64))
        if head_b is not None:
            pr_b.append(head_b(make_pair_input(x, mode_b)).cpu().numpy().astype(np.float64))
    ph = np.concatenate(ph)
    pr_pose = np.concatenate(pr_pose)
    pr_c = np.concatenate(pr_c)

    # ---- 连续配置 ----
    configs: dict[str, np.ndarray] = {"pose_head_only": pr_pose, "range_C": pr_c}
    if head_b is not None:
        pr_b = np.concatenate(pr_b)
        configs["range_B"] = pr_b
        # 门控:用**近距头**的预测判阈值,超过则换远距头(与 pairuav.gate_merge 同规则)
        configs["gate"] = np.where(np.abs(pr_c) > args.threshold, pr_b, pr_c)

    # ---- MAP-hard 解码(内部完成,复用 postproc_maphard.map_decode)----
    grid_d, grid_h, grid_r = grid_from_d_values(None)
    w = json.loads(Path(args.map_weights or default_weights_path()).read_text(encoding="utf-8"))

    results: dict[str, Any] = {}
    for name, pr in configs.items():
        results[name] = _metrics(ph, pr, gt_h, gt_r, alpha_a, rho_a, p_gt)
        if name == "pose_head_only":
            continue                                   # 位姿头自读的 range 不值得解码
        d_hat = map_decode(ph, pr, grid_d, grid_h, grid_r, float(w["w_h"]), float(w["w_r"]))
        dh, dr = wrap180(4.0 * d_hat.astype(np.float64)), -0.5 * d_hat.astype(np.float64)
        r = _metrics(dh, dr, gt_h, gt_r, alpha_a, rho_a, p_gt)
        r["D_hit"] = round(float((d_hat == lab["d"].astype(np.int64)).mean() * 100.0), 2)
        results[f"{name}_maphard"] = r

    main = "gate" if head_b is not None else "range_C"
    return {
        "val_cache": str(args.val_cache),
        "val_geom": str(args.val_geom),
        "n_pairs": int(feats.shape[0]),
        "checkpoints": {
            "angle": str(args.angle_ckpt),
            "range_close": str(args.range_ckpt),
            "range_far": str(args.range2_ckpt) if args.range2_ckpt else None,
        },
        "gate_threshold_m": args.threshold if head_b is not None else None,
        "gate_switched_rows": int((np.abs(pr_c) > args.threshold).sum()) if head_b is not None else None,
        "primary_system": f"{main} + maphard",
        "submitted_system": "gate + maphard" if head_b is not None else None,
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="在验证 cache 上组合评测角度头、一个/两个距离头和 MAP-hard。")
    p.add_argument("--val-cache", type=Path, required=True)
    p.add_argument("--val-geom", type=Path, required=True)
    p.add_argument("--angle-ckpt", type=Path, required=True, help="角度头 .pt(配置从同目录自动读)")
    p.add_argument("--range-ckpt", type=Path, required=True, help="近距距离头 .pt")
    p.add_argument("--range2-ckpt", type=Path, default=None, help="远距距离头 .pt;给了才启用门控")
    p.add_argument("--threshold", type=float, default=80.0, help="门控阈值(米);|近距头预测| > 阈值时换远距头")
    p.add_argument("--map-weights", type=Path, default=None, help="MAP-hard 权重 json;缺省用随包资源")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch algorithms (default: enabled)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(
        args.seed,
        deterministic=args.deterministic,
        matmul_precision="high",
    )
    res = evaluate(args)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)
    if args.out is not None:
        write_json(args.out, res)


if __name__ == "__main__":
    main()
