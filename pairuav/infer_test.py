"""连续 test 推理入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairListDataset, image_pair_collate, load_test_pairs
from .features import build_vggt, extract_pooled_features
from .heads import RangeMLP, SixDofHead, heading_from_rot_torch, make_pair_input
from .metrics import wrap180


EXPECTED_TEST_PAIRS = 2_773_116


def load_sixdof_head(run_root: Path, name: str, ckpt: str, device: torch.device) -> SixDofHead:
    cfg = json.loads((Path(run_root) / name / "config.json").read_text(encoding="utf-8"))
    model = SixDofHead(
        feat_dim=4096,
        hidden_dim=int(cfg["hidden_dim"]),
        input_mode=str(cfg["input_mode"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(torch.load(Path(run_root) / name / ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval()


def load_range_head(run_dir: Path, ckpt: Path, device: torch.device) -> tuple[RangeMLP, dict]:
    result_path = Path(run_dir) / "result.json"
    if result_path.exists():
        cfg = json.loads(result_path.read_text(encoding="utf-8"))["config"]
    else:
        cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    in_dims = {"ab": 8192, "ab_diff_prod": 16384, "diff_prod": 8192}
    model = RangeMLP(
        in_dim=in_dims[str(cfg["input_mode"])],
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
        range_limit=132.0,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to(device).eval(), cfg


def truncate_to_complete_lines(path: Path) -> int:
    if not path.exists():
        return 0
    data = path.read_bytes()
    if not data:
        return 0
    last_newline = data.rfind(b"\n")
    if last_newline != len(data) - 1:
        path.write_bytes(data[: last_newline + 1])
        data = data[: last_newline + 1]
    return data.count(b"\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 VGGT + 外置任务头的连续 test 推理。")
    parser.add_argument("--test-json-dir", type=Path, required=True)
    parser.add_argument("--test-image-dir", type=Path, required=True)
    parser.add_argument("--vggt-weight", type=Path, required=True)
    parser.add_argument("--angle-run-root", type=Path, required=True)
    parser.add_argument("--angle-name", default="S0_rich_noc")
    parser.add_argument("--angle-ckpt", default="head_best_angle.pt")
    parser.add_argument("--range-run-dir", type=Path, required=True)
    parser.add_argument("--range-ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-cache", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_test_pairs(args.test_json_dir, args.pairs_cache)
    if args.limit:
        pairs = pairs[: args.limit]
    elif len(pairs) != EXPECTED_TEST_PAIRS:
        raise ValueError(f"pair count {len(pairs)} != expected {EXPECTED_TEST_PAIRS}")

    done = truncate_to_complete_lines(args.out) if args.resume else 0
    if done >= len(pairs):
        print(f"[done] {args.out} already has {done} lines", flush=True)
        return
    pairs_todo = pairs[done:]

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    vggt = build_vggt(device, args.vggt_weight)
    angle_head = load_sixdof_head(args.angle_run_root, args.angle_name, args.angle_ckpt, device)
    range_head, range_cfg = load_range_head(args.range_run_dir, args.range_ckpt, device)
    range_input_mode = str(range_cfg["input_mode"])

    dataset = PairListDataset(pairs_todo, args.test_image_dir, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=image_pair_collate)
    mode = "a" if done > 0 else "w"
    with args.out.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for batch_idx, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            feats = extract_pooled_features(vggt, images, dtype=dtype)
            rot, _trans = angle_head(feats)
            heading = heading_from_rot_torch(rot).detach().cpu().numpy().astype(np.float64)
            range_m = range_head(make_pair_input(feats, range_input_mode)).detach().cpu().numpy().astype(np.float64)
            heading = wrap180(heading)
            handle.writelines(f"{h:.6f} {r:.6f}\n" for h, r in zip(heading, range_m))
            done += int(images.shape[0])
            if batch_idx == 0 or batch_idx % 100 == 0:
                handle.flush()
                print(f"[infer] {done}/{len(pairs)}", flush=True)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
