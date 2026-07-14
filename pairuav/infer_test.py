"""连续 test 推理入口。

单距离头时输出两列 `heading range`;同时给出 `--range2-run-dir/--range2-ckpt` 时输出三列
`heading range range2`(供 `pairuav.gate_merge` 做双距离头门控合成)。
`--start/--end` 用于多卡/多机分片:各分片推理 pair 清单的 [start, end) 区间,产物按行拼接或交给
gate_merge 按偏移合并。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairListDataset, image_pair_collate, load_test_pairs, pair_list_sha256
from .features import build_vggt, extract_pooled_features, file_sha256
from .head_io import config_beside_checkpoint, config_in_directory, load_range_head, load_sixdof_head
from .heads import heading_from_rot_torch, make_pair_input
from .metrics import wrap180
from .reproducibility import DEFAULT_SEED, dataloader_generator, seed_everything, seed_worker


EXPECTED_TEST_PAIRS = 2_773_116


def write_json_atomic(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def file_record(label: str, path: Path) -> dict[str, str]:
    path = Path(path)
    print(f"[hash] {label}: {path}", flush=True)
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}


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
    parser.add_argument("--range2-run-dir", type=Path, default=None, help="第二距离头(门控用);与 --range2-ckpt 成对出现")
    parser.add_argument("--range2-ckpt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs-cache", type=Path, default=None)
    parser.add_argument(
        "--trust-pairs-cache",
        action="store_true",
        help="allow a legacy pairs cache without its .meta.json sidecar",
    )
    parser.add_argument("--start", type=int, default=0, help="分片起点(含),对完整 pair 清单切片")
    parser.add_argument("--end", type=int, default=0, help="分片终点(不含);0 = 到末尾")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output instead of resuming it")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="request deterministic PyTorch algorithms (default: enabled)",
    )
    args = parser.parse_args()
    if (args.range2_run_dir is None) != (args.range2_ckpt is None):
        parser.error("--range2-run-dir 与 --range2-ckpt 必须成对出现")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pairs = load_test_pairs(
        args.test_json_dir,
        args.pairs_cache,
        trust_existing_cache=args.trust_pairs_cache,
    )
    selected_start = 0
    selected_end = len(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
        selected_end = len(pairs)
    elif len(pairs) != EXPECTED_TEST_PAIRS:
        raise ValueError(f"pair count {len(pairs)} != expected {EXPECTED_TEST_PAIRS}")
    if args.start or args.end:
        end = args.end or len(pairs)
        if args.start < 0 or end < args.start or end > len(pairs):
            raise ValueError(f"invalid shard [{args.start}, {end}) for {len(pairs)} pairs")
        selected_start, selected_end = args.start, end
        pairs = pairs[args.start: end]
        print(f"[shard] rows [{args.start}, {end}) -> {len(pairs)} pairs", flush=True)

    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    if args.resume and args.out.exists() != meta_path.exists():
        raise RuntimeError(
            f"cannot resume a partial state; output and metadata must both exist: {args.out}, {meta_path}"
        )
    if args.resume and not args.out.exists():
        raise FileNotFoundError(f"cannot resume because output does not exist: {args.out}")
    if not args.resume and (args.out.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(f"output exists; use --resume or --overwrite: {args.out}")

    angle_ckpt = Path(args.angle_run_root) / args.angle_name / args.angle_ckpt
    angle_config = config_beside_checkpoint(angle_ckpt)
    range_config = config_in_directory(args.range_run_dir)
    range2_config = config_in_directory(args.range2_run_dir) if args.range2_run_dir is not None else None
    run_meta = {
        "test_json_dir": str(Path(args.test_json_dir).resolve()),
        "test_image_dir": str(Path(args.test_image_dir).resolve()),
        "pair_count": len(pairs),
        "pairs_sha256": pair_list_sha256(pairs),
        "selected_start": selected_start,
        "selected_end": selected_end,
        "image_size": args.image_size,
        "output_columns": 3 if args.range2_ckpt is not None else 2,
        "seed": args.seed,
        "angle_checkpoint": file_record("angle checkpoint", angle_ckpt),
        "angle_config": file_record("angle config", angle_config),
        "range_checkpoint": file_record("range checkpoint", args.range_ckpt),
        "range_config": file_record("range config", range_config),
        "range2_checkpoint": (
            file_record("range2 checkpoint", args.range2_ckpt)
            if args.range2_ckpt is not None else None
        ),
        "range2_config": (
            file_record("range2 config", range2_config)
            if range2_config is not None else None
        ),
        "vggt_weight": file_record("VGGT weight", args.vggt_weight),
    }
    if args.resume:
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing_meta != run_meta:
            raise RuntimeError("resume metadata differs from the requested inference run")

    done = truncate_to_complete_lines(args.out) if args.resume else 0
    if done > len(pairs):
        raise RuntimeError(f"output has {done} lines but this run expects only {len(pairs)}")
    if done == len(pairs):
        print(f"[done] {args.out} already has {done} lines", flush=True)
        return
    pairs_todo = pairs[done:]

    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    vggt = build_vggt(device, args.vggt_weight)
    angle_head, _angle_cfg, _angle_source = load_sixdof_head(
        angle_ckpt,
        4096,
        device,
        config=angle_config,
    )
    range_head, range_cfg, _range_source = load_range_head(
        args.range_ckpt,
        4096,
        device,
        config=range_config,
    )
    range_input_mode = str(range_cfg["input_mode"])
    range2_head = None
    range2_input_mode = ""
    if args.range2_run_dir is not None:
        assert args.range2_ckpt is not None and range2_config is not None
        range2_head, range2_cfg, _range2_source = load_range_head(
            args.range2_ckpt,
            4096,
            device,
            config=range2_config,
        )
        range2_input_mode = str(range2_cfg["input_mode"])

    if not args.resume:
        # Do not replace metadata for an existing output until all weights load successfully.
        write_json_atomic(meta_path, run_meta)

    dataset = PairListDataset(pairs_todo, args.test_image_dir, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=image_pair_collate,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(args.seed),
    )
    mode = "a" if done > 0 else "w"
    with args.out.open(mode, encoding="utf-8") as handle, torch.inference_mode():
        for batch_idx, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            feats = extract_pooled_features(vggt, images, dtype=dtype)
            rot, _trans = angle_head(feats)
            heading = heading_from_rot_torch(rot).detach().cpu().numpy().astype(np.float64)
            range_m = range_head(make_pair_input(feats, range_input_mode)).detach().cpu().numpy().astype(np.float64)
            heading = wrap180(heading)
            if range2_head is None:
                handle.writelines(f"{h:.6f} {r:.6f}\n" for h, r in zip(heading, range_m))
            else:
                range2_m = range2_head(make_pair_input(feats, range2_input_mode)).detach().cpu().numpy().astype(np.float64)
                handle.writelines(f"{h:.6f} {r:.6f} {r2:.6f}\n" for h, r, r2 in zip(heading, range_m, range2_m))
            done += int(images.shape[0])
            if batch_idx == 0 or batch_idx % 100 == 0:
                handle.flush()
                print(f"[infer] {done}/{len(pairs)}", flush=True)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
