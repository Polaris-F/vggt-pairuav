"""冻结 VGGT 特征抽取与 cache 写入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairImageDataset, pair_collate
from .reproducibility import DEFAULT_SEED, dataloader_generator, seed_everything, seed_worker

if TYPE_CHECKING:
    from vggt.models.vggt import VGGT


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_vggt(device: torch.device, weight: Path) -> "VGGT":
    """加载并冻结 VGGT。"""

    from vggt.models.vggt import VGGT

    model = VGGT()
    state = torch.load(weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    for param in model.parameters():
        param.requires_grad_(False)
    return model.to(device).eval()


@torch.no_grad()
def extract_pooled_features(vggt: "VGGT", images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """使用 VGGT aggregator 最后一层 patch tokens 的 mean/max pooling。"""

    with torch.amp.autocast("cuda", dtype=dtype, enabled=images.is_cuda):
        aggregated_tokens_list, patch_start_idx = vggt.aggregator(images)
        final_tokens = aggregated_tokens_list[-1]
        patch_tokens = final_tokens[:, :, patch_start_idx:, :]
        mean_pool = patch_tokens.mean(dim=2)
        max_pool = patch_tokens.max(dim=2).values
        return torch.cat([mean_pool, max_pool], dim=-1)


def cache_complete(cache_dir: Path) -> bool:
    required = [
        "features.npy",
        "heading.npy",
        "range.npy",
        "json_paths.json",
        "meta.json",
    ]
    if not all((cache_dir / name).exists() for name in required):
        return False
    try:
        meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
        paths = json.loads((cache_dir / "json_paths.json").read_text(encoding="utf-8"))
        samples = int(meta["samples"])
        features = np.load(cache_dir / "features.npy", mmap_mode="r")
        heading = np.load(cache_dir / "heading.npy", mmap_mode="r")
        range_m = np.load(cache_dir / "range.npy", mmap_mode="r")
        return (
            isinstance(paths, list)
            and len(paths) == samples
            and features.ndim == 3
            and features.shape[:2] == (samples, 2)
            and heading.shape == (samples,)
            and range_m.shape == (samples,)
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def prepare_feature_cache(
    *,
    name: str,
    json_dir: Path,
    image_dir: Path,
    cache_dir: Path,
    image_size: int,
    max_pairs: int | None,
    batch_size: int,
    workers: int,
    device: torch.device,
    vggt_weight: Path,
    vggt_weight_sha256: str,
    seed: int,
    reproducibility: dict[str, Any],
    force: bool = False,
) -> dict[str, Any]:
    """抽取冻结 VGGT 特征并写入 cache 目录。"""

    cache_dir = Path(cache_dir)
    expected_files = {"features.npy", "heading.npy", "range.npy", "json_paths.json", "meta.json"}
    existing = sorted(path.name for path in cache_dir.iterdir() if path.name in expected_files) if cache_dir.exists() else []
    if cache_complete(cache_dir):
        if force:
            raise RuntimeError(
                f"refusing to overwrite complete cache {cache_dir}; move it aside or choose a new cache root"
            )
        return json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
    if existing:
        raise RuntimeError(
            f"refusing to reuse incomplete cache {cache_dir}; move it aside before retrying. existing={existing}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = PairImageDataset(json_dir, image_dir, image_size=image_size, max_pairs=max_pairs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=pair_collate,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=dataloader_generator(seed),
    )
    model = build_vggt(device, vggt_weight)
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16

    t0 = time.time()
    features_arr = None
    heading_arr = np.lib.format.open_memmap(cache_dir / "heading.npy", mode="w+", dtype=np.float32, shape=(len(dataset),))
    range_arr = np.lib.format.open_memmap(cache_dir / "range.npy", mode="w+", dtype=np.float32, shape=(len(dataset),))
    json_paths: list[str] = []
    offset = 0
    pooled_dim = None
    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        feats = extract_pooled_features(model, images, dtype=dtype).detach().cpu().to(torch.float16).numpy()
        if features_arr is None:
            pooled_dim = int(feats.shape[-1])
            features_arr = np.lib.format.open_memmap(
                cache_dir / "features.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(dataset), 2, pooled_dim),
            )
        bs = int(feats.shape[0])
        features_arr[offset: offset + bs] = feats
        heading_arr[offset: offset + bs] = batch["heading"].numpy().astype(np.float32)
        range_arr[offset: offset + bs] = batch["range"].numpy().astype(np.float32)
        json_paths.extend(batch["json_path"])
        offset += bs
        if step == 0 or (step + 1) % 25 == 0 or offset == len(dataset):
            elapsed = time.time() - t0
            print(f"[cache:{name}] {offset}/{len(dataset)} elapsed={elapsed:.1f}s rate={offset/max(elapsed, 1e-9):.2f}/s", flush=True)

    assert features_arr is not None and pooled_dim is not None
    features_arr.flush()
    heading_arr.flush()
    range_arr.flush()
    meta = {
        "name": name,
        "json_dir": str(Path(json_dir)),
        "image_dir": str(Path(image_dir)),
        "image_size": int(image_size),
        "max_pairs": max_pairs,
        "samples": int(len(dataset)),
        "pooled_dim": int(pooled_dim),
        "feature": "VGGT aggregator final patch tokens per-view mean+max",
        "dtype": "float16",
        "vggt_weight": str(Path(vggt_weight)),
        "vggt_weight_sha256": vggt_weight_sha256,
        "elapsed_sec": time.time() - t0,
        "range_mean": float(np.asarray(range_arr).mean()),
        "range_std": float(np.asarray(range_arr).std()),
        "heading_mean": float(np.asarray(heading_arr).mean()),
        "seed": int(seed),
        "reproducibility": reproducibility,
        "json_paths_sha256": hashlib.sha256(
            ("\n".join(json_paths) + "\n").encode("utf-8")
        ).hexdigest(),
    }
    write_json(cache_dir / "meta.json", meta)
    write_json(cache_dir / "json_paths.json", json_paths)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return meta


def parse_cache_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抽取冻结 VGGT pair 特征 cache。")
    parser.add_argument("--train-json-dir", type=Path, required=True)
    parser.add_argument("--val-json-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--vggt-weight", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-val-pairs", type=int, default=None)
    parser.add_argument(
        "--force-cache",
        action="store_true",
        help="deprecated safety guard: existing caches are never overwritten; move them aside first",
    )
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
    args = parse_cache_args()
    if args.force_cache:
        raise RuntimeError(
            "--force-cache does not overwrite caches; move the old cache aside or choose a new cache root"
        )
    reproducibility = seed_everything(args.seed, deterministic=args.deterministic)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_tag = f"train_n{args.max_train_pairs or 'full'}_s{args.image_size}"
    val_tag = f"val_n{args.max_val_pairs or 'full'}_s{args.image_size}"
    train_cache = args.cache_root / train_tag
    val_cache = args.cache_root / val_tag
    needs_weight = not (cache_complete(train_cache) and cache_complete(val_cache))
    if needs_weight:
        print(f"[weights] sha256 {args.vggt_weight}", flush=True)
        vggt_weight_sha256 = file_sha256(args.vggt_weight)
    else:
        vggt_weight_sha256 = "already recorded in existing cache metadata"
    prepare_feature_cache(
        name="train",
        json_dir=args.train_json_dir,
        image_dir=args.image_dir,
        cache_dir=train_cache,
        image_size=args.image_size,
        max_pairs=args.max_train_pairs,
        batch_size=args.extract_batch_size,
        workers=args.workers,
        device=device,
        vggt_weight=args.vggt_weight,
        vggt_weight_sha256=vggt_weight_sha256,
        seed=args.seed,
        reproducibility=reproducibility,
        force=args.force_cache,
    )
    prepare_feature_cache(
        name="val",
        json_dir=args.val_json_dir,
        image_dir=args.image_dir,
        cache_dir=val_cache,
        image_size=args.image_size,
        max_pairs=args.max_val_pairs,
        batch_size=args.extract_batch_size,
        workers=args.workers,
        device=device,
        vggt_weight=args.vggt_weight,
        vggt_weight_sha256=vggt_weight_sha256,
        seed=args.seed,
        reproducibility=reproducibility,
        force=args.force_cache,
    )


if __name__ == "__main__":
    main()
