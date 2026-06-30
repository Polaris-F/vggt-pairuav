"""PairUAV 数据 split index 的生成、重建与校验工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np


PAIR_RE = re.compile(r"^(\d{2})_(\d{2})\.json$")


def extract_int(value: object) -> int | float:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else math.inf


def json_sort_key(path: Path) -> tuple[int | float, str, int | float, str]:
    path = Path(path)
    return (extract_int(path.parent.name), path.parent.name, extract_int(path.stem), path.stem)


def iter_json_paths(json_dir: Path) -> list[Path]:
    json_dir = Path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(json_dir)
    paths = sorted((path for path in json_dir.rglob("*.json") if PAIR_RE.match(path.name)), key=json_sort_key)
    if not paths:
        raise RuntimeError(f"No JSON files under {json_dir}")
    return paths


def read_index(path: Path) -> list[str]:
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    if not lines:
        raise RuntimeError(f"empty index: {path}")
    return lines


def write_index(path: Path, rel_paths: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")


def sha256_lines(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def path_suffix(path: str | Path, depth: int = 2) -> str:
    parts = Path(path).parts
    if len(parts) < depth:
        return str(Path(*parts))
    return str(Path(*parts[-depth:]))


def build_manifest(name: str, rel_paths: list[str]) -> dict[str, Any]:
    buildings = sorted({Path(p).parts[0] for p in rel_paths})
    frames = []
    for rel in rel_paths:
        match = PAIR_RE.match(Path(rel).name)
        if match:
            frames.append((int(match.group(1)), int(match.group(2))))
    return {
        "name": name,
        "count": len(rel_paths),
        "sha256": sha256_lines(rel_paths),
        "buildings": len(buildings),
        "first": rel_paths[:5],
        "last": rel_paths[-5:],
        "frame_pair_min": min(frames) if frames else None,
        "frame_pair_max": max(frames) if frames else None,
    }


def command_export(args: argparse.Namespace) -> None:
    json_dir = Path(args.json_dir)
    rel_paths = [str(path.relative_to(json_dir)) for path in iter_json_paths(json_dir)]
    if args.limit:
        rel_paths = rel_paths[: args.limit]
    write_index(args.out, rel_paths)
    manifest = build_manifest(args.name or args.out.stem, rel_paths)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def command_materialize(args: argparse.Namespace) -> None:
    index = read_index(args.index)
    source = Path(args.source_json_dir)
    out = Path(args.out_json_dir)
    copied = 0
    for rel in index:
        src = source / rel
        dst = out / rel
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    manifest = build_manifest(args.name or args.index.stem, index)
    manifest.update({"source": str(source), "out": str(out), "copied": copied})
    (out / "index_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def command_verify_json(args: argparse.Namespace) -> None:
    index = read_index(args.index)
    json_dir = Path(args.json_dir)
    missing = [rel for rel in index if not (json_dir / rel).exists()]
    out = {
        "index": str(args.index),
        "json_dir": str(json_dir),
        "index_count": len(index),
        "missing_count": len(missing),
        "index_sha256": sha256_lines(index),
        "first_missing": missing[:10],
    }
    same_order = True
    if args.require_same_order:
        actual = [str(path.relative_to(json_dir)) for path in iter_json_paths(json_dir)]
        same_order = actual == index
        out.update({
            "json_count": len(actual),
            "same_order": same_order,
            "json_sha256": sha256_lines(actual),
        })
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    if missing or (args.require_same_order and not same_order):
        raise SystemExit(1)


def command_verify_cache(args: argparse.Namespace) -> None:
    index = read_index(args.index)
    cache_dir = Path(args.cache_dir)
    cached_paths_file = cache_dir / "json_paths.json"
    if not cached_paths_file.exists():
        raise FileNotFoundError(cached_paths_file)
    cached_paths = json.loads(cached_paths_file.read_text(encoding="utf-8"))
    cached_rel = [path_suffix(path, depth=2) for path in cached_paths]
    out: dict[str, Any] = {
        "index": str(args.index),
        "cache_dir": str(cache_dir),
        "index_count": len(index),
        "cache_count": len(cached_rel),
        "same_order": cached_rel == index,
        "index_sha256": sha256_lines(index),
        "cache_sha256": sha256_lines(cached_rel),
        "first_index": index[:5],
        "first_cache": cached_rel[:5],
        "last_index": index[-5:],
        "last_cache": cached_rel[-5:],
    }
    if (cache_dir / "features.npy").exists():
        features = np.load(cache_dir / "features.npy", mmap_mode="r")
        out["features_shape"] = list(features.shape)
        out["features_dtype"] = str(features.dtype)
    if (cache_dir / "heading.npy").exists():
        out["heading_shape"] = list(np.load(cache_dir / "heading.npy", mmap_mode="r").shape)
    if (cache_dir / "range.npy").exists():
        out["range_shape"] = list(np.load(cache_dir / "range.npy", mmap_mode="r").shape)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    if not out["same_order"]:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PairUAV split index 工具。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export", help="从 json split 目录导出相对路径 index")
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=command_export)

    p = sub.add_parser("materialize", help="按 index 从源 json 目录复制出固定 split")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--source-json-dir", type=Path, required=True)
    p.add_argument("--out-json-dir", type=Path, required=True)
    p.add_argument("--name", default=None)
    p.set_defaults(func=command_materialize)

    p = sub.add_parser("verify-json", help="校验 json 目录是否覆盖 index")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--json-dir", type=Path, required=True)
    p.add_argument("--require-same-order", action="store_true")
    p.set_defaults(func=command_verify_json)

    p = sub.add_parser("verify-cache", help="校验特征 cache 与 index 的 pair 顺序是否一致")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.set_defaults(func=command_verify_cache)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
