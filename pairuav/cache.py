"""冻结特征 cache 的轻量工具。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


CACHE_ARRAYS = ("features.npy", "heading.npy", "range.npy")
REQUIRED_CACHE_FILES = (*CACHE_ARRAYS, "json_paths.json", "meta.json")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_empty_output(out: Path) -> None:
    out = Path(out)
    if out.exists():
        if not out.is_dir() or any(out.iterdir()):
            raise FileExistsError(f"output directory must be empty: {out}")
        return
    out.mkdir(parents=True)


def require_complete_cache(source: Path) -> None:
    missing = [name for name in REQUIRED_CACHE_FILES if not (Path(source) / name).exists()]
    if missing:
        raise FileNotFoundError(f"cache is incomplete: {source}; missing={missing}")


def slice_cache(source: Path, out: Path, limit: int, offset: int = 0) -> dict[str, Any]:
    source = Path(source)
    out = Path(out)
    if limit <= 0:
        raise ValueError("--limit must be positive")
    start = int(offset)
    if start < 0:
        raise ValueError("--offset must be non-negative")
    end = start + int(limit)
    require_complete_cache(source)
    prepare_empty_output(out)
    for name in CACHE_ARRAYS:
        arr = np.load(source / name, mmap_mode="r")
        if end > arr.shape[0]:
            raise ValueError(f"slice [{start}:{end}] exceeds {name} length {arr.shape[0]}")
        np.save(out / name, np.asarray(arr[start:end]))
    json_paths = json.loads((source / "json_paths.json").read_text(encoding="utf-8"))
    write_json(out / "json_paths.json", json_paths[start:end])
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    meta.update({
        "source_cache": str(source),
        "slice_offset": start,
        "slice_limit": int(limit),
        "samples": int(limit),
    })
    write_json(out / "meta.json", meta)
    return meta


def link_cache(source: Path, out: Path) -> None:
    source = Path(source)
    out = Path(out)
    require_complete_cache(source)
    prepare_empty_output(out)
    for name in REQUIRED_CACHE_FILES:
        dst = out / name
        dst.symlink_to((source / name).resolve())


def copy_cache_meta(source: Path, out: Path) -> None:
    source = Path(source)
    out = Path(out)
    require_complete_cache(source)
    prepare_empty_output(out)
    for name in ("json_paths.json", "meta.json"):
        shutil.copy2(source / name, out / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结特征 cache 工具。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("slice", help="从已有 cache 切出小样本 cache")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--limit", type=int, required=True)
    p.add_argument("--offset", type=int, default=0)
    p.set_defaults(func=lambda args: print(json.dumps(
        slice_cache(args.cache_dir, args.out_dir, args.limit, args.offset),
        indent=2,
        ensure_ascii=False,
    )))

    p = sub.add_parser("link", help="为已有 cache 创建文件级软链接")
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.set_defaults(func=lambda args: link_cache(args.cache_dir, args.out_dir))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
