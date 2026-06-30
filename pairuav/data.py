"""PairUAV 数据读取与确定性 pair 顺序。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


PAIR_JSON_RE = re.compile(r"^\d{2}_\d{2}\.json$")


def extract_int(value: object) -> int | float:
    """从字符串中提取第一个整数,用于自然排序。"""

    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else math.inf


def json_sort_key(path: Path) -> tuple[int | float, str, int | float, str]:
    path = Path(path)
    return (extract_int(path.parent.name), path.parent.name, extract_int(path.stem), path.stem)


def iter_json_paths(json_dir: Path) -> list[Path]:
    """按 PairUAV/Codabench 约定顺序列出 json pair 文件。"""

    json_dir = Path(json_dir)
    if not json_dir.exists():
        raise FileNotFoundError(json_dir)
    paths: list[Path] = []
    for child in sorted(json_dir.iterdir(), key=lambda p: (extract_int(p.name), p.name)):
        if child.is_dir():
            paths.extend(sorted((p for p in child.glob("*.json") if PAIR_JSON_RE.match(p.name)), key=lambda p: (extract_int(p.stem), p.stem)))
        elif child.is_file() and PAIR_JSON_RE.match(child.name):
            paths.append(child)
    paths = sorted(paths, key=json_sort_key)
    if not paths:
        raise RuntimeError(f"No JSON files under {json_dir}")
    return paths


def read_pair_meta(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_image_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((int(image_size), int(image_size)), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])


def resolve_image_path(image_root: Path, rel: str | Path) -> Path:
    """解析 json 中的图像路径。

    json 里可能保存相对路径,也可能只保存文件名。这里按常见两种方式尝试。
    """

    image_root = Path(image_root)
    rel_path = Path(str(rel))
    candidates = [image_root / rel_path, image_root / rel_path.name]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for {rel}; tried {candidates}")


def load_rgb_tensor(path: Path, transform: T.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


class PairImageDataset(Dataset):
    """训练/验证 pair 数据集,返回两张图像及 heading/range 标签。"""

    def __init__(
        self,
        json_dir: Path,
        image_dir: Path,
        image_size: int,
        max_pairs: int | None = None,
    ) -> None:
        self.json_paths = iter_json_paths(Path(json_dir))
        if max_pairs is not None:
            self.json_paths = self.json_paths[: int(max_pairs)]
        self.image_dir = Path(image_dir)
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.json_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.json_paths[int(idx)]
        meta = read_pair_meta(path)
        image_a = load_rgb_tensor(resolve_image_path(self.image_dir, meta["image_a"]), self.transform)
        image_b = load_rgb_tensor(resolve_image_path(self.image_dir, meta["image_b"]), self.transform)
        return {
            "images": torch.stack([image_a, image_b], dim=0),
            "heading": torch.tensor(float(meta["heading_num"]), dtype=torch.float32),
            "range": torch.tensor(float(meta["range_num"]), dtype=torch.float32),
            "json_path": str(path),
        }


def pair_collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": torch.stack([item["images"] for item in batch], dim=0),
        "heading": torch.stack([item["heading"] for item in batch], dim=0),
        "range": torch.stack([item["range"] for item in batch], dim=0),
        "json_path": [str(item["json_path"]) for item in batch],
    }


def load_test_pairs(json_dir: Path, pairs_cache: Path | None = None) -> list[tuple[str, str]]:
    """读取测试 pair 的图像路径顺序,可用文本 cache 加速重复运行。"""

    if pairs_cache is not None:
        pairs_cache = Path(pairs_cache)
        if pairs_cache.exists():
            pairs: list[tuple[str, str]] = []
            with pairs_cache.open("r", encoding="utf-8") as handle:
                for line in handle:
                    a, b = line.split()
                    pairs.append((a, b))
            return pairs

    pairs = []
    for path in iter_json_paths(Path(json_dir)):
        meta = read_pair_meta(path)
        pairs.append((str(meta["image_a"]), str(meta["image_b"])))

    if pairs_cache is not None:
        pairs_cache.parent.mkdir(parents=True, exist_ok=True)
        with pairs_cache.open("w", encoding="utf-8") as handle:
            handle.writelines(f"{a} {b}\n" for a, b in pairs)
    return pairs


class PairListDataset(Dataset):
    """给定已排序 image pair 列表的推理数据集。"""

    def __init__(self, pairs: Sequence[tuple[str, str]], image_dir: Path, image_size: int) -> None:
        self.pairs = list(pairs)
        self.image_dir = Path(image_dir)
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image_a, image_b = self.pairs[int(idx)]
        tensor_a = load_rgb_tensor(resolve_image_path(self.image_dir, image_a), self.transform)
        tensor_b = load_rgb_tensor(resolve_image_path(self.image_dir, image_b), self.transform)
        return torch.stack([tensor_a, tensor_b], dim=0)


def image_pair_collate(batch: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(batch), dim=0)
