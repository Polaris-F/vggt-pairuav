"""PairUAV 官方相对误差指标的本地实现。"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
PAIR_JSON_RE = re.compile(r"^\d{2}_\d{2}\.json$")


def extract_int(value: object) -> int | float:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else math.inf


def json_sort_key(path: Path) -> tuple[int | float, str, int | float, str]:
    path = Path(path)
    return (extract_int(path.parent.name), path.parent.name, extract_int(path.stem), path.stem)


def iter_json_paths(json_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for child in sorted(Path(json_dir).iterdir(), key=lambda p: (extract_int(p.name), p.name)):
        if child.is_dir():
            paths.extend(sorted((p for p in child.glob("*.json") if PAIR_JSON_RE.match(p.name)), key=lambda p: (extract_int(p.stem), p.stem)))
        elif child.is_file() and PAIR_JSON_RE.match(child.name):
            paths.append(child)
    return sorted(paths, key=json_sort_key)


def read_pair_meta(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def wrap180(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def circular_angle_abs_error(pred_angle: np.ndarray, gt_angle: np.ndarray) -> np.ndarray:
    return np.abs(wrap180(np.asarray(pred_angle, dtype=np.float64) - np.asarray(gt_angle, dtype=np.float64)))


def compute_metrics(
    gt_pairs: Sequence[tuple[float, float, str]],
    pred_pairs: Sequence[tuple[float, float]],
    eps: float = 1e-12,
) -> dict[str, float | int]:
    if len(gt_pairs) != len(pred_pairs):
        raise ValueError(f"line count mismatch: gt={len(gt_pairs)}, pred={len(pred_pairs)}")

    gt_h = np.asarray([x[0] for x in gt_pairs], dtype=np.float64)
    gt_r = np.asarray([x[1] for x in gt_pairs], dtype=np.float64)
    pred_h = np.asarray([x[0] for x in pred_pairs], dtype=np.float64)
    pred_r = np.asarray([x[1] for x in pred_pairs], dtype=np.float64)
    return compute_metrics_np(pred_h, pred_r, gt_h, gt_r, eps=eps)


def compute_metrics_np(
    pred_h: np.ndarray,
    pred_r: np.ndarray,
    gt_h: np.ndarray,
    gt_r: np.ndarray,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    pred_h = np.asarray(pred_h, dtype=np.float64)
    pred_r = np.asarray(pred_r, dtype=np.float64)
    gt_h = np.asarray(gt_h, dtype=np.float64)
    gt_r = np.asarray(gt_r, dtype=np.float64)
    if not (len(pred_h) == len(pred_r) == len(gt_h) == len(gt_r)):
        raise ValueError("pred/gt arrays must have the same length")

    angle_abs = circular_angle_abs_error(pred_h, gt_h)
    range_abs = np.abs(pred_r - gt_r)
    gt_h_norm = gt_h % 360.0

    angle_rel = np.full_like(angle_abs, np.nan, dtype=np.float64)
    distance_rel = np.full_like(range_abs, np.nan, dtype=np.float64)

    nonzero_h = np.abs(gt_h_norm) > eps
    zero_h_ok = (~nonzero_h) & (angle_abs <= eps)
    angle_rel[nonzero_h] = angle_abs[nonzero_h] / np.abs(gt_h_norm[nonzero_h])
    angle_rel[zero_h_ok] = 0.0

    nonzero_r = np.abs(gt_r) > eps
    zero_r_ok = (~nonzero_r) & (range_abs <= eps)
    distance_rel[nonzero_r] = range_abs[nonzero_r] / np.abs(gt_r[nonzero_r])
    distance_rel[zero_r_ok] = 0.0

    final = (angle_rel + distance_rel) * 0.5
    angle_valid = np.isfinite(angle_rel)
    distance_valid = np.isfinite(distance_rel)
    final_valid = np.isfinite(final)

    def mean_or_nan(values: np.ndarray) -> float:
        return float(values.mean()) if len(values) else math.nan

    return {
        "total_samples": int(len(gt_h)),
        "angle_valid_samples": int(angle_valid.sum()),
        "distance_valid_samples": int(distance_valid.sum()),
        "final_valid_samples": int(final_valid.sum()),
        "skipped_angle_samples": int((~angle_valid).sum()),
        "skipped_distance_samples": int((~distance_valid).sum()),
        "skipped_final_samples": int((~final_valid).sum()),
        "angle_rel_error": mean_or_nan(angle_rel[angle_valid]),
        "distance_rel_error": mean_or_nan(distance_rel[distance_valid]),
        "final_score": mean_or_nan(final[final_valid]),
    }


def load_gt_from_json_dir(json_dir: Path) -> list[tuple[float, float, str]]:
    gt = []
    for path in iter_json_paths(Path(json_dir)):
        data = read_pair_meta(path)
        gt.append((float(data["heading_num"]), float(data["range_num"]), str(path)))
    return gt


def parse_prediction_line(line: str, line_no: int) -> tuple[float, float]:
    values = FLOAT_RE.findall(line)
    if len(values) != 2:
        raise ValueError(f"line {line_no}: expected exactly two numeric values, got {len(values)}")
    return float(values[0]), float(values[1])


def load_predictions(pred_txt: Path) -> list[tuple[float, float]]:
    preds = []
    with Path(pred_txt).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                preds.append(parse_prediction_line(stripped, line_no))
    return preds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="本地计算 PairUAV 官方相对误差指标。")
    parser.add_argument("--gt-json-dir", required=True)
    parser.add_argument("--pred-txt", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = compute_metrics(load_gt_from_json_dir(Path(args.gt_json_dir)), load_predictions(Path(args.pred_txt)))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
