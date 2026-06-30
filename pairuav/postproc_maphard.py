"""MAP-hard 轨迹采样点后处理。"""

from __future__ import annotations

import argparse
import json
import zipfile
from importlib.resources import files
from pathlib import Path

import numpy as np

from .geometry import grid_from_d_values
from .metrics import wrap180


EXPECTED_TEST_PAIRS = 2_773_116


def circ_abs(angle_a: np.ndarray, angle_b: np.ndarray) -> np.ndarray:
    diff = np.abs(np.mod(angle_a, 360.0) - np.mod(angle_b, 360.0))
    return np.minimum(diff, 360.0 - diff)


def default_weights_path() -> Path:
    return Path(str(files("pairuav").joinpath("resources/p348_map_weights.json")))


def map_decode(
    heading: np.ndarray,
    range_m: np.ndarray,
    grid_d: np.ndarray,
    grid_heading: np.ndarray,
    grid_range: np.ndarray,
    weight_h: float,
    weight_r: float,
    chunk: int = 65536,
) -> np.ndarray:
    out = np.empty(len(heading), dtype=np.int64)
    grid_heading_2d = grid_heading[None, :].astype(np.float64)
    grid_range_2d = grid_range[None, :].astype(np.float64)
    for start in range(0, len(heading), chunk):
        end = min(start + chunk, len(heading))
        dh = circ_abs(heading[start:end].astype(np.float64)[:, None], grid_heading_2d) / 4.0
        dr = (range_m[start:end].astype(np.float64)[:, None] - grid_range_2d) / 0.5
        cost = weight_h * dh * dh + weight_r * dr * dr
        out[start:end] = grid_d[np.argmin(cost, axis=1)]
    return out


def write_submission(out_dir: Path, heading: np.ndarray, range_m: np.ndarray, expected_lines: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_txt = out_dir / "result.txt"
    np.savetxt(result_txt, np.stack([heading, range_m], axis=1), fmt="%.6f %.6f")
    line_count = sum(1 for _ in result_txt.open("r", encoding="utf-8"))
    if line_count != expected_lines:
        raise ValueError(f"line count {line_count} != expected {expected_lines}")
    result_zip = out_dir / "result.zip"
    if result_zip.exists():
        result_zip.unlink()
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(result_txt, arcname="result.txt")
    print(f"[write] {result_zip}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对连续预测执行 MAP-hard 后处理。")
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--d-values", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--expected-lines", type=int, default=EXPECTED_TEST_PAIRS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred = np.loadtxt(args.pred, dtype=np.float64)
    if pred.shape != (args.expected_lines, 2):
        raise ValueError(f"bad pred shape {pred.shape}, expected ({args.expected_lines}, 2)")
    grid_d, grid_heading, grid_range = grid_from_d_values(args.d_values)
    weights_path = args.weights or default_weights_path()
    weights = json.loads(Path(weights_path).read_text(encoding="utf-8"))
    decoded_d = map_decode(pred[:, 0], pred[:, 1], grid_d, grid_heading, grid_range, float(weights["w_h"]), float(weights["w_r"]))
    write_submission(
        args.out_dir,
        wrap180(4.0 * decoded_d.astype(np.float64)),
        -0.5 * decoded_d.astype(np.float64),
        expected_lines=args.expected_lines,
    )


if __name__ == "__main__":
    main()
