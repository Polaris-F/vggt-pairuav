"""公开螺旋采集过程下的规范 object-centric 几何。

本实现依据 University-1652 公开的螺旋采集过程,定义一个规范的
object-centric 轨迹参数化;给定配对帧索引,即可闭式生成用于训练的
6DoF 辅助标签。该标签用于监督外置任务头,最终预测仍投影为 PairUAV
评测所需的 heading/range。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from importlib.resources import files
from pathlib import Path

import numpy as np


PAIR_RE = re.compile(r"^(\d{2})_(\d{2})\.json$")
COS45 = math.cos(math.radians(45.0))
SIN45 = math.sin(math.radians(45.0))
TRANS_SCALE = 256.0


def wrap180(value: np.ndarray | float) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def frame_step(frame_idx: int) -> int:
    """标准 54 帧到轨迹 step 的映射。"""

    frame_idx = int(frame_idx)
    return 0 if frame_idx == 1 else 4 + 5 * (frame_idx - 2)


def d_from_frames(frame_a: int, frame_b: int) -> int:
    """相对轨迹步长 D = step_b - step_a。"""

    return frame_step(frame_b) - frame_step(frame_a)


def frames_from_json_path(path: str | Path) -> tuple[int, int]:
    match = PAIR_RE.match(Path(path).name)
    if not match:
        raise ValueError(f"cannot parse frame pair from {path}")
    return int(match.group(1)), int(match.group(2))


def labels_from_d(d_value: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """由相对轨迹步长投影到 PairUAV 的 heading/range 标签。"""

    d = np.asarray(d_value, dtype=np.float64)
    return wrap180(4.0 * d), -0.5 * d


def camera_position_for_step(step: np.ndarray | float) -> np.ndarray:
    """规范目标坐标系下的相机位置,返回 shape (..., 3)。"""

    k = np.asarray(step, dtype=np.float64)
    rho = 256.0 - 0.5 * k
    theta = np.deg2rad(4.0 * k)
    horizontal = rho * SIN45
    east = horizontal * np.sin(theta)
    north = horizontal * np.cos(theta)
    up = rho * COS45
    return np.stack([east, north, up], axis=-1)


def yaw_matrix_deg(deg: np.ndarray | float) -> np.ndarray:
    """生成绕 z 轴的 yaw 旋转矩阵,shape (..., 3, 3)。"""

    angle = np.deg2rad(np.asarray(deg, dtype=np.float64))
    c = np.cos(angle)
    s = np.sin(angle)
    out = np.zeros(angle.shape + (3, 3), dtype=np.float64)
    out[..., 0, 0] = c
    out[..., 0, 1] = -s
    out[..., 1, 0] = s
    out[..., 1, 1] = c
    out[..., 2, 2] = 1.0
    return out


def heading_from_yaw_matrix(rot: np.ndarray) -> np.ndarray:
    rot = np.asarray(rot, dtype=np.float64)
    return wrap180(np.rad2deg(np.arctan2(rot[..., 1, 0], rot[..., 0, 0])))


def range_from_translation_world(trans_world: np.ndarray) -> np.ndarray:
    """将规范坐标系下的竖直位移投影为 PairUAV signed range。"""

    return np.asarray(trans_world, dtype=np.float64)[..., 2] / COS45


def make_pair_geometry(frame_a: np.ndarray, frame_b: np.ndarray) -> dict[str, np.ndarray]:
    fa = np.asarray(frame_a, dtype=np.int64)
    fb = np.asarray(frame_b, dtype=np.int64)
    step_a = np.vectorize(frame_step)(fa).astype(np.float64)
    step_b = np.vectorize(frame_step)(fb).astype(np.float64)
    d_value = step_b - step_a
    heading, range_m = labels_from_d(d_value)
    pos_a = camera_position_for_step(step_a)
    pos_b = camera_position_for_step(step_b)
    trans_world = pos_b - pos_a
    rot = yaw_matrix_deg(heading)
    return {
        "frame_a": fa.astype(np.int16),
        "frame_b": fb.astype(np.int16),
        "step_a": step_a.astype(np.float32),
        "step_b": step_b.astype(np.float32),
        "d": d_value.astype(np.float32),
        "heading": heading.astype(np.float32),
        "range": range_m.astype(np.float32),
        "rot": rot.astype(np.float32),
        "trans_world": trans_world.astype(np.float32),
    }


def build_geometry_from_json_paths(paths: list[str]) -> dict[str, np.ndarray]:
    frames = np.asarray([frames_from_json_path(path) for path in paths], dtype=np.int16)
    return make_pair_geometry(frames[:, 0], frames[:, 1])


def default_d_values_path() -> Path:
    return Path(str(files("pairuav").joinpath("resources/d_values.json")))


def load_d_values(path: Path | None = None) -> np.ndarray:
    """读取公开轨迹采样步长表。"""

    source = Path(path) if path is not None else default_d_values_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = next(iter(data.values()))
    return np.asarray(data, dtype=np.int64)


def grid_from_d_values(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_values = np.sort(load_d_values(path))
    heading, range_m = labels_from_d(d_values.astype(np.float64))
    return d_values, heading, range_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="由特征 cache 的 json_paths.json 生成 6DoF 辅助标签。")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_paths_file = args.cache_dir / "json_paths.json"
    if not json_paths_file.exists():
        raise FileNotFoundError(f"{json_paths_file} 不存在;请先抽取特征 cache")
    paths = json.loads(json_paths_file.read_text(encoding="utf-8"))
    geom = build_geometry_from_json_paths([str(path) for path in paths])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **geom)
    print(f"[geometry] {len(paths)} pairs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
