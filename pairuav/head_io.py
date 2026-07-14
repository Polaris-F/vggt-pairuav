"""Task-head configuration and checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .heads import RangeMLP, SixDofHead, pair_input_dim


def read_head_config(path: Path) -> dict[str, Any]:
    """Read either a plain config JSON or a training result containing `config`."""

    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", data) if isinstance(data, dict) else None
    if not isinstance(config, dict):
        raise ValueError(f"invalid head config in {path}")
    return config


def config_in_directory(directory: Path) -> Path:
    """Locate a recorded task-head config in a run directory."""

    directory = Path(directory)
    for name in ("result.json", "config.json"):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing result.json/config.json under {directory}")


def config_beside_checkpoint(checkpoint: Path) -> Path:
    """Locate the recorded config beside a task-head checkpoint."""

    return config_in_directory(Path(checkpoint).parent)


def resolve_head_config(checkpoint: Path, config: Path | None = None) -> tuple[dict[str, Any], Path]:
    source = Path(config) if config is not None else config_beside_checkpoint(checkpoint)
    return read_head_config(source), source


def load_sixdof_head(
    checkpoint: Path,
    feat_dim: int,
    device: torch.device,
    *,
    config: Path | None = None,
) -> tuple[SixDofHead, dict[str, Any], Path]:
    cfg, source = resolve_head_config(checkpoint, config)
    model = SixDofHead(
        feat_dim=int(feat_dim),
        hidden_dim=int(cfg["hidden_dim"]),
        input_mode=str(cfg["input_mode"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval(), cfg, source


def load_range_head(
    checkpoint: Path,
    feat_dim: int,
    device: torch.device,
    *,
    config: Path | None = None,
    range_limit: float = 132.0,
) -> tuple[RangeMLP, dict[str, Any], Path]:
    cfg, source = resolve_head_config(checkpoint, config)
    model = RangeMLP(
        in_dim=pair_input_dim(int(feat_dim), str(cfg["input_mode"])),
        hidden_dim=int(cfg["hidden_dim"]),
        depth=int(cfg["depth"]),
        dropout=float(cfg["dropout"]),
        range_limit=float(range_limit),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return model.to(device).eval(), cfg, source
