"""Shared reproducibility controls for PairUAV command-line entry points."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


DEFAULT_SEED = 20260712


def seed_everything(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and PyTorch and return settings for run metadata."""

    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

    return {
        "seed": seed,
        "deterministic_algorithms": bool(deterministic),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def dataloader_generator(seed: int) -> torch.Generator:
    """Return a CPU generator whose shuffle order is independent of model RNG."""

    return torch.Generator().manual_seed(int(seed))


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python in a DataLoader worker from its PyTorch seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
