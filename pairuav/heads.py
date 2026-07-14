"""PairUAV 外置任务头。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import geometry as G


PAIR_INPUT_MULTIPLIERS = {
    "ab": 2,
    "ab_diff_prod": 4,
    "diff_prod": 2,
}


def pair_input_dim(feat_dim: int, mode: str) -> int:
    try:
        multiplier = PAIR_INPUT_MULTIPLIERS[mode]
    except KeyError as exc:
        raise ValueError(f"unknown input mode: {mode}") from exc
    return int(feat_dim) * multiplier


def make_pair_input(features: torch.Tensor, mode: str) -> torch.Tensor:
    feat_a = features[:, 0].float()
    feat_b = features[:, 1].float()
    if mode == "ab":
        return torch.cat([feat_a, feat_b], dim=1)
    if mode == "ab_diff_prod":
        return torch.cat([feat_a, feat_b, feat_a - feat_b, feat_a * feat_b], dim=1)
    if mode == "diff_prod":
        return torch.cat([feat_a - feat_b, feat_a * feat_b], dim=1)
    raise ValueError(f"unknown input mode: {mode}")


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-6)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def geodesic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    rel = pred.transpose(1, 2) @ target
    trace = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()


def heading_from_rot_torch(rot: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.atan2(rot[:, 1, 0], rot[:, 0, 0]))


def sincos_angle_loss(pred_h: torch.Tensor, gt_h: torch.Tensor) -> torch.Tensor:
    pred_rad = torch.deg2rad(pred_h)
    gt_rad = torch.deg2rad(gt_h)
    return F.mse_loss(torch.sin(pred_rad), torch.sin(gt_rad)) + F.mse_loss(torch.cos(pred_rad), torch.cos(gt_rad))


def circ_sincos_loss_deg(angle_a: torch.Tensor, angle_b: torch.Tensor) -> torch.Tensor:
    rad_a = torch.deg2rad(angle_a)
    rad_b = torch.deg2rad(angle_b)
    return F.mse_loss(torch.sin(rad_a), torch.sin(rad_b)) + F.mse_loss(torch.cos(rad_a), torch.cos(rad_b))


class SixDofHead(nn.Module):
    """6DoF 辅助任务头,最终主要使用旋转 yaw 作为 heading。"""

    def __init__(
        self,
        feat_dim: int = 4096,
        hidden_dim: int = 512,
        input_mode: str = "ab",
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_mode = input_mode
        layers: list[nn.Module] = []
        in_dim = pair_input_dim(feat_dim, input_mode)
        for index in range(int(depth)):
            layers.append(nn.Linear(in_dim if index == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.head_rot6d = nn.Linear(hidden_dim, 6)
        self.head_xy = nn.Linear(hidden_dim, 2)
        self.head_z = nn.Linear(hidden_dim, 1)

        nn.init.zeros_(self.head_rot6d.weight)
        self.head_rot6d.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
        nn.init.zeros_(self.head_xy.weight)
        nn.init.zeros_(self.head_xy.bias)
        nn.init.zeros_(self.head_z.weight)
        nn.init.zeros_(self.head_z.bias)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.mlp(make_pair_input(features, self.input_mode))
        rot = rot6d_to_matrix(self.head_rot6d(hidden))
        xy = 384.0 * torch.tanh(self.head_xy(hidden))
        z = (132.0 * G.COS45) * torch.tanh(self.head_z(hidden))
        trans = torch.cat([xy, z], dim=1)
        return rot, trans


class RangeMLP(nn.Module):
    """独立距离头。"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 512,
        depth: int = 2,
        dropout: float = 0.0,
        range_limit: float = 132.0,
    ) -> None:
        super().__init__()
        self.range_limit = float(range_limit)
        layers: list[nn.Module] = []
        cur = int(in_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(nn.ReLU(True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(cur, 1)

    def init_mean(self, range_mean: float) -> None:
        clipped = max(-0.95, min(0.95, float(range_mean) / self.range_limit))
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.fill_(math.atanh(clipped))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.range_limit * torch.tanh(self.head(self.mlp(x))).squeeze(1)
