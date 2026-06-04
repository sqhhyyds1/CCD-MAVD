from __future__ import annotations

import torch
from torch import nn, Tensor


class MILBaseline(nn.Module):
    """Small A0 baseline: concatenate available modalities and score each snippet."""

    def __init__(self, rgb_dim: int, flow_dim: int, audio_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        in_dim = rgb_dim + flow_dim + audio_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rgb: Tensor, flow: Tensor, audio: Tensor) -> Tensor:
        x = torch.cat([rgb, flow, audio], dim=-1)
        return self.net(x).squeeze(-1)
