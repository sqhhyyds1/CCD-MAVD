from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn, Tensor


class MILBaseline(nn.Module):
    """Small MIL baseline: concatenate selected modalities and score each snippet."""

    def __init__(
        self,
        rgb_dim: int,
        flow_dim: int,
        audio_dim: int,
        hidden_dim: int = 256,
        modalities: Iterable[str] = ("rgb", "flow", "audio"),
    ) -> None:
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities:
            raise ValueError("at least one modality is required")
        dims = {"rgb": int(rgb_dim), "flow": int(flow_dim), "audio": int(audio_dim)}
        invalid = sorted(set(self.modalities) - set(dims))
        if invalid:
            raise ValueError(f"unknown modalities: {invalid}")
        for name in self.modalities:
            if dims[name] <= 0:
                raise ValueError(f"{name}_dim must be positive")
        self.input_dim = sum(dims[name] for name in self.modalities)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, rgb: Tensor, flow: Tensor, audio: Tensor) -> Tensor:
        tensors = {"rgb": rgb, "flow": flow, "audio": audio}
        x = torch.cat([tensors[name] for name in self.modalities], dim=-1)
        return self.net(x).squeeze(-1)
