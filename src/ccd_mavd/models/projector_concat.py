from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


MODALITY_ORDER = ("rgb", "flow", "audio")


class ProjectorConcatBaseline(nn.Module):
    """Project selected modalities independently, concatenate them, and score snippets."""

    def __init__(
        self,
        rgb_dim: int,
        flow_dim: int,
        audio_dim: int,
        projector_dim: int = 256,
        hidden_dim: int = 256,
        modalities: Iterable[str] = ("rgb", "flow"),
        dropout: float = 0.1,
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
        self.projector_dim = int(projector_dim)
        self.projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dims[name]),
                    nn.Linear(dims[name], self.projector_dim),
                    nn.ReLU(),
                )
                for name in self.modalities
            }
        )
        self.input_dim = self.projector_dim * len(self.modalities)
        self.head = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _availability(self, name: str, batch_size: int, modality_mask: Tensor | None, device: torch.device) -> Tensor:
        if modality_mask is None:
            return torch.ones(batch_size, device=device)
        index = MODALITY_ORDER.index(name)
        return modality_mask[:, index].to(device=device, dtype=torch.float32)

    def forward_with_aux(
        self,
        rgb: Tensor,
        flow: Tensor,
        audio: Tensor,
        modality_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, dict[str, Tensor]]]:
        tensors = {"rgb": rgb, "flow": flow, "audio": audio}
        projected = []
        projector_norms: dict[str, Tensor] = {}
        modality_availability: dict[str, Tensor] = {}
        batch_size = rgb.shape[0]
        for name in self.modalities:
            available = self._availability(name, batch_size, modality_mask, tensors[name].device)
            features = self.projectors[name](tensors[name]) * available[:, None, None]
            projected.append(features)
            projector_norms[name] = torch.linalg.vector_norm(features, dim=-1)
            modality_availability[name] = available
        logits = self.head(torch.cat(projected, dim=-1)).squeeze(-1)
        return logits, {
            "projector_norms": projector_norms,
            "modality_availability": modality_availability,
        }

    def forward(self, rgb: Tensor, flow: Tensor, audio: Tensor, modality_mask: Tensor | None = None) -> Tensor:
        logits, _ = self.forward_with_aux(rgb, flow, audio, modality_mask=modality_mask)
        return logits
