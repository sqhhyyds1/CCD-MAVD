from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def topk_video_logits(snippet_logits: Tensor, k: int) -> Tensor:
    if snippet_logits.ndim != 2:
        raise ValueError(f"expected [B,T] logits, got {tuple(snippet_logits.shape)}")
    k_eff = max(1, min(int(k), snippet_logits.shape[1]))
    return torch.topk(snippet_logits, k=k_eff, dim=1).values.mean(dim=1)


def mil_bce_loss(snippet_logits: Tensor, video_labels: Tensor, k: int) -> Tensor:
    video_logits = topk_video_logits(snippet_logits, k=k)
    return F.binary_cross_entropy_with_logits(video_logits, video_labels.float())


def smoothness_loss(snippet_logits: Tensor) -> Tensor:
    if snippet_logits.shape[1] < 2:
        return snippet_logits.new_tensor(0.0)
    scores = torch.sigmoid(snippet_logits)
    return torch.mean((scores[:, 1:] - scores[:, :-1]) ** 2)


def sparsity_loss(snippet_logits: Tensor) -> Tensor:
    return torch.sigmoid(snippet_logits).mean()


def pairwise_topk_ranking_loss(snippet_logits: Tensor, video_labels: Tensor, k: int, margin: float = 1.0) -> Tensor:
    video_logits = topk_video_logits(snippet_logits, k=k)
    labels = video_labels.float()
    positive_logits = video_logits[labels > 0.5]
    negative_logits = video_logits[labels <= 0.5]
    if positive_logits.numel() == 0 or negative_logits.numel() == 0:
        return snippet_logits.new_tensor(0.0)
    violations = float(margin) - positive_logits[:, None] + negative_logits[None, :]
    return F.relu(violations).mean()
