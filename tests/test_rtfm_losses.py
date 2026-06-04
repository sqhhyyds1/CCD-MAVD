from __future__ import annotations

import torch

from ccd_mavd.losses import pairwise_topk_ranking_loss


def test_pairwise_topk_ranking_loss_is_zero_when_positive_scores_are_higher():
    logits = torch.tensor([
        [3.0, 2.8, 2.5, 2.0],
        [-2.0, -2.5, -2.8, -3.0],
    ])
    labels = torch.tensor([1.0, 0.0])

    loss = pairwise_topk_ranking_loss(logits, labels, k=2, margin=0.5)

    assert loss.item() == 0.0


def test_pairwise_topk_ranking_loss_penalizes_negative_scores_above_positive_scores():
    logits = torch.tensor([
        [-2.0, -2.5, -2.8, -3.0],
        [3.0, 2.8, 2.5, 2.0],
    ])
    labels = torch.tensor([1.0, 0.0])

    loss = pairwise_topk_ranking_loss(logits, labels, k=2, margin=0.5)

    assert loss.item() > 0.0


def test_pairwise_topk_ranking_loss_returns_zero_for_single_class_batch():
    logits = torch.tensor([
        [3.0, 2.8, 2.5, 2.0],
        [1.0, 0.8, 0.5, 0.2],
    ])
    labels = torch.tensor([1.0, 1.0])

    loss = pairwise_topk_ranking_loss(logits, labels, k=2, margin=0.5)

    assert loss.item() == 0.0
