"""Loss functions for weakly supervised VAD."""

from .mil import mil_bce_loss, pairwise_topk_ranking_loss, smoothness_loss, sparsity_loss, topk_video_logits

__all__ = ["mil_bce_loss", "pairwise_topk_ranking_loss", "smoothness_loss", "sparsity_loss", "topk_video_logits"]
