from __future__ import annotations

import numpy as np


def interpolate_scores_to_frames(scores: np.ndarray, num_frames: int) -> np.ndarray:
    """Linearly expand snippet scores to exactly ``num_frames`` frame scores."""

    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("scores must contain at least one snippet score")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if values.size == 1:
        return np.full((num_frames,), float(values[0]), dtype=np.float32)

    old_x = np.linspace(0.0, float(num_frames - 1), num=values.size, dtype=np.float32)
    new_x = np.arange(num_frames, dtype=np.float32)
    expanded = np.interp(new_x, old_x, values).astype(np.float32)
    return expanded
