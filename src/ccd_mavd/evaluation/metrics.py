from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def frame_average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(labels).reshape(-1)
    y_score = np.asarray(scores).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"label/score length mismatch: {y_true.shape[0]} != {y_score.shape[0]}")
    return float(average_precision_score(y_true, y_score))


def frame_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(labels).reshape(-1)
    y_score = np.asarray(scores).reshape(-1)
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"label/score length mismatch: {y_true.shape[0]} != {y_score.shape[0]}")
    return float(roc_auc_score(y_true, y_score))
