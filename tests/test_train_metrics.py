from __future__ import annotations

import numpy as np

from scripts.train import score_metrics_from_eval


def test_score_metrics_from_eval_reports_score_distribution_and_label_means():
    eval_result = {
        'frame_ap': 0.42,
        'snippet_scores': np.asarray([[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]], dtype=np.float32),
        'video_labels': np.asarray([0.0, 1.0], dtype=np.float32),
    }

    metrics = score_metrics_from_eval(eval_result, frame_ap_best=0.5)

    assert metrics['frame_ap_last'] == 0.42
    assert metrics['frame_ap_best'] == 0.5
    assert metrics['neg_video_score_mean'] == np.float32(0.3).item()
    assert metrics['pos_video_score_mean'] == np.float32(0.9).item()
    assert metrics['score_min'] == np.float32(0.1).item()
    assert metrics['score_max'] == np.float32(0.9).item()
    assert metrics['score_mean'] == np.asarray(eval_result['snippet_scores']).mean().item()
    assert metrics['score_std'] == np.asarray(eval_result['snippet_scores']).std().item()


def test_score_metrics_from_eval_reports_projector_norms_and_modality_availability():
    eval_result = {
        'frame_ap': 0.42,
        'snippet_scores': np.asarray([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32),
        'video_labels': np.asarray([0.0, 1.0], dtype=np.float32),
        'projector_norms': {
            'rgb': np.asarray([1.0, 3.0], dtype=np.float32),
            'flow': np.asarray([2.0, 4.0], dtype=np.float32),
            'audio': np.asarray([0.0, 0.0], dtype=np.float32),
        },
        'modality_availability': {
            'rgb': np.asarray([1.0, 1.0], dtype=np.float32),
            'flow': np.asarray([1.0, 1.0], dtype=np.float32),
            'audio': np.asarray([0.0, 0.0], dtype=np.float32),
        },
    }

    metrics = score_metrics_from_eval(eval_result, frame_ap_best=0.5)

    assert metrics['projector_norm_rgb_mean'] == 2.0
    assert metrics['projector_norm_flow_mean'] == 3.0
    assert metrics['projector_norm_audio_mean'] == 0.0
    assert metrics['modality_availability_audio'] == 0.0
