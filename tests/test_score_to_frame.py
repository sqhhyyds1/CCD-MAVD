from __future__ import annotations

import numpy as np

from ccd_mavd.evaluation.score_to_frame import interpolate_scores_to_frames


def test_interpolate_scores_to_frames_matches_requested_length():
    scores = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    expanded = interpolate_scores_to_frames(scores, num_frames=7)

    assert expanded.shape == (7,)
    assert np.isclose(expanded[0], 0.0)
    assert np.isclose(expanded[3], 1.0)
    assert np.isclose(expanded[-1], 0.0)


def test_interpolate_scores_to_frames_handles_single_snippet():
    expanded = interpolate_scores_to_frames(np.array([0.25], dtype=np.float32), num_frames=4)

    assert expanded.shape == (4,)
    assert np.allclose(expanded, 0.25)
