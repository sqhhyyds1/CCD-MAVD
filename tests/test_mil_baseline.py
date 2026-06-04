from __future__ import annotations

import pytest
import torch

from ccd_mavd.models import MILBaseline


def test_mil_baseline_requires_explicit_feature_dims():
    with pytest.raises(TypeError):
        MILBaseline()


def test_mil_baseline_uses_explicit_feature_dims():
    model = MILBaseline(rgb_dim=4, flow_dim=3, audio_dim=2, hidden_dim=5)

    logits = model(
        rgb=torch.zeros(2, 7, 4),
        flow=torch.zeros(2, 7, 3),
        audio=torch.zeros(2, 7, 2),
    )

    assert logits.shape == (2, 7)
