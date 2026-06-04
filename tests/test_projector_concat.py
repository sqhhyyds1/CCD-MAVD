from __future__ import annotations

import torch

from ccd_mavd.models import ProjectorConcatBaseline


def test_projector_concat_rgb_flow_outputs_logits_and_projector_norms():
    model = ProjectorConcatBaseline(
        rgb_dim=4,
        flow_dim=3,
        audio_dim=2,
        projector_dim=5,
        hidden_dim=6,
        modalities=("rgb", "flow"),
    )

    logits, aux = model.forward_with_aux(
        rgb=torch.zeros(2, 7, 4),
        flow=torch.zeros(2, 7, 3),
        audio=torch.zeros(2, 7, 2),
        modality_mask=torch.ones(2, 3),
    )

    assert logits.shape == (2, 7)
    assert model.input_dim == 10
    assert set(aux["projector_norms"]) == {"rgb", "flow"}
    assert aux["projector_norms"]["rgb"].shape == (2, 7)
    assert aux["projector_norms"]["flow"].shape == (2, 7)
    assert aux["modality_availability"]["rgb"].shape == (2,)
    assert aux["modality_availability"]["flow"].shape == (2,)


def test_projector_concat_audio_mask_zero_removes_audio_effect():
    torch.manual_seed(0)
    model = ProjectorConcatBaseline(
        rgb_dim=4,
        flow_dim=3,
        audio_dim=2,
        projector_dim=5,
        hidden_dim=6,
        modalities=("rgb", "flow", "audio"),
    )
    model.eval()
    rgb = torch.randn(1, 7, 4)
    flow = torch.randn(1, 7, 3)
    audio_a = torch.randn(1, 7, 2)
    audio_b = torch.randn(1, 7, 2) * 10.0
    mask_without_audio = torch.tensor([[1.0, 1.0, 0.0]])

    logits_a, aux_a = model.forward_with_aux(rgb, flow, audio_a, modality_mask=mask_without_audio)
    logits_b, aux_b = model.forward_with_aux(rgb, flow, audio_b, modality_mask=mask_without_audio)

    assert torch.allclose(logits_a, logits_b, atol=1e-6)
    assert torch.count_nonzero(aux_a["projector_norms"]["audio"]) == 0
    assert torch.count_nonzero(aux_b["projector_norms"]["audio"]) == 0
