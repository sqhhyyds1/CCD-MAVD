from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from scripts.eval import apply_audio_controls, build_audio_permutation


def test_build_audio_permutation_avoids_identity_when_possible():
    permutation = build_audio_permutation(8, seed=0)

    assert sorted(permutation.tolist()) == list(range(8))
    assert all(int(src) != idx for idx, src in enumerate(permutation))


def test_apply_audio_controls_disables_audio_and_mask():
    audio = torch.ones(1, 4, 2)
    mask = torch.ones(1, 3)

    controlled_audio, controlled_mask = apply_audio_controls(
        audio=audio,
        modality_mask=mask,
        disable_audio=True,
        shuffled_audio=None,
    )

    assert torch.count_nonzero(controlled_audio) == 0
    assert controlled_mask.tolist() == [[1.0, 1.0, 0.0]]


def test_apply_audio_controls_uses_shuffled_audio():
    audio = torch.ones(1, 4, 2)
    mask = torch.ones(1, 3)
    shuffled = torch.full((1, 4, 2), 7.0)

    controlled_audio, controlled_mask = apply_audio_controls(
        audio=audio,
        modality_mask=mask,
        disable_audio=False,
        shuffled_audio=shuffled,
        shuffled_modality_mask=None,
    )

    assert torch.equal(controlled_audio, shuffled)
    assert controlled_mask.tolist() == [[1.0, 1.0, 1.0]]


def test_apply_audio_controls_uses_shuffled_audio_mask():
    audio = torch.ones(1, 4, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    shuffled = torch.full((1, 4, 2), 7.0)
    shuffled_mask = torch.tensor([[1.0, 1.0, 1.0]])

    controlled_audio, controlled_mask = apply_audio_controls(
        audio=audio,
        modality_mask=mask,
        disable_audio=False,
        shuffled_audio=shuffled,
        shuffled_modality_mask=shuffled_mask,
    )

    assert torch.equal(controlled_audio, shuffled)
    assert controlled_mask.tolist() == [[1.0, 1.0, 1.0]]


def test_eval_script_help_runs_as_direct_script():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, 'scripts/eval.py', '--help'],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert '--disable-audio' in result.stdout
    assert '--shuffle-audio' in result.stdout
