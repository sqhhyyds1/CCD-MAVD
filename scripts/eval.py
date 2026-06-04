from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

from ccd_mavd.data.xd_violence import XDFeatureDataset, load_subset_manifest
from ccd_mavd.evaluation.metrics import frame_average_precision
from ccd_mavd.evaluation.score_to_frame import interpolate_scores_to_frames
from scripts.train import build_model, forward_model, score_metrics_from_eval


def build_audio_permutation(count: int, seed: int) -> np.ndarray:
    if count <= 1:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(count)
    for _ in range(count + 1):
        if np.all(permutation != np.arange(count)):
            return permutation.astype(np.int64)
        permutation = np.roll(permutation, 1)
    return np.roll(np.arange(count, dtype=np.int64), 1)


def apply_audio_controls(
    audio: torch.Tensor,
    modality_mask: torch.Tensor,
    disable_audio: bool,
    shuffled_audio: torch.Tensor | None,
    shuffled_modality_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    controlled_audio = shuffled_audio.clone() if shuffled_audio is not None else audio.clone()
    controlled_mask = modality_mask.clone()
    if shuffled_modality_mask is not None:
        controlled_mask[:, 2] = shuffled_modality_mask[:, 2]
    if disable_audio:
        controlled_audio = torch.zeros_like(controlled_audio)
        controlled_mask[:, 2] = 0.0
    return controlled_audio, controlled_mask


def _dataset_from_config(config: dict[str, object], repo: Path, subset_manifest: dict[str, object] | None) -> XDFeatureDataset:
    test_subset_ids = subset_manifest["test_video_ids"] if subset_manifest else None
    return XDFeatureDataset(
        feature_root=repo / str(config["feature_root"]),
        list_root=repo / str(config["list_root"]),
        split="test",
        temporal_size=int(config.get("segments", 32)),
        limit_videos=int(config["limit_test_videos"]) if config.get("limit_test_videos") is not None else None,
        balanced_limit=bool(config.get("balanced_test_limit", False)),
        balanced_seed=int(config.get("seed", 0)) + 1000003,
        subset_video_ids=test_subset_ids,
    )


def _load_subset_manifest(config: dict[str, object], repo: Path) -> dict[str, object] | None:
    manifest_path = config.get("subset_manifest")
    if not manifest_path:
        return None
    return load_subset_manifest(repo / str(manifest_path))


@torch.no_grad()
def evaluate_checkpoint(
    config: dict[str, object],
    checkpoint_path: Path,
    output_dir: Path,
    disable_audio: bool = False,
    shuffle_audio: bool = False,
    shuffle_seed: int = 0,
) -> dict[str, object]:
    if disable_audio and shuffle_audio:
        raise ValueError("--disable-audio and --shuffle-audio are mutually exclusive")
    repo = Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_manifest = _load_subset_manifest(config, repo)
    dataset = _dataset_from_config(config, repo, subset_manifest)
    dim_sample = dataset[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, dim_sample).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    audio_permutation = build_audio_permutation(len(dataset), shuffle_seed) if shuffle_audio else None
    video_ids = []
    snippet_scores = []
    frame_scores = []
    video_labels = []
    projector_norms: dict[str, list[np.ndarray]] = {}
    modality_availability: dict[str, list[np.ndarray]] = {}

    for index in range(len(dataset)):
        sample = dataset[index]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        flow = sample["flow"].unsqueeze(0).to(device)
        audio = sample["audio"].unsqueeze(0).to(device)
        modality_mask = sample["modality_mask"].unsqueeze(0).to(device)
        shuffled_audio = None
        shuffled_modality_mask = None
        if audio_permutation is not None:
            shuffled_sample = dataset[int(audio_permutation[index])]
            shuffled_audio = shuffled_sample["audio"].unsqueeze(0).to(device)
            shuffled_modality_mask = shuffled_sample["modality_mask"].unsqueeze(0).to(device)
        audio, modality_mask = apply_audio_controls(
            audio=audio,
            modality_mask=modality_mask,
            disable_audio=disable_audio,
            shuffled_audio=shuffled_audio,
            shuffled_modality_mask=shuffled_modality_mask,
        )
        logits, aux = forward_model(model, rgb, flow, audio, modality_mask=modality_mask, with_aux=True)
        if aux:
            for name, values in aux.get("projector_norms", {}).items():
                projector_norms.setdefault(name, []).append(values.detach().cpu().reshape(-1).numpy())
            for name, values in aux.get("modality_availability", {}).items():
                modality_availability.setdefault(name, []).append(values.detach().cpu().reshape(-1).numpy())
        scores = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
        record = dataset.records[index]
        video_ids.append(record.video_id)
        snippet_scores.append(scores)
        frame_scores.append(interpolate_scores_to_frames(scores, record.frame_count or record.snippet_count * 16))
        video_labels.append(record.video_label)

    frame_scores_arr = np.concatenate(frame_scores) if frame_scores else np.zeros((0,), dtype=np.float32)
    frame_labels = dataset.index.frame_labels_for(dataset.records).astype(np.float32)
    eval_result = {
        "video_ids": np.asarray(video_ids),
        "snippet_scores": np.stack(snippet_scores, axis=0) if snippet_scores else np.zeros((0, 0), dtype=np.float32),
        "frame_scores": frame_scores_arr,
        "video_labels": np.asarray(video_labels, dtype=np.float32),
        "frame_labels": frame_labels,
        "frame_ap": frame_average_precision(frame_labels, frame_scores_arr) if frame_labels.size else None,
        "projector_norms": {name: np.concatenate(chunks) for name, chunks in projector_norms.items()},
        "modality_availability": {name: np.concatenate(chunks) for name, chunks in modality_availability.items()},
    }
    np.savez(
        output_dir / "test_scores.npz",
        video_ids=eval_result["video_ids"],
        snippet_scores=eval_result["snippet_scores"],
        frame_scores=eval_result["frame_scores"],
        video_labels=eval_result["video_labels"],
        frame_labels=eval_result["frame_labels"],
    )
    summary = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": str(config.get("run_name", "")),
        "checkpoint": str(checkpoint_path),
        "disable_audio": disable_audio,
        "shuffle_audio": shuffle_audio,
        "shuffle_seed": int(shuffle_seed),
        "video_count": len(dataset),
        "gt_used_for_training": False,
    }
    summary.update(score_metrics_from_eval(eval_result, frame_ap_best=eval_result["frame_ap"], video_score_k=int(config.get("topk", 3))))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "summary.md").write_text(
        "# XD Evaluation Summary\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in summary.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained XD checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--disable-audio", action="store_true")
    parser.add_argument("--shuffle-audio", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    evaluate_checkpoint(
        config=config,
        checkpoint_path=Path(args.checkpoint),
        output_dir=Path(args.output_dir),
        disable_audio=bool(args.disable_audio),
        shuffle_audio=bool(args.shuffle_audio),
        shuffle_seed=int(args.shuffle_seed),
    )


if __name__ == "__main__":
    main()
