from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ccd_mavd.data.xd_violence import (
    XDFeatureDataset,
    XDFeatureIndex,
    build_balanced_subset_manifest,
    load_subset_manifest,
    save_subset_manifest,
)
from ccd_mavd.evaluation.metrics import frame_average_precision
from ccd_mavd.evaluation.score_to_frame import interpolate_scores_to_frames
from ccd_mavd.losses import mil_bce_loss, pairwise_topk_ranking_loss, smoothness_loss, sparsity_loss
from ccd_mavd.models import MILBaseline, ProjectorConcatBaseline
from ccd_mavd.utils import set_seed


def unique_run_dir(output_root: Path, run_name: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = output_root / f"{run_name}_{stamp}"
    counter = 1
    while candidate.exists():
        candidate = output_root / f"{run_name}_{stamp}_{counter}"
        counter += 1
    return candidate


def score_metrics_from_eval(eval_result: dict[str, object], frame_ap_best: float | None, video_score_k: int = 1) -> dict[str, float | None]:
    snippet_scores = np.asarray(eval_result["snippet_scores"], dtype=np.float32)
    video_labels = np.asarray(eval_result["video_labels"], dtype=np.float32)
    projector_norms = eval_result.get("projector_norms", {})
    modality_availability = eval_result.get("modality_availability", {})
    aux_metrics: dict[str, float | None] = {}
    for name in ("rgb", "flow", "audio"):
        aux_metrics[f"projector_norm_{name}_mean"] = _mean_optional(projector_norms.get(name)) if isinstance(projector_norms, dict) else None
        aux_metrics[f"modality_availability_{name}"] = _mean_optional(modality_availability.get(name)) if isinstance(modality_availability, dict) else None
    if snippet_scores.size == 0:
        return {
            "frame_ap_last": eval_result["frame_ap"],
            "frame_ap_best": frame_ap_best,
            "pos_video_score_mean": None,
            "neg_video_score_mean": None,
            "score_mean": None,
            "score_std": None,
            "score_min": None,
            "score_max": None,
            **aux_metrics,
        }
    k_eff = max(1, min(int(video_score_k), snippet_scores.shape[1]))
    video_scores = np.sort(snippet_scores, axis=1)[:, -k_eff:].mean(axis=1)
    pos_scores = video_scores[video_labels > 0.5]
    neg_scores = video_scores[video_labels <= 0.5]
    return {
        "frame_ap_last": eval_result["frame_ap"],
        "frame_ap_best": frame_ap_best,
        "pos_video_score_mean": float(pos_scores.mean()) if pos_scores.size else None,
        "neg_video_score_mean": float(neg_scores.mean()) if neg_scores.size else None,
        "score_mean": float(snippet_scores.mean()),
        "score_std": float(snippet_scores.std()),
        "score_min": float(snippet_scores.min()),
        "score_max": float(snippet_scores.max()),
        **aux_metrics,
    }


def resolve_subset_manifest(config: dict[str, object], repo: Path, run_dir: Path) -> dict[str, object] | None:
    manifest_config = config.get("subset_manifest")
    if not manifest_config:
        return None
    manifest_path = repo / str(manifest_config)
    if manifest_path.exists():
        manifest = load_subset_manifest(manifest_path)
    else:
        index = XDFeatureIndex.build(
            feature_root=repo / str(config["feature_root"]),
            list_root=repo / str(config["list_root"]),
        )
        manifest = build_balanced_subset_manifest(
            index=index,
            name=str(config.get("subset_name", manifest_path.stem)),
            seed=int(config.get("seed", 0)),
            train_limit=int(config["limit_train_videos"]),
            test_limit=int(config["limit_test_videos"]),
        )
        save_subset_manifest(manifest, manifest_path)
    save_subset_manifest(manifest, run_dir / "subset_manifest.json")
    return manifest


def build_model(config: dict[str, object], dim_sample: dict[str, object]) -> torch.nn.Module:
    modalities = tuple(config.get("modalities", ["rgb", "flow", "audio"]))
    model_name = str(config.get("model", "mil_baseline"))
    dims = {
        "rgb_dim": int(dim_sample["rgb"].shape[-1]),
        "flow_dim": int(dim_sample["flow"].shape[-1]),
        "audio_dim": int(dim_sample["audio"].shape[-1]),
    }
    if model_name == "mil_baseline":
        return MILBaseline(
            **dims,
            hidden_dim=int(config.get("hidden_dim", 256)),
            modalities=modalities,
        )
    if model_name == "projector_concat":
        return ProjectorConcatBaseline(
            **dims,
            projector_dim=int(config.get("projector_dim", 256)),
            hidden_dim=int(config.get("hidden_dim", 256)),
            modalities=modalities,
            dropout=float(config.get("dropout", 0.1)),
        )
    raise ValueError(f"unknown model: {model_name}")


def forward_model(
    model: torch.nn.Module,
    rgb: torch.Tensor,
    flow: torch.Tensor,
    audio: torch.Tensor,
    modality_mask: torch.Tensor | None = None,
    with_aux: bool = False,
) -> tuple[torch.Tensor, dict[str, object] | None]:
    if with_aux and hasattr(model, "forward_with_aux"):
        logits, aux = model.forward_with_aux(rgb, flow, audio, modality_mask=modality_mask)
        return logits, aux
    if hasattr(model, "forward_with_aux"):
        logits, _ = model.forward_with_aux(rgb, flow, audio, modality_mask=modality_mask)
        return logits, None
    return model(rgb, flow, audio), None


def _mean_optional(values: object) -> float | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size else None


def write_env(path: Path) -> None:
    lines = [
        f"python={os.sys.version}",
        f"torch={torch.__version__}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
    ]
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True, stdout=subprocess.PIPE, check=False).stdout.strip()
        lines.append(f"git_head={head}")
    except Exception as exc:
        lines.append(f"git_head_error={exc!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_mil_stage(config: dict[str, object], config_path: Path, command: str) -> Path:
    set_seed(int(config.get("seed", 0)))
    repo = Path.cwd()
    output_root = repo / str(config.get("output_root", "experiments/xd_violence"))
    run_dir = unique_run_dir(output_root, str(config.get("run_name", "xd_a0_mil")))
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    write_env(run_dir / "env.txt")
    subset_manifest = resolve_subset_manifest(config, repo, run_dir)
    train_subset_ids = subset_manifest["train_video_ids"] if subset_manifest else None
    test_subset_ids = subset_manifest["test_video_ids"] if subset_manifest else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = XDFeatureDataset(
        feature_root=repo / str(config["feature_root"]),
        list_root=repo / str(config["list_root"]),
        split="train",
        temporal_size=int(config.get("segments", 32)),
        limit_videos=int(config["limit_train_videos"]) if config.get("limit_train_videos") is not None else None,
        balanced_limit=bool(config.get("balanced_train_limit", False)),
        balanced_seed=int(config.get("seed", 0)),
        subset_video_ids=train_subset_ids,
    )
    test_ds = XDFeatureDataset(
        feature_root=repo / str(config["feature_root"]),
        list_root=repo / str(config["list_root"]),
        split="test",
        temporal_size=int(config.get("segments", 32)),
        limit_videos=int(config["limit_test_videos"]) if config.get("limit_test_videos") is not None else None,
        balanced_limit=bool(config.get("balanced_test_limit", False)),
        balanced_seed=int(config.get("seed", 0)) + 1000003,
        subset_video_ids=test_subset_ids,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(config.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(config.get("num_workers", 0)),
    )
    dim_sample = train_ds[0]
    modalities = tuple(config.get("modalities", ["rgb", "flow", "audio"]))
    model = build_model(config, dim_sample).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    metrics_path = run_dir / "metrics.jsonl"
    train_log = []
    best_ap = -1.0
    for epoch in range(1, int(config.get("epochs", 1)) + 1):
        model.train()
        losses = []
        mil_losses = []
        ranking_losses = []
        rank_valid_batches = 0
        rank_skipped_batches = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            flow = batch["flow"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["video_label"].to(device)
            modality_mask = batch["modality_mask"].to(device)
            logits, _ = forward_model(model, rgb, flow, audio, modality_mask=modality_mask)
            loss_mil = mil_bce_loss(logits, labels, k=int(config.get("topk", 3)))
            has_positive = bool(torch.any(labels > 0.5).item())
            has_negative = bool(torch.any(labels <= 0.5).item())
            if has_positive and has_negative:
                loss_rank = pairwise_topk_ranking_loss(
                    logits,
                    labels,
                    k=int(config.get("topk", 3)),
                    margin=float(config.get("rank_margin", 1.0)),
                )
                rank_valid_batches += 1
            else:
                loss_rank = logits.new_tensor(0.0)
                rank_skipped_batches += 1
            loss = loss_mil + float(config.get("lambda_rank", 0.0)) * loss_rank
            if float(config.get("lambda_smooth", 0.0)):
                loss = loss + float(config["lambda_smooth"]) * smoothness_loss(logits)
            if float(config.get("lambda_sparse", 0.0)):
                loss = loss + float(config["lambda_sparse"]) * sparsity_loss(logits)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            mil_losses.append(float(loss_mil.detach().cpu()))
            ranking_losses.append(float(loss_rank.detach().cpu()))
        eval_result = evaluate(model, test_ds, device, k=int(config.get("topk", 3)))
        frame_ap_last = eval_result["frame_ap"]
        frame_ap_best = None
        if frame_ap_last is not None:
            frame_ap_best = max(best_ap, float(frame_ap_last)) if best_ap >= 0 else float(frame_ap_last)
        rank_total_batches = rank_valid_batches + rank_skipped_batches
        row = {
            "epoch": epoch,
            "loss_total": float(np.mean(losses)) if losses else None,
            "loss_mil": float(np.mean(mil_losses)) if mil_losses else None,
            "loss_rank": float(np.mean(ranking_losses)) if ranking_losses else None,
            "val_frame_ap": frame_ap_last,
            "rank_valid_batches": rank_valid_batches,
            "rank_skipped_batches": rank_skipped_batches,
            "rank_valid_ratio": (rank_valid_batches / rank_total_batches) if rank_total_batches else None,
            "lr": optimizer.param_groups[0]["lr"],
        }
        row.update(score_metrics_from_eval(eval_result, frame_ap_best=frame_ap_best, video_score_k=int(config.get("topk", 3))))
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        train_log.append(json.dumps(row))
        torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "last.pt")
        if frame_ap_last is not None and float(frame_ap_last) >= best_ap:
            best_ap = float(frame_ap_last)
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch}, run_dir / "checkpoints" / "best.pt")
            np.savez(
                run_dir / "predictions" / "test_scores.npz",
                video_ids=eval_result["video_ids"],
                snippet_scores=eval_result["snippet_scores"],
                frame_scores=eval_result["frame_scores"],
                video_labels=eval_result["video_labels"],
                frame_labels=eval_result["frame_labels"],
            )
    (run_dir / "train.log").write_text("\n".join(train_log) + "\n", encoding="utf-8")
    summary = {
        "run_dir": str(run_dir),
        "stage": config.get("stage", "A0"),
        "train_videos": len(train_ds),
        "test_videos": len(test_ds),
        "epochs": int(config.get("epochs", 1)),
        "modalities": ",".join(modalities),
        "lambda_rank": float(config.get("lambda_rank", 0.0)),
        "best_frame_ap": best_ap,
        "gt_used_for_training": False,
        "subset_manifest": str(run_dir / "subset_manifest.json") if subset_manifest else None,
    }
    (run_dir / "reports" / "summary.md").write_text(
        f"# XD {config.get('stage', 'A0')} MIL Tiny Summary\n\n"
        + "\n".join(f"- {key}: {value}" for key, value in summary.items())
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return run_dir


@torch.no_grad()
def evaluate(model: MILBaseline, dataset: XDFeatureDataset, device: torch.device, k: int) -> dict[str, object]:
    model.eval()
    video_ids = []
    snippet_scores = []
    frame_scores = []
    video_labels = []
    projector_norms: dict[str, list[np.ndarray]] = {}
    modality_availability: dict[str, list[np.ndarray]] = {}
    records = dataset.records
    for i in range(len(dataset)):
        sample = dataset[i]
        rgb = sample["rgb"].unsqueeze(0).to(device)
        flow = sample["flow"].unsqueeze(0).to(device)
        audio = sample["audio"].unsqueeze(0).to(device)
        modality_mask = sample["modality_mask"].unsqueeze(0).to(device)
        logits, aux = forward_model(model, rgb, flow, audio, modality_mask=modality_mask, with_aux=True)
        if aux:
            for name, values in aux.get("projector_norms", {}).items():
                projector_norms.setdefault(name, []).append(values.detach().cpu().reshape(-1).numpy())
            for name, values in aux.get("modality_availability", {}).items():
                modality_availability.setdefault(name, []).append(values.detach().cpu().reshape(-1).numpy())
        scores = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
        record = records[i]
        video_ids.append(record.video_id)
        snippet_scores.append(scores)
        frame_scores.append(interpolate_scores_to_frames(scores, record.frame_count or record.snippet_count * 16))
        video_labels.append(record.video_label)
    frame_scores_arr = np.concatenate(frame_scores) if frame_scores else np.zeros((0,), dtype=np.float32)
    frame_labels = dataset.index.frame_labels_for(records).astype(np.float32)
    frame_ap = frame_average_precision(frame_labels, frame_scores_arr) if frame_labels.size else None
    return {
        "video_ids": np.asarray(video_ids),
        "snippet_scores": np.stack(snippet_scores, axis=0) if snippet_scores else np.zeros((0, 0), dtype=np.float32),
        "frame_scores": frame_scores_arr,
        "video_labels": np.asarray(video_labels, dtype=np.float32),
        "frame_labels": frame_labels,
        "frame_ap": frame_ap,
        "projector_norms": {name: np.concatenate(chunks) for name, chunks in projector_norms.items()},
        "modality_availability": {name: np.concatenate(chunks) for name, chunks in modality_availability.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CCD-MAVD stages.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("stage") not in {"A0", "A1", "A2a", "A2b"}:
        raise SystemExit("Only A0/A1/A2a/A2b MIL-style stages are implemented in this runner.")
    train_mil_stage(config, config_path, f"{sys.executable} " + " ".join(sys.argv))


if __name__ == "__main__":
    main()
