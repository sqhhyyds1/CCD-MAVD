from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ccd_mavd.data.xd_violence import XDFeatureIndex


def build_report(manifest: dict[str, object]) -> str:
    lines = [
        "# XD-Violence M0 Audit Report",
        "",
        f"- Train videos: {manifest['train_video_count']}",
        f"- Test videos: {manifest['test_video_count']}",
        f"- Test snippets: {manifest['test_snippet_count']}",
        f"- GT frames: {manifest['gt_num_frames']}",
        f"- GT/snippet ratio: {manifest['gt_over_snippet_ratio']}",
        f"- Missing modality records: {manifest['missing_modality_count']}",
        f"- 5-crop grouping pass: {manifest['five_crop_grouping_pass']}",
        f"- Modality alignment pass: {manifest['modality_alignment_pass']}",
        f"- Train labels: {manifest['train_label_counts']}",
        f"- Test labels: {manifest['test_label_counts']}",
        "",
        "## Gate",
        "",
        f"Protocol audit pass: {manifest['protocol_audit_pass']}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit XD-Violence feature/list/GT protocol.")
    parser.add_argument("--feature-root", default="data/features/xd_violence")
    parser.add_argument("--list-root", default="data/lists/xd_violence")
    parser.add_argument("--output-dir", default="experiments/audit")
    args = parser.parse_args()

    repo = Path.cwd()
    index = XDFeatureIndex.build(repo / args.feature_root, repo / args.list_root)
    train_labels = Counter(record.video_label for record in index.train_videos)
    test_labels = Counter(record.video_label for record in index.test_videos)
    ratio = index.gt_num_frames / index.test_snippet_count if index.test_snippet_count else None
    feature_root = repo / args.feature_root
    rgb_train_files = len(list((feature_root / "rgb" / "train").glob("*.npy")))
    rgb_test_files = len(list((feature_root / "rgb" / "test").glob("*.npy")))
    flow_train_files = len(list((feature_root / "flow" / "train").glob("*.npy")))
    flow_test_files = len(list((feature_root / "flow" / "test").glob("*.npy")))
    audio_train_files = len(list((feature_root / "audio" / "train").glob("*.npy")))
    audio_test_files = len(list((feature_root / "audio" / "test").glob("*.npy")))
    five_crop_pass = (
        rgb_train_files == len(index.train_videos) * 5
        and rgb_test_files == len(index.test_videos) * 5
        and flow_train_files == len(index.train_videos) * 5
        and flow_test_files == len(index.test_videos) * 5
    )
    modality_alignment_pass = (
        audio_train_files == len(index.train_videos)
        and audio_test_files == len(index.test_videos)
        and len(index.missing_modalities) == 0
    )
    manifest = {
        "dataset": "XD-Violence",
        "feature_root": str(repo / args.feature_root),
        "list_root": str(repo / args.list_root),
        "train_video_count": len(index.train_videos),
        "test_video_count": len(index.test_videos),
        "rgb_train_file_count": rgb_train_files,
        "rgb_test_file_count": rgb_test_files,
        "flow_train_file_count": flow_train_files,
        "flow_test_file_count": flow_test_files,
        "audio_train_file_count": audio_train_files,
        "audio_test_file_count": audio_test_files,
        "five_crop_grouping_pass": five_crop_pass,
        "modality_alignment_pass": modality_alignment_pass,
        "test_snippet_count": index.test_snippet_count,
        "gt_path": str(index.gt_path),
        "gt_num_frames": index.gt_num_frames,
        "gt_over_snippet_ratio": ratio,
        "missing_modality_count": len(index.missing_modalities),
        "missing_modalities": index.missing_modalities,
        "train_label_counts": dict(train_labels),
        "test_label_counts": dict(test_labels),
        "score_to_frame_rule": "linear interpolation from T snippet scores to N annotated frames",
        "gt_used_for_training": False,
        "protocol_audit_pass": (
            len(index.train_videos) == 3954
            and len(index.test_videos) == 800
            and len(index.missing_modalities) == 0
            and five_crop_pass
            and modality_alignment_pass
            and index.gt_num_frames == index.test_snippet_count * 16
        ),
    }
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xd_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "xd_audit_report.md").write_text(build_report(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not manifest["protocol_audit_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
