from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

import numpy as np

from ccd_mavd.data.xd_violence import XDFeatureIndex
from ccd_mavd.evaluation.metrics import frame_average_precision
from ccd_mavd.evaluation.score_to_frame import interpolate_scores_to_frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XD metric smoke test with dummy scores.")
    parser.add_argument("--feature-root", default="data/features/xd_violence")
    parser.add_argument("--list-root", default="data/lists/xd_violence")
    parser.add_argument("--output-dir", default="experiments/audit")
    parser.add_argument("--limit-videos", type=int, default=None)
    args = parser.parse_args()

    repo = Path.cwd()
    index = XDFeatureIndex.build(repo / args.feature_root, repo / args.list_root)
    records = index.test_videos[: args.limit_videos] if args.limit_videos else index.test_videos
    frame_scores = []
    for record in records:
        dummy = np.full((32,), 0.5, dtype=np.float32)
        frame_scores.append(interpolate_scores_to_frames(dummy, record.frame_count or record.snippet_count * 16))
    scores = np.concatenate(frame_scores) if frame_scores else np.zeros((0,), dtype=np.float32)
    labels = index.frame_labels_for(records)
    ap = frame_average_precision(labels, scores)
    result = {
        "video_count": len(records),
        "frame_score_count": int(scores.shape[0]),
        "frame_label_count": int(labels.shape[0]),
        "dummy_frame_ap": ap,
        "score_to_frame_rule": "linear interpolation from 32 dummy snippet scores to N annotated frames",
        "shape_match": int(scores.shape[0]) == int(labels.shape[0]),
    }
    output_dir = repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "xd_metric_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["shape_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
