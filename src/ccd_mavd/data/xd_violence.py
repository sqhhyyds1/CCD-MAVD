from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


VISUAL_RE = re.compile(r"^(?P<base>.+)__(?P<crop>[0-4])\.npy$")
AUDIO_RE = re.compile(r"^(?P<base>.+)__vggish\.npy$")
LABEL_RE = re.compile(r"_label_(?P<label>[^_]+)(?:__|$)")


@dataclass(frozen=True)
class XDVideoRecord:
    video_id: str
    split: str
    label_text: str
    video_label: int
    rgb_paths: tuple[Path, Path, Path, Path, Path]
    flow_paths: tuple[Path, Path, Path, Path, Path]
    audio_path: Path
    snippet_count: int
    frame_count: int | None = None
    frame_offset: int | None = None


def label_from_video_id(video_id: str) -> tuple[str, int]:
    match = LABEL_RE.search(video_id)
    label_text = match.group("label") if match else "UNKNOWN"
    return label_text, 0 if label_text == "A" else 1


def normalize_list_line(line: str) -> str:
    name = Path(line.strip()).name
    visual = VISUAL_RE.match(name)
    if visual:
        return visual.group("base")
    audio = AUDIO_RE.match(name)
    if audio:
        return audio.group("base")
    return re.sub(r"\.npy$", "", name)


def ordered_unique_video_ids(list_path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for line in list_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        base = normalize_list_line(line)
        if base not in seen:
            seen.add(base)
            ordered.append(base)
    return ordered


def _group_visual(directory: Path) -> dict[str, tuple[Path, Path, Path, Path, Path]]:
    grouped: dict[str, dict[int, Path]] = defaultdict(dict)
    for path in directory.glob("*.npy"):
        match = VISUAL_RE.match(path.name)
        if match:
            grouped[match.group("base")][int(match.group("crop"))] = path
    complete: dict[str, tuple[Path, Path, Path, Path, Path]] = {}
    for base, crops in grouped.items():
        if sorted(crops) == [0, 1, 2, 3, 4]:
            complete[base] = tuple(crops[i] for i in range(5))  # type: ignore[assignment]
    return complete


def _group_audio(directory: Path) -> dict[str, Path]:
    grouped: dict[str, Path] = {}
    for path in directory.glob("*.npy"):
        match = AUDIO_RE.match(path.name)
        if match:
            grouped[match.group("base")] = path
    return grouped


def _first_dim(path: Path) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


@dataclass
class XDFeatureIndex:
    feature_root: Path
    list_root: Path
    train_videos: list[XDVideoRecord]
    test_videos: list[XDVideoRecord]
    gt_path: Path
    gt_num_frames: int
    test_snippet_count: int
    missing_modalities: dict[str, list[str]]

    @classmethod
    def build(cls, feature_root: Path, list_root: Path) -> "XDFeatureIndex":
        feature_root = Path(feature_root)
        list_root = Path(list_root)
        gt_path = list_root / "gt.npy"
        gt_num_frames = int(np.load(gt_path, mmap_mode="r").shape[0]) if gt_path.exists() else 0
        missing: dict[str, list[str]] = {}

        def build_split(split: str) -> list[XDVideoRecord]:
            rgb = _group_visual(feature_root / "rgb" / split)
            flow = _group_visual(feature_root / "flow" / split)
            audio = _group_audio(feature_root / "audio" / split)
            list_name = "video_train.list" if split == "train" else "video_test.list"
            ordered = ordered_unique_video_ids(list_root / list_name)
            records: list[XDVideoRecord] = []
            frame_offset = 0
            for video_id in ordered:
                absent = []
                if video_id not in rgb:
                    absent.append("rgb")
                if video_id not in flow:
                    absent.append("flow")
                if video_id not in audio:
                    absent.append("audio")
                if absent:
                    missing[video_id] = absent
                    continue
                rgb_len = _first_dim(rgb[video_id][0])
                flow_len = _first_dim(flow[video_id][0])
                audio_len = _first_dim(audio[video_id])
                if len({rgb_len, flow_len, audio_len}) != 1:
                    missing[video_id] = [f"length:{rgb_len}/{flow_len}/{audio_len}"]
                    continue
                label_text, label = label_from_video_id(video_id)
                frame_count = rgb_len * 16 if split == "test" else None
                records.append(
                    XDVideoRecord(
                        video_id=video_id,
                        split=split,
                        label_text=label_text,
                        video_label=label,
                        rgb_paths=rgb[video_id],
                        flow_paths=flow[video_id],
                        audio_path=audio[video_id],
                        snippet_count=rgb_len,
                        frame_count=frame_count,
                        frame_offset=frame_offset if split == "test" else None,
                    )
                )
                if split == "test":
                    frame_offset += frame_count or 0
            return records

        train = build_split("train")
        test = build_split("test")
        test_snippets = sum(v.snippet_count for v in test)
        return cls(
            feature_root=feature_root,
            list_root=list_root,
            train_videos=train,
            test_videos=test,
            gt_path=gt_path,
            gt_num_frames=gt_num_frames,
            test_snippet_count=test_snippets,
            missing_modalities=missing,
        )

    def records_for_split(self, split: str) -> list[XDVideoRecord]:
        if split == "train":
            return self.train_videos
        if split == "test":
            return self.test_videos
        raise ValueError(f"unknown split: {split}")

    def frame_labels_for(self, records: Iterable[XDVideoRecord]) -> np.ndarray:
        labels = np.load(self.gt_path, mmap_mode="r")
        chunks = []
        for record in records:
            if record.frame_offset is None or record.frame_count is None:
                raise ValueError("frame labels are only available for test records")
            start = record.frame_offset
            end = start + record.frame_count
            chunks.append(np.asarray(labels[start:end], dtype=np.float32))
        return np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.float32)


def mean_pool_crops(paths: tuple[Path, Path, Path, Path, Path]) -> np.ndarray:
    arrays = [np.load(path).astype(np.float32, copy=False) for path in paths]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)


def resample_temporal(features: np.ndarray, temporal_size: int) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected [T,D] features, got shape {arr.shape}")
    if temporal_size <= 0:
        raise ValueError("temporal_size must be positive")
    if arr.shape[0] == temporal_size:
        return arr.copy()
    if arr.shape[0] == 1:
        return np.repeat(arr, temporal_size, axis=0)
    old_x = np.linspace(0.0, 1.0, num=arr.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=temporal_size, dtype=np.float32)
    out = np.empty((temporal_size, arr.shape[1]), dtype=np.float32)
    for dim in range(arr.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, arr[:, dim]).astype(np.float32)
    return out


class XDFeatureDataset(Dataset):
    def __init__(
        self,
        feature_root: Path,
        list_root: Path,
        split: str,
        temporal_size: int = 32,
        limit_videos: int | None = None,
    ) -> None:
        self.index = XDFeatureIndex.build(feature_root=feature_root, list_root=list_root)
        records = self.index.records_for_split(split)
        self.records = records[:limit_videos] if limit_videos is not None else records
        self.temporal_size = temporal_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        rgb = resample_temporal(mean_pool_crops(record.rgb_paths), self.temporal_size)
        flow = resample_temporal(mean_pool_crops(record.flow_paths), self.temporal_size)
        audio = resample_temporal(np.load(record.audio_path).astype(np.float32, copy=False), self.temporal_size)
        return {
            "rgb": torch.from_numpy(rgb),
            "flow": torch.from_numpy(flow),
            "audio": torch.from_numpy(audio),
            "modality_mask": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "video_label": torch.tensor(record.video_label, dtype=torch.float32),
            "video_id": record.video_id,
            "snippet_count": torch.tensor(record.snippet_count, dtype=torch.long),
        }
