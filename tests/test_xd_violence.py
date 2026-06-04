from __future__ import annotations

from pathlib import Path

from ccd_mavd.data.xd_violence import XDFeatureDataset, XDFeatureIndex


ROOT = Path('/home/han/CCD-MAVD')


def test_xd_feature_index_counts_and_alignment():
    index = XDFeatureIndex.build(
        feature_root=ROOT / 'data/features/xd_violence',
        list_root=ROOT / 'data/lists/xd_violence',
    )

    assert len(index.train_videos) == 3954
    assert len(index.test_videos) == 800
    assert index.gt_num_frames == 2330384
    assert index.test_snippet_count == 145649
    assert index.gt_num_frames == index.test_snippet_count * 16
    assert index.missing_modalities == {}


def test_xd_dataset_returns_resampled_multimodal_sample():
    dataset = XDFeatureDataset(
        feature_root=ROOT / 'data/features/xd_violence',
        list_root=ROOT / 'data/lists/xd_violence',
        split='train',
        temporal_size=32,
        limit_videos=2,
    )

    sample = dataset[0]

    assert sample['rgb'].shape == (32, 1024)
    assert sample['flow'].shape == (32, 1024)
    assert sample['audio'].shape == (32, 128)
    assert sample['modality_mask'].tolist() == [1.0, 1.0, 1.0]
    assert sample['video_label'].item() in (0, 1)
    assert isinstance(sample['video_id'], str)
