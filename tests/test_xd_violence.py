from __future__ import annotations

from pathlib import Path

import pytest

from ccd_mavd.data.xd_violence import (
    XDFeatureDataset,
    XDFeatureIndex,
    build_balanced_subset_manifest,
    label_from_video_id,
    load_subset_manifest,
    records_by_video_ids,
    save_subset_manifest,
)


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


def test_xd_train_balanced_limit_keeps_positive_and_negative_bags():
    dataset = XDFeatureDataset(
        feature_root=ROOT / 'data/features/xd_violence',
        list_root=ROOT / 'data/lists/xd_violence',
        split='train',
        temporal_size=32,
        limit_videos=16,
        balanced_limit=True,
    )

    labels = [record.video_label for record in dataset.records]

    assert labels.count(1) == 8
    assert labels.count(0) == 8


def test_balanced_subset_manifest_saves_ids_and_label_counts(tmp_path):
    index = XDFeatureIndex.build(
        feature_root=ROOT / 'data/features/xd_violence',
        list_root=ROOT / 'data/lists/xd_violence',
    )

    manifest = build_balanced_subset_manifest(
        index=index,
        name='xd_bal64_seed0',
        seed=0,
        train_limit=64,
        test_limit=64,
    )
    manifest_path = tmp_path / 'xd_bal64_seed0.json'
    save_subset_manifest(manifest, manifest_path)
    loaded = load_subset_manifest(manifest_path)

    assert loaded['name'] == 'xd_bal64_seed0'
    assert loaded['seed'] == 0
    assert len(loaded['train_video_ids']) == 64
    assert len(loaded['test_video_ids']) == 64
    assert loaded['num_train_normal'] == 32
    assert loaded['num_train_abnormal'] == 32
    assert loaded['num_test_normal'] == 32
    assert loaded['num_test_abnormal'] == 32

    test_records = records_by_video_ids(index.test_videos, loaded['test_video_ids'])
    assert {record.video_label for record in test_records} == {0, 1}


def test_label_from_video_id_rejects_unparseable_label():
    with pytest.raises(ValueError, match='Cannot parse XD label'):
        label_from_video_id('video_without_target_token')
