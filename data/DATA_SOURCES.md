# Data Sources

Copied from existing server-local datasets on 2026-06-03.

## XD-Violence

Source: `/home/han/CQAF-VAD/XDData`

Copied subsets:

- `RGB` -> `data/features/xd_violence/rgb/train`
- `RGBTest` -> `data/features/xd_violence/rgb/test`
- `Flow` -> `data/features/xd_violence/flow/train`
- `FlowTest` -> `data/features/xd_violence/flow/test`
- `vggish-features` -> `data/features/xd_violence/audio`

Not copied: `downloads/vggish-features.zip`.

## ShanghaiTech

Source: `/home/han/cv/data/processed/shanghaitech_deep/p2_4b_v1_clip_i3d`

Copied subset:

- `train.npz`, `test.npz`, and `feature_manifest.json` -> `data/features/shanghaitech/i3d_clip`

Not copied: raw videos/frames and the small OpenCV histogram feature set.
