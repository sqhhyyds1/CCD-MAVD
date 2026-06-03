# CCD-MAVD Complete Experiment Design

## Material Passport

- Origin: Codex experiment-design pass using local proposal documents and source-backed literature verification
- Date: 2026-06-03
- Project root: `/home/han/CCD-MAVD`
- Verification status: DESIGN_VERIFIED, RESULTS_UNVERIFIED
- Result boundary: this document defines the experiment; it does not report experimental results.
- Revision scope: dataset availability is assumed for XD-Violence（暴力检测数据集）, ShanghaiTech（上海科技大学校园异常检测数据集）, and UCF-Crime（犯罪异常检测数据集）; upload/staging status is not treated as a methodological limitation.
- Required wording boundary: use `causal-inspired context debiasing`（因果启发的上下文去偏）, not strict `causal identification`（因果识别）.


## 1. Research Objective

CCD-MAVD aims to test whether a lightweight, dataset-aware multimodal weakly supervised VAD（视频异常检测） framework can improve robustness beyond pure feature fusion by separating event evidence from visual context shortcuts.

The central research question is:

> In weakly supervised multimodal video anomaly or violence detection, can an alignment-first event branch plus visual context debiasing improve detection and cross-scene robustness without adding a heavy fusion Transformer（融合 Transformer）?

The experiment plan assumes the three target datasets are available when each stage is executed:

- XD-Violence（暴力检测数据集） is the main multimodal benchmark using RGB（视觉外观）, Flow（光流）, and Audio（音频） features.
- ShanghaiTech（上海科技大学校园异常检测数据集） is the scene/context benchmark using visual features and scene identifiers or documented scene proxies.
- UCF-Crime（犯罪异常检测数据集） is the weak-supervised generalization benchmark using RGB-first and optional Flow（光流） features.

Dataset upload or local staging status is an engineering logistics item, not a design limitation. The experiment design should judge protocol compatibility, feature/annotation schemas, and evaluation fairness, not whether a dataset is currently present on disk.

## 2. Source-Backed Facts and Design Inferences

### 2.1 Verified Source Facts

- XD-Violence is a large-scale multi-scene dataset with 4754 untrimmed videos, 217 hours, audio signals, weak labels, and official I3D RGB/Flow plus VGGish feature downloads. Source: [XD-Violence official page](https://roc-ng.github.io/XD-Violence/).
- RTFM（鲁棒时序特征幅值学习） formulates weakly supervised VAD（视频异常检测） as MIL（多实例学习） over video snippets and uses top-k feature magnitude learning to reduce positive-bag noise and improve temporal discrimination. Source: [RTFM ICCV 2021 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf).
- CoMo（上下文-运动关系学习） motivates context-sensitive anomaly decisions: the same motion may be normal or abnormal depending on surrounding scene context. Source: [CoMo CVPR 2023 page](https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.html).
- MAVD（先对齐再融合的多模态暴力检测方法） reports an alignment-first multimodal pipeline and 86.07 AP（平均精度） on XD-Violence. Source: [MAVD arXiv](https://arxiv.org/abs/2501.07496) and [MAVD GitHub](https://github.com/xjpp2016/MAVD).
- CLIP（图文对比预训练模型） provides transferable image-text visual representations and released code/weights; it can be used as a frozen source of visual context embeddings. Source: [CLIP arXiv](https://arxiv.org/abs/2103.00020) and [OpenAI CLIP page](https://openai.com/research/clip).
- DANN（领域对抗神经网络） introduces GRL（梯度反转层） to learn task-relevant but domain-invariant representations. Source: [DANN arXiv](https://arxiv.org/abs/1505.07818).
- Barlow Twins（冗余降低自监督学习） motivates correlation-reduction objectives; CCD-MAVD uses a simpler orthogonal/covariance penalty as an engineering-friendly event/context decorrelation regularizer. Source: [Barlow Twins arXiv](https://arxiv.org/abs/2103.03230).


### 2.2 Design Inferences

These are design choices, not established dataset facts:

- CLIP/KMeans（CLIP 特征加 K 均值聚类） context labels（上下文标签） are pseudo or latent context signals, not ground-truth scene annotations.
- Context debiasing（上下文去偏） is causal-inspired because the datasets do not provide interventions, randomized contexts, or a validated causal graph.
- Expected ablation（消融实验） gains are hypotheses; they must not be written as results until experiments are run.
- Dataset availability is assumed for planning; the methodological question is whether each run uses the correct official split, modality set, annotation granularity, and frame-level metric protocol.
- ShanghaiTech（上海科技大学校园异常检测数据集） must be reported under the exact protocol used in the run. If a normal-only diagnostic split is used, call it a diagnostic stage; if the weakly supervised 238/199 split is used, report it separately as the comparable weak-supervised setting.


## 3. Current Server Data Layout

The copied datasets are recorded in `/home/han/CCD-MAVD/data/DATA_SOURCES.md`. This section is only an implementation snapshot. Absence from the snapshot must not be described as a methodological weakness, because all three datasets are assumed available for the final experiment plan.

```text
/home/han/CCD-MAVD/data/features/xd_violence/
  rgb/train/      # copied from /home/han/CQAF-VAD/XDData/RGB
  rgb/test/       # copied from /home/han/CQAF-VAD/XDData/RGBTest
  flow/train/     # copied from /home/han/CQAF-VAD/XDData/Flow
  flow/test/      # copied from /home/han/CQAF-VAD/XDData/FlowTest
  audio/train/    # under copied vggish-features/train
  audio/test/     # under copied vggish-features/test

/home/han/CCD-MAVD/data/features/shanghaitech/i3d_clip/
  train.npz
  test.npz
  feature_manifest.json
```

Verified copied sizes in the current snapshot:

| Dataset | Target path | Files | Size |
|---|---|---:|---:|
| XD-Violence（暴力检测数据集） | `data/features/xd_violence` | 52295 | 45.102 GB |
| ShanghaiTech（上海科技大学校园异常检测数据集） | `data/features/shanghaitech/i3d_clip` | 3 | 3.022 GB |

The raw videos/frames（原始视频/帧） and unrelated archive files were intentionally not copied in this snapshot. If a later stage needs UCF-Crime（犯罪异常检测数据集） or raw-frame CLIP（图文对比预训练模型） extraction, upload/staging should be handled as routine infrastructure work and documented in the run manifest.


## 4. Dataset Protocols

| Dataset | Role | Assumption | Modalities（模态） | Context source（上下文来源） | Primary metric（主指标） | Protocol note |
|---|---|---|---|---|---|---|
| XD-Violence | Main multimodal benchmark | Available for final runs | RGB + Flow + Audio | CLIP/KMeans pseudo context（伪上下文） or feature-cluster context（特征聚类上下文） | frame-level AP（帧级平均精度） | Use official feature-style setup; compare against XD multimodal baselines with matched modalities. |
| ShanghaiTech | Scene/context diagnostic and weak-supervised benchmark | Available for final runs | I3D + CLIP-derived views, or the official feature protocol used by the baseline | scene ids when present; otherwise documented audit_scene_proxy（审计场景代理） | frame-level ROC-AUC（帧级受试者工作特征曲线下面积） | Report normal-only diagnostic and weak-supervised split results as separate protocols. |
| UCF-Crime | Weak-supervised generalization benchmark | Available for final runs | RGB first, Flow optional | CLIP/KMeans latent context（潜在上下文） or documented scene proxy | frame-level ROC-AUC（帧级受试者工作特征曲线下面积） | Use shared modalities for cross-dataset comparisons; do not mix multimodal XD gains with RGB-only UCF claims. |

## 5. Experimental Hypotheses

H1. Alignment-first multimodal event modeling improves XD-Violence AP（平均精度） over a naive concatenation or single-modality MIL（多实例学习） baseline.

H2. Mask-aware gated fusion（掩码感知门控融合） improves robustness to missing or noisy modalities, especially under audio/flow dropout（模态丢弃） tests.

H3. Visual context debiasing（视觉上下文去偏） via GRL（梯度反转层） and counterfactual context swap（反事实上下文交换） reduces scene shortcut（场景捷径） dependence without removing useful event evidence.

H4. Orthogonal decorrelation（正交去相关） between event and context representations improves cross-scene stability more reliably than heavier MI（互信息） estimation in the first implementation.

## 6. Model Design


### 6.1 High-Level Architecture

```text
Input snippets
  -> modality projectors: RGB / Flow / Audio
  -> event branch: projected modality features
  -> local alignment loss across available modalities
  -> mask-aware gated fusion
  -> temporal encoder
  -> event representation z_t

Visual features
  -> visual context branch
  -> context representation c_t
  -> context classifier trains c_t to contain context signal
  -> GRL adversary discourages z_t from predicting context

z_t
  -> score head
  -> snippet anomaly scores
  -> top-k MIL video score
```

The default main model keeps the anomaly score head（异常分数头） event-only: `score_t = h(z_t)`. The context representation `c_t` is used for supervision, adversarial debiasing, diagnostics, and optional ablations（消融实验）, but it does not directly enter the main score head. This avoids replacing one context shortcut（上下文捷径） path with another.

Optional context-conditioned ablations（上下文条件消融） may use `score_t = h([z_t ; beta * g(c_t)])`, but these runs must be reported as ablations, not as the default debiased model.


### 6.2 Components

| Component | Input | Output | Purpose |
|---|---|---|---|
| RGB projector（RGB 投影头） | RGB feature | 256-d event feature | Put visual appearance into event space. |
| Flow projector（光流投影头） | Flow feature | 256-d event feature | Add motion evidence when available. |
| Audio projector（音频投影头） | VGGish feature | 256-d event feature | Add audio event evidence for XD-Violence. |
| Alignment projector（对齐投影头） | event feature | 128-d normalized feature | Compute local-window multimodal alignment on a shared temporal grid（时间网格）. |
| Context branch（上下文分支） | RGB or visual semantic features | 128-d context feature | Represent visual scene/context; audio is excluded. |
| Mask-aware gate（掩码感知门控） | available event features + modality mask | modality weights | Fuse only available modalities. |
| Temporal encoder（时序编码器） | fused event sequence | contextualized event sequence | Model snippet temporal dependence. |
| Score head（异常分数头） | event representation `z_t` | snippet score | Predict anomaly score without direct context input in the main model. |
| Context-conditioned score head（上下文条件异常分数头） | `[z_t ; beta * g(c_t)]` | snippet score | Ablation only; tests whether direct context improves scores or creates shortcuts. |
| GRL adversary（梯度反转对抗器） | pooled event representation | context prediction | Penalize context leakage from event features. |
| Context classifier（上下文分类器） | pooled context representation | context label prediction | Ensure context branch is meaningful. |


### 6.3 Recommended Default Dimensions

| Module | Default |
|---|---|
| Event hidden dim | 256 |
| Alignment dim | 128 |
| Context dim | 128 |
| Context residual dim | 64, ablation only |
| Temporal encoder | 2-layer TransformerEncoder（Transformer 编码器）, 4 heads, FFN 512 |
| Training temporal grid（训练时间网格） | `T = 32` snippets after deterministic resampling |
| top-k（前 k 个片段） | Pilot default `k = 3`; final report must include fixed-k or ratio-k sensitivity |

All input feature dimensions must be read from data or config. Do not hard-code 1024, 2048, 512, or 128 globally.

## 7. Loss Design

Let `s_t` be snippet anomaly scores, `z_t` event features, and `c_t` context features.


### 7.1 MIL Loss（多实例学习损失）

Use top-k video score after each video has been mapped to the same training temporal grid（训练时间网格）:

```text
S(v) = mean(topk(s_1, ..., s_T), k)
L_mil = BCE(S(v_pos), 1) + BCE(S(v_neg), 0)
```

Default pilot values: `T = 32`, `k = 3`. Final experiments must report one of the following and keep it fixed across baselines and ablations（消融实验）:

- Fixed-k sweep（固定 k 扫描）: `k in {1, 3, 5, 8}`.
- Ratio-k rule（比例 k 规则）: `k = max(1, floor(T / r))`, with `r in {8, 10}`.

For unpaired batching, use per-video BCE over video labels first. Add paired ranking only when the sampler reliably pairs positive and negative bags in the same batch（批量）.

### 7.2 Ranking Loss（排序损失）

```text
L_rank = max(0, margin - S(v_pos) + S(v_neg))
```

Default margin: 1.0. Weight: `lambda_rank = 0.1`.

### 7.3 Smoothness and Sparsity（平滑与稀疏）

```text
L_smooth = mean((s_t - s_{t+1})^2)
L_sparse = mean(s_t)
```

Default weights: `lambda_smooth = 1e-4`, `lambda_sparse = 1e-4`.


### 7.4 Local Alignment（局部对齐）

All available modalities must first be mapped onto a shared temporal grid（时间网格） of length `T = 32` for training. For modality pair `(m, n)`, use a local positive window `|u - t| <= delta` and contrast it against all snippets from the paired sequence.

Default values:

- `delta = 1`
- `tau_align = 0.07`
- `lambda_align = 0.10`

Implementation rules:

1. For 5-crop visual features（五裁剪视觉特征）, group crop-suffixed files by video id before alignment; use the mean over crops for the default run and keep crop dimension only in an explicitly named ablation.
2. Resample RGB（视觉外观）, Flow（光流）, and Audio（音频） to the same temporal grid before computing `L_align`.
3. If one modality is missing, too short, or cannot be mapped to the grid, disable the alignment term for that modality pair only.
4. Do not treat crop suffixes such as `__0.npy` to `__4.npy` as different videos.

Apply only to available modality pairs. For ShanghaiTech without audio, use RGB/Flow or the documented visual feature pair if both are available. For single-view runs, disable this loss.

### 7.5 Context Classification and GRL（上下文分类与梯度反转）

```text
L_ctx = CE(context_classifier(avg(c_t)), context_label)
L_adv = CE(adversary(GRL(avg(z_t))), context_label)
```

The optimizer minimizes `L_adv`, while GRL reverses gradients into the event branch. Default weights:

- `lambda_ctx = 1.0`
- `lambda_grl = 0.20`, linearly ramped from 0 after warm-up.


### 7.6 Counterfactual Context Swap（反事实上下文交换）

Counterfactual context swap is not part of the default main objective. The main score head is event-only:

```text
score_t = h(z_t)
```

Use context swap only in a late-stage ablation（后期消融实验） with a context-conditioned score head:

```text
score_t = h([z_t ; beta * tanh(W_c c_t)])
L_cf = Huber(score_i - score_i_with_context_j)
```

Safety rules for `L_cf`（反事实损失）:

- Enable only after the MIL（多实例学习） score is stable, for example after epoch 20 or after validation metrics plateau.
- Use stop-gradient teacher scores（停止梯度教师分数） or an EMA teacher（指数滑动平均教师模型） when selecting high-confidence snippets.
- Use all normal snippets plus abnormal top-k snippets only when their confidence exceeds a documented threshold.
- Report whether enabling `L_cf` improves or harms AP/AUC（平均精度/曲线下面积）; if it lowers performance or increases context shortcut behavior, keep it out of the main model.

Default values for the ablation only:

- `beta = 0.10`
- `lambda_cf = 0.05`


### 7.7 Orthogonal Decorrelation（正交去相关）

Use a covariance/correlation penalty between event and projected context features only as a regularized debiasing ablation（正则化去偏消融）:

```text
L_orth = || Z_norm^T C_norm / (B*T) ||_F^2
```

Default weight in the main model: `lambda_orth = 0`. Ablation weight: `lambda_orth = 0.01`.

MI（互信息） estimation is not part of the first implementation. It can be added later as an appendix ablation because it is heavier and less stable.


### 7.8 Total Objective（总目标）

Use a staged objective instead of enabling every loss in the first full model.

Default event-only CCD-MAVD objective:

```text
L_main = L_mil
  + lambda_rank * L_rank
  + lambda_smooth * L_smooth
  + lambda_sparse * L_sparse
  + lambda_align * L_align
  + lambda_ctx * L_ctx
  + lambda_grl * L_adv
```

Optional regularized debiasing ablations（正则化去偏消融）:

```text
L_cf_ablation = L_main + lambda_cf * L_cf
L_orth_ablation = L_main + lambda_orth * L_orth
L_full_regularized_ablation = L_main + lambda_cf * L_cf + lambda_orth * L_orth
```

Recommended model ladder:

| Version | Objective | Purpose |
|---|---|---|
| V0: MIL baseline（多实例学习基线） | `L_mil + L_rank + L_smooth + L_sparse` | Reproduce weakly supervised training behavior. |
| V1: MAVD-style event fusion（事件融合） | V0 + `L_align` + gated fusion（门控融合） | Test multimodal alignment/fusion. |
| V2: context debias main（上下文去偏主模型） | V1 + `L_ctx + L_adv` with event-only score head | Main CCD-MAVD claim. |
| V3: regularized debias ablation（正则化去偏消融） | V2 + `L_cf` or `L_orth` | Appendix or late-stage robustness test. |

Do not present V3 as the default full model unless it is more stable than V2 across seeds, context probes, and frame-level metrics.

## 8. Training Protocol

### 8.1 Global Defaults

| Item | Default |
|---|---|
| Python（脚本解释器） | `/home/han/miniconda3/envs/hh/bin/python` |
| GPU（图形处理器） | Set `CUDA_VISIBLE_DEVICES` explicitly for every run |
| Epochs（训练轮数） | 50 |
| Optimizer（优化器） | AdamW for CCD-MAVD; preserve baseline optimizer when reproducing baselines |
| LR（学习率） | 3e-4 for CCD-MAVD |
| Weight decay（权重衰减） | 1e-4 |
| Gradient clip（梯度裁剪） | 5.0 |
| Batch size（批量大小） | XD 64 if memory allows; Shanghai 32; reduce to 16/32 if needed |
| Seeds（随机种子） | 5 final seeds: 0, 1, 2, 3, 4 |

Server environment checked on 2026-06-03:

```text
Python 3.10.20
PyTorch 2.11.0+cu128
CUDA available: True
GPU count: 2
GPU 0: NVIDIA GeForce RTX 4090
```


### 8.2 Staged Training

Stage A: baseline warm-up implementation.

- Train MIL-only（仅多实例学习） single-dataset model.
- Verify feature loading, labels, snippet sampling, AP/AUC calculation, score-to-frame expansion（分数到帧扩展）, and checkpoint saving.

Stage B: RTFM-style baseline.

- Add top-k feature magnitude/ranking-compatible behavior.
- Keep model simple; use it as a reproducibility anchor.

Stage C: CCD-MAVD event branch.

- Add modality projectors.
- Add mask-aware gated fusion（掩码感知门控融合）.
- Add modality dropout（模态丢弃） tests.

Stage D: alignment and context diagnostics.

- Add local alignment（局部模态对齐） on the shared temporal grid（时间网格）.
- Add visual context branch（视觉上下文分支）.
- Generate and store context labels（上下文标签）.
- Run context quality audit（上下文质量审计） before treating context labels as useful supervision.

Stage E: main debiasing.

- Add GRL（梯度反转层） after warm-up.
- Keep the main score head event-only: `score_t = h(z_t)`.
- Report context leakage probes（上下文泄漏探针） and anomaly retention probes（异常保留探针）.

Stage F: optional regularized debiasing ablations.

- Add counterfactual context swap（反事实上下文交换） only after stable warm-up or epoch 20.
- Add orthogonal decorrelation（正交去相关） only as an ablation unless it improves stability.

Stage G: final evaluation.

- Run 5 seeds.
- Produce mean/std（均值/标准差）.
- Run paired bootstrap（配对自助法） for confidence intervals.


### 8.3 Warm-Up Schedule

Epochs 1-5:

```text
L_mil + L_rank + L_smooth + L_sparse + optional L_align
```

Epochs 6-50 for the default main model:

```text
L_main, with lambda_grl ramped linearly to target value
```

Optional late-stage ablations after epoch 20 or after validation stabilization:

```text
L_cf_ablation or L_orth_ablation
```

Do not enable GRL（梯度反转层）, counterfactual swap（反事实上下文交换）, and orthogonal decorrelation（正交去相关） from epoch 1. Do not put `L_cf`（反事实损失） or `L_orth`（正交损失） into the first default full model unless the V2 main model has already been shown stable.


## 9. Context Label Generation

### 9.1 XD-Violence

Default first route:

- Generate CLIP（图文对比预训练模型） embeddings from sampled video frames if raw frames/videos are available for that run.
- If only current I3D/VGGish features are used, create a temporary pseudo-context from visual feature clustering and mark it as `feature_cluster_context`（特征聚类上下文）, not `scene`（场景）.
- Use KMeans（K 均值聚类） with `K in {8, 12, 16}` and choose by training-split silhouette score（训练集轮廓系数） if stable.
- Fit KMeans only on the training split; assign validation/test context labels by nearest training centroid. Do not fit pseudo-context labels on train + test together.
- Store in `data/context_labels/xd_violence/context_labels.pkl` or `.npz`.

### 9.2 ShanghaiTech

Actions:

- Inspect whether `scene_ids` exist inside the `.npz` files.
- If scene ids exist, use them for context labels and cross-scene diagnostics.
- If scene ids do not exist, parse video names or derive audit strata from metadata; mark as `audit_scene_proxy`（审计场景代理）.
- Report the exact ShanghaiTech protocol used by each run: normal-only diagnostic, weakly supervised 238/199 split, or another documented split.

### 9.3 UCF-Crime

Planned route:

- Use CLIP/KMeans（CLIP 特征加 K 均值聚类） latent context or a documented scene proxy.
- Fit pseudo-context labels on the training split only and assign held-out videos by nearest centroid.
- Keep GRL（梯度反转层） weaker than ShanghaiTech when context labels are noisy.
- Use RGB-only first for fair cross-dataset comparison, then add Flow（光流） if the compared baselines use it.

### 9.4 Context Quality Audit（上下文质量审计）

Before using pseudo-context labels as supervision, run and report the following checks:

| Check | Purpose |
|---|---|
| `NMI(context, video_label)`（上下文与视频标签归一化互信息） | Detect whether context labels are strongly tied to normal/abnormal labels. |
| `context distribution per anomaly class`（每类异常的上下文分布） | Detect whether clusters are actually anomaly categories. |
| `context distribution per train/test split`（训练/测试上下文分布） | Detect split leakage or dataset-source artifacts. |
| `linear probe: context -> anomaly`（线性探针：上下文到异常） | Check whether context alone predicts anomaly too well. |
| `visual samples per cluster`（每簇视觉样本） | Human audit: decide whether clusters look like scenes, quality artifacts, or semantic events. |

If `context -> anomaly` is high, treat context as a possible shortcut（捷径） and reduce claim strength. If pseudo-context looks like compression quality, source id, or anomaly class rather than scene/context, do not use it as the main debiasing label.


## 10. Evaluation Protocol

### 10.1 Metrics（指标）

| Dataset | Primary metric（主指标） | Secondary metrics（辅指标） |
|---|---|---|
| XD-Violence | frame-level AP（帧级平均精度） | frame-level ROC-AUC（帧级曲线下面积）, modality-missing AP |
| ShanghaiTech | frame-level ROC-AUC（帧级曲线下面积） under the declared protocol | AP（平均精度）, per-scene AUC, cross-scene retention |
| UCF-Crime | frame-level ROC-AUC（帧级曲线下面积） | AP/PR-AUC（平均精度/精确率-召回率曲线下面积）, cross-dataset retention |

### 10.2 Score-to-Frame Expansion（分数到帧扩展）

The model outputs `T` snippet scores, but frame-level AP/AUC（帧级平均精度/曲线下面积） requires one score per annotated frame. Every run, baseline, and ablation must use the same expansion rule.

Default rule:

```text
Given T snippet scores and N annotated frames:
1. linearly interpolate snippet scores to length N;
2. clip scores to the valid evaluation range if the metric wrapper requires it;
3. save both original snippet scores and expanded frame scores.
```

Allowed alternative for a dataset-specific official protocol:

```text
repeat each snippet score over its temporal segment [l_t, r_t]
```

Choose the rule before running experiments, record it in `config.yaml`, and do not change it mid-study. If a cited baseline uses a different expansion rule, report that as a protocol difference.

### 10.3 Fairness Rules

- Do not compare multimodal XD-Violence runs directly against RGB-only UCF/ShanghaiTech methods in a single leaderboard.
- For cross-dataset runs, use shared modalities only: RGB-only or RGB+Flow-only.
- If CCD-MAVD uses CLIP（图文对比预训练模型） or pseudo-context labels（伪上下文标签）, include baselines with access to the same extra information but without the proposed debiasing mechanism.
- Baseline re-runs should preserve their original optimizer, epochs, and feature protocol where feasible.
- CCD-MAVD can use a unified AdamW（AdamW 优化器） setup, but this must be disclosed separately from baseline settings.

### 10.4 Statistical Testing（统计检验）

Final results must include:

- 5 seeds（随机种子） for main runs.
- Mean ± std（均值加减标准差）.
- Paired bootstrap（配对自助法） over test videos, 10,000 resamples if runtime allows.
- 95% confidence interval（95% 置信区间） for metric differences against the strongest fair baseline.

Single-seed results are allowed only for smoke tests（冒烟测试） and preliminary ablations（初步消融）.


## 11. Baselines and Comparison Plan

### 11.1 Internal Baselines

| Name | Purpose | Required before CCD-MAVD claim |
|---|---|---|
| MIL-only | sanity baseline | Yes |
| RTFM-style | weak-supervised reproducibility anchor | Yes |
| concat-fusion | simple multimodal fusion baseline | Yes for XD |
| mask-gated fusion only | isolate fusion effect | Yes |
| context branch without debias | isolate context modeling | Yes |
| same-context no-GRL baseline（同上下文无梯度反转基线） | controls for CLIP/KMeans information gain | Yes when CCD-MAVD uses pseudo-context labels |
| context-conditioned score head baseline（上下文条件分数头基线） | tests whether direct context creates a shortcut | Yes as a diagnostic ablation |

### 11.2 External Baselines to Cite or Reproduce

| Method | Dataset role | Use |
|---|---|---|
| Sultani MIL | UCF-Crime foundation | cite and reproduce if protocol/features match |
| RTFM | UCF/Shanghai/XD anchor | reproduce or cite with protocol match |
| CoMo | context-motion motivation | cite; compare on Shanghai/UCF if protocol available |
| TPWNG | text/normality guidance critique of one-stage MIL | cite as related work and possible later baseline |
| PEL4VAD | prompt-enhanced context VAD | cite and maybe compare if features/protocol match |
| MACIL-SD | audio-visual asynchrony | cite/compare on XD |
| MSBT | heavy multimodal fusion baseline | cite/compare on XD |
| MAVD | alignment-first reference | run official code at least once if claiming improvement; otherwise cite official number and clearly mark protocol differences |

### 11.3 Fair Information Baseline Set（同等信息量基线）

If CCD-MAVD uses additional CLIP（图文对比预训练模型） frame context, KMeans（K 均值聚类） context labels, or context branch features, report this minimum set:

| Baseline | Role |
|---|---|
| Original/reimplemented MAVD | Original alignment/fusion reference. |
| MAVD + CLIP/context no debias | Controls for extra CLIP/context information. |
| MAVD + context branch no GRL | Controls for the context branch itself. |
| CCD-MAVD main | Tests whether event-only score head + context debiasing is useful. |

This table should appear in the main paper if the proposed method relies on extra context information. Otherwise, reviewers may attribute gains to extra inputs rather than debiasing.


## 12. Ablation Matrix

Run ablations in this order. Each row should inherit the same data split, seed, optimizer, score-to-frame expansion rule, and evaluation code.

| ID | Modules enabled | Main question |
|---|---|---|
| A0 | MIL-only | Is the data/metric pipeline valid? |
| A1 | A0 + RTFM-style top-k/ranking | Does top-k ranking stabilize weak supervision? |
| A2 | A1 + modality projectors + concat fusion | Does multimodal evidence help without gating? |
| A3 | A1 + mask-aware gated fusion | Does gating beat concatenation? |
| A4 | A3 + local alignment | Does alignment help multimodal temporal consistency? |
| A5 | A4 + context branch + `L_ctx` only | Does explicit context modeling provide a meaningful diagnostic branch? |
| A6 | A5 + GRL, event-only score head | Main debiasing model: does event representation become less context-predictive while retaining anomaly signal? |
| A7 | A5 + direct context-conditioned score head, no GRL | Shortcut diagnostic: does direct context inflate scores? |
| A8 | A6 + late `L_cf` | Optional: are scores stable under context replacement after warm-up? |
| A9 | A6 + `L_orth` | Optional: does event/context leakage decrease without hurting AP/AUC? |
| A10 | best stable model + modality dropout | Does robustness improve under missing modality tests? |

No ablation gain should be described as a finding until the corresponding experiment log and metrics file exist. Do not promote A8/A9 to the default full model unless they improve both detection metrics and diagnostic probes across seeds.

## 13. Robustness and Diagnostic Tests

### 13.1 Missing Modality Tests（缺失模态测试）

For XD-Violence:

- Full: RGB + Flow + Audio.
- No audio: RGB + Flow.
- No flow: RGB + Audio.
- RGB-only.
- Random modality dropout at evaluation with probabilities 0.1, 0.2, 0.3.


### 13.2 Context Leakage and Anomaly Retention Tests（上下文泄漏与异常保留测试）

Train frozen linear probes（线性探针） from the same frozen checkpoint to measure what information remains in each representation:

| Probe | Expected pattern |
|---|---|
| `z -> context`（事件表征到上下文） | Lower after GRL（梯度反转层） and optional decorrelation. |
| `c -> context`（上下文表征到上下文） | Remains high; otherwise the context branch is not meaningful. |
| `z -> anomaly`（事件表征到异常） | Should not clearly decrease after debiasing; otherwise useful event evidence was removed. |
| `c -> anomaly`（上下文表征到异常） | Should not be too high; if high, context labels themselves may be shortcuts. |

Report probe accuracy/AUC（准确率/曲线下面积） before and after GRL. The key claim is not just that `z` loses context information, but that `z` still keeps anomaly-relevant information while `c` does not become a standalone anomaly predictor.

If `c -> anomaly` is high, downgrade claims and treat context debiasing as risky because the context label may be tightly coupled to abnormal labels.

### 13.3 Cross-Scene Diagnostic（跨场景诊断）

For ShanghaiTech:

- per-scene AUC（逐场景曲线下面积）.
- macro average over scenes（场景宏平均）.
- worst-scene AUC（最差场景曲线下面积）.
- retention = worst-scene AUC / overall AUC.

If weakly supervised ShanghaiTech split is staged later, add leave-scene-out（留场景外测试） or k-fold cross-scene（跨场景 K 折） experiments.

### 13.4 Cross-Dataset Diagnostic（跨数据集诊断）

For UCF-Crime cross-dataset runs:

- Train XD RGB-only -> test UCF RGB-only.
- Train UCF RGB-only -> test XD RGB-only.
- Keep modality set, score-to-frame expansion（分数到帧扩展）, and preprocessing rules matched before interpreting cross-dataset retention.
- Optionally RGB+Flow if both datasets provide compatible flow features.

Do not include audio in cross-dataset comparisons unless both datasets have comparable audio features.

## 14. Output Artifacts

Every run should create a timestamped directory:

```text
experiments/{dataset}/{run_id}/
  config.yaml
  command.txt
  env.txt
  train.log
  metrics.jsonl
  checkpoints/best.pt
  checkpoints/last.pt
  predictions/test_scores.npz
  reports/summary.md
```

`metrics.jsonl` should contain one JSON object per epoch:

```json
{
  "epoch": 1,
  "loss_total": 0.0,
  "loss_mil": 0.0,
  "loss_align": 0.0,
  "loss_ctx": 0.0,
  "loss_adv": 0.0,
  "loss_cf": 0.0,
  "loss_orth": 0.0,
  "val_frame_ap": null,
  "val_frame_auc": null,
  "lr": 0.0003
}
```

Final aggregate files:

```text
experiments/aggregate/{date}/
  main_table.csv
  ablation_table.csv
  seed_summary.csv
  bootstrap_ci.json
  paper_ready_tables.md
```

## 15. Reproducibility Rules

- Always set `CUDA_VISIBLE_DEVICES`（可见 GPU 环境变量） explicitly.
- Always use `/home/han/miniconda3/envs/hh/bin/python` unless the environment changes and is documented.
- Always record `git rev-parse HEAD`（Git 提交哈希） if the repository has commits.
- Always record `pip freeze` or a compact dependency manifest.
- Do not overwrite run directories; create a new `run_id`.
- Save configs copied into each run directory.
- Save random seeds for Python, NumPy（数值计算库）, and PyTorch（深度学习框架）.

Recommended future training command shape:

```bash
CUDA_VISIBLE_DEVICES=0 /home/han/miniconda3/envs/hh/bin/python scripts/train.py \
  --config configs/xd.yaml \
  --run-name xd_a0_mil_seed0 \
  --seed 0 \
  --output experiments/xd_violence/xd_a0_mil_seed0
```

Before launching long training, verify:

```bash
pgrep -af train.py
nvidia-smi
```


## 16. Implementation Milestones

M0. Protocol audit.

- Inspect `.npy` and `.npz` feature shapes for each dataset used in the run.
- Generate manifest files for XD-Violence, ShanghaiTech, and UCF-Crime.
- Document official split, video-level labels, frame-level annotations, frame counts, and score-to-frame expansion（分数到帧扩展） rule.
- Confirm XD 5-crop features are grouped by video id before model input.

M1. Metrics and loaders.

- Implement AP（平均精度） and ROC-AUC（曲线下面积） wrappers.
- Implement deterministic score-to-frame expansion（分数到帧扩展）.
- Implement XD feature grouping: RGB/Flow files are crop-suffixed; group `__0.npy` to `__4.npy` correctly.
- Implement ShanghaiTech and UCF-Crime loaders under their declared protocols.

M2. MIL-only baseline.

- Test feature loading with a tiny subset.
- Train one seed on each planned dataset protocol.
- Verify metrics file, snippet score export, and expanded frame score export.

M3. RTFM-style top-k.

- Add top-k and ranking logic.
- Run fixed-k or ratio-k sensitivity.
- Compare against MIL-only.

M4. CCD-MAVD main modules.

- Add projectors, gated fusion, local alignment, context branch, GRL, and event-only score head.
- Add targeted unit tests for each loss and tensor shape.
- Run context quality audit（上下文质量审计） and probe diagnostics（探针诊断）.

M5. Optional regularizers.

- Add counterfactual context swap and orthogonal decorrelation only as ablations.
- Promote neither to default unless they improve metrics and probes across seeds.

M6. Experiments.

- Run ablation ladder on XD-Violence.
- Run ShanghaiTech diagnostic and weak-supervised protocols separately when both are used.
- Run UCF-Crime weak-supervised and cross-dataset protocols.

M7. Paper tables.

- Run final 5 seeds.
- Generate aggregate tables and confidence intervals.


## 17. Known Risks and Mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Causal overclaim | High | Use causal-inspired wording; report assumptions and diagnostics. |
| Context-conditioned shortcut | High | Keep main score head event-only; direct context score head is an ablation only. |
| Pseudo-context noise or leakage | High | Fit KMeans only on training split; run context quality audit and `c -> anomaly` probe. |
| Score-to-frame ambiguity | High | Fix interpolation or segment-repeat rule before experiments and apply it to all methods. |
| Baseline information mismatch | High | Add same-context/no-debias baselines when CCD-MAVD uses CLIP or pseudo-context. |
| Loss interference | Medium-High | Use staged objectives; keep `L_cf` and `L_orth` as late ablations unless proven stable. |
| Over-debiasing | Medium | Ramp GRL, monitor AP/AUC and context/anomaly probes together. |
| Modality misalignment | Medium | Use shared temporal grid, grouped 5-crop visual features, and local window alignment. |
| Feature dimension mismatch | Medium | Infer dims from files; do not hard-code feature dimensions. |
| Protocol mixing across datasets | Medium | Separate dataset/modalities/protocol tables; use shared modalities for cross-dataset claims. |
| Disk pressure | Low-Medium | Do not copy raw videos unless needed; document any raw-frame extraction separately. |


## 18. Decision Gates

Gate 1: Protocol audit pass.

- Each dataset used in the run exposes feature tensors, video labels, split ids, frame counts, and frame-level annotations through a documented loader.
- XD RGB/Flow 5-crop files are grouped correctly before model input.
- The score-to-frame expansion（分数到帧扩展） rule is fixed and saved in config.

Gate 2: Baseline pass.

- MIL-only can train for at least one epoch and export snippet/frame predictions.
- Metric computation completes without shape mismatch.
- top-k sensitivity or ratio-k choice is documented.

Gate 3: Module pass.

- Each CCD-MAVD loss has a unit test.
- Full model forward pass works with missing modality masks.
- Main score head is event-only unless running an explicitly named context-conditioned ablation.

Gate 4: Diagnostic pass.

- Context quality audit is complete for pseudo-context labels.
- Frozen probes report `z -> context`, `c -> context`, `z -> anomaly`, and `c -> anomaly`.
- Debiasing is not considered successful if anomaly retention drops sharply or context alone predicts anomaly too well.

Gate 5: Experiment pass.

- At least one full XD run, one ShanghaiTech run under the declared protocol, and one UCF-Crime run finish.
- Logs contain config, command, environment, metrics, score expansion rule, and checkpoint.

Gate 6: Paper readiness.

- Main results are 5-seed mean/std.
- Best baseline comparison is fair by dataset, modality, and information access.
- Claims are marked as source fact, experimental finding, or design inference.

## 19. Source Index

- XD-Violence official project page: https://roc-ng.github.io/XD-Violence/
- RTFM ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/papers/Tian_Weakly-Supervised_Video_Anomaly_Detection_With_Robust_Temporal_Feature_Magnitude_Learning_ICCV_2021_paper.pdf
- CoMo CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Cho_Look_Around_for_Anomalies_Weakly-Supervised_Anomaly_Detection_via_Context-Motion_Relational_CVPR_2023_paper.html
- MAVD arXiv: https://arxiv.org/abs/2501.07496
- MAVD GitHub: https://github.com/xjpp2016/MAVD
- MACIL-SD arXiv: https://arxiv.org/abs/2207.05500
- MSBT arXiv: https://arxiv.org/abs/2405.05130
- CLIP arXiv: https://arxiv.org/abs/2103.00020
- OpenAI CLIP page: https://openai.com/research/clip
- DANN arXiv: https://arxiv.org/abs/1505.07818
- Barlow Twins arXiv: https://arxiv.org/abs/2103.03230
- ShanghaiTech COVE summary: https://cove.thecvf.com/datasets/562
- UCF-Crime original paper: https://arxiv.org/abs/1801.04264

## 20. Immediate Next Step

Implement M0 protocol audit first. The audit should read the declared dataset feature files, print feature keys/shapes, group XD 5-crop files by video id, document frame counts and annotation schemas, and fix the score-to-frame expansion（分数到帧扩展） rule before any long training run. This is a protocol correctness gate, not a judgment about dataset availability.
