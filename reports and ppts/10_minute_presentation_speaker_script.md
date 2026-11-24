# BTP Research Project — Complete Chronological Technical Handover

**Project:** Hyperspectral Image Super-Resolution (HSI-SR)  
**Coverage:** December 2025 – November 2026  
**Purpose:** Transfer the research history, technical decisions, implementations, experiments, results and pending work to Microsoft Copilot so that research can continue without repeating completed work.

---

# 1. Project overview

The central technical problem investigated in this project is Hyperspectral Image Super-Resolution (HSI-SR).

A hyperspectral image contains spatial information and a large number of spectral bands. Unlike RGB images, which contain three colour channels, an HSI records information across dozens or hundreds of wavelength bands.

The central challenge is that hyperspectral sensors often sacrifice spatial resolution to capture detailed spectral information.

HSI-SR attempts to reconstruct a high-spatial-resolution hyperspectral image from a low-resolution input while preserving spectral information.

The two major requirements are:

1. **Spatial fidelity:** Recover edges, textures, structures and fine spatial details.
2. **Spectral fidelity:** Preserve the spectral signatures of different materials and avoid spectral distortion.

The project explored multiple approaches to this problem, including conventional CNN and attention architectures, Transformer-based methods, diffusion models and regional sparse attention.

## Major research directions

The project history contains several related but distinct directions:

| Research direction | Main objective | Status in available records |
|---|---|---|
| HSI-SR literature review | Understand existing architectures and research gaps | Studied and documented |
| GEWDiff | Establish a diffusion-based HSI-SR baseline | Baseline reproduced |
| GEWDiff + CFG | Investigate classifier-free guidance | Implemented and experimentally evaluated |
| Alternative HSI-SR architectures | Investigate EUNet and other methods | Primarily research and planning |
| SDANet | Establish a Transformer/attention-based baseline | Implemented and evaluated |
| SADS | Introduce regional sparse attention into SDANet | Implemented and evaluated |
| Fine-gating and upsampling | Investigate additional architectural improvements | Experimental ablations conducted |
| Feature fusion | Investigate combinations of model outputs | Experimental comparisons conducted |
| Matched-training comparison | Separate regional-attention effects from additional training | Conducted; aggregate results do not establish a regional advantage |

These directions should not be presented as one single architecture. GEWDiff and SDANet are separate research implementations.

---

# 2. Chronological research history

## December 2025 — Initial research topics

The initial research topics documented in the project history were:

1. Self-supervised learning for hyperspectral imaging data.
2. Large-scale foundation models for spectral data.
3. Stable diffusion frameworks for video surveillance.
4. Multimodal data fusion for aerial sensing data.
5. Solving vision problems using deep learning.

These topics formed the initial research exploration.

The available records do not establish that all five became implementation objectives. The subsequent work became increasingly focused on hyperspectral image super-resolution and deep-learning-based spectral-spatial reconstruction.

**Important distinction:** These five topics should be treated as initial research topics, not five completed implementations.

## February 2026 — Diffusion-based HSI-SR research

On 17 February 2026, a major discussion focused on the paper:

**DiffHSR: Unleashing Diffusion Priors in Hyperspectral Image Super-Resolution**

The objective was to understand the entire paper rather than simply obtain a summary.

The discussion covered:
- Research motivation.
- Problem formulation.
- Diffusion-model fundamentals.
- Network architecture.
- Individual architectural components.
- Training and inference.
- Mathematical formulations.
- Experimental results.
- Advantages and limitations.

This established an understanding of how diffusion priors could be applied to hyperspectral reconstruction.

The paper was studied as research literature. The available records do not establish that its architecture was independently implemented in our repository.

## March 2026 — Architectural fundamentals

On 3 March 2026, we discussed why outputs from multiple units or branches are concatenated.

The discussion covered:
- Feature concatenation.
- Channel dimensions.
- How multiple feature representations are combined.
- Why concatenation preserves information from different branches.
- How subsequent layers process the combined representation.

This was relevant to understanding multi-branch neural architectures and spatial-spectral feature extraction.

The important distinction was between concatenation and addition:

- Addition combines corresponding feature values and generally requires compatible dimensions.
- Concatenation joins feature tensors along a selected dimension, often the channel dimension.

This discussion supported the later architectural analysis of attention-based HSI-SR networks.

## April 2026 — Research paper presentation and GEWDiff literature

On 29 April 2026, a new diffusion-related research paper was discussed. The paper was described as published at AAAI in March 2026.

The discussion focused on presenting its basic architecture and results and understanding its relationship to the ongoing research direction.

During this period, the research also involved:
- Understanding the GEWDiff paper.
- Studying the motivation for wavelet-based compression.
- Understanding the geometry-enhanced diffusion process.
- Understanding the role of diffusion in reconstructing spatial details.
- Preparing research presentations and faculty updates.

On 1 May 2026, additional technical explanations covered:
- Namespace in HDFS.
- MapReduce and HDFS layers.
- System architecture diagrams.
- Data-flow and architectural concepts.

These were separate technical learning discussions and should not be represented as HSI-SR implementations.

## May 2026 — GEWDiff technical understanding and report preparation

May involved substantial work on understanding the diffusion architecture and preparing technical documentation.

Important discussions included:

### 1. Model distillation

We examined why model distillation is useful.

The important distinction was that distillation is not limited to easy deployment. It can also reduce:
- Model complexity.
- Inference time.
- Memory consumption.
- Sampling or computation requirements.

For diffusion models, distillation can potentially reduce the number of reverse-sampling steps while attempting to preserve reconstruction quality.

### 2. EDM and NDVI masking

We discussed:
- EDM sampling.
- The meaning of mathematical terms in the diffusion equations.
- The role of noise and variance.
- How masks affect image reconstruction.
- The distinction between NDVI-related masks and diffusion sampling mechanisms.

A specific equation involving a normalized variance term and noise was examined in the context of preserving sharp contours during reverse sampling.

We also discussed whether EDM itself provides an NDVI masking mechanism. The important distinction was that a mask-based mechanism should not automatically be attributed to the underlying EDM formulation.

### 3. Technical report and presentation

On 9 May 2026, we worked on report preparation and presentation material.

This included:
- Comparing research methods.
- Preparing figures.
- Creating LaTeX report content.
- Identifying missing illustrations.
- Preparing comparative performance visualizations.
- Reviewing the report and presentation for faculty evaluation.

A key requirement was to ensure that figure descriptions and numerical comparisons were supported by the actual tables rather than invented values.

## July 2026 — GEWDiff evaluation and future research directions

An evaluation report titled `HSI_SR_End_Evaluation_Report.pdf` was created on 31 July 2026.

This document reviewed GEWDiff and discussed its broader research implications.

The documented future directions included:

### 1. Hyperspectral perceptual loss

A limitation identified was the use of conventional perceptual features that may not represent hyperspectral spectral relationships adequately.

A possible direction was to investigate an HSI-aware perceptual feature extractor.

### 2. Multi-scale mask conditioning

This was proposed as a possible way to provide more spatially informed conditioning.

### 3. Model distillation

The report discussed the computational cost of GEWDiff.

The documented estimates were:
- Model/checkpoint size: approximately 4.55 GB.
- Sampling steps: 50.
- Inference time: approximately 28.7 seconds per image.

A possible future direction was to reduce the sampling steps to approximately 4–8 using consistency or progressive distillation.

These are reported estimates and proposed future work, not evidence that distillation was implemented.

### 4. Other future research directions

The report also discussed:
- Hyperspectral foundation models.
- Satellite-to-airborne resolution enhancement.
- Uncertainty quantification.
- Video hyperspectral super-resolution.

These were future research opportunities rather than completed experiments.

---

# 3. August 2026 — GEWDiff baseline and research implementation

August represents an important documented implementation phase.

The project was organized around the GEWDiff repository.

## 3.1 Original repository

Original research implementation:

`https://github.com/zhu-xlab/GEWDiff`

Our project repository:

`https://github.com/hemanth-sai-2203/GEWDiff-BTP`

Local project path:

`/home/aidslab/hemanth-ug4/GEWDiff-BTP`

The project was intended to follow a controlled research workflow:

1. Understand the original GEWDiff architecture.
2. Reproduce the original baseline.
3. Establish reliable evaluation metrics.
4. Identify limitations.
5. Introduce a research modification.
6. Train and evaluate the modified model.
7. Compare the modified model against the baseline.
8. Analyze the results.
9. Document the contribution.

## 3.2 Repository organization

The intended repository structure included:

- `src/GEWDiff/` — original model implementation.
- `experiments/` — notebooks and experiments.
- `docs/` — project notes, results, changes and research decisions.
- `results/` — experiment outputs.

Documentation files were planned for:
- `CHANGES.md`
- `EXPERIMENTS.md`
- `IDEAS.md`
- `LOGBOOK.md`
- `PAPER_NOTES.md`
- `RESULTS.md`
- `TODO.md`

## 3.3 Git repository issue

An early issue occurred when the original GEWDiff repository was cloned inside our repository as an embedded Git repository.

Git reported:

`warning: adding embedded git repository: src/GEWDiff`

The problem was that the nested repository was being tracked as an embedded repository rather than as ordinary source files.

The nested Git tracking was removed from the parent index using:

`git rm --cached -r src/GEWDiff`

The intention was to preserve the actual GEWDiff source files inside our repository.

## 3.4 Kaggle and WDC baseline

The initial major milestone was reproducing the original GEWDiff pipeline on the WDC dataset.

The purpose was to establish a baseline before introducing any modification.

An early handover record reports the following baseline:

| Metric | Baseline value |
|---|---:|
| MPSNR | 33.7827 dB |
| MSSIM | 0.6920 |
| SAM | 9.1510° |
| CC | 0.6275 |
| RMSE | 0.05695 |
| FID | 47.8211 |

Additional recorded variance values:

| Measurement | Value |
|---|---:|
| Predicted local variance | 0.005330 |
| True local variance | 0.007440 |

The handover documented that:
- All 50 diffusion sampling steps completed.
- PCA/RWA reconstruction executed.
- Quantitative evaluation completed.

These values belong to the early baseline record. They must not be silently merged with later evaluation results.

## 3.5 GEWDiff architecture

The original GEWDiff pipeline was understood as:

1. Input low-resolution HSI.
2. Wavelet/RWA processing.
3. PCA spectral compression.
4. Conditional diffusion.
5. 3D U-Net noise prediction.
6. EDM sampling.
7. Inverse PCA.
8. Inverse wavelet/RWA reconstruction.
9. High-resolution HSI output.

The purpose of the compression stages is to make diffusion processing more computationally manageable while retaining important spectral-spatial information.

### Architecture configuration recorded later

| Parameter | Value |
|---|---:|
| U-Net layers per block | 4 |
| Channel progression | 128 → 128 → 256 → 256 → 512 → 512 |
| Approximate parameters | 376.85 million |
| Diffusion training timesteps | 500 |
| Sampling steps | 50 |
| Sigma minimum | 0.002 |
| Sigma maximum | 80 |
| Sigma data | 0.5 |
| Rho | 7 |

The architecture used 3D operations, cross-attention and upsampling/downsampling components.

---

# 4. September 2026 — GEWDiff with Classifier-Free Guidance

September was the main documented CFG implementation and evaluation phase.

## 4.1 Research motivation

The original GEWDiff implementation was conditional.

It used the LR hyperspectral image as conditioning information.

The research hypothesis was that allowing the model to learn both conditional and unconditional behaviour could provide additional control over the generation process.

The goal was to investigate whether classifier-free guidance could improve reconstruction quality.

This was a modification to the original GEWDiff pipeline, not a replacement of the complete architecture.

## 4.2 CFG training modification

During fine-tuning, conditioning was dropped for approximately 10% of training samples.

The configuration was conceptually:

| Training sample proportion | Conditioning |
|---|---|
| Approximately 90% | Normal LR-HSI conditioning |
| Approximately 10% | Conditioning removed |

This enabled the model to learn both conditional and unconditional noise predictions.

## 4.3 CFG inference

During inference, two predictions are obtained:

- Conditional noise prediction.
- Unconditional noise prediction.

They are combined using:

\[
\epsilon_{\mathrm{guided}}
=
\epsilon_{\mathrm{uncond}}
+
s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}})
\]

Where:

- \(\epsilon_{\mathrm{guided}}\): guided noise prediction.
- \(\epsilon_{\mathrm{uncond}}\): unconditional noise prediction.
- \(\epsilon_{\mathrm{cond}}\): conditional noise prediction.
- \(s\): guidance scale.

The guidance scale controls how strongly the conditional prediction influences the guided result.

A guidance scale of 1 corresponds to the conditional prediction under this formulation.

Higher values increase the influence of the difference between conditional and unconditional predictions. Excessive guidance can introduce artifacts or distortions.

## 4.4 Guidance schedule

A dynamic guidance schedule was investigated.

One documented configuration used approximately:

\[
s: 1.0 \rightarrow 1.5
\]

over 50 sampling steps.

This was intended to vary the guidance strength throughout reverse sampling rather than apply one fixed value at every step.

## 4.5 Implementation files

The recorded source files included:

- `src/GEWDiff/cfg_core.py`
- `src/GEWDiff/cfg_wdc_test.py`
- `src/GEWDiff/kaggle_cfg_finetune.py`
- `src/GEWDiff/kaggle_hf_stream.py`
- `src/GEWDiff/model/edm.py`
- `src/GEWDiff/model/unet3d.py`

Project branch:

`cfg-integration`

Project path:

`/home/aidslab/hemanth-ug4/GEWDiff-BTP`

Original checkpoint:

`checkpoints/epoch_200.pth`

Later CFG checkpoint:

`results/cfg_finetune_fp32/cfg_step_0037300.pth`

## 4.6 CFG training progress

The documented training checkpoints included:
- 28,000 steps.
- 31,000 steps.
- 37,300 steps.

The training was manually interrupted at approximately 37,302 steps.

The last documented valid checkpoint was:

`cfg_step_0037300.pth`

The handover reported finite losses and no observed NaN/Inf values.

This does not establish that every planned training or evaluation experiment was completed.

## 4.7 Baseline versus CFG results

A later seeded evaluation record reported the following baseline:

| Metric | Baseline |
|---|---:|
| MPSNR | 34.35384563140723 |
| MSSIM | 0.7028253628790538 |
| SAM | 8.960673° |
| CC | 0.6856752 |
| RMSE | 0.05543631025308507 |
| FID | 110.380661 |
| Predicted local variance | 0.005810253 |
| True local variance | 0.007439957 |

The CFG checkpoint at 28,000 steps reported:

| Metric | CFG at 28K |
|---|---:|
| MPSNR | 35.32744838140157 |
| MSSIM | 0.7516379594670375 |
| SAM | 7.987999° |
| CC | 0.75597554 |
| RMSE | 0.04844052893333606 |
| FID | 272.73706 |
| Predicted local variance | 0.00216859 |

### Interpretation

Compared with the later baseline evaluation, the 28K CFG result showed:

- Higher MPSNR.
- Higher MSSIM.
- Lower SAM.
- Higher CC.
- Lower RMSE.
- Much higher FID.

Therefore, CFG produced a mixed result.

It improved several reconstruction metrics, but the FID result deteriorated substantially.

It would be incorrect to conclude that CFG improved every aspect of reconstruction quality.

### Important evaluation qualification

The early baseline values and later seeded baseline values differ.

For example:

- Early baseline MPSNR: 33.7827.
- Later seeded baseline MPSNR: 34.35384563140723.

These must be treated as separate evaluation records until their configurations and evaluation protocols are verified.

The paper-reported values recorded in the handover were approximately:

| Metric | Paper reference |
|---|---:|
| MPSNR | 35.837 |
| MSSIM | 0.7747 |
| SAM | 7.4735° |
| CC | 0.7906 |
| RMSE | 0.0468 |
| FID | 238.12 |

The evaluation protocol was not established as identical to our implementation. Therefore, these values should not be used to claim direct superiority or inferiority without verifying comparability.

## 4.8 Engineering difficulties

The CFG implementation involved several debugging and environment issues:

- Missing `einops` dependency.
- Python import problems.
- PyTorch checkpoint-loading compatibility.
- GPU detection.
- Diffusers UNet3D compatibility.
- Hyperspectral band mismatches.
- Checkpoint size and GPU memory limitations.
- Kaggle T4 memory constraints.
- Session limitations.
- Adapting evaluation scripts from their original hardcoded paths.

A dataset band mismatch was also documented, with some data containing 191 bands and other data containing 242 bands.

The model configuration recorded 41 input channels and 20 output channels for the relevant compressed representation.

These dataset and channel details must be checked against the actual experiment configuration before inclusion in a final academic methodology.

---

# 5. October 2026 — Transition to SDANet and SADS research

The project history contains a substantial SDANet research phase in October.

The repository was:

`https://github.com/hemanth-sai-2203/SDANet`

Local path:

`~/hemanth-ug4/SDANet_repo`

Research branch:

`research-sads`

The objective was to investigate regional sparse attention within SDANet.

## 5.1 Original SDANet baseline

The original architecture was preserved in:

`basicsr/archs/SDAnet_arch.py`

Recorded parameter count:

\[
3,086,272
\]

The baseline architecture was intentionally kept separate from the SADS implementation.

This was important because the baseline had to remain a stable reference for controlled comparisons.

## 5.2 SADS research motivation

SADS introduced regional sparsity into the DSCA attention mechanism.

The research idea was to introduce spatially adaptive attention rather than rely exclusively on a globally applied sparsity mechanism.

The intended benefit was to allow different spatial regions to use different attention selections.

This could potentially improve the modelling of local spatial-spectral relationships.

## 5.3 SADS implementation

New implementation:

`basicsr/archs/SDAnet_sads_arch.py`

Model class:

`SDANET_SADS`

Main components:

- `RegionalDSCA`
- `SDABlockSADS`
- `SDAStageSADS`

The documented implementation:
- Computes Q/K/V globally.
- Calculates global channel-attention scores.
- Applies region-specific sparsity masks.
- Uses region-sliced values.
- Combines the regional outputs.

Spatial partition:

\[
2 \times 2
\]

The parameter count remained:

\[
3,086,272
\]

This matched the original SDANet parameter count.

## 5.4 Initial verification

The following checks were documented as successful:

- Output shapes were correct.
- Model outputs were finite.
- Gradients were finite.
- Baseline checkpoint loading succeeded.
- No missing keys were reported.
- No unexpected keys were reported.

This established that the modified implementation was structurally compatible with the baseline checkpoint.

It did not, by itself, establish that the regional-attention method improved performance.

## 5.5 Training configuration

Full training configuration:

`options/train/HSI/SDANetRegionalFullChux4.yml`

Test configuration:

`options/test/HSI/test_SDANetRegionalFullChux4.yml`

Pilot configurations:

- `options/train/HSI/SDANetRegionalPilotChux4.yml`
- `options/test/HSI/test_SDANetRegionalPilotChux4.yml`

Baseline configuration:

`options/train/HSI/SDAnetChux4.yml`

The full SADS experiment was:

- Run name: `SDANetRegionalFull_spatial_split`
- Initialization: original baseline checkpoint at 33K.
- Training iterations: 33,000.
- Training time: approximately 1 hour 50 minutes.
- Final checkpoint: `net_g_33000.pth`.

The model was evaluated on Chikusei ×4 using a spatial-split dataset configuration.

## 5.6 Full SADS results

### Evaluation set

| Metric | Baseline, 33K | Full SADS |
|---|---:|---:|
| SSIM | 0.9461 | 0.9485 |
| MPSNR | 41.3063 | 41.5480 |
| SAM | 2.2810 | 2.2082 |
| ERGAS | 4.6320 | 4.4802 |
| CC | 0.9433 | 0.9462 |

Important qualifications:
- SADS SSIM value 0.9485 was recorded at iteration 30K.
- MPSNR and CC were recorded at iteration 33K.
- SAM and ERGAS were taken as the actual final values.
- The training log's best-value tracking for SAM and ERGAS was incorrect because their direction was configured as higher-is-better.

The values above should be used instead of the incorrectly tracked best entries.

### Held-out test set

| Metric | Baseline | Full SADS |
|---|---:|---:|
| SSIM | 0.9409 | 0.9431 |
| MPSNR | 40.0314 | 40.2252 |
| SAM | 2.4037 | 2.3355 |
| ERGAS | 5.1656 | 5.0292 |
| CC | 0.9532 | 0.9553 |

### Difference: SADS minus baseline

| Metric | Difference | Interpretation |
|---|---:|---|
| SSIM | +0.0022 | Improvement |
| MPSNR | +0.1938 dB | Improvement |
| SAM | −0.0682 | Improvement |
| ERGAS | −0.1364 | Improvement |
| CC | +0.0021 | Improvement |

The full SADS run improved all five reported aggregate held-out metrics.

However, this was one full SADS training run evaluated on four held-out scenes. It did not establish statistical robustness or consistent gains on every individual scene.

## 5.7 Pilot experiment

A 500-iteration regional pilot was also documented.

| Metric | Pilot evaluation |
|---|---:|
| SSIM | 0.9461 |
| MPSNR | 41.3114 |
| SAM | 2.2811 |
| ERGAS | 4.6285 |
| CC | 0.9434 |

Held-out pilot results:

| Metric | Pilot held-out test |
|---|---:|
| SSIM | 0.9409 |
| MPSNR | 40.0377 |
| SAM | 2.4025 |
| ERGAS | 5.1631 |
| CC | 0.9533 |

The pilot was essentially tied with the baseline.

It should not be confused with the full 33K SADS result.

## 5.8 Comparison with paper-reported results

The paper values recorded in the handover were:

| Metric | Paper |
|---|---:|
| MPSNR | 40.5060 |
| SSIM | 0.9472 |
| SAM | 2.2606 |
| CC | 0.9580 |
| ERGAS | 4.8567 |

Our full SADS held-out results were:

| Metric | SADS |
|---|---:|
| MPSNR | 40.2252 |
| SSIM | 0.9431 |
| SAM | 2.3355 |
| CC | 0.9553 |
| ERGAS | 5.0292 |

Our results were not better than these paper-reported values.

Additionally, evaluation protocols and dataset splits may differ. A direct performance claim should not be made without verifying equivalence.

## 5.9 Git commit

The documented SADS implementation was committed and pushed.

- Branch: `research-sads`
- Commit: `487e327`
- Commit message: `Add regional attention SADS experiment`

The commit included:

- `basicsr/archs/SDAnet_sads_arch.py`
- `options/test/HSI/test_SDANetRegionalFullChux4.yml`
- `options/train/HSI/SDANetRegionalFullChux4.yml`

The handover also documented uncommitted local changes and result directories. Their current status must be checked before any Git operation.

---

# 6. November 2026 — SDANet ablations and further research

November continued the SDANet investigation.

The important shift was from simply showing that SADS improved over the original baseline to determining whether the improvement was genuinely caused by regional sparsity.

This required more controlled experiments.

## 6.1 Matched-training comparison

A concern with the initial SADS comparison was that differences in training duration or optimization could influence the observed improvement.

A matched global-attention control was therefore introduced.

The purpose was to compare:
- Global SDANet.
- Regional SADS.

Under a more closely matched training schedule.

The documented global configuration:

`SDANetGlobalMatchedContinue66KChux4.yml`

The documented regional continuation configuration:

`SDANetRegionalContinue66K_spatial_split`

Both were initialized from the original baseline checkpoint at 33K and used 66K additional iterations.

Important settings included:
- Chikusei ×4.
- Spatial-split training.
- Batch size 16.
- Patch size 128.
- Adam optimizer.
- Initial learning rate \(10^{-4}\).
- L1 reconstruction loss.
- SAM loss weight 0.2.
- EMA decay 0.999.
- Cosine annealing with two 33K cycles.

This was designed to isolate the contribution of the attention mechanism more reliably.

## 6.2 Matched 66K comparison conclusion

The available project history records that the matched global and regional models produced nearly identical aggregate results, with the global model marginally better across the aggregate metrics.

Therefore:

**The matched comparison did not demonstrate a consistent overall advantage for regional SADS over global SDANet.**

This is an important result and must not be hidden.

It means the earlier 33K SADS improvement cannot automatically be attributed to regional sparsity itself.

The additional training and optimization history may have contributed to the observed difference.

The exact matched-66K metric values and per-scene results must be extracted from the experiment output files before being inserted into the final numerical results table. They are not reproduced here because the exact values are not established in the available summary.

## 6.3 Fine-gating ablation

A further investigation examined fine-gating modifications.

The objective was to determine whether the gating mechanism could improve feature selection or reconstruction quality.

The documented outcome was:

- Differences between control and treatment were very small.
- Results were mixed across metrics.
- No robust overall benefit was established.

Therefore, fine-gating should not be presented as a demonstrated performance improvement.

## 6.4 Independent upsampler

An independent upsampler was investigated as an additional reconstruction component.

The documented outcome was that the independent upsampler was weaker when evaluated alone.

This suggests that its use should not be justified as an independently superior reconstruction architecture based on the available results.

## 6.5 Feature-fusion experiments

Different fusion configurations were investigated.

The recorded findings were:

- Fixed fusion weight \(\alpha=0.8\) was the best aggregate configuration among the compared variants.
- A compact dynamic gate produced only very small gains.
- Dynamic gating increased the model footprint.

The important interpretation is that fixed fusion was the stronger practical choice among the tested variants.

The dynamic gate did not establish a sufficiently strong benefit to justify claiming it as a major improvement.

The exact per-variant metrics, parameter counts and inference measurements should be recovered from the corresponding result files before publication.

## 6.6 Evaluation limitations

Several methodological limitations were identified during the SDANet experiments:

1. The matched comparison produced nearly identical global and regional results.
2. Aggregate metrics can hide scene-level variation.
3. Some experiments used four tiles from one Chikusei scene rather than four independent scenes.
4. Test data had been inspected during development, so it should not be described as a completely pristine blind test.
5. A single training run is insufficient to establish statistical robustness.
6. Results from different checkpoints or training schedules must not be mixed.

These limitations should be explicitly addressed in the final research report.

---

# 7. Consolidated technical implementation inventory

## 7.1 GEWDiff

| Item | Recorded information |
|---|---|
| Repository | `hemanth-sai-2203/GEWDiff-BTP` |
| Local path | `/home/aidslab/hemanth-ug4/GEWDiff-BTP` |
| Main branch during CFG | `cfg-integration` |
| Original source | `src/GEWDiff/` |
| Original checkpoint | `checkpoints/epoch_200.pth` |
| CFG checkpoint | `cfg_step_0037300.pth` |
| Research modification | Classifier-Free Guidance |
| Main dataset discussed | WDC |
| Main metrics | MPSNR, MSSIM, SAM, CC, RMSE, FID |

## 7.2 SDANet

| Item | Recorded information |
|---|---|
| Repository | `hemanth-sai-2203/SDANet` |
| Local path | `~/hemanth-ug4/SDANet_repo` |
| Branch | `research-sads` |
| Baseline architecture | `basicsr/archs/SDAnet_arch.py` |
| SADS architecture | `basicsr/archs/SDAnet_sads_arch.py` |
| SADS class | `SDANET_SADS` |
| Parameter count | 3,086,272 |
| Spatial partition | 2×2 |
| Dataset | Chikusei ×4 |
| Framework | BasicSR-based implementation |
| GPU recorded | NVIDIA RTX 4000 Ada |
| PyTorch recorded | 2.11.0+cu128 |
| Main metrics | SSIM, MPSNR, SAM, ERGAS, CC |

---

# 8. Important research conclusions

The following conclusions are supported by the available project records.

### GEWDiff and CFG

- The original GEWDiff baseline was reproduced.
- CFG was integrated through conditioning dropout and guided noise prediction.
- The 28K CFG evaluation improved several reconstruction metrics.
- FID became substantially worse.
- CFG therefore produced a mixed result rather than a universal improvement.
- The 37.3K checkpoint was documented as the last valid checkpoint before manual interruption.

### SDANet and SADS

- The original SDANet baseline was preserved separately.
- Regional SADS was implemented with matching parameter count.
- Structural and gradient smoke tests passed.
- The full 33K SADS model improved the recorded aggregate held-out metrics.
- The matched 66K global-versus-regional comparison did not establish a regional advantage.
- Fine-gating did not demonstrate robust improvement.
- The independent upsampler was weaker alone.
- Fixed fusion with \(\alpha=0.8\) was the best aggregate fusion configuration among those compared.
- Dynamic gating provided only very small gains while increasing model footprint.

### Overall interpretation

The project has produced multiple implemented research directions and several useful negative or mixed experimental findings.

The strongest academic reporting approach is to present:
- The original baseline.
- The proposed modification.
- The exact implementation.
- The controlled experimental setup.
- The complete results.
- The limitations.
- The conclusions supported by the evidence.

It would be scientifically incorrect to claim that every proposed modification improved the model.

---

# 9. Items that must not be confused

The following distinctions are particularly important for Copilot.

| Information | Correct interpretation |
|---|---|
| GEWDiff paper results | Original publication results |
| GEWDiff early baseline | Our early WDC baseline evaluation |
| GEWDiff later baseline | A separate seeded evaluation |
| CFG 28K | Our modified GEWDiff evaluation |
| SDANet baseline | Original attention-based reference model |
| SADS pilot | 500-iteration pilot |
| SADS full | Full 33K regional model |
| SADS matched 66K | Controlled continuation comparison |
| Fine-gating | Separate ablation |
| Independent upsampler | Separate component experiment |
| Fixed fusion \(\alpha=0.8\) | Best aggregate tested fusion configuration |
| Dynamic gate | Small gains with additional model footprint |
| EUNet/DASCC | Research direction and proposal, not confirmed completed implementation |

---

# 10. Current remaining SDANet work

The immediate continuation should focus on completing and verifying the SDANet research record rather than restarting the architecture or repeating completed experiments.

## Phase 1 — Repository and result freeze

Before changing code:

1. Verify the active branch.
2. Inspect the working tree.
3. Identify uncommitted modifications.
4. Verify all relevant checkpoints.
5. Verify the baseline and SADS evaluation outputs.
6. Preserve all existing result directories.
7. Confirm that the original baseline architecture remains unchanged.

Do not run `git add .` blindly.

Do not delete existing outputs or checkpoints.

## Phase 2 — Complete the matched 66K analysis

Recover the exact results for:
- Global SDANet matched 66K.
- Regional SADS matched 66K.

Extract:
- Aggregate metrics.
- Per-scene metrics.
- Per-quadrant metrics, if generated.
- Prediction comparisons.
- Checkpoint identities.
- Evaluation configuration.
- Training configuration.

The existing analysis script is:

`create_matched66k_analysis.py`

Its documented purpose is to reproduce the matched 66K comparison from existing checkpoints, test MAT data and saved predictions.

Output directory:

`results/Matched66K_Analysis/`

The script reads existing results and writes analysis artifacts. It should not require retraining.

## Phase 3 — Complete per-scene evaluation

The initial 33K SADS aggregate improvement must be examined at the scene level.

Determine:
- Which scenes improved.
- Which scenes deteriorated.
- Whether the gains are consistent.
- Whether one scene dominates the aggregate result.
- Whether spectral fidelity and spatial fidelity move together.

Do not claim consistent generalization without this analysis.

## Phase 4 — Finalize ablation analysis

Prepare one consolidated table containing:

- Baseline SDANet.
- Full SADS.
- Matched global 66K.
- Matched regional 66K.
- Fine-gating control.
- Fine-gating treatment.
- Independent upsampler.
- Fixed fusion variants.
- Dynamic fusion gate.

For every variant, include:
- Parameter count.
- Training schedule.
- Checkpoint.
- Dataset.
- Evaluation metrics.
- Inference cost, if available.
- Conclusion.

Use `Not reported` when information is unavailable.

## Phase 5 — Reproducibility and final checks

Verify:
- Baseline checkpoint loading.
- SADS checkpoint loading.
- Exact evaluation configurations.
- Dataset split.
- Metric direction.
- Metric implementation.
- Random seeds.
- Checkpoint selection.
- Output normalization.
- Result aggregation.

In particular, SAM and ERGAS must be treated as lower-is-better metrics.

## Phase 6 — Final research interpretation

After the above checks, decide which contribution can legitimately be claimed.

The current evidence supports saying that:

- SADS was successfully implemented.
- Its initial full 33K experiment improved aggregate held-out metrics over the recorded baseline.
- The matched 66K comparison did not demonstrate an overall regional-attention advantage.
- Additional gating and fusion experiments produced mixed or limited benefits.

Do not claim regional attention is conclusively superior until the matched and per-scene analysis supports that statement.

---

# 11. Source files and evidence inventory

The following saved records were used to prepare this handover:

| Source | Main information |
|---|---|
| `Pasted text(1).txt`, August 2026 | GEWDiff initial project organization, Git history and baseline work |
| `Pasted markdown (2).md`, August 2026 | Extended GEWDiff handover |
| `Pasted markdown(20260921-061921).md` | GEWDiff CFG architecture, training and results |
| `HSI_SR_End_Evaluation_Report.pdf` | GEWDiff evaluation and future research directions |
| `Pasted text(20261023-045027).txt` | SDANet/SADS implementation, results and Git handover |
| `Pasted text(20261023-050252).txt` | Regional attention design and ablation planning |
| `SDANetGlobalMatchedContinue66KChux4.yml` | Matched global training configuration |
| `test_SDANetRegionalMatched66KChux4.yml` | Matched regional testing configuration |
| `create_matched66k_analysis.py` | Matched 66K analysis script |
| `Pasted text(20261116-074238).txt` | November SDANet continuation configuration and work |

Additional research documents included:
- `HSI-SR Research Project Plan.md`
- `Advanced HSI-SR Baseline Search for High-End Workstation.md`
- `37978-Article Text-42070-1-2-20260314.pdf`

Some of these are planning documents or literature sources. They should not automatically be treated as evidence of completed experiments.

---

# 12. Instructions for Microsoft Copilot

Copilot should use this document as a project handover, not as unquestionable ground truth.

The following rules must be followed:

1. Do not invent missing files, results, metrics, checkpoints or implementation details.
2. Inspect the actual SDANet repository before proposing code modifications.
3. Preserve the original baseline architecture.
4. Do not repeat experiments already completed unless there is a clear reproducibility reason.
5. Do not train another model before checking whether the required checkpoints and predictions already exist.
6. Do not combine results from different datasets, checkpoints or evaluation protocols.
7. Treat negative and inconclusive results as valid research findings.
8. Keep GEWDiff and SDANet as separate implementation tracks.
9. Do not treat proposed architectures such as EUNet or DASCC as completed work.
10. Give exact commands, file paths and code changes when implementation is required.
11. Work on one phase at a time.
12. Verify all numerical values against original result files before preparing the final academic report.

## Immediate task for Copilot

Begin with the SDANet repository and perform a read-only audit.

Inspect:
- Current branch.
- Git status.
- Latest commits.
- Architecture files.
- Training and testing configurations.
- Checkpoint inventory.
- Existing results.
- Matched 66K analysis outputs.
- Fine-gating and fusion experiment outputs.

Do not modify code or start training during this initial audit.

Produce a concise inventory of what exists, what is verified, and what remains incomplete.

Then continue with the matched 66K and per-scene result analysis.

---

# 13. Final handover summary

The project progressed from broad research exploration and paper understanding into practical HSI-SR implementation and controlled experimentation.

The GEWDiff direction established a diffusion-based baseline and investigated classifier-free guidance. CFG improved several reconstruction metrics in one documented evaluation but substantially worsened FID, producing a mixed result.

The SDANet direction established a separate attention-based baseline and introduced SADS, a regional sparse-attention modification. Its full 33K experiment improved the aggregate held-out metrics, but the subsequent matched 66K comparison did not establish a clear advantage for regional attention over global attention.

Further fine-gating, upsampling and fusion investigations produced mixed or limited results.

The immediate priority is therefore not to invent another architecture. It is to complete the SDANet evidence, verify the matched comparisons, finish the per-scene analysis, consolidate the ablations and establish the strongest scientifically defensible conclusion.

**The next research phase should begin with a read-only SDANet repository and experiment audit, followed by completion of the matched 66K and per-scene analyses.**
