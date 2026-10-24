# Research Checkpoint — 2026-10-23

## Project
Task-Aware Low-Light Image Restoration for Robust Computer Vision

## Baseline
- Model: pretrained DarkIR-m
- Dataset: LOL-Blur test split, 1,800 paired images
- PSNR: 27.00247
- SSIM: 0.882912
- LPIPS: 0.162013
- Parameters: approximately 3.32M
- Compute: 7.25 GMac

The measured restoration metrics reproduce the reported baseline closely.

## Initial Fine-Tuning Experiments
Both variants started from the same pretrained checkpoint and used
1,000 steps, batch size 2, and 256x256 crops.

| Variant | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| Fine-tuned restoration control | 27.22182 | 0.881808 | 0.164713 |
| Task-aware prototype | 27.17017 | 0.880252 | 0.159633 |

The task-aware prototype adds frozen Faster R-CNN feature consistency
to an L1 restoration objective, with task weight 0.1.

## Preliminary Detection Evaluation
An initial zero-shot COCO-detector evaluation was run on the ExDark
validation split. Its metrics are provisional and must not be treated
as final results: the evaluator's AP matching and confidence-threshold
handling were subsequently identified as issues, and the training
feature preprocessing did not match the detector's inference transform.

## Remaining Work
1. Validate the corrected evaluation implementation and metric protocol.
2. Validate detector-matched preprocessing with a short training pilot.
3. Re-run the controlled task-aware experiment if the pilot passes.
4. Measure downstream detection performance with a validated protocol.
5. Decide whether adaptive feature modulation and further ablations are warranted.

No claim of improved object-detection performance has been established.
