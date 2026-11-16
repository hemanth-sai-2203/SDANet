# Matched 66K Global SDANet vs Regional SADS Analysis

Generated (UTC): 2026-11-16T12:14:32.817854+00:00

## Protocol

- Test data: `/home/aidslab/hemanth-ug4/datasets/Chikusei_x4_sspsr_spatial_split/Chikusei_test.mat`
- Four tiles, each 512×512×128; same ground truth for both models.
- EMA weights (`params_ema`) from the matched 66K checkpoints.
- BasicSR metric implementations/settings reproduced from the test configuration.
- Aggregate values are arithmetic means over the four equally sized tiles.

## Aggregate metrics

| Metric | Direction | Global SDANet | Regional SADS | Regional − Global |
|---|---|---:|---:|---:|
| SSIM | higher | 0.9440142758 | 0.9439837754 | -0.0000305005 |
| MPSNR | higher | 40.3146499110 | 40.3133546366 | -0.0012952744 |
| SAM | lower | 2.3110660626 | 2.3117169583 | +0.0006508957 |
| ERGAS | lower | 4.9669785293 | 4.9673637710 | +0.0003852417 |
| CC | higher | 0.9562024623 | 0.9561763108 | -0.0000261515 |
| MSE | lower | 0.0001363429 | 0.0001363964 | +0.0000000536 |
| MAE | lower | 0.0064575747 | 0.0064606824 | +0.0000031078 |

## Quadrant-wise MSE audit

- Regional lower MSE: **5/16** quadrants.
- Global lower MSE: **11/16** quadrants.
- Ties: **0/16** quadrants.
- Unweighted mean Regional MSE improvement: **-0.06473%**.
- Positive improvement means Regional MSE is lower than Global MSE.
- Pixel-win rate compares per-pixel MSE averaged over 128 spectral bands.

| Tile | Quadrant | Global MSE | Regional MSE | Regional improvement | Regional pixel-win rate |
|---:|---|---:|---:|---:|---:|
| 0 | TOP_LEFT | 0.0001792526 | 0.0001796033 | -0.19560% | 49.35% |
| 0 | TOP_RIGHT | 0.0001635165 | 0.0001634654 | +0.03130% | 51.40% |
| 0 | BOTTOM_LEFT | 0.0003293102 | 0.0003302604 | -0.28856% | 49.18% |
| 0 | BOTTOM_RIGHT | 0.0002338702 | 0.0002338784 | -0.00349% | 49.95% |
| 1 | TOP_LEFT | 0.0001375272 | 0.0001381517 | -0.45405% | 49.27% |
| 1 | TOP_RIGHT | 0.0001532493 | 0.0001522950 | +0.62267% | 49.67% |
| 1 | BOTTOM_LEFT | 0.0001172303 | 0.0001173450 | -0.09781% | 49.73% |
| 1 | BOTTOM_RIGHT | 0.0001157190 | 0.0001141500 | +1.35584% | 49.07% |
| 2 | TOP_LEFT | 0.0001294299 | 0.0001299447 | -0.39774% | 47.89% |
| 2 | TOP_RIGHT | 0.0001298458 | 0.0001301165 | -0.20849% | 49.25% |
| 2 | BOTTOM_LEFT | 0.0000478438 | 0.0000481355 | -0.60954% | 48.99% |
| 2 | BOTTOM_RIGHT | 0.0000434964 | 0.0000436705 | -0.40028% | 49.48% |
| 3 | TOP_LEFT | 0.0000803272 | 0.0000798711 | +0.56781% | 49.79% |
| 3 | TOP_RIGHT | 0.0000567643 | 0.0000572846 | -0.91658% | 49.04% |
| 3 | BOTTOM_LEFT | 0.0001200812 | 0.0001200390 | +0.03513% | 50.41% |
| 3 | BOTTOM_RIGHT | 0.0001440219 | 0.0001441318 | -0.07632% | 49.78% |

## Prediction differences

| Tile | Mean absolute difference | RMSE between predictions | Maximum absolute difference |
|---:|---:|---:|---:|
| 0 | 0.00079235 | 0.00133084 | 0.02730364 |
| 1 | 0.00052570 | 0.00099852 | 0.02750695 |
| 2 | 0.00042268 | 0.00079691 | 0.01828712 |
| 3 | 0.00040422 | 0.00076262 | 0.01737496 |

## Interpretation guardrails

- These results are descriptive for the four held-out spatial tiles from one Chikusei scene.
- They do not establish statistical significance or generalization across independent scenes.
- A lower error in selected quadrants does not establish a consistent benefit of regional attention.
- Retain the full per-tile and per-quadrant records; do not report only the aggregate means.

## Reproducibility files

- `tile_metrics.csv`: one row per tile and model.
- `tile_comparison.csv`: paired Global/Regional metrics and differences per tile.
- `quadrant_metrics.csv`: 16 spatial quadrant comparisons.
- `prediction_difference.csv`: direct prediction difference per tile.
- `summary.json`: aggregate results, checkpoint hashes, dataset details, and software versions.
