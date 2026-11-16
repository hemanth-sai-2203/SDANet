# Matched 66K Spectral Band-Range Consistency Analysis

Generated (UTC): 2026-11-16T12:27:38.622471+00:00

## Method

- Uses the saved per-tile RMSE/MAE values for four tiles and 128 bands.
- Per-tile improvement is `(Global error - Regional error) / Global error × 100`; positive values favour Regional SADS.
- Aggregate per-band RMSE/MAE is calculated from the arithmetic mean across the four equally sized tiles.
- Tile vote counts compare each tile independently at each band.
- Band indices are zero-based (0–127); no wavelength mapping is assumed.

## Aggregate spectral consistency

- Regional has lower aggregate RMSE in 32/128 bands; Global in 96/128.
- Mean aggregate RMSE improvement across bands: -0.012580%.
- Regional has lower aggregate MAE in 9/128 bands; Global in 119/128.
- Mean aggregate MAE improvement across bands: -0.042973%.
- At least 3 of 4 tiles favour Regional RMSE in 47 bands; at least 3 favour Global in 59 bands; 22 bands are split.

## Contiguous aggregate RMSE runs (minimum length 3)

- Bands 1–3 (3 bands): REGIONAL
- Bands 6–16 (11 bands): GLOBAL
- Bands 17–29 (13 bands): REGIONAL
- Bands 30–46 (17 bands): GLOBAL
- Bands 48–57 (10 bands): GLOBAL
- Bands 58–62 (5 bands): REGIONAL
- Bands 63–118 (56 bands): GLOBAL
- Bands 119–127 (9 bands): REGIONAL

## Per-tile RMSE results across bands

| Tile | Regional RMSE wins | Global RMSE wins | Mean RMSE improvement |
|---:|---:|---:|---:|
| 0 | 48/128 | 80/128 | -0.049740% |
| 1 | 128/128 | 0/128 | +0.139672% |
| 2 | 0/128 | 128/128 | -0.246343% |
| 3 | 68/128 | 60/128 | +0.096037% |

## Exploratory focus ranges

The ranges below were selected after inspecting the earlier top/bottom aggregate-band list. They are post-hoc checks, not pre-registered findings and not independent statistical tests.

| Band range | Aggregate RMSE improvement | Regional aggregate wins | Mean tile RMSE improvement | Tile-wise RMSE improvement |
|---:|---:|---:|---:|---|
| 19–27 | +0.018559% | 9/9 | +0.016610% | T0: +0.0359% (9/9); T1: +0.0786% (9/9); T2: -0.4018% (0/9); T3: +0.3537% (9/9) |
| 80–93 | -0.040417% | 0/14 | -0.042330% | T0: -0.1298% (0/14); T1: +0.1827% (14/14); T2: -0.1437% (0/14); T3: -0.0786% (0/14) |

## Interpretation cautions

- A positive band-level improvement is descriptive, not evidence of statistical significance.
- Bands are correlated spectral channels; adjacent bands are not independent observations.
- The four tiles are regions from one Chikusei scene, not four independent scenes.
- Contiguous ranges are exploratory and must not be interpreted as physical wavelength/material ranges without verified wavelength metadata.
- The two focus ranges are post-hoc and should not be used as confirmatory evidence.

## Generated files

- `band_range_consistency.csv`: one row per band with aggregate values and all four tile-level improvements.
- `band_contiguous_runs.csv`: contiguous sign/vote runs for aggregate and tile-wise RMSE/MAE.
- `band_focus_range_summary.csv`: descriptive summaries for the two exploratory ranges.
- `band_range_consistency_summary.json`: input hashes, definitions, and summary statistics.
