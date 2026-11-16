# Matched 66K Spatial and Spectral Diagnostics

Generated (UTC): 2026-11-16T12:19:47.704249+00:00

## Spatial error maps

- Each map compares per-pixel spectral RMSE, calculated across all 128 bands.
- The improvement map is `(Global pixel MSE - Regional pixel MSE) / Global pixel MSE × 100`.
- Positive values favor Regional SADS; negative values favor Global SDANet.
- Error-map color scales are robustly limited for visualization; underlying arrays are stored in NPZ files.

| Tile | Global mean pixel RMSE | Regional mean pixel RMSE | Regional MSE improvement | Regional pixel-win rate |
|---:|---:|---:|---:|---:|
| 0 | 0.010398377 | 0.010400625 | -0.13884% | 49.97% |
| 1 | 0.007930154 | 0.007925764 | +0.34066% | 49.44% |
| 2 | 0.006502988 | 0.006517358 | -0.35687% | 48.90% |
| 3 | 0.007079761 | 0.007081983 | -0.03288% | 49.75% |

## Spectral-band analysis

- Per-band RMSE and MAE are calculated over each 512×512 tile.
- The four-tile band summary is the arithmetic mean of the four tile-level values.
- Bands are labeled with zero-based indices 0–127 because no wavelength mapping is assumed.

## Sliding-window diagnostic

- Window size: 64×64; stride: 32 pixels.
- Total overlapping windows: 900.
- Regional lower-MSE windows: 389.
- Global lower-MSE windows: 511.
- Windows are spatially overlapping and must not be treated as independent samples.

### Five windows with largest Regional improvement

| Tile | Top | Left | Regional MSE improvement |
|---:|---:|---:|---:|
| 1 | 352 | 256 | +11.75830% |
| 1 | 352 | 224 | +8.72374% |
| 1 | 320 | 256 | +6.73811% |
| 1 | 384 | 256 | +6.24112% |
| 1 | 384 | 224 | +6.07169% |

### Five windows with largest Global improvement

| Tile | Top | Left | Regional MSE improvement |
|---:|---:|---:|---:|
| 1 | 256 | 416 | -10.04955% |
| 1 | 320 | 352 | -6.70988% |
| 0 | 0 | 32 | -6.42045% |
| 3 | 160 | 352 | -6.34005% |
| 1 | 288 | 352 | -5.82391% |

## Files

- `tile_0..3_spatial_error_maps.png`: paired error maps and Regional-vs-Global improvement.
- `tile_0..3_spatial_error_maps.npz`: unclipped per-pixel error arrays.
- `tile_band_metrics.csv`: per-tile, per-band RMSE and MAE for both models.
- `mean_band_metrics.csv`: four-tile mean per-band RMSE and MAE comparison.
- `four_tile_band_rmse_comparison.png`: aggregate band curves.
- `per_tile_band_rmse_improvement.png`: band improvement curves for each tile.
- `spatial_error_summary.csv`: per-tile spatial summary.
- `sliding_window_metrics_64x64_stride32.csv`: overlapping local-region comparisons.
- `spatial_spectral_summary.json`: inputs, checksums, definitions, and key regions.

## Interpretation cautions

- The four test tiles are spatial regions from one Chikusei scene, not independent scenes.
- The 128 bands are indexed only; this script does not assume wavelengths.
- Sliding windows overlap and are exploratory diagnostics, not independent statistical samples.
- Color-map percentile limits are for visualization; saved NPZ arrays retain unclipped values.
- These are descriptive diagnostics for four spatial tiles from one scene; they do not establish statistical significance or cross-scene generalization.
