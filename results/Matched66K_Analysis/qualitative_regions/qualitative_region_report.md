# Matched 66K Qualitative Region Analysis

Generated (UTC): 2026-11-16T12:35:02.744336+00:00

## Selection protocol

Regions are selected from the previously computed 64x64, stride-32 sliding-window diagnostic:
- Strongest Regional-favouring window across all tiles.
- Strongest Global-favouring window across all tiles.
- Strongest Global-favouring window within each of tiles 0, 2 and 3.

These are deliberately selected illustrative examples, not an unbiased sample. The windows overlap with the broader diagnostic grid and must not be treated as independent observations.

## Display interpretation

The first row shows ground truth and predictions averaged over all 128 bands for spatial context; it is not an RGB or wavelength-calibrated composite.
The second row shows per-pixel spectral RMSE and relative Regional-vs-Global MSE improvement. Positive improvement favours Regional SADS.
Improvement-map color limits are robustly clipped for display only; NPZ arrays retain the full values.

## Selected regions

| Case | Tile | Top | Left | Recomputed Regional MSE improvement | Regional pixel-win rate | PNG |
|---|---:|---:|---:|---:|---:|---|
| tile_0_global_favouring_window | 0 | 0 | 32 | -6.42045% | 44.04% | `tile_0_global_favouring_window_tile0_top0_left32.png` |
| strongest_regional_window | 1 | 352 | 256 | +11.75830% | 52.47% | `strongest_regional_window_tile1_top352_left256.png` |
| strongest_global_window | 1 | 256 | 416 | -10.04955% | 42.53% | `strongest_global_window_tile1_top256_left416.png` |
| tile_2_global_favouring_window | 2 | 224 | 32 | -5.08297% | 46.14% | `tile_2_global_favouring_window_tile2_top224_left32.png` |
| tile_3_global_favouring_window | 3 | 160 | 352 | -6.34005% | 45.29% | `tile_3_global_favouring_window_tile3_top160_left352.png` |

## Caution

The four test tiles are spatial regions from one Chikusei scene. These selected crops provide qualitative examples only and do not establish cross-scene generalization or statistical significance.
