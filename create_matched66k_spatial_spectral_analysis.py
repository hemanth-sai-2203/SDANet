#!/usr/bin/env python3
"""Spatial and spectral diagnostic analysis for matched 66K SDANet models.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python create_matched66k_spatial_spectral_analysis.py

Reads the existing Chikusei test GT and saved Global/Regional MAT predictions.
Writes diagnostics to results/Matched66K_Analysis/spatial_spectral/.
Does not modify checkpoints, predictions, configs, or training artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "Matched66K_Analysis" / "spatial_spectral"
GT_PATH = Path(
    "/home/aidslab/hemanth-ug4/datasets/"
    "Chikusei_x4_sspsr_spatial_split/Chikusei_test.mat"
)

MODELS = {
    "GLOBAL": {
        "directory": RESULTS_ROOT / "SDANetGlobalMatched66K_Chikusei_x4_test/visualization/Chikusei_test",
        "suffix": "SDANetGlobalMatched66K_Chikusei_x4_test",
        "label": "Global SDANet",
    },
    "REGIONAL": {
        "directory": RESULTS_ROOT / "SDANetRegionalMatched66K_Chikusei_x4_test/visualization/Chikusei_test",
        "suffix": "SDANetRegionalMatched66K_Chikusei_x4_test",
        "label": "Regional SADS",
    },
}

TILE_COUNT = 4
HEIGHT = WIDTH = 512
BANDS = 128
WINDOW_SIZE = 64
WINDOW_STRIDE = 32


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prediction_path(model_key: str, tile: int) -> Path:
    model = MODELS[model_key]
    return model["directory"] / f"test_img_{tile}_{model['suffix']}.mat"


def load_prediction(model_key: str, tile: int) -> tuple[np.ndarray, Path]:
    path = prediction_path(model_key, tile)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model_key} tile {tile}: {path}")
    loaded = loadmat(path, variable_names=["sr_data"])
    if "sr_data" not in loaded:
        raise KeyError(f"Missing sr_data in {path}")
    arr = np.asarray(loaded["sr_data"])
    if arr.shape != (HEIGHT, WIDTH, BANDS):
        raise ValueError(f"Unexpected prediction shape in {path}: {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite prediction values in {path}")
    return arr, path


def save_tile_error_map(
    tile: int,
    global_pixel_mse: np.ndarray,
    regional_pixel_mse: np.ndarray,
) -> tuple[Path, dict[str, float]]:
    global_rmse = np.sqrt(global_pixel_mse)
    regional_rmse = np.sqrt(regional_pixel_mse)
    eps = 1e-15
    improvement_pct = (
        (global_pixel_mse - regional_pixel_mse)
        / np.maximum(global_pixel_mse, eps)
        * 100.0
    )

    # Persist the underlying arrays as well as the visual summary.
    npz_path = OUTPUT_DIR / f"tile_{tile}_spatial_error_maps.npz"
    np.savez_compressed(
        npz_path,
        global_pixel_rmse=global_rmse.astype(np.float32),
        regional_pixel_rmse=regional_rmse.astype(np.float32),
        regional_mse_improvement_pct=improvement_pct.astype(np.float32),
    )

    # Comparable error-map scale within a tile; robust limits reduce the
    # influence of a very small number of extreme pixels.
    error_limit = float(np.percentile(np.concatenate([
        global_rmse.ravel(), regional_rmse.ravel()
    ]), 99.0))
    delta_limit = float(np.percentile(np.abs(improvement_pct), 98.0))
    if not np.isfinite(delta_limit) or delta_limit <= 0:
        delta_limit = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), constrained_layout=True)
    im0 = axes[0].imshow(global_rmse, cmap="magma", vmin=0, vmax=error_limit)
    axes[0].set_title("Global SDANet: per-pixel spectral RMSE")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="RMSE")

    im1 = axes[1].imshow(regional_rmse, cmap="magma", vmin=0, vmax=error_limit)
    axes[1].set_title("Regional SADS: per-pixel spectral RMSE")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="RMSE")

    im2 = axes[2].imshow(
        improvement_pct, cmap="RdBu", vmin=-delta_limit, vmax=delta_limit
    )
    axes[2].set_title("Regional MSE improvement over Global")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")
    fig.colorbar(
        im2, ax=axes[2], fraction=0.046, pad=0.04,
        label="Improvement (%) — positive favors Regional"
    )
    fig.suptitle(
        f"Chikusei test tile {tile}: spatial reconstruction-error comparison",
        fontsize=13,
    )
    png_path = OUTPUT_DIR / f"tile_{tile}_spatial_error_maps.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    stats = {
        "tile": tile,
        "global_mean_pixel_rmse": float(np.mean(global_rmse)),
        "regional_mean_pixel_rmse": float(np.mean(regional_rmse)),
        "global_pixel_mse": float(np.mean(global_pixel_mse)),
        "regional_pixel_mse": float(np.mean(regional_pixel_mse)),
        "regional_mse_improvement_pct": float(
            (np.mean(global_pixel_mse) - np.mean(regional_pixel_mse))
            / np.mean(global_pixel_mse) * 100.0
        ),
        "regional_pixel_win_rate_pct": float(
            np.mean(regional_pixel_mse < global_pixel_mse) * 100.0
        ),
        "improvement_map_p02_pct": float(np.percentile(improvement_pct, 2)),
        "improvement_map_median_pct": float(np.median(improvement_pct)),
        "improvement_map_p98_pct": float(np.percentile(improvement_pct, 98)),
    }
    return png_path, stats


def sliding_window_rows(
    tile: int, global_pixel_mse: np.ndarray, regional_pixel_mse: np.ndarray
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for top in range(0, HEIGHT - WINDOW_SIZE + 1, WINDOW_STRIDE):
        for left in range(0, WIDTH - WINDOW_SIZE + 1, WINDOW_STRIDE):
            ys = slice(top, top + WINDOW_SIZE)
            xs = slice(left, left + WINDOW_SIZE)
            g_mse = float(np.mean(global_pixel_mse[ys, xs]))
            r_mse = float(np.mean(regional_pixel_mse[ys, xs]))
            improvement = (g_mse - r_mse) / g_mse * 100.0 if g_mse > 0 else 0.0
            rows.append({
                "tile": tile,
                "top": top,
                "left": left,
                "height": WINDOW_SIZE,
                "width": WINDOW_SIZE,
                "global_mse": g_mse,
                "regional_mse": r_mse,
                "regional_mse_improvement_pct": improvement,
            })
    return rows


def main() -> None:
    if not GT_PATH.is_file():
        raise FileNotFoundError(f"Test GT MAT file not found: {GT_PATH}")
    for key in MODELS:
        for tile in range(TILE_COUNT):
            if not prediction_path(key, tile).is_file():
                raise FileNotFoundError(f"Missing prediction: {prediction_path(key, tile)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Matched 66K spatial + spectral diagnostics")
    print(f"Repository: {REPO_ROOT}")
    print(f"Output:    {OUTPUT_DIR}")
    print("Loading test ground truth...")
    gt_data = loadmat(GT_PATH, variable_names=["gt"])
    gt_all = np.asarray(gt_data["gt"])
    if gt_all.shape != (TILE_COUNT, HEIGHT, WIDTH, BANDS):
        raise ValueError(f"Unexpected GT shape: {gt_all.shape}")
    if not np.isfinite(gt_all).all():
        raise ValueError("Ground truth contains non-finite values")

    tile_band_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    band_rmse_by_model = {
        key: np.zeros((TILE_COUNT, BANDS), dtype=np.float64) for key in MODELS
    }
    band_mae_by_model = {
        key: np.zeros((TILE_COUNT, BANDS), dtype=np.float64) for key in MODELS
    }
    input_records: list[dict[str, Any]] = []

    for tile in range(TILE_COUNT):
        print(f"Processing tile {tile}...")
        gt = gt_all[tile].astype(np.float64, copy=False)
        global_pred, global_path = load_prediction("GLOBAL", tile)
        regional_pred, regional_path = load_prediction("REGIONAL", tile)
        input_records.extend([
            {"tile": tile, "model": "GLOBAL", "path": str(global_path),
             "size_bytes": global_path.stat().st_size,
             "sha256": sha256_file(global_path)},
            {"tile": tile, "model": "REGIONAL", "path": str(regional_path),
             "size_bytes": regional_path.stat().st_size,
             "sha256": sha256_file(regional_path)},
        ])

        global_error = global_pred.astype(np.float64) - gt
        regional_error = regional_pred.astype(np.float64) - gt
        global_pixel_mse = np.mean(global_error ** 2, axis=2)
        regional_pixel_mse = np.mean(regional_error ** 2, axis=2)

        # Per-tile, per-band RMSE and MAE. Band indices are 0..127; no
        # wavelength mapping is assumed because this MAT file has no labels.
        for band in range(BANDS):
            g_band = global_error[:, :, band]
            r_band = regional_error[:, :, band]
            g_rmse = float(np.sqrt(np.mean(g_band ** 2)))
            r_rmse = float(np.sqrt(np.mean(r_band ** 2)))
            g_mae = float(np.mean(np.abs(g_band)))
            r_mae = float(np.mean(np.abs(r_band)))
            band_rmse_by_model["GLOBAL"][tile, band] = g_rmse
            band_rmse_by_model["REGIONAL"][tile, band] = r_rmse
            band_mae_by_model["GLOBAL"][tile, band] = g_mae
            band_mae_by_model["REGIONAL"][tile, band] = r_mae
            tile_band_rows.extend([
                {"tile": tile, "band_index_zero_based": band, "model": "GLOBAL",
                 "rmse": g_rmse, "mae": g_mae},
                {"tile": tile, "band_index_zero_based": band, "model": "REGIONAL",
                 "rmse": r_rmse, "mae": r_mae},
            ])

        png_path, spatial_stats = save_tile_error_map(
            tile, global_pixel_mse, regional_pixel_mse
        )
        spatial_rows.append(spatial_stats)
        tile_windows = sliding_window_rows(
            tile, global_pixel_mse, regional_pixel_mse
        )
        window_rows.extend(tile_windows)
        print(
            f"  Tile {tile}: Regional MSE improvement="
            f"{spatial_stats['regional_mse_improvement_pct']:+.5f}%; "
            f"pixel win rate={spatial_stats['regional_pixel_win_rate_pct']:.2f}%"
        )
        print(f"  Saved {png_path.name}")

        del global_error, regional_error, global_pred, regional_pred
        del global_pixel_mse, regional_pixel_mse, gt

    # Four-tile arithmetic mean per band, giving each tile equal weight.
    aggregate_band_rows: list[dict[str, Any]] = []
    for band in range(BANDS):
        g_rmse = float(np.mean(band_rmse_by_model["GLOBAL"][:, band]))
        r_rmse = float(np.mean(band_rmse_by_model["REGIONAL"][:, band]))
        g_mae = float(np.mean(band_mae_by_model["GLOBAL"][:, band]))
        r_mae = float(np.mean(band_mae_by_model["REGIONAL"][:, band]))
        aggregate_band_rows.append({
            "band_index_zero_based": band,
            "global_mean_tile_rmse": g_rmse,
            "regional_mean_tile_rmse": r_rmse,
            "regional_rmse_improvement_pct": (
                (g_rmse - r_rmse) / g_rmse * 100.0 if g_rmse > 0 else 0.0
            ),
            "global_mean_tile_mae": g_mae,
            "regional_mean_tile_mae": r_mae,
            "regional_mae_improvement_pct": (
                (g_mae - r_mae) / g_mae * 100.0 if g_mae > 0 else 0.0
            ),
        })

    # Band-wise comparison plot, plus separate tile traces.
    band_indices = np.arange(BANDS)
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True, constrained_layout=True)
    g_mean_rmse = np.mean(band_rmse_by_model["GLOBAL"], axis=0)
    r_mean_rmse = np.mean(band_rmse_by_model["REGIONAL"], axis=0)
    axes[0].plot(band_indices, g_mean_rmse, label="Global SDANet", linewidth=1.7)
    axes[0].plot(band_indices, r_mean_rmse, label="Regional SADS", linewidth=1.7)
    axes[0].set_ylabel("Mean tile RMSE")
    axes[0].set_title("Four-tile mean per-band reconstruction RMSE")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    rmse_improvement = (g_mean_rmse - r_mean_rmse) / np.maximum(g_mean_rmse, 1e-15) * 100
    axes[1].plot(band_indices, rmse_improvement, linewidth=1.5)
    axes[1].axhline(0, linewidth=1, linestyle="--")
    axes[1].set_xlabel("Spectral band index (0–127; no wavelength mapping applied)")
    axes[1].set_ylabel("Regional RMSE improvement (%)")
    axes[1].set_title("Per-band relative RMSE improvement; positive favors Regional")
    axes[1].grid(True, alpha=0.25)
    band_plot_path = OUTPUT_DIR / "four_tile_band_rmse_comparison.png"
    fig.savefig(band_plot_path, dpi=180)
    plt.close(fig)

    # One overview figure with individual tile band-RMSE differences.
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)
    for tile, ax in enumerate(axes.flat):
        g = band_rmse_by_model["GLOBAL"][tile]
        r = band_rmse_by_model["REGIONAL"][tile]
        improvement = (g - r) / np.maximum(g, 1e-15) * 100
        ax.plot(band_indices, improvement, linewidth=1.2)
        ax.axhline(0, linewidth=1, linestyle="--")
        ax.set_title(f"Tile {tile}")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("Band index")
        ax.set_ylabel("Regional RMSE improvement (%)")
    tile_band_plot_path = OUTPUT_DIR / "per_tile_band_rmse_improvement.png"
    fig.savefig(tile_band_plot_path, dpi=180)
    plt.close(fig)

    # Sort sliding windows to flag candidate regions for later visual inspection.
    sorted_windows = sorted(
        window_rows, key=lambda row: row["regional_mse_improvement_pct"], reverse=True
    )
    regional_wins = sum(row["regional_mse_improvement_pct"] > 0 for row in window_rows)
    global_wins = sum(row["regional_mse_improvement_pct"] < 0 for row in window_rows)

    write_csv(OUTPUT_DIR / "tile_band_metrics.csv", tile_band_rows)
    write_csv(OUTPUT_DIR / "mean_band_metrics.csv", aggregate_band_rows)
    write_csv(OUTPUT_DIR / "spatial_error_summary.csv", spatial_rows)
    write_csv(OUTPUT_DIR / "sliding_window_metrics_64x64_stride32.csv", window_rows)

    summary = {
        "title": "Matched 66K spatial and spectral diagnostics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ground_truth_path": str(GT_PATH),
        "ground_truth_sha256": sha256_file(GT_PATH),
        "ground_truth_shape": [TILE_COUNT, HEIGHT, WIDTH, BANDS],
        "tile_count": TILE_COUNT,
        "band_count": BANDS,
        "band_indexing": "Zero-based 0..127; wavelength mapping not assumed.",
        "spatial_metric_definition": "Per-pixel MSE/RMSE averaged over the 128 spectral bands.",
        "regional_improvement_definition": "(Global error - Regional error) / Global error * 100; positive favors Regional.",
        "window_analysis": {
            "window_size": WINDOW_SIZE,
            "stride": WINDOW_STRIDE,
            "windows_per_tile": len(window_rows) // TILE_COUNT,
            "total_windows": len(window_rows),
            "windows_with_regional_lower_mse": regional_wins,
            "windows_with_global_lower_mse": global_wins,
        },
        "spatial_summary": spatial_rows,
        "input_predictions": input_records,
        "top_5_regional_windows": sorted_windows[:5],
        "top_5_global_windows": list(reversed(sorted_windows[-5:])),
        "interpretation_cautions": [
            "The four test tiles are spatial regions from one Chikusei scene, not independent scenes.",
            "The 128 bands are indexed only; this script does not assume wavelengths.",
            "Sliding windows overlap and are exploratory diagnostics, not independent statistical samples.",
            "Color-map percentile limits are for visualization; saved NPZ arrays retain unclipped values.",
        ],
    }
    with (OUTPUT_DIR / "spatial_spectral_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")

    report = [
        "# Matched 66K Spatial and Spectral Diagnostics",
        "",
        f"Generated (UTC): {summary['created_at_utc']}",
        "",
        "## Spatial error maps",
        "",
        "- Each map compares per-pixel spectral RMSE, calculated across all 128 bands.",
        "- The improvement map is `(Global pixel MSE - Regional pixel MSE) / Global pixel MSE × 100`.",
        "- Positive values favor Regional SADS; negative values favor Global SDANet.",
        "- Error-map color scales are robustly limited for visualization; underlying arrays are stored in NPZ files.",
        "",
        "| Tile | Global mean pixel RMSE | Regional mean pixel RMSE | Regional MSE improvement | Regional pixel-win rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in spatial_rows:
        report.append(
            f"| {row['tile']} | {row['global_mean_pixel_rmse']:.9f} | "
            f"{row['regional_mean_pixel_rmse']:.9f} | "
            f"{row['regional_mse_improvement_pct']:+.5f}% | "
            f"{row['regional_pixel_win_rate_pct']:.2f}% |"
        )
    report += [
        "",
        "## Spectral-band analysis",
        "",
        "- Per-band RMSE and MAE are calculated over each 512×512 tile.",
        "- The four-tile band summary is the arithmetic mean of the four tile-level values.",
        "- Bands are labeled with zero-based indices 0–127 because no wavelength mapping is assumed.",
        "",
        "## Sliding-window diagnostic",
        "",
        f"- Window size: {WINDOW_SIZE}×{WINDOW_SIZE}; stride: {WINDOW_STRIDE} pixels.",
        f"- Total overlapping windows: {len(window_rows)}.",
        f"- Regional lower-MSE windows: {regional_wins}.",
        f"- Global lower-MSE windows: {global_wins}.",
        "- Windows are spatially overlapping and must not be treated as independent samples.",
        "",
        "### Five windows with largest Regional improvement",
        "",
        "| Tile | Top | Left | Regional MSE improvement |",
        "|---:|---:|---:|---:|",
    ]
    for row in sorted_windows[:5]:
        report.append(
            f"| {row['tile']} | {row['top']} | {row['left']} | "
            f"{row['regional_mse_improvement_pct']:+.5f}% |"
        )
    report += [
        "",
        "### Five windows with largest Global improvement",
        "",
        "| Tile | Top | Left | Regional MSE improvement |",
        "|---:|---:|---:|---:|",
    ]
    for row in reversed(sorted_windows[-5:]):
        report.append(
            f"| {row['tile']} | {row['top']} | {row['left']} | "
            f"{row['regional_mse_improvement_pct']:+.5f}% |"
        )
    report += [
        "",
        "## Files",
        "",
        "- `tile_0..3_spatial_error_maps.png`: paired error maps and Regional-vs-Global improvement.",
        "- `tile_0..3_spatial_error_maps.npz`: unclipped per-pixel error arrays.",
        "- `tile_band_metrics.csv`: per-tile, per-band RMSE and MAE for both models.",
        "- `mean_band_metrics.csv`: four-tile mean per-band RMSE and MAE comparison.",
        "- `four_tile_band_rmse_comparison.png`: aggregate band curves.",
        "- `per_tile_band_rmse_improvement.png`: band improvement curves for each tile.",
        "- `spatial_error_summary.csv`: per-tile spatial summary.",
        "- `sliding_window_metrics_64x64_stride32.csv`: overlapping local-region comparisons.",
        "- `spatial_spectral_summary.json`: inputs, checksums, definitions, and key regions.",
        "",
        "## Interpretation cautions",
        "",
        *[f"- {item}" for item in summary["interpretation_cautions"]],
        "- These are descriptive diagnostics for four spatial tiles from one scene; they do not establish statistical significance or cross-scene generalization.",
    ]
    (OUTPUT_DIR / "spatial_spectral_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\nSpatial/spectral artifacts written:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(f"  {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")
    print("\nTop 5 windows where Regional has lower MSE:")
    for row in sorted_windows[:5]:
        print(
            f"  tile={row['tile']} top={row['top']} left={row['left']} "
            f"improvement={row['regional_mse_improvement_pct']:+.4f}%"
        )
    print("\nTop 5 windows where Global has lower MSE:")
    for row in reversed(sorted_windows[-5:]):
        print(
            f"  tile={row['tile']} top={row['top']} left={row['left']} "
            f"Regional improvement={row['regional_mse_improvement_pct']:+.4f}%"
        )
    print("\nNo checkpoint, prediction, dataset, training, or test configuration was modified.")


if __name__ == "__main__":
    main()
