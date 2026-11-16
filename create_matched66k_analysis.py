#!/usr/bin/env python3
"""Reproduce the matched 66K Global SDANet vs Regional SADS analysis.

Run from the SDANet repository root with:
  PYTHONPATH="$PWD" .venv/bin/python create_matched66k_analysis.py

Reads existing checkpoints, test MAT data, and saved predictions only.
Writes analysis artifacts beneath results/Matched66K_Analysis/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.io import loadmat

from basicsr.metrics import calculate_metric


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "Matched66K_Analysis"

DATASET_PATH = Path(
    "/home/aidslab/hemanth-ug4/datasets/"
    "Chikusei_x4_sspsr_spatial_split/Chikusei_test.mat"
)

MODELS = {
    "GLOBAL": {
        "name": "Global SDANet",
        "result_dir": RESULTS_ROOT
        / "SDANetGlobalMatched66K_Chikusei_x4_test/visualization/Chikusei_test",
        "suffix": "SDANetGlobalMatched66K_Chikusei_x4_test",
        "checkpoint": REPO_ROOT
        / "experiments/SDANetGlobalMatchedContinue66K_spatial_split/models/net_g_66000.pth",
    },
    "REGIONAL": {
        "name": "Regional SADS",
        "result_dir": RESULTS_ROOT
        / "SDANetRegionalMatched66K_Chikusei_x4_test/visualization/Chikusei_test",
        "suffix": "SDANetRegionalMatched66K_Chikusei_x4_test",
        "checkpoint": REPO_ROOT
        / "experiments/SDANetRegionalContinue66K_spatial_split/models/net_g_66000.pth",
    },
}

METRIC_OPTIONS = {
    "SSIM": {"type": "calculate_ssim_float", "crop_border": 0, "data_range": 1},
    "MPSNR": {"type": "calculate_mpsnr_hsi", "crop_border": 0, "data_range": 1},
    "SAM": {"type": "calculate_sam_hsi", "crop_border": 0},
    "ERGAS": {"type": "calculate_ergas_hsi", "crop_border": 0},
    "CC": {"type": "calculate_corr_hsi", "crop_border": 0},
}

QUADRANTS = {
    "TOP_LEFT": (slice(0, 256), slice(0, 256)),
    "TOP_RIGHT": (slice(0, 256), slice(256, 512)),
    "BOTTOM_LEFT": (slice(256, 512), slice(0, 256)),
    "BOTTOM_RIGHT": (slice(256, 512), slice(256, 512)),
}

DIRECTIONS = {
    "SSIM": "higher",
    "MPSNR": "higher",
    "SAM": "lower",
    "ERGAS": "lower",
    "CC": "higher",
    "MSE": "lower",
    "MAE": "lower",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_values(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, options in METRIC_OPTIONS.items():
        values[name] = float(
            calculate_metric({"img": pred, "img2": gt}, options)
        )
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    values["MSE"] = float(np.mean(diff**2))
    values["MAE"] = float(np.mean(np.abs(diff)))
    return values


def load_prediction(model_key: str, tile: int) -> tuple[np.ndarray, Path]:
    model = MODELS[model_key]
    path = model["result_dir"] / (
        f"test_img_{tile}_{model['suffix']}.mat"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing {model_key} prediction: {path}")
    data = loadmat(path, variable_names=["sr_data"])
    if "sr_data" not in data:
        raise KeyError(f"MAT file has no sr_data variable: {path}")
    pred = np.asarray(data["sr_data"])
    if pred.shape != (512, 512, 128):
        raise ValueError(f"Unexpected shape in {path}: {pred.shape}")
    if not np.isfinite(pred).all():
        raise ValueError(f"Non-finite prediction values in {path}")
    return pred, path


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Test dataset not found: {DATASET_PATH}")

    print("Matched 66K analysis")
    print(f"Repository: {REPO_ROOT}")
    print(f"Test MAT:  {DATASET_PATH}")
    print(f"Output:    {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_data = loadmat(DATASET_PATH, variable_names=["gt"])
    if "gt" not in gt_data:
        raise KeyError("Test MAT file does not contain variable 'gt'")
    gt_all = np.asarray(gt_data["gt"])
    if gt_all.shape != (4, 512, 512, 128):
        raise ValueError(f"Unexpected GT shape: {gt_all.shape}")
    if not np.isfinite(gt_all).all():
        raise ValueError("Ground truth contains non-finite values")

    tile_rows: list[dict[str, Any]] = []
    paired_tile_rows: list[dict[str, Any]] = []
    quadrant_rows: list[dict[str, Any]] = []
    tile_prediction_difference: list[dict[str, Any]] = []
    all_metric_values: dict[str, dict[str, list[float]]] = {
        key: {metric: [] for metric in DIRECTIONS} for key in MODELS
    }

    for tile in range(4):
        gt = gt_all[tile].astype(np.float32, copy=False)
        predictions: dict[str, np.ndarray] = {}
        values_by_model: dict[str, dict[str, float]] = {}
        prediction_paths: dict[str, Path] = {}

        for model_key in MODELS:
            pred, pred_path = load_prediction(model_key, tile)
            if pred.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for tile {tile} {model_key}: "
                    f"prediction={pred.shape}, GT={gt.shape}"
                )
            predictions[model_key] = pred
            prediction_paths[model_key] = pred_path
            values = metric_values(pred, gt)
            values_by_model[model_key] = values

            row: dict[str, Any] = {
                "tile": tile,
                "model": model_key,
                "model_name": MODELS[model_key]["name"],
                **values,
            }
            tile_rows.append(row)
            for metric, value in values.items():
                all_metric_values[model_key][metric].append(value)

        paired_row: dict[str, Any] = {"tile": tile}
        for metric in DIRECTIONS:
            global_value = values_by_model["GLOBAL"][metric]
            regional_value = values_by_model["REGIONAL"][metric]
            paired_row[f"global_{metric}"] = global_value
            paired_row[f"regional_{metric}"] = regional_value
            paired_row[f"regional_minus_global_{metric}"] = (
                regional_value - global_value
            )
            if metric in {"MSE", "MAE", "SAM", "ERGAS"}:
                paired_row[f"regional_relative_change_pct_{metric}"] = (
                    (regional_value - global_value) / global_value * 100
                    if global_value != 0 else None
                )
        paired_tile_rows.append(paired_row)

        pred_delta = (
            predictions["REGIONAL"].astype(np.float64)
            - predictions["GLOBAL"].astype(np.float64)
        )
        tile_prediction_difference.append(
            {
                "tile": tile,
                "mean_absolute_difference": float(np.mean(np.abs(pred_delta))),
                "rmse_between_predictions": float(np.sqrt(np.mean(pred_delta**2))),
                "maximum_absolute_difference": float(np.max(np.abs(pred_delta))),
            }
        )

        for quadrant, (ys, xs) in QUADRANTS.items():
            gt_q = gt[ys, xs, :].astype(np.float64, copy=False)
            g_q = predictions["GLOBAL"][ys, xs, :].astype(np.float64, copy=False)
            r_q = predictions["REGIONAL"][ys, xs, :].astype(np.float64, copy=False)
            g_error = g_q - gt_q
            r_error = r_q - gt_q
            g_mse = float(np.mean(g_error**2))
            r_mse = float(np.mean(r_error**2))
            g_mae = float(np.mean(np.abs(g_error)))
            r_mae = float(np.mean(np.abs(r_error)))
            g_pixel_mse = np.mean(g_error**2, axis=2)
            r_pixel_mse = np.mean(r_error**2, axis=2)
            row = {
                "tile": tile,
                "quadrant": quadrant,
                "global_mse": g_mse,
                "regional_mse": r_mse,
                "regional_mse_improvement_pct": (
                    (g_mse - r_mse) / g_mse * 100 if g_mse > 0 else None
                ),
                "global_mae": g_mae,
                "regional_mae": r_mae,
                "regional_mae_improvement_pct": (
                    (g_mae - r_mae) / g_mae * 100 if g_mae > 0 else None
                ),
                "regional_pixel_win_rate_pct": float(
                    np.mean(r_pixel_mse < g_pixel_mse) * 100
                ),
                "spatial_pixel_count": int(g_pixel_mse.size),
                "spectral_bands": int(gt_q.shape[2]),
            }
            quadrant_rows.append(row)

    # Equal-weight four-tile means, matching the BasicSR test aggregation.
    aggregate: dict[str, dict[str, float]] = {}
    for model_key in MODELS:
        aggregate[model_key] = {
            metric: float(np.mean(values))
            for metric, values in all_metric_values[model_key].items()
        }

    quadrant_wins = sum(
        row["regional_mse_improvement_pct"] > 0 for row in quadrant_rows
    )
    quadrant_losses = sum(
        row["regional_mse_improvement_pct"] < 0 for row in quadrant_rows
    )
    quadrant_ties = len(quadrant_rows) - quadrant_wins - quadrant_losses
    mean_quadrant_improvement = float(
        np.mean([row["regional_mse_improvement_pct"] for row in quadrant_rows])
    )

    summary: dict[str, Any] = {
        "title": "Matched 66K Global SDANet vs Regional SADS evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET_PATH),
        "dataset_shape": list(gt_all.shape),
        "tile_count": 4,
        "tile_shape": [512, 512, 128],
        "inference_weights": "params_ema",
        "metric_aggregation": "Arithmetic mean of the four equally sized test tiles",
        "models": {
            key: {
                "name": MODELS[key]["name"],
                "architecture": "SDANET" if key == "GLOBAL" else "SDANET_SADS",
                "parameter_count": 3086272,
                "checkpoint": str(MODELS[key]["checkpoint"]),
                "checkpoint_sha256": (
                    sha256_file(MODELS[key]["checkpoint"])
                    if MODELS[key]["checkpoint"].is_file() else None
                ),
                "aggregate_metrics": aggregate[key],
            }
            for key in MODELS
        },
        "regional_minus_global_aggregate": {
            metric: aggregate["REGIONAL"][metric] - aggregate["GLOBAL"][metric]
            for metric in DIRECTIONS
        },
        "quadrant_analysis": {
            "quadrants_where_regional_mse_is_lower": quadrant_wins,
            "quadrants_where_global_mse_is_lower": quadrant_losses,
            "quadrant_ties": quadrant_ties,
            "total_quadrants": len(quadrant_rows),
            "unweighted_mean_regional_mse_improvement_pct": mean_quadrant_improvement,
            "pixel_win_rate_definition": (
                "Percentage of spatial pixels where Regional has lower mean-squared "
                "error across the 128 spectral bands than Global."
            ),
        },
        "metric_directions": DIRECTIONS,
        "caveats": [
            "The four test tiles are spatial regions from one Chikusei scene, not four independent scenes.",
            "Small metric differences are descriptive and are not evidence of statistical significance.",
            "Quadrant-level MSE is a spatial diagnostic; it is not a separate model evaluation set.",
        ],
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }

    # Write machine-readable and human-readable records.
    write_csv(OUTPUT_DIR / "tile_metrics.csv", tile_rows)
    write_csv(OUTPUT_DIR / "tile_comparison.csv", paired_tile_rows)
    write_csv(OUTPUT_DIR / "quadrant_metrics.csv", quadrant_rows)
    write_csv(OUTPUT_DIR / "prediction_difference.csv", tile_prediction_difference)

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")

    report_lines = [
        "# Matched 66K Global SDANet vs Regional SADS Analysis",
        "",
        f"Generated (UTC): {summary['created_at_utc']}",
        "",
        "## Protocol",
        "",
        f"- Test data: `{DATASET_PATH}`",
        "- Four tiles, each 512×512×128; same ground truth for both models.",
        "- EMA weights (`params_ema`) from the matched 66K checkpoints.",
        "- BasicSR metric implementations/settings reproduced from the test configuration.",
        "- Aggregate values are arithmetic means over the four equally sized tiles.",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Direction | Global SDANet | Regional SADS | Regional − Global |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, direction in DIRECTIONS.items():
        g = aggregate["GLOBAL"][metric]
        r = aggregate["REGIONAL"][metric]
        report_lines.append(
            f"| {metric} | {direction} | {g:.10f} | {r:.10f} | {r-g:+.10f} |"
        )

    report_lines += [
        "",
        "## Quadrant-wise MSE audit",
        "",
        f"- Regional lower MSE: **{quadrant_wins}/16** quadrants.",
        f"- Global lower MSE: **{quadrant_losses}/16** quadrants.",
        f"- Ties: **{quadrant_ties}/16** quadrants.",
        f"- Unweighted mean Regional MSE improvement: **{mean_quadrant_improvement:+.5f}%**.",
        "- Positive improvement means Regional MSE is lower than Global MSE.",
        "- Pixel-win rate compares per-pixel MSE averaged over 128 spectral bands.",
        "",
        "| Tile | Quadrant | Global MSE | Regional MSE | Regional improvement | Regional pixel-win rate |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in quadrant_rows:
        report_lines.append(
            f"| {row['tile']} | {row['quadrant']} | {row['global_mse']:.10f} | "
            f"{row['regional_mse']:.10f} | {row['regional_mse_improvement_pct']:+.5f}% | "
            f"{row['regional_pixel_win_rate_pct']:.2f}% |"
        )

    report_lines += [
        "",
        "## Prediction differences",
        "",
        "| Tile | Mean absolute difference | RMSE between predictions | Maximum absolute difference |",
        "|---:|---:|---:|---:|",
    ]
    for row in tile_prediction_difference:
        report_lines.append(
            f"| {row['tile']} | {row['mean_absolute_difference']:.8f} | "
            f"{row['rmse_between_predictions']:.8f} | "
            f"{row['maximum_absolute_difference']:.8f} |"
        )

    report_lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- These results are descriptive for the four held-out spatial tiles from one Chikusei scene.",
        "- They do not establish statistical significance or generalization across independent scenes.",
        "- A lower error in selected quadrants does not establish a consistent benefit of regional attention.",
        "- Retain the full per-tile and per-quadrant records; do not report only the aggregate means.",
        "",
        "## Reproducibility files",
        "",
        "- `tile_metrics.csv`: one row per tile and model.",
        "- `tile_comparison.csv`: paired Global/Regional metrics and differences per tile.",
        "- `quadrant_metrics.csv`: 16 spatial quadrant comparisons.",
        "- `prediction_difference.csv`: direct prediction difference per tile.",
        "- `summary.json`: aggregate results, checkpoint hashes, dataset details, and software versions.",
    ]
    (OUTPUT_DIR / "analysis_report.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\nAnalysis artifacts written:")
    for name in [
        "tile_metrics.csv",
        "tile_comparison.csv",
        "quadrant_metrics.csv",
        "prediction_difference.csv",
        "summary.json",
        "analysis_report.md",
    ]:
        path = OUTPUT_DIR / name
        print(f"  {path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")

    print("\nAggregate metrics (four-tile arithmetic mean):")
    for metric in DIRECTIONS:
        g = aggregate["GLOBAL"][metric]
        r = aggregate["REGIONAL"][metric]
        print(f"  {metric:6s} Global={g:.10f} Regional={r:.10f} Delta={r-g:+.10f}")
    print(
        f"\nQuadrant MSE: Regional wins {quadrant_wins}/16, "
        f"Global wins {quadrant_losses}/16, mean improvement {mean_quadrant_improvement:+.5f}%"
    )
    print("\nNo checkpoint, prediction, training, or test configuration was modified.")


if __name__ == "__main__":
    main()
