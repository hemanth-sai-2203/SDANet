#!/usr/bin/env python3
"""Create scene-wise false-colour and spectral-error figures for SDANet HSI-SR.

Run from the repository root:
  .venv/bin/python visualize_sdanet_hsi.py \
    --root results/final_repro \
    --data-mat ../datasets/Chikusei_x4_sspsr_spatial_split/Chikusei_test.mat

Expected prediction layout (produced by evaluate_sdanet_scenes.py):
  <root>/<run>/predictions/test_img_<i>_prediction.mat
with variable `sr_data` in HWC float32 format.

Figures use one shared GT-derived display range per selected band and shared
error-map scales across the four models within each scene. This is intended for
visual comparison, not for calculating the numerical metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

RUNS = {
    "Original SDANet (33K)": "SDAnet_Chikusei_x4_test",
    "RegionalFull (66K)": "SDANetRegionalFull_Chikusei_x4_test",
    "Fine-Gate Control (66K+5K)": "SDANetRegionalFineGateControl5K_Chikusei_x4_test",
    "Fine-Gate Treatment (66K+5K)": "SDANetRegionalFineGateTreatment5K_Chikusei_x4_test",
}


def load_hwc(path: Path, variable: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    mat = sio.loadmat(path, variable_names=[variable])
    if variable not in mat:
        raise KeyError(f"Variable {variable!r} not found in {path}")
    array = np.asarray(mat[variable], dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected HWC 3-D array in {path}, found shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {path}")
    return array


def false_colour(image: np.ndarray, bands: tuple[int, int, int], limits: list[tuple[float, float]]) -> np.ndarray:
    channels = []
    for band, (low, high) in zip(bands, limits):
        plane = image[:, :, band]
        scale = max(high - low, 1e-12)
        channels.append(np.clip((plane - low) / scale, 0.0, 1.0))
    return np.stack(channels, axis=-1)


def pixel_sam_degrees(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    ref = reference.astype(np.float64, copy=False)
    pred = prediction.astype(np.float64, copy=False)
    dot = np.sum(ref * pred, axis=2)
    denom = np.linalg.norm(ref, axis=2) * np.linalg.norm(pred, axis=2)
    cosine = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    angles = np.degrees(np.arccos(cosine))
    angles[denom <= 1e-12] = np.nan
    return angles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/final_repro"))
    parser.add_argument("--data-mat", type=Path, required=True,
                        help="Chikusei_test.mat containing variable `gt` with shape N,H,W,C")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bands", default="90,50,20",
                        help="0-based band indices used as R,G,B in the false-colour composite")
    parser.add_argument("--percentiles", default="1,99",
                        help="GT percentiles used for shared false-colour contrast limits")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    data_mat = args.data_mat.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "visualizations"
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing visualizations: {out_dir}")

    try:
        bands = tuple(int(x.strip()) for x in args.bands.split(","))
        percentiles = tuple(float(x.strip()) for x in args.percentiles.split(","))
    except ValueError as exc:
        raise ValueError("Could not parse --bands or --percentiles") from exc
    if len(bands) != 3:
        raise ValueError("--bands must contain exactly three comma-separated indices")
    if len(percentiles) != 2 or not (0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("--percentiles must be two increasing values between 0 and 100")

    gt_mat = sio.loadmat(data_mat, variable_names=["gt"])
    if "gt" not in gt_mat:
        raise KeyError(f"Variable 'gt' not found in {data_mat}")
    ground_truth = np.asarray(gt_mat["gt"], dtype=np.float32)
    if ground_truth.ndim != 4:
        raise ValueError(f"Expected GT with shape N,H,W,C, found {ground_truth.shape}")
    if not np.isfinite(ground_truth).all():
        raise ValueError("Ground-truth array contains non-finite values")

    scene_count, height, width, channels = ground_truth.shape
    if max(bands) >= channels or min(bands) < 0:
        raise ValueError(f"Band indices {bands} invalid for {channels} channels")

    # Verify all required model/scene prediction files before creating output.
    predictions: dict[str, list[np.ndarray]] = {}
    for label, run_name in RUNS.items():
        run_dir = root / run_name / "predictions"
        predictions[label] = []
        for scene_index in range(scene_count):
            scene_id = f"test_img_{scene_index}"
            prediction_path = run_dir / f"{scene_id}_prediction.mat"
            prediction = load_hwc(prediction_path, "sr_data")
            if prediction.shape != ground_truth[scene_index].shape:
                raise ValueError(
                    f"Shape mismatch for {label}/{scene_id}: "
                    f"prediction={prediction.shape}, GT={ground_truth[scene_index].shape}"
                )
            predictions[label].append(prediction)

    out_dir.mkdir(parents=True, exist_ok=False)
    print(f"Scenes: {scene_count}; image shape: {height}x{width}x{channels}")
    print(f"False-colour bands (0-based R,G,B): {bands}; GT percentiles: {percentiles}")

    for scene_index in range(scene_count):
        gt = ground_truth[scene_index]
        scene_predictions = {label: values[scene_index] for label, values in predictions.items()}

        # Use GT-only contrast limits so every row is displayed on the same scale.
        limits = [
            tuple(float(v) for v in np.percentile(gt[:, :, band], percentiles))
            for band in bands
        ]
        colour_gt = false_colour(gt, bands, limits)

        mae_maps = {
            label: np.mean(np.abs(pred - gt), axis=2)
            for label, pred in scene_predictions.items()
        }
        sam_maps = {
            label: pixel_sam_degrees(gt, pred)
            for label, pred in scene_predictions.items()
        }
        all_mae = np.concatenate([v.ravel() for v in mae_maps.values()])
        all_sam = np.concatenate([v[np.isfinite(v)] for v in sam_maps.values()])
        mae_max = max(float(np.percentile(all_mae, 99.5)), 1e-12)
        sam_max = max(float(np.percentile(all_sam, 99.5)), 1e-12)

        fig, axes = plt.subplots(
            nrows=len(RUNS) + 1,
            ncols=3,
            figsize=(15, 3.1 * (len(RUNS) + 1)),
            constrained_layout=True,
        )
        column_titles = [
            f"False-colour composite (bands {bands[0]}, {bands[1]}, {bands[2]})",
            "Mean absolute spectral error",
            "Per-pixel spectral angle (degrees)",
        ]
        for col, title in enumerate(column_titles):
            axes[0, col].set_title(title, fontsize=11)

        axes[0, 0].imshow(colour_gt)
        axes[0, 0].set_ylabel("Ground truth", fontsize=10)
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

        for row_index, (label, prediction) in enumerate(scene_predictions.items(), start=1):
            axes[row_index, 0].imshow(false_colour(prediction, bands, limits))
            mae_image = axes[row_index, 1].imshow(
                mae_maps[label], cmap="magma", vmin=0.0, vmax=mae_max
            )
            sam_image = axes[row_index, 2].imshow(
                np.ma.masked_invalid(sam_maps[label]), cmap="viridis", vmin=0.0, vmax=sam_max
            )
            axes[row_index, 0].set_ylabel(label, fontsize=9)
            if row_index == 1:
                fig.colorbar(mae_image, ax=axes[row_index, 1], fraction=0.046, pad=0.03)
                fig.colorbar(sam_image, ax=axes[row_index, 2], fraction=0.046, pad=0.03)

        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Chikusei test scene {scene_index}", fontsize=15)
        output_path = out_dir / f"scene_{scene_index}_qualitative_comparison.png"
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    readme = out_dir / "README.txt"
    readme.write_text(
        "SDANet HSI-SR qualitative comparisons\n"
        f"Scenes: {scene_count}\n"
        f"Image shape: {height}x{width}x{channels}\n"
        f"False-colour bands (0-based R,G,B): {bands}\n"
        f"False-colour contrast: GT percentiles {percentiles}, shared across models per scene\n"
        "Error maps use a shared 99.5th-percentile display maximum across the four models per scene.\n"
        "MAE map: mean absolute error over spectral bands at each pixel.\n"
        "SAM map: spectral angle in degrees at each pixel.\n"
        "These figures are qualitative; use the saved CSVs for numerical comparisons.\n",
        encoding="utf-8",
    )
    print(f"\nAll figures saved under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
