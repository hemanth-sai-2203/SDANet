#!/usr/bin/env python3
"""Create reproducible qualitative crops for matched Global vs Regional 66K HSI-SR."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ANALYSIS = RESULTS / "Matched66K_Analysis"
SPATIAL = ANALYSIS / "spatial_spectral"
OUT = ANALYSIS / "qualitative_regions"
GT_PATH = Path(
    "/home/aidslab/hemanth-ug4/datasets/"
    "Chikusei_x4_sspsr_spatial_split/Chikusei_test.mat"
)
GLOBAL_DIR = RESULTS / "SDANetGlobalMatched66K_Chikusei_x4_test/visualization/Chikusei_test"
REGIONAL_DIR = RESULTS / "SDANetRegionalMatched66K_Chikusei_x4_test/visualization/Chikusei_test"
GLOBAL_SUFFIX = "SDANetGlobalMatched66K_Chikusei_x4_test"
REGIONAL_SUFFIX = "SDANetRegionalMatched66K_Chikusei_x4_test"
WINDOW_CSV = SPATIAL / "sliding_window_metrics_64x64_stride32.csv"
WINDOW = 64


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_windows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"tile", "top", "left", "height", "width", "regional_mse_improvement_pct"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected sliding-window CSV schema: {path}")
    parsed = []
    for row in rows:
        item = {k: int(row[k]) for k in ("tile", "top", "left", "height", "width")}
        item["improvement"] = float(row["regional_mse_improvement_pct"])
        parsed.append(item)
    return parsed


def select_regions(rows: list[dict]) -> list[dict]:
    # The first two are the strongest overall Regional and Global-favouring windows.
    selected: list[dict] = []
    best_regional = max(rows, key=lambda r: r["improvement"])
    worst_regional = min(rows, key=lambda r: r["improvement"])
    selected.append({**best_regional, "case": "strongest_regional_window"})
    selected.append({**worst_regional, "case": "strongest_global_window"})

    # Add one strongly Global-favouring example from each remaining tile.
    for tile in (0, 2, 3):
        tile_rows = [r for r in rows if r["tile"] == tile]
        if tile_rows:
            candidate = min(tile_rows, key=lambda r: r["improvement"])
            selected.append({**candidate, "case": f"tile_{tile}_global_favouring_window"})
    # De-duplicate any coincident coordinates while retaining order.
    unique = []
    seen = set()
    for item in selected:
        key = (item["tile"], item["top"], item["left"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def load_prediction(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = loadmat(path, variable_names=["sr_data"])["sr_data"]
    if data.shape != (512, 512, 128):
        raise ValueError(f"Unexpected prediction shape in {path}: {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite prediction values in {path}")
    return data.astype(np.float32, copy=False)


def main() -> None:
    for path in (GT_PATH, WINDOW_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Matched 66K qualitative-region analysis")
    print(f"Repository: {ROOT}")
    print(f"Output:    {OUT}")
    print("Loading test ground truth and selected-window metadata...")
    gt_all = loadmat(GT_PATH, variable_names=["gt"])["gt"]
    if gt_all.shape != (4, 512, 512, 128):
        raise ValueError(f"Unexpected GT shape: {gt_all.shape}")
    rows = read_windows(WINDOW_CSV)
    chosen = select_regions(rows)

    metadata_rows = []
    for tile in sorted({r["tile"] for r in chosen}):
        print(f"Loading predictions for tile {tile}...")
        gt = gt_all[tile].astype(np.float32, copy=False)
        g_path = GLOBAL_DIR / f"test_img_{tile}_{GLOBAL_SUFFIX}.mat"
        r_path = REGIONAL_DIR / f"test_img_{tile}_{REGIONAL_SUFFIX}.mat"
        global_pred = load_prediction(g_path)
        regional_pred = load_prediction(r_path)

        for region in [r for r in chosen if r["tile"] == tile]:
            top, left = region["top"], region["left"]
            height, width = region["height"], region["width"]
            if height != WINDOW or width != WINDOW:
                raise ValueError(f"Expected {WINDOW}x{WINDOW} window, got {height}x{width}")
            ys, xs = slice(top, top + height), slice(left, left + width)
            gt_crop = gt[ys, xs, :].astype(np.float32, copy=False)
            g_crop = global_pred[ys, xs, :]
            r_crop = regional_pred[ys, xs, :]

            g_error = g_crop.astype(np.float64) - gt_crop.astype(np.float64)
            r_error = r_crop.astype(np.float64) - gt_crop.astype(np.float64)
            g_pixel_mse = np.mean(g_error ** 2, axis=2)
            r_pixel_mse = np.mean(r_error ** 2, axis=2)
            g_rmse_map = np.sqrt(g_pixel_mse)
            r_rmse_map = np.sqrt(r_pixel_mse)
            improvement_map = np.divide(
                g_pixel_mse - r_pixel_mse,
                g_pixel_mse,
                out=np.zeros_like(g_pixel_mse),
                where=g_pixel_mse > 0,
            ) * 100.0
            g_mse = float(np.mean(g_error ** 2))
            r_mse = float(np.mean(r_error ** 2))
            improvement = (g_mse - r_mse) / g_mse * 100.0 if g_mse > 0 else 0.0
            pixel_win = float(np.mean(r_pixel_mse < g_pixel_mse) * 100.0)

            # A mean-over-bands intensity view provides spatial context only; it is not a wavelength composite.
            gt_mean = np.mean(gt_crop, axis=2)
            g_mean = np.mean(g_crop, axis=2)
            r_mean = np.mean(r_crop, axis=2)
            display_stack = np.stack([gt_mean, g_mean, r_mean])
            lo, hi = np.percentile(display_stack, [1, 99])
            if hi <= lo:
                lo, hi = float(display_stack.min()), float(display_stack.max() + 1e-8)

            safe_case = region["case"]
            stem = f"{safe_case}_tile{tile}_top{top}_left{left}"
            npz_path = OUT / f"{stem}.npz"
            np.savez_compressed(
                npz_path,
                gt_mean_over_bands=gt_mean,
                global_mean_over_bands=g_mean,
                regional_mean_over_bands=r_mean,
                global_pixel_spectral_rmse=g_rmse_map,
                regional_pixel_spectral_rmse=r_rmse_map,
                regional_pixel_mse_improvement_pct=improvement_map,
                gt_crop=gt_crop,
                global_crop=g_crop,
                regional_crop=r_crop,
            )

            fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
            fig.suptitle(
                f"{safe_case.replace('_', ' ').title()} | Tile {tile} | "
                f"64x64 at (top={top}, left={left})\n"
                f"Regional MSE improvement: {improvement:+.3f}% | "
                f"Regional pixel-win rate: {pixel_win:.2f}%",
                fontsize=13,
            )
            panels = [
                (gt_mean, "GT mean over 128 bands", "gray", lo, hi),
                (g_mean, "Global mean over 128 bands", "gray", lo, hi),
                (r_mean, "Regional mean over 128 bands", "gray", lo, hi),
                (g_rmse_map, "Global per-pixel spectral RMSE", "magma", None, None),
                (r_rmse_map, "Regional per-pixel spectral RMSE", "magma", None, None),
                (improvement_map, "Regional MSE improvement (%)", "coolwarm", None, None),
            ]
            lim = float(np.percentile(np.abs(improvement_map), 98))
            lim = max(lim, 1e-8)
            for ax, (array, title, cmap, vmin, vmax) in zip(axes.flat, panels):
                if title == "Regional MSE improvement (%)":
                    im = ax.imshow(array, cmap=cmap, vmin=-lim, vmax=lim)
                else:
                    im = ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("x within crop")
                ax.set_ylabel("y within crop")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            png_path = OUT / f"{stem}.png"
            fig.savefig(png_path, dpi=180)
            plt.close(fig)

            metadata_rows.append({
                "case": safe_case,
                "tile": tile,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
                "sliding_window_improvement_pct": region["improvement"],
                "recomputed_mse_improvement_pct": improvement,
                "regional_pixel_win_rate_pct": pixel_win,
                "global_mse": g_mse,
                "regional_mse": r_mse,
                "png": png_path.name,
                "npz": npz_path.name,
            })
            print(
                f"  {safe_case}: tile={tile}, top={top}, left={left}, "
                f"Regional MSE improvement={improvement:+.5f}%, "
                f"pixel-win rate={pixel_win:.2f}%"
            )

    csv_path = OUT / "selected_region_manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    report_path = OUT / "qualitative_region_report.md"
    lines = [
        "# Matched 66K Qualitative Region Analysis",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Selection protocol",
        "",
        "Regions are selected from the previously computed 64x64, stride-32 sliding-window diagnostic:",
        "- Strongest Regional-favouring window across all tiles.",
        "- Strongest Global-favouring window across all tiles.",
        "- Strongest Global-favouring window within each of tiles 0, 2 and 3.",
        "",
        "These are deliberately selected illustrative examples, not an unbiased sample. The windows overlap with the broader diagnostic grid and must not be treated as independent observations.",
        "",
        "## Display interpretation",
        "",
        "The first row shows ground truth and predictions averaged over all 128 bands for spatial context; it is not an RGB or wavelength-calibrated composite.",
        "The second row shows per-pixel spectral RMSE and relative Regional-vs-Global MSE improvement. Positive improvement favours Regional SADS.",
        "Improvement-map color limits are robustly clipped for display only; NPZ arrays retain the full values.",
        "",
        "## Selected regions",
        "",
        "| Case | Tile | Top | Left | Recomputed Regional MSE improvement | Regional pixel-win rate | PNG |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in metadata_rows:
        lines.append(
            f"| {r['case']} | {r['tile']} | {r['top']} | {r['left']} | "
            f"{r['recomputed_mse_improvement_pct']:+.5f}% | "
            f"{r['regional_pixel_win_rate_pct']:.2f}% | `{r['png']}` |"
        )
    lines += [
        "",
        "## Caution",
        "",
        "The four test tiles are spatial regions from one Chikusei scene. These selected crops provide qualitative examples only and do not establish cross-scene generalization or statistical significance.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(GT_PATH),
        "dataset_sha256": sha256(GT_PATH),
        "window_csv": str(WINDOW_CSV),
        "window_csv_sha256": sha256(WINDOW_CSV),
        "window_size": WINDOW,
        "selected_regions": metadata_rows,
        "selection_is_exploratory_and_nonrandom": True,
        "test_tiles_are_spatial_regions_of_one_scene": True,
        "no_training_or_model_evaluation_performed": True,
    }
    summary_path = OUT / "qualitative_region_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nQualitative-region artifacts written:")
    for path in sorted(OUT.iterdir()):
        if path.is_file():
            print(f"  {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    print("\nNo checkpoints, predictions, datasets, training or test configurations were modified.")


if __name__ == "__main__":
    main()
