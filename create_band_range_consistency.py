#!/usr/bin/env python3
"""Analyze spectral-band consistency for the matched 66K Global/Regional models.

Reads the existing tile_band_metrics.csv and mean_band_metrics.csv. It does not
load model checkpoints, rerun inference, or modify source experiment artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
INPUT_DIR = REPO / "results/Matched66K_Analysis/spatial_spectral"
OUTPUT_DIR = INPUT_DIR
TILE_FILE = INPUT_DIR / "tile_band_metrics.csv"
MEAN_FILE = INPUT_DIR / "mean_band_metrics.csv"
EXPECTED_TILES = [0, 1, 2, 3]
EXPECTED_BANDS = list(range(128))
FOCUS_RANGES = [(19, 27), (80, 93)]  # exploratory ranges flagged in prior top/bottom-band inspection


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required input not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def improvement(global_value: float, regional_value: float) -> float:
    if global_value == 0:
        return 0.0 if regional_value == 0 else float("nan")
    return (global_value - regional_value) / global_value * 100.0


def sign_label(value: float) -> str:
    if value > 0:
        return "REGIONAL"
    if value < 0:
        return "GLOBAL"
    return "TIE"


def contiguous_runs(values: list[tuple[int, str]], category: str, metric: str) -> list[dict]:
    if not values:
        return []
    runs = []
    start_band, current = values[0]
    prev_band = start_band
    for band, label in values[1:]:
        if label != current or band != prev_band + 1:
            runs.append({
                "metric": metric, "classification": category, "direction": current,
                "start_band": start_band, "end_band": prev_band,
                "length_bands": prev_band - start_band + 1,
            })
            start_band, current = band, label
        prev_band = band
    runs.append({
        "metric": metric, "classification": category, "direction": current,
        "start_band": start_band, "end_band": prev_band,
        "length_bands": prev_band - start_band + 1,
    })
    return runs


def summarize_range(rows_by_band: dict[int, dict], start: int, end: int) -> dict:
    chosen = [rows_by_band[b] for b in range(start, end + 1)]
    result = {
        "start_band": start,
        "end_band": end,
        "length_bands": len(chosen),
        "mean_aggregate_rmse_improvement_pct": statistics.mean(
            r["aggregate_rmse_improvement_pct"] for r in chosen
        ),
        "median_aggregate_rmse_improvement_pct": statistics.median(
            r["aggregate_rmse_improvement_pct"] for r in chosen
        ),
        "regional_aggregate_rmse_better_bands": sum(
            r["aggregate_rmse_improvement_pct"] > 0 for r in chosen
        ),
        "global_aggregate_rmse_better_bands": sum(
            r["aggregate_rmse_improvement_pct"] < 0 for r in chosen
        ),
        "mean_tile_rmse_improvement_pct": statistics.mean(
            r["mean_tile_rmse_improvement_pct"] for r in chosen
        ),
        "mean_tile_mae_improvement_pct": statistics.mean(
            r["mean_tile_mae_improvement_pct"] for r in chosen
        ),
    }
    for tile in EXPECTED_TILES:
        result[f"tile_{tile}_mean_rmse_improvement_pct"] = statistics.mean(
            r[f"tile_{tile}_rmse_improvement_pct"] for r in chosen
        )
        result[f"tile_{tile}_regional_rmse_win_bands"] = sum(
            r[f"tile_{tile}_rmse_improvement_pct"] > 0 for r in chosen
        )
    result["interpretation_note"] = "exploratory_posthoc_range" if (start, end) in FOCUS_RANGES else "data_derived_range"
    return result


def main() -> None:
    print("Matched 66K spectral-band range consistency analysis")
    print(f"Repository: {REPO}")
    print(f"Input:      {INPUT_DIR}")
    print(f"Output:     {OUTPUT_DIR}")

    tile_rows = read_csv(TILE_FILE)
    mean_rows = read_csv(MEAN_FILE)

    values: dict[tuple[int, int, str], dict[str, float]] = {}
    for row in tile_rows:
        tile = int(row["tile"])
        band = int(row["band_index_zero_based"])
        model = row["model"].strip().upper()
        if model not in {"GLOBAL", "REGIONAL"}:
            raise ValueError(f"Unexpected model label: {model}")
        key = (tile, band, model)
        if key in values:
            raise ValueError(f"Duplicate tile/band/model record: {key}")
        values[key] = {"rmse": float(row["rmse"]), "mae": float(row["mae"])}
        if not all(math.isfinite(v) for v in values[key].values()):
            raise ValueError(f"Non-finite metric in record: {key}")

    expected = {(t, b, m) for t in EXPECTED_TILES for b in EXPECTED_BANDS for m in ("GLOBAL", "REGIONAL")}
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(f"Incomplete/unexpected records. Missing={len(missing)}, extra={len(extra)}")

    mean_by_band = {int(row["band_index_zero_based"]): row for row in mean_rows}
    if set(mean_by_band) != set(EXPECTED_BANDS):
        raise ValueError("mean_band_metrics.csv must contain exactly bands 0 through 127")

    per_band_rows = []
    rows_by_band: dict[int, dict] = {}
    for band in EXPECTED_BANDS:
        g_rmse = [values[(t, band, "GLOBAL")]["rmse"] for t in EXPECTED_TILES]
        r_rmse = [values[(t, band, "REGIONAL")]["rmse"] for t in EXPECTED_TILES]
        g_mae = [values[(t, band, "GLOBAL")]["mae"] for t in EXPECTED_TILES]
        r_mae = [values[(t, band, "REGIONAL")]["mae"] for t in EXPECTED_TILES]
        aggregate_rmse_improvement = improvement(statistics.mean(g_rmse), statistics.mean(r_rmse))
        aggregate_mae_improvement = improvement(statistics.mean(g_mae), statistics.mean(r_mae))
        tile_rmse_imps = [improvement(g, r) for g, r in zip(g_rmse, r_rmse)]
        tile_mae_imps = [improvement(g, r) for g, r in zip(g_mae, r_mae)]

        row = {
            "band_index_zero_based": band,
            "global_mean_tile_rmse": statistics.mean(g_rmse),
            "regional_mean_tile_rmse": statistics.mean(r_rmse),
            "aggregate_rmse_improvement_pct": aggregate_rmse_improvement,
            "global_mean_tile_mae": statistics.mean(g_mae),
            "regional_mean_tile_mae": statistics.mean(r_mae),
            "aggregate_mae_improvement_pct": aggregate_mae_improvement,
            "mean_tile_rmse_improvement_pct": statistics.mean(tile_rmse_imps),
            "median_tile_rmse_improvement_pct": statistics.median(tile_rmse_imps),
            "regional_rmse_tile_wins": sum(v > 0 for v in tile_rmse_imps),
            "global_rmse_tile_wins": sum(v < 0 for v in tile_rmse_imps),
            "regional_mae_tile_wins": sum(v > 0 for v in tile_mae_imps),
            "global_mae_tile_wins": sum(v < 0 for v in tile_mae_imps),
            "mean_tile_mae_improvement_pct": statistics.mean(tile_mae_imps),
        }
        for tile, rmse_imp, mae_imp in zip(EXPECTED_TILES, tile_rmse_imps, tile_mae_imps):
            row[f"tile_{tile}_rmse_improvement_pct"] = rmse_imp
            row[f"tile_{tile}_mae_improvement_pct"] = mae_imp
        row["aggregate_rmse_direction"] = sign_label(aggregate_rmse_improvement)
        row["rmse_tile_vote"] = (
            "REGIONAL_3_OR_4" if row["regional_rmse_tile_wins"] >= 3 else
            "GLOBAL_3_OR_4" if row["global_rmse_tile_wins"] >= 3 else "MIXED"
        )
        per_band_rows.append(row)
        rows_by_band[band] = row

        # Validate that the previous aggregate CSV agrees with our recomputed means.
        mean_row = mean_by_band[band]
        for col, calc in [
            ("global_mean_tile_rmse", row["global_mean_tile_rmse"]),
            ("regional_mean_tile_rmse", row["regional_mean_tile_rmse"]),
            ("global_mean_tile_mae", row["global_mean_tile_mae"]),
            ("regional_mean_tile_mae", row["regional_mean_tile_mae"]),
        ]:
            saved = float(mean_row[col])
            if not math.isclose(saved, calc, rel_tol=1e-7, abs_tol=1e-12):
                raise ValueError(f"Aggregate CSV mismatch at band {band}, column {col}: {saved} vs {calc}")

    per_band_fields = list(per_band_rows[0].keys())
    write_csv(OUTPUT_DIR / "band_range_consistency.csv", per_band_rows, per_band_fields)

    run_rows = []
    for metric, value_key in [("RMSE", "aggregate_rmse_direction"), ("RMSE", "rmse_tile_vote")]:
        seq = [(b, rows_by_band[b][value_key]) for b in EXPECTED_BANDS]
        run_rows.extend(contiguous_runs(seq, value_key, metric))
    for metric in ["RMSE", "MAE"]:
        for tile in EXPECTED_TILES:
            key = f"tile_{tile}_{metric.lower()}_improvement_pct"
            seq = [(b, sign_label(rows_by_band[b][key])) for b in EXPECTED_BANDS]
            run_rows.extend(contiguous_runs(seq, f"TILE_{tile}_SIGN", metric))
    write_csv(
        OUTPUT_DIR / "band_contiguous_runs.csv",
        run_rows,
        ["metric", "classification", "direction", "start_band", "end_band", "length_bands"],
    )

    range_rows = [summarize_range(rows_by_band, start, end) for start, end in FOCUS_RANGES]
    write_csv(OUTPUT_DIR / "band_focus_range_summary.csv", range_rows, list(range_rows[0].keys()))

    rmse_aggregate = [r["aggregate_rmse_improvement_pct"] for r in per_band_rows]
    mae_aggregate = [r["aggregate_mae_improvement_pct"] for r in per_band_rows]
    tile_rmse_vote_counts = {
        "regional_3_or_4": sum(r["regional_rmse_tile_wins"] >= 3 for r in per_band_rows),
        "global_3_or_4": sum(r["global_rmse_tile_wins"] >= 3 for r in per_band_rows),
        "mixed": sum(r["rmse_tile_vote"] == "MIXED" for r in per_band_rows),
    }

    def runs_for(metric: str, key: str) -> list[dict]:
        seq = [(b, rows_by_band[b][key]) for b in EXPECTED_BANDS]
        return contiguous_runs(seq, key, metric)

    report_lines = [
        "# Matched 66K Spectral Band-Range Consistency Analysis",
        "",
        f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Method",
        "",
        "- Uses the saved per-tile RMSE/MAE values for four tiles and 128 bands.",
        "- Per-tile improvement is `(Global error - Regional error) / Global error × 100`; positive values favour Regional SADS.",
        "- Aggregate per-band RMSE/MAE is calculated from the arithmetic mean across the four equally sized tiles.",
        "- Tile vote counts compare each tile independently at each band.",
        "- Band indices are zero-based (0–127); no wavelength mapping is assumed.",
        "",
        "## Aggregate spectral consistency",
        "",
        f"- Regional has lower aggregate RMSE in {sum(v > 0 for v in rmse_aggregate)}/128 bands; Global in {sum(v < 0 for v in rmse_aggregate)}/128.",
        f"- Mean aggregate RMSE improvement across bands: {statistics.mean(rmse_aggregate):+.6f}%.",
        f"- Regional has lower aggregate MAE in {sum(v > 0 for v in mae_aggregate)}/128 bands; Global in {sum(v < 0 for v in mae_aggregate)}/128.",
        f"- Mean aggregate MAE improvement across bands: {statistics.mean(mae_aggregate):+.6f}%.",
        f"- At least 3 of 4 tiles favour Regional RMSE in {tile_rmse_vote_counts['regional_3_or_4']} bands; at least 3 favour Global in {tile_rmse_vote_counts['global_3_or_4']} bands; {tile_rmse_vote_counts['mixed']} bands are split.",
        "",
        "## Contiguous aggregate RMSE runs (minimum length 3)",
        "",
    ]
    agg_runs = [r for r in contiguous_runs(
        [(b, rows_by_band[b]["aggregate_rmse_direction"]) for b in EXPECTED_BANDS],
        "aggregate_rmse_direction", "RMSE") if r["length_bands"] >= 3]
    if agg_runs:
        for run in agg_runs:
            report_lines.append(
                f"- Bands {run['start_band']}–{run['end_band']} ({run['length_bands']} bands): {run['direction']}"
            )
    else:
        report_lines.append("- No contiguous directional run of at least three bands.")

    report_lines += ["", "## Per-tile RMSE results across bands", "", "| Tile | Regional RMSE wins | Global RMSE wins | Mean RMSE improvement |", "|---:|---:|---:|---:|"]
    for tile in EXPECTED_TILES:
        vals = [rows_by_band[b][f"tile_{tile}_rmse_improvement_pct"] for b in EXPECTED_BANDS]
        report_lines.append(
            f"| {tile} | {sum(v > 0 for v in vals)}/128 | {sum(v < 0 for v in vals)}/128 | {statistics.mean(vals):+.6f}% |"
        )

    report_lines += [
        "", "## Exploratory focus ranges", "",
        "The ranges below were selected after inspecting the earlier top/bottom aggregate-band list. They are post-hoc checks, not pre-registered findings and not independent statistical tests.",
        "",
        "| Band range | Aggregate RMSE improvement | Regional aggregate wins | Mean tile RMSE improvement | Tile-wise RMSE improvement |",
        "|---:|---:|---:|---:|---|"]
    for r in range_rows:
        tile_detail = "; ".join(
            f"T{t}: {r[f'tile_{t}_mean_rmse_improvement_pct']:+.4f}% ({r[f'tile_{t}_regional_rmse_win_bands']}/{r['length_bands']})"
            for t in EXPECTED_TILES
        )
        report_lines.append(
            f"| {r['start_band']}–{r['end_band']} | {r['mean_aggregate_rmse_improvement_pct']:+.6f}% | {r['regional_aggregate_rmse_better_bands']}/{r['length_bands']} | {r['mean_tile_rmse_improvement_pct']:+.6f}% | {tile_detail} |"
        )

    report_lines += [
        "", "## Interpretation cautions", "",
        "- A positive band-level improvement is descriptive, not evidence of statistical significance.",
        "- Bands are correlated spectral channels; adjacent bands are not independent observations.",
        "- The four tiles are regions from one Chikusei scene, not four independent scenes.",
        "- Contiguous ranges are exploratory and must not be interpreted as physical wavelength/material ranges without verified wavelength metadata.",
        "- The two focus ranges are post-hoc and should not be used as confirmatory evidence.",
        "", "## Generated files", "",
        "- `band_range_consistency.csv`: one row per band with aggregate values and all four tile-level improvements.",
        "- `band_contiguous_runs.csv`: contiguous sign/vote runs for aggregate and tile-wise RMSE/MAE.",
        "- `band_focus_range_summary.csv`: descriptive summaries for the two exploratory ranges.",
        "- `band_range_consistency_summary.json`: input hashes, definitions, and summary statistics.",
    ]
    (OUTPUT_DIR / "band_range_consistency_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "matched_66k_spectral_band_range_consistency",
        "inputs": {
            TILE_FILE.name: {"sha256": sha256(TILE_FILE), "rows": len(tile_rows)},
            MEAN_FILE.name: {"sha256": sha256(MEAN_FILE), "rows": len(mean_rows)},
        },
        "definitions": {
            "improvement_pct": "(global_error - regional_error) / global_error * 100; positive favours Regional",
            "aggregate_per_band": "arithmetic mean across four equal-sized tiles",
            "band_indexing": "zero-based 0 through 127; wavelength metadata not assumed",
            "focus_ranges": "post-hoc exploratory ranges, not confirmatory tests",
        },
        "aggregate_rmse": {
            "regional_better_bands": sum(v > 0 for v in rmse_aggregate),
            "global_better_bands": sum(v < 0 for v in rmse_aggregate),
            "mean_improvement_pct": statistics.mean(rmse_aggregate),
            "median_improvement_pct": statistics.median(rmse_aggregate),
        },
        "aggregate_mae": {
            "regional_better_bands": sum(v > 0 for v in mae_aggregate),
            "global_better_bands": sum(v < 0 for v in mae_aggregate),
            "mean_improvement_pct": statistics.mean(mae_aggregate),
            "median_improvement_pct": statistics.median(mae_aggregate),
        },
        "per_band_tile_vote_counts_rmse": tile_rmse_vote_counts,
        "tile_rmse_band_wins": {
            str(t): sum(rows_by_band[b][f"tile_{t}_rmse_improvement_pct"] > 0 for b in EXPECTED_BANDS)
            for t in EXPECTED_TILES
        },
        "focus_ranges": range_rows,
        "validation": {
            "tile_records_expected": 4 * 128 * 2,
            "tile_records_read": len(tile_rows),
            "mean_file_rows": len(mean_rows),
            "aggregate_file_consistency": "passed",
        },
    }
    (OUTPUT_DIR / "band_range_consistency_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Optional compact heatmap. Numerical CSV/JSON/Markdown outputs do not depend on matplotlib.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        heat = np.array([
            [rows_by_band[b][f"tile_{t}_rmse_improvement_pct"] for b in EXPECTED_BANDS]
            for t in EXPECTED_TILES
        ], dtype=float)
        lim = float(np.nanpercentile(np.abs(heat), 98))
        lim = max(lim, 1e-9)
        fig, ax = plt.subplots(figsize=(15, 4.5))
        im = ax.imshow(heat, aspect="auto", interpolation="nearest", cmap="RdBu", vmin=-lim, vmax=lim)
        ax.set_yticks(range(4), [f"Tile {t}" for t in EXPECTED_TILES])
        ax.set_xlabel("Zero-based spectral band index")
        ax.set_title("Regional RMSE improvement by tile and spectral band")
        ax.set_xticks(range(0, 128, 8))
        ax.grid(False)
        fig.colorbar(im, ax=ax, label="Regional RMSE improvement (%)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "band_range_consistency_heatmap.png", dpi=180)
        plt.close(fig)
        print("Optional plot: band_range_consistency_heatmap.png")
    except ImportError:
        print("Matplotlib unavailable; numerical outputs were still generated.")

    print("\nArtifacts written:")
    for name in [
        "band_range_consistency.csv",
        "band_contiguous_runs.csv",
        "band_focus_range_summary.csv",
        "band_range_consistency_report.md",
        "band_range_consistency_summary.json",
    ]:
        path = OUTPUT_DIR / name
        print(f"  {path.relative_to(REPO)} ({path.stat().st_size} bytes)")
    print("\nAggregate band results:")
    print(f"  RMSE: Regional better={sum(v > 0 for v in rmse_aggregate)}/128; Global better={sum(v < 0 for v in rmse_aggregate)}/128; mean improvement={statistics.mean(rmse_aggregate):+.6f}%")
    print(f"  MAE:  Regional better={sum(v > 0 for v in mae_aggregate)}/128; Global better={sum(v < 0 for v in mae_aggregate)}/128; mean improvement={statistics.mean(mae_aggregate):+.6f}%")
    print("  At least 3/4 tiles favour Regional (RMSE):", tile_rmse_vote_counts["regional_3_or_4"], "bands")
    print("  At least 3/4 tiles favour Global (RMSE):", tile_rmse_vote_counts["global_3_or_4"], "bands")
    print("\nNo checkpoints, predictions, datasets, training or test configurations were modified.")


if __name__ == "__main__":
    main()
