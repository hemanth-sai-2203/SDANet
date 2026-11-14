#!/usr/bin/env python3
"""Combine evaluate_sdanet_scenes.py outputs into comparison CSV/Markdown files.

Example:
  .venv/bin/python summarize_sdanet_evaluations.py \
    --root results/final_repro \
    --output-dir results/final_repro/summary

The summary directory must be separate from the run folders. Existing outputs
are not overwritten unless --overwrite is explicitly supplied.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = ("ssim", "mpsnr", "sam", "ergas", "CC")
DIRECTIONS = {"ssim": "higher", "mpsnr": "higher", "sam": "lower", "ergas": "lower", "CC": "higher"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/final_repro"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Evaluation root not found: {root}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else root / "summary"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Summary output already exists: {output_dir}. Use --overwrite to replace summary files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(root.rglob("per_scene_metrics.csv"))
    result_files = [path for path in result_files if output_dir not in path.parents]
    if not result_files:
        raise FileNotFoundError(f"No per_scene_metrics.csv files found under {root}")

    run_summaries: list[dict[str, Any]] = []
    per_scene_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []

    for csv_path in result_files:
        rows = read_csv(csv_path)
        if not rows:
            continue
        run_name = rows[0].get("run") or csv_path.parent.name
        for row in rows:
            scene_row: dict[str, Any] = {
                "run": run_name,
                "scene_index": int(row["scene_index"]),
                "scene_id": row["scene_id"],
            }
            for metric in METRICS:
                scene_row[metric] = float(row[metric])
            per_scene_rows.append(scene_row)

        summary: dict[str, Any] = {"run": run_name, "scene_count": len(rows)}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_scene_std"] = statistics.pstdev(values)
        run_summaries.append(summary)

        metadata_path = csv_path.parent / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            run_metadata.append({
                "run": run_name,
                "network_type": metadata.get("network_type"),
                "fine_gate_enabled": metadata.get("fine_gate_enabled"),
                "checkpoint_path": metadata.get("checkpoint_path"),
                "checkpoint_sha256": metadata.get("checkpoint_sha256"),
                "scene_count": metadata.get("scene_count"),
                "device": metadata.get("device"),
                "torch_version": metadata.get("torch_version"),
                "backend_settings": json.dumps(metadata.get("backend_settings", {}), sort_keys=True),
            })

    if not run_summaries:
        raise RuntimeError("No non-empty scene metric files were found.")

    aggregate_fields = ["run", "scene_count"] + [
        name for metric in METRICS for name in (f"{metric}_mean", f"{metric}_scene_std")
    ]
    write_csv(output_dir / "aggregate_comparison.csv", aggregate_fields, run_summaries)

    per_scene_fields = ["run", "scene_index", "scene_id", *METRICS]
    write_csv(output_dir / "per_scene_comparison.csv", per_scene_fields, per_scene_rows)

    if run_metadata:
        write_csv(
            output_dir / "run_metadata.csv",
            ["run", "network_type", "fine_gate_enabled", "checkpoint_path", "checkpoint_sha256",
             "scene_count", "device", "torch_version", "backend_settings"],
            run_metadata,
        )

    lines = [
        "# SDANet evaluation comparison",
        "",
        f"Evaluation folders scanned: `{root}`",
        "",
        "Values are arithmetic means across the evaluated scenes. The accompanying scene standard deviation "
        "describes between-scene variation; it is not a confidence interval.",
        "",
        "| Run | Scenes | SSIM ↑ | MPSNR ↑ | SAM ↓ | ERGAS ↓ | CC ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_summaries:
        lines.append(
            f"| {row['run']} | {row['scene_count']} | "
            f"{fmt(row['ssim_mean'])} | {fmt(row['mpsnr_mean'])} | "
            f"{fmt(row['sam_mean'])} | {fmt(row['ergas_mean'])} | {fmt(row['CC_mean'])} |"
        )

    lines += [
        "",
        "## Interpretation notes",
        "",
        "- Higher is better for SSIM, MPSNR and CC; lower is better for SAM and ERGAS.",
        "- Compare Fine-Gate Control and Treatment as the matched ablation pair only after confirming their "
        "checkpoint provenance and identical evaluation settings.",
        "- Comparisons against the original baseline or longer-trained RegionalFull checkpoints also reflect "
        "differences in architecture and/or training duration; do not attribute those differences solely to the fine gate.",
        "- Small metric differences should be interpreted cautiously, especially with only four test scenes and "
        "previous exposure of this split during model development.",
        "",
        "## Files",
        "",
        "- `aggregate_comparison.csv`: aggregate metrics and between-scene standard deviations.",
        "- `per_scene_comparison.csv`: all per-scene metric values.",
        "- `run_metadata.csv`: checkpoint hashes and evaluation environment, when metadata files are present.",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Found {len(run_summaries)} evaluation runs:")
    for row in run_summaries:
        print(
            f"- {row['run']}: {row['scene_count']} scenes; "
            + ", ".join(f"{metric}={row[f'{metric}_mean']:.6f}" for metric in METRICS)
        )
    print(f"\nSummary written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
