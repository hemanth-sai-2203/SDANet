#!/usr/bin/env python3
"""Validation-only sweep of fixed scalar blends versus a learned regional gate.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python evaluate_validation_fixed_blends.py --check-only
  PYTHONPATH="$PWD" .venv/bin/python evaluate_validation_fixed_blends.py

The script evaluates only the configured `evals` validation directory. It loads
both frozen SDANet checkpoints and the existing compact fusion gate. It never
opens Chikusei_test.mat and never uses validation GT to generate predictions
or gate weights. GT is used only for metrics after predictions are complete.

Fixed alpha means:
  prediction = alpha * RegionalFull + (1-alpha) * IndependentUpsampler

The predeclared alpha grid is descriptive. Choosing the best alpha on this
same validation split is model selection, not an unbiased final evaluation.
The 62 patches are spatially correlated and must not be treated as IID.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.metrics import calculate_metric

from train_dynamic_regional_fusion_compact import (
    METRIC_NAMES,
    REGION_NAMES,
    DynamicRegionalFusionGate,
    apply_regional_fusion,
    build_frozen_model,
    build_gate_features_and_oracle,
    choose_dataset_entry,
    get_sample_name,
    load_yaml,
    make_dataset,
    resolve_path,
    set_reproducible_settings,
    tensor_to_hwc,
)

DEFAULT_ALPHAS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.0)
ALL_METRICS = (*METRIC_NAMES, "mse", "mae")
DIRECTIONS = {
    "ssim": "higher",
    "mpsnr": "higher",
    "sam": "lower",
    "ergas": "lower",
    "CC": "higher",
    "mse": "lower",
    "mae": "lower",
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_gate(checkpoint_path: Path, device: torch.device) -> tuple[DynamicRegionalFusionGate, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Gate checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"state_dict", "input_dim", "hidden_dim", "region_names"}
    if not isinstance(payload, dict):
        raise TypeError("Gate checkpoint must be a dictionary")
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"Gate checkpoint missing fields: {sorted(missing)}")
    if tuple(payload["region_names"]) != tuple(REGION_NAMES):
        raise ValueError(
            f"Gate region order mismatch: {payload['region_names']} != {REGION_NAMES}"
        )
    gate = DynamicRegionalFusionGate(
        input_dim=int(payload["input_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    ).to(device)
    gate.load_state_dict(payload["state_dict"], strict=True)
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate, payload


def image_metrics(prediction: torch.Tensor, ground_truth: torch.Tensor,
                  metric_options: dict[str, Any]) -> dict[str, float]:
    pred_hwc = tensor_to_hwc(prediction)
    gt_hwc = tensor_to_hwc(ground_truth)
    if pred_hwc.shape != gt_hwc.shape:
        raise RuntimeError(
            f"Prediction/GT shape mismatch: {pred_hwc.shape} vs {gt_hwc.shape}"
        )
    if not np.isfinite(pred_hwc).all():
        raise FloatingPointError("Non-finite fused prediction")
    metric_data = {"img": pred_hwc, "img2": gt_hwc}
    values: dict[str, float] = {}
    for name in METRIC_NAMES:
        value = float(calculate_metric(metric_data, metric_options[name]))
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite {name} value: {value}")
        values[name] = value
    residual = pred_hwc.astype(np.float64) - gt_hwc.astype(np.float64)
    values["mse"] = float(np.mean(residual * residual))
    values["mae"] = float(np.mean(np.abs(residual)))
    return values


def aggregate(per_sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = list(dict.fromkeys(row["model"] for row in per_sample_rows))
    baseline = {
        row["sample_id"]: row
        for row in per_sample_rows
        if row["model"] == "RegionalFull"
    }
    output: list[dict[str, Any]] = []
    for model_name in models:
        rows = [row for row in per_sample_rows if row["model"] == model_name]
        result: dict[str, Any] = {"model": model_name, "sample_count": len(rows)}
        for metric in ALL_METRICS:
            values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std_across_samples"] = float(values.std(ddof=0))
            if model_name == "RegionalFull":
                result[f"{metric}_wins_vs_regional"] = 0
                result[f"{metric}_mean_delta_vs_regional"] = 0.0
            else:
                paired = [(row, baseline[row["sample_id"]]) for row in rows]
                deltas = np.asarray(
                    [float(a[metric]) - float(b[metric]) for a, b in paired],
                    dtype=np.float64,
                )
                if DIRECTIONS[metric] == "higher":
                    wins = int(np.sum(deltas > 0))
                else:
                    wins = int(np.sum(deltas < 0))
                result[f"{metric}_wins_vs_regional"] = wins
                result[f"{metric}_mean_delta_vs_regional"] = float(deltas.mean())
        output.append(result)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regional-config", type=Path,
        default=Path("options/test/HSI/test_SDANetRegionalFullChux4.yml"),
    )
    parser.add_argument(
        "--independent-config", type=Path,
        default=Path("options/test/HSI/test_SDANetIndependentUpsamplerFinetuneChux4.yml"),
    )
    parser.add_argument(
        "--data-config", type=Path,
        default=Path("options/train/HSI/SDANetRegionalFullChux4.yml"),
    )
    parser.add_argument(
        "--gate-checkpoint", type=Path,
        default=Path("results/dynamic_regional_fusion_compact/dynamic_regional_fusion_gate.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/validation_fixed_blend_sweep"),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS),
        help="Fixed weights on RegionalFull; default is a predeclared broad sweep.",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Load checkpoints and evaluate one validation sample; create no reports.",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if not (repo_root / "basicsr").is_dir():
        raise RuntimeError("Run from SDANet repository root (the folder containing basicsr/).")
    if not (repo_root / "train_dynamic_regional_fusion_compact.py").is_file():
        raise FileNotFoundError("Required helper script train_dynamic_regional_fusion_compact.py is missing.")

    alphas = [float(a) for a in args.alphas]
    if not alphas or any(not np.isfinite(a) or a < 0.0 or a > 1.0 for a in alphas):
        raise ValueError("Every --alphas value must be finite and in [0,1].")
    if len(set(alphas)) != len(alphas):
        raise ValueError("Duplicate fixed alpha values are not allowed.")

    regional_config_path = resolve_path(args.regional_config, repo_root)
    independent_config_path = resolve_path(args.independent_config, repo_root)
    data_config_path = resolve_path(args.data_config, repo_root)
    gate_checkpoint_path = resolve_path(args.gate_checkpoint, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)

    regional_config = load_yaml(regional_config_path)
    independent_config = load_yaml(independent_config_path)
    data_config = load_yaml(data_config_path)
    if int(regional_config.get("scale", 0)) != int(independent_config.get("scale", -1)):
        raise ValueError("Backbone scale settings differ")
    if regional_config.get("path", {}).get("strict_load_g", True) is not True:
        raise ValueError("RegionalFull config must use strict_load_g: true")
    if independent_config.get("path", {}).get("strict_load_g", True) is not True:
        raise ValueError("Independent config must use strict_load_g: true")

    val_keys = [key for key in data_config.get("datasets", {})
                if str(key).split("_", 1)[0] == "val"]
    if len(val_keys) != 1:
        raise ValueError(f"Expected one validation dataset entry, found {val_keys}")
    val_options = choose_dataset_entry(data_config, val_keys[0])
    val_dataset, val_root = make_dataset(
        val_options, "val", int(regional_config["scale"]), repo_root
    )
    if val_root.name.lower() != "evals" or "test" in str(val_root).lower():
        raise RuntimeError(f"Refusing non-validation data path: {val_root}")
    if len(val_dataset) != 62:
        raise RuntimeError(f"Expected the established 62 validation patches, found {len(val_dataset)}")

    metric_options = regional_config.get("val", {}).get("metrics") or {}
    missing_metrics = [name for name in METRIC_NAMES if name not in metric_options]
    if missing_metrics:
        raise ValueError(f"RegionalFull test config lacks metrics: {missing_metrics}")

    reproducibility = set_reproducible_settings(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    print("Loading frozen models and compact gate...", flush=True)
    model_a, _, meta_a = build_frozen_model(
        regional_config_path, device, repo_root, "RegionalFull"
    )
    model_b, _, meta_b = build_frozen_model(
        independent_config_path, device, repo_root, "IndependentUpsampler"
    )
    gate, gate_payload = load_gate(gate_checkpoint_path, device)
    expected_dim = int(regional_config["network_g"].get("n_colors", 128)) * 6 + 4
    if gate.input_dim != expected_dim:
        raise ValueError(f"Gate input dimension {gate.input_dim} != expected {expected_dim}")

    print(f"RegionalFull: {meta_a['checkpoint_path']}", flush=True)
    print(f"Independent:  {meta_b['checkpoint_path']}", flush=True)
    print(f"Gate:         {gate_checkpoint_path} (hidden={gate.hidden_dim})", flush=True)
    print(f"Validation:   {val_root} ({len(val_dataset)} samples)", flush=True)
    print(f"Fixed alphas: {alphas}", flush=True)
    print(f"Device:       {device}", flush=True)
    print("Test MAT file is not opened by this script.", flush=True)

    if args.check_only:
        sample = val_dataset[0]
        sample_id = Path(str(sample.get("lq_path", "eval_0"))).stem
        lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
        lms = sample["lms"].unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = sample["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            pred_a = model_a(lq, lms)
            pred_b = model_b(lq, lms)
            features, _ = build_gate_features_and_oracle(pred_a, pred_b, None)
            gate_alpha = gate(features)
            dynamic = apply_regional_fusion(pred_a, pred_b, gate_alpha)
        if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
            raise RuntimeError(f"{sample_id}: output shape mismatch")
        outputs = {"RegionalFull": pred_a, "IndependentUpsampler": pred_b,
                   "DynamicRegionalGate": dynamic}
        for alpha in alphas:
            outputs[f"FixedAlpha_{alpha:.3f}"] = alpha * pred_a + (1.0 - alpha) * pred_b
        checked = {}
        for name, prediction in outputs.items():
            values = image_metrics(prediction, gt, metric_options)
            checked[name] = {key: round(value, 7) for key, value in values.items()}
        print(f"CHECK ONLY PASSED: {sample_id}; metrics={json.dumps(checked, sort_keys=True)}", flush=True)
        print("One validation sample only. No test data accessed; no reports created.", flush=True)
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    per_sample_rows: list[dict[str, Any]] = []
    gate_weight_rows: list[dict[str, Any]] = []
    model_a.eval()
    model_b.eval()
    gate.eval()

    for index, batch in enumerate(loader):
        sample_id = get_sample_name(batch, index)
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        lms = batch["lms"].to(device=device, dtype=torch.float32)
        gt = batch["gt"].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            pred_a = model_a(lq, lms)
            pred_b = model_b(lq, lms)
            if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
                raise RuntimeError(
                    f"{sample_id}: A={tuple(pred_a.shape)}, B={tuple(pred_b.shape)}, GT={tuple(gt.shape)}"
                )
            features, _ = build_gate_features_and_oracle(pred_a, pred_b, None)
            gate_alpha = gate(features)
            dynamic_prediction = apply_regional_fusion(pred_a, pred_b, gate_alpha)

            predictions: list[tuple[str, torch.Tensor]] = [
                ("RegionalFull", pred_a),
                ("IndependentUpsampler", pred_b),
            ]
            for alpha in alphas:
                predictions.append((f"FixedAlpha_{alpha:.3f}",
                                    alpha * pred_a + (1.0 - alpha) * pred_b))
            predictions.append(("DynamicRegionalGate", dynamic_prediction))

        for model_name, prediction in predictions:
            values = image_metrics(prediction, gt, metric_options)
            per_sample_rows.append({"sample_id": sample_id, "model": model_name, **values})

        alpha_values = gate_alpha.squeeze(0).detach().cpu().numpy().astype(np.float64)
        weight_row: dict[str, Any] = {"sample_id": sample_id}
        for region_index, region_name in enumerate(REGION_NAMES):
            weight_row[region_name] = float(alpha_values[region_index])
        gate_weight_rows.append(weight_row)

        print(
            f"Validation [{index + 1}/{len(val_dataset)}] {sample_id}: "
            f"gate_alpha=" + ",".join(f"{value:.3f}" for value in alpha_values),
            flush=True,
        )
        del lq, lms, gt, pred_a, pred_b, features, gate_alpha, dynamic_prediction, predictions
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = aggregate(per_sample_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "validation_per_sample_metrics.csv", per_sample_rows)
    write_csv(output_dir / "validation_blend_summary.csv", summary_rows)
    write_csv(output_dir / "validation_dynamic_gate_weights.csv", gate_weight_rows)

    # Compact alpha-only table, sorted in the original predeclared grid order.
    alpha_summary = [row for row in summary_rows if row["model"].startswith("FixedAlpha_")]
    write_csv(output_dir / "fixed_alpha_sweep_summary.csv", alpha_summary)

    metadata = {
        "experiment": "validation_only_fixed_scalar_blend_sweep",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "regional_model": meta_a,
        "independent_model": meta_b,
        "gate_checkpoint": str(gate_checkpoint_path),
        "gate_checkpoint_sha256": sha256_file(gate_checkpoint_path),
        "gate_hidden_dim": gate.hidden_dim,
        "validation_dataset": str(val_root),
        "validation_sample_count": len(val_dataset),
        "fixed_alpha_values_on_regional_full": alphas,
        "fixed_blend_definition": "alpha * RegionalFull + (1-alpha) * IndependentUpsampler",
        "metrics": list(ALL_METRICS),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reproducibility_settings": reproducibility,
        "test_data_accessed": False,
        "ground_truth_usage": "metrics only; never passed into either backbone or dynamic gate",
        "model_selection_caveat": (
            "The alpha grid is evaluated on the same validation split. Selecting the best alpha here is "
            "validation model selection and does not provide an unbiased final generalization estimate."
        ),
        "statistical_caveat": (
            "The 62 validation patches are extracted on an overlapping spatial grid; patch metrics are "
            "spatially correlated and are descriptive, not IID samples or a formal significance test."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nValidation blend summary:", flush=True)
    for row in summary_rows:
        print(
            f"{row['model']:24s} n={row['sample_count']:2d} | "
            f"SSIM={row['ssim_mean']:.6f} | MPSNR={row['mpsnr_mean']:.6f} | "
            f"SAM={row['sam_mean']:.6f} | ERGAS={row['ergas_mean']:.6f} | "
            f"CC={row['CC_mean']:.6f} | MSE={row['mse_mean']:.8f} | "
            f"MAE={row['mae_mean']:.7f}",
            flush=True,
        )
    print(f"\nReports saved to: {output_dir}", flush=True)
    print("Validation-only comparison complete; no test MAT file was opened.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
