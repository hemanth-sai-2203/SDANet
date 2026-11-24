#!/usr/bin/env python3
"""Validation-only regional error-complementarity audit for two frozen SDANet models.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python audit_validation_complementarity.py --check-only
  PYTHONPATH="$PWD" .venv/bin/python audit_validation_complementarity.py

This script deliberately reads ONLY the `evals` dataset configured in
SDANetRegionalFullChux4.yml. It loads model architectures/checkpoints from the
two test YAMLs but never opens their Chikusei_test MAT file paths.

The validation ground truth is used only to calculate diagnostic errors and
oracle blending coefficients. It is never passed into either model or the
fusion gate. Results are descriptive; validation patches are spatially
correlated and must not be treated as IID observations.
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

from train_dynamic_regional_fusion_compact import (
    REGION_NAMES,
    DynamicRegionalFusionGate,
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


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_gate(checkpoint_path: Path, device: torch.device) -> tuple[DynamicRegionalFusionGate, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fusion gate checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"state_dict", "input_dim", "hidden_dim", "region_names"}
    if not isinstance(checkpoint, dict):
        raise TypeError("Fusion gate checkpoint must contain a dictionary")
    missing = required.difference(checkpoint)
    if missing:
        raise KeyError(f"Fusion gate checkpoint is missing fields: {sorted(missing)}")
    if tuple(checkpoint["region_names"]) != tuple(REGION_NAMES):
        raise ValueError(
            f"Gate region order mismatch: checkpoint={checkpoint['region_names']}, "
            f"expected={REGION_NAMES}"
        )

    gate = DynamicRegionalFusionGate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    gate.load_state_dict(checkpoint["state_dict"], strict=True)
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    return gate, checkpoint


def region_slices(height: int, width: int):
    if height % 2 or width % 2:
        raise ValueError(f"Prediction dimensions must be even, got {height}x{width}")
    half_h, half_w = height // 2, width // 2
    return (
        (slice(0, half_h), slice(0, half_w)),
        (slice(0, half_h), slice(half_w, width)),
        (slice(half_h, height), slice(0, half_w)),
        (slice(half_h, height), slice(half_w, width)),
    )


def region_diagnostics(
    prediction_a: np.ndarray,
    prediction_b: np.ndarray,
    ground_truth: np.ndarray,
    gate_alpha: np.ndarray,
    sample_id: str,
) -> list[dict[str, Any]]:
    """Calculate paired per-region errors and a validation-only convex oracle."""
    if prediction_a.shape != prediction_b.shape or prediction_a.shape != ground_truth.shape:
        raise ValueError(
            f"Shape mismatch: A={prediction_a.shape}, B={prediction_b.shape}, "
            f"GT={ground_truth.shape}"
        )
    if prediction_a.ndim != 3:
        raise ValueError(f"Expected HWC arrays, got {prediction_a.shape}")
    if len(gate_alpha) != 4:
        raise ValueError(f"Expected four gate weights, got {gate_alpha}")

    height, width, _ = ground_truth.shape
    slices = region_slices(height, width)
    rows: list[dict[str, Any]] = []

    for region_index, (ys, xs) in enumerate(slices):
        a = prediction_a[ys, xs, :].astype(np.float64, copy=False)
        b = prediction_b[ys, xs, :].astype(np.float64, copy=False)
        gt = ground_truth[ys, xs, :].astype(np.float64, copy=False)

        error_a = a - gt
        error_b = b - gt
        mse_a = float(np.mean(np.square(error_a)))
        mse_b = float(np.mean(np.square(error_b)))
        mae_a = float(np.mean(np.abs(error_a)))
        mae_b = float(np.mean(np.abs(error_b)))

        delta = a - b
        target_delta = gt - b
        denominator = float(np.sum(np.square(delta)))
        if denominator <= 1e-12:
            oracle_alpha = 0.5
        else:
            oracle_alpha = float(np.clip(np.sum(delta * target_delta) / denominator, 0.0, 1.0))
        oracle_prediction = oracle_alpha * a + (1.0 - oracle_alpha) * b
        oracle_mse = float(np.mean(np.square(oracle_prediction - gt)))

        norm_product = float(np.linalg.norm(error_a.ravel()) * np.linalg.norm(error_b.ravel()))
        error_cosine = (
            float(np.dot(error_a.ravel(), error_b.ravel()) / norm_product)
            if norm_product > 1e-12 else float("nan")
        )

        # Compare absolute spectral error at each spatial pixel after averaging
        # over 128 bands, as well as over all band-pixel scalar elements.
        pixel_error_a = np.mean(np.abs(error_a), axis=2)
        pixel_error_b = np.mean(np.abs(error_b), axis=2)
        pixel_b_better = float(np.mean(pixel_error_b < pixel_error_a))
        element_b_better = float(np.mean(np.abs(error_b) < np.abs(error_a)))

        rows.append({
            "sample_id": sample_id,
            "region": REGION_NAMES[region_index],
            "mse_regional_full": mse_a,
            "mse_independent": mse_b,
            "mse_independent_minus_regional": mse_b - mse_a,
            "mse_change_percent_vs_regional": 100.0 * (mse_b - mse_a) / max(mse_a, 1e-12),
            "mae_regional_full": mae_a,
            "mae_independent": mae_b,
            "independent_better_by_mse": int(mse_b < mse_a),
            "fraction_spatial_pixels_independent_lower_mae": pixel_b_better,
            "fraction_band_pixel_elements_independent_lower_abs_error": element_b_better,
            "residual_error_cosine": error_cosine,
            "oracle_alpha_regional_full": oracle_alpha,
            "oracle_blend_mse": oracle_mse,
            "oracle_mse_gain_percent_vs_regional": 100.0 * (mse_a - oracle_mse) / max(mse_a, 1e-12),
            "predicted_gate_alpha": float(gate_alpha[region_index]),
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for region in (*REGION_NAMES, "ALL_REGIONS"):
        selected = rows if region == "ALL_REGIONS" else [r for r in rows if r["region"] == region]
        if not selected:
            continue
        def mean(key: str) -> float:
            vals = np.asarray([r[key] for r in selected], dtype=np.float64)
            return float(np.nanmean(vals))
        def median(key: str) -> float:
            vals = np.asarray([r[key] for r in selected], dtype=np.float64)
            return float(np.nanmedian(vals))

        output.append({
            "region": region,
            "sample_region_count": len(selected),
            "regional_full_mse_mean": mean("mse_regional_full"),
            "independent_mse_mean": mean("mse_independent"),
            "mse_difference_mean_independent_minus_regional": mean("mse_independent_minus_regional"),
            "mse_change_percent_mean": mean("mse_change_percent_vs_regional"),
            "independent_better_count": int(sum(r["independent_better_by_mse"] for r in selected)),
            "independent_better_fraction": float(np.mean([r["independent_better_by_mse"] for r in selected])),
            "spatial_pixel_fraction_independent_lower_mae_mean": mean("fraction_spatial_pixels_independent_lower_mae"),
            "band_pixel_fraction_independent_lower_abs_error_mean": mean("fraction_band_pixel_elements_independent_lower_abs_error"),
            "residual_error_cosine_mean": mean("residual_error_cosine"),
            "oracle_alpha_mean": mean("oracle_alpha_regional_full"),
            "oracle_alpha_median": median("oracle_alpha_regional_full"),
            "oracle_alpha_std": float(np.nanstd([r["oracle_alpha_regional_full"] for r in selected], ddof=0)),
            "oracle_blend_mse_mean": mean("oracle_blend_mse"),
            "oracle_mse_gain_percent_mean": mean("oracle_mse_gain_percent_vs_regional"),
            "predicted_gate_alpha_mean": mean("predicted_gate_alpha"),
            "predicted_gate_alpha_std": float(np.nanstd([r["predicted_gate_alpha"] for r in selected], ddof=0)),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-config", type=Path,
                        default=Path("options/test/HSI/test_SDANetRegionalFullChux4.yml"))
    parser.add_argument("--independent-config", type=Path,
                        default=Path("options/test/HSI/test_SDANetIndependentUpsamplerFinetuneChux4.yml"))
    parser.add_argument("--data-config", type=Path,
                        default=Path("options/train/HSI/SDANetRegionalFullChux4.yml"))
    parser.add_argument("--gate-checkpoint", type=Path,
                        default=Path("results/dynamic_regional_fusion_compact/dynamic_regional_fusion_gate.pth"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/validation_complementarity_audit"))
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--check-only", action="store_true",
                        help="Load both models, the gate and one validation sample; create no reports")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if not (repo_root / "basicsr").is_dir():
        raise RuntimeError("Run this script from the SDANet repository root (the folder containing basicsr/).")
    if not (repo_root / "train_dynamic_regional_fusion_compact.py").is_file():
        raise FileNotFoundError(
            "train_dynamic_regional_fusion_compact.py must be in the repository root because "
            "this audit reuses its verified model/dataset/gate helpers."
        )

    regional_config_path = resolve_path(args.regional_config, repo_root)
    independent_config_path = resolve_path(args.independent_config, repo_root)
    data_config_path = resolve_path(args.data_config, repo_root)
    gate_checkpoint_path = resolve_path(args.gate_checkpoint, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)

    regional_config = load_yaml(regional_config_path)
    independent_config = load_yaml(independent_config_path)
    data_config = load_yaml(data_config_path)
    if int(regional_config.get("scale", 0)) != int(independent_config.get("scale", -1)):
        raise ValueError("RegionalFull and IndependentUpsampler scale settings differ")
    scale = int(regional_config["scale"])

    val_keys = [k for k in data_config.get("datasets", {}) if str(k).split("_", 1)[0] == "val"]
    if len(val_keys) != 1:
        raise ValueError(f"Expected exactly one validation dataset entry; found {val_keys}")
    val_options = choose_dataset_entry(data_config, val_keys[0])
    val_dataset, val_root = make_dataset(val_options, "val", scale, repo_root)

    # Explicitly refuse any dataset path that could identify a test split.
    if "test" in str(val_root).lower() or val_root.name.lower() not in {"evals", "validation", "val"}:
        raise RuntimeError(f"Safety check failed: expected a validation/evals directory, got {val_root}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"Validation dataset is empty: {val_root}")

    reproducibility = set_reproducible_settings(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print("Loading frozen models from the existing strict test configurations...", flush=True)
    model_a, _, meta_a = build_frozen_model(regional_config_path, device, repo_root, "RegionalFull")
    model_b, _, meta_b = build_frozen_model(independent_config_path, device, repo_root, "IndependentUpsampler")
    gate, gate_payload = load_gate(gate_checkpoint_path, device)

    expected_dim = int(regional_config["network_g"].get("n_colors", 128)) * 6 + 4
    if gate.input_dim != expected_dim:
        raise ValueError(f"Gate input dimension {gate.input_dim} != expected {expected_dim}")
    if gate.hidden_dim != 8:
        print(f"WARNING: loaded gate hidden_dim={gate.hidden_dim}, expected compact hidden_dim=8", flush=True)

    print(f"RegionalFull checkpoint: {meta_a['checkpoint_path']}", flush=True)
    print(f"Independent checkpoint:  {meta_b['checkpoint_path']}", flush=True)
    print(f"Gate checkpoint:         {gate_checkpoint_path}", flush=True)
    print(f"Validation data:         {val_root} ({len(val_dataset)} samples)", flush=True)
    print(f"Device:                  {device}", flush=True)
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
            alpha = gate(features)
        if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
            raise RuntimeError(f"Shape mismatch on {sample_id}: A={pred_a.shape}, B={pred_b.shape}, GT={gt.shape}")
        if features.shape != (1, 4, expected_dim) or alpha.shape != (1, 4):
            raise RuntimeError(f"Unexpected feature/alpha shapes: {features.shape}, {alpha.shape}")
        if not torch.isfinite(pred_a).all() or not torch.isfinite(pred_b).all() or not torch.isfinite(alpha).all():
            raise FloatingPointError("Non-finite output detected during preflight")
        print(
            f"CHECK ONLY PASSED: sample={sample_id}, output={tuple(pred_a.shape)}, "
            f"gate_features={tuple(features.shape)}, alpha={alpha.detach().cpu().numpy().round(4).tolist()}",
            flush=True,
        )
        print("No test data accessed. No report files created.", flush=True)
        return 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {output_dir}")

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    per_region_rows: list[dict[str, Any]] = []
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
            features, _ = build_gate_features_and_oracle(pred_a, pred_b, None)
            alpha = gate(features)

        if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
            raise RuntimeError(
                f"{sample_id}: prediction/GT shapes differ: A={tuple(pred_a.shape)}, "
                f"B={tuple(pred_b.shape)}, GT={tuple(gt.shape)}"
            )
        if not torch.isfinite(pred_a).all() or not torch.isfinite(pred_b).all() or not torch.isfinite(alpha).all():
            raise FloatingPointError(f"Non-finite predictions or gate weights for {sample_id}")

        a_hwc = tensor_to_hwc(pred_a)
        b_hwc = tensor_to_hwc(pred_b)
        gt_hwc = tensor_to_hwc(gt)
        alpha_np = alpha.squeeze(0).detach().cpu().numpy().astype(np.float64)
        per_region_rows.extend(region_diagnostics(a_hwc, b_hwc, gt_hwc, alpha_np, sample_id))

        print(
            f"Validation [{index + 1}/{len(val_dataset)}] {sample_id}: "
            f"oracle_alpha=" + ",".join(
                f"{r['oracle_alpha_regional_full']:.3f}" for r in per_region_rows[-4:]
            ) + " | gate_alpha=" + ",".join(f"{x:.3f}" for x in alpha_np),
            flush=True,
        )
        del lq, lms, gt, pred_a, pred_b, features, alpha
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_rows = summarize(per_region_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "validation_per_sample_region_complementarity.csv", per_region_rows)
    write_csv(output_dir / "validation_complementarity_summary.csv", summary_rows)

    metadata = {
        "experiment": "validation_only_regional_error_complementarity_audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repo_root),
        "regional_model": meta_a,
        "independent_model": meta_b,
        "gate_checkpoint": str(gate_checkpoint_path),
        "gate_checkpoint_sha256": sha256_file(gate_checkpoint_path),
        "gate_input_dim": gate.input_dim,
        "gate_hidden_dim": gate.hidden_dim,
        "validation_dataset": str(val_root),
        "validation_sample_count": len(val_dataset),
        "regional_target_count": len(per_region_rows),
        "region_order": list(REGION_NAMES),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reproducibility_settings": reproducibility,
        "validation_ground_truth_usage": "diagnostics only; never passed to either model or fusion gate",
        "test_data_accessed": False,
        "interpretation_caveat": (
            "The validation patches were extracted on an overlapping spatial grid and adjacent grid rows "
            "were buffered; patch metrics and four regions within a patch are correlated. Treat outputs as "
            "descriptive paired diagnostics, not IID samples or a formal significance test."
        ),
        "oracle_alpha_definition": (
            "Per-sample, per-quadrant MSE-optimal convex coefficient; uses validation GT for diagnosis only "
            "and is not a deployable gate prediction."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nValidation complementarity summary:", flush=True)
    for row in summary_rows:
        print(
            f"{row['region']:12s} n={row['sample_region_count']:3d} | "
            f"MSE A={row['regional_full_mse_mean']:.7f} | "
            f"MSE B={row['independent_mse_mean']:.7f} | "
            f"B-better={row['independent_better_count']}/{row['sample_region_count']} | "
            f"oracle alpha={row['oracle_alpha_mean']:.4f}±{row['oracle_alpha_std']:.4f} | "
            f"gate alpha={row['predicted_gate_alpha_mean']:.4f}±{row['predicted_gate_alpha_std']:.4f} | "
            f"oracle gain={row['oracle_mse_gain_percent_mean']:.4f}%",
            flush=True,
        )
    print(f"\nAudit files saved to: {output_dir}", flush=True)
    print("Validation-only audit complete; no test MAT file was opened.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
