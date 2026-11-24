#!/usr/bin/env python3
"""Evaluate the validation-selected fixed 0.80 fusion on the existing Chikusei test tiles.

Compares four frozen inference choices:
  - RegionalFull
  - IndependentUpsampler
  - FixedAlpha_0.800: 0.8 * RegionalFull + 0.2 * IndependentUpsampler
  - DynamicRegionalGate: the previously trained compact gate

The fixed alpha is locked from validation before this evaluation. Test GT is
used only for metrics. The existing four test tiles were previously examined
during development, so this is not a pristine blind evaluation.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python evaluate_fixed080_fusion_test.py --check-only
  PYTHONPATH="$PWD" .venv/bin/python evaluate_fixed080_fusion_test.py

The check-only path loads and validates checkpoints/configurations, then runs
synthetic tensor checks. It does NOT construct or read the test dataset.
Existing output directories are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric
from train_dynamic_regional_fusion import (
    METRIC_NAMES,
    REGION_NAMES,
    DynamicRegionalFusionGate,
    apply_regional_fusion,
    build_frozen_model,
    build_gate_features_and_oracle,
    load_yaml,
    make_dataset,
    resolve_path,
    set_reproducible_settings,
    tensor_to_hwc,
    write_csv,
)

FIXED_ALPHA = 0.80


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_gate(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Fusion gate checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"state_dict", "input_dim", "hidden_dim", "region_names"}
    missing = required.difference(checkpoint.keys() if isinstance(checkpoint, dict) else {})
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


def sample_name(batch: dict[str, Any], index: int) -> str:
    value = batch.get("lq_path", f"test_img_{index}")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return Path(str(value)).stem or f"test_img_{index}"


def aggregate(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_names = (
        "RegionalFull",
        "IndependentUpsampler",
        "FixedAlpha_0.800",
        "DynamicRegionalGate",
    )
    output: list[dict[str, Any]] = []
    for model_name in model_names:
        selected = [row for row in metric_rows if row["model"] == model_name]
        if not selected:
            continue
        row: dict[str, Any] = {"model": model_name, "scene_count": len(selected)}
        for metric_name in METRIC_NAMES:
            values = np.asarray([item[metric_name] for item in selected], dtype=np.float64)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std_across_scenes"] = float(values.std(ddof=0))
        output.append(row)
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
        "--gate-checkpoint", type=Path,
        default=Path("results/dynamic_regional_fusion_compact/dynamic_regional_fusion_gate.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/fixed080_fusion_test_eval"),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--check-only", action="store_true",
        help="Strictly load configs/checkpoints and test fusion on synthetic tensors; do not read test MAT data",
    )
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if not (repo_root / "basicsr").is_dir():
        raise RuntimeError("Run this script from the SDANet repository root (folder containing basicsr/).")

    regional_config_path = resolve_path(args.regional_config, repo_root)
    independent_config_path = resolve_path(args.independent_config, repo_root)
    gate_checkpoint_path = resolve_path(args.gate_checkpoint, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    if not args.check_only and output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new --output-dir if another run is intentional."
        )

    regional_options = load_yaml(regional_config_path)
    independent_options = load_yaml(independent_config_path)
    for label, options in (("RegionalFull", regional_options), ("Independent", independent_options)):
        if options.get("model_type") != "SDAModel":
            raise ValueError(f"{label} config must specify model_type: SDAModel")
        if int(options.get("scale", 0)) != 4:
            raise ValueError(f"{label} config must specify scale: 4")
        if "test_1" not in (options.get("datasets") or {}):
            raise KeyError(f"{label} config must contain datasets.test_1")

    metrics_options = regional_options.get("val", {}).get("metrics") or {}
    independent_metrics = independent_options.get("val", {}).get("metrics") or {}
    missing = [name for name in METRIC_NAMES if name not in metrics_options]
    if missing:
        raise KeyError(f"RegionalFull test config is missing metrics: {missing}")
    if metrics_options != independent_metrics:
        raise ValueError("RegionalFull and Independent configs use different metric definitions")

    dataset_value = regional_options["datasets"]["test_1"].get("dataroot_gt")
    if not dataset_value:
        raise ValueError("RegionalFull datasets.test_1.dataroot_gt is missing")
    test_mat_path = resolve_path(dataset_value, repo_root)
    if not test_mat_path.is_file():
        raise FileNotFoundError(f"Test MAT file not found: {test_mat_path}")

    reproducibility = set_reproducible_settings(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print("Loading frozen models and compact gate...", flush=True)
    model_a, _, meta_a = build_frozen_model(
        regional_config_path, device, repo_root, "RegionalFull"
    )
    model_b, _, meta_b = build_frozen_model(
        independent_config_path, device, repo_root, "IndependentUpsampler"
    )
    gate, gate_meta = load_gate(gate_checkpoint_path, device)

    expected_input_dim = int(regional_options["network_g"].get("n_colors", 128)) * 6 + 4
    if gate.input_dim != expected_input_dim:
        raise ValueError(
            f"Gate input dimension mismatch: {gate.input_dim} vs expected {expected_input_dim}"
        )
    if meta_a["scale"] != 4 or meta_b["scale"] != 4:
        raise ValueError("Both loaded backbones must be x4 models")

    print(f"RegionalFull checkpoint: {meta_a['checkpoint_path']}", flush=True)
    print(f"Independent checkpoint:  {meta_b['checkpoint_path']}", flush=True)
    print(f"Gate checkpoint:         {gate_checkpoint_path}", flush=True)
    print(f"Gate: hidden={gate.hidden_dim}, parameters={sum(p.numel() for p in gate.parameters()):,}", flush=True)
    print(f"Test MAT path:           {test_mat_path}", flush=True)
    print(f"Fixed blend:             {FIXED_ALPHA:.2f} * RegionalFull + {1-FIXED_ALPHA:.2f} * Independent", flush=True)
    print(f"Device:                  {device}", flush=True)

    if args.check_only:
        # Synthetic check validates gate feature, fixed blend and regional fusion paths.
        generator = torch.Generator(device=device).manual_seed(args.seed)
        prediction_a = torch.rand((1, 128, 32, 32), generator=generator, device=device)
        prediction_b = torch.rand((1, 128, 32, 32), generator=generator, device=device)
        features, targets = build_gate_features_and_oracle(prediction_a, prediction_b, None)
        alpha = gate(features)
        fixed = FIXED_ALPHA * prediction_a + (1.0 - FIXED_ALPHA) * prediction_b
        dynamic = apply_regional_fusion(prediction_a, prediction_b, alpha)
        if targets is not None:
            raise RuntimeError("Synthetic check unexpectedly produced oracle targets")
        for label, tensor in (("features", features), ("alpha", alpha), ("fixed", fixed), ("dynamic", dynamic)):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Non-finite values in synthetic {label}")
        if fixed.shape != prediction_a.shape or dynamic.shape != prediction_a.shape or alpha.shape != (1, 4):
            raise RuntimeError("Synthetic fusion shape check failed")
        print(
            f"CHECK ONLY PASSED: checkpoints strict-loaded; synthetic output={tuple(fixed.shape)}, "
            f"gate_features={tuple(features.shape)}, alpha={alpha.detach().cpu().tolist()}",
            flush=True,
        )
        print("Test MAT file was not opened; no reports were created.", flush=True)
        return 0

    print("\nOpening existing four-tile Chikusei test dataset for the locked comparison...", flush=True)
    test_options = dict(regional_options["datasets"]["test_1"])
    test_dataset, test_root = make_dataset(
        test_options, "test", int(regional_options["scale"]), repo_root
    )
    if len(test_dataset) != 4:
        raise RuntimeError(f"Expected four test tiles, found {len(test_dataset)}")

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    metric_rows: list[dict[str, Any]] = []
    gate_weight_rows: list[dict[str, Any]] = []

    model_a.eval()
    model_b.eval()
    gate.eval()
    for index, batch in enumerate(loader):
        scene_id = sample_name(batch, index)
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        lms = batch["lms"].to(device=device, dtype=torch.float32)
        gt = batch["gt"].to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            prediction_a = model_a(lq, lms)
            prediction_b = model_b(lq, lms)
            if prediction_a.shape != gt.shape or prediction_b.shape != gt.shape:
                raise RuntimeError(
                    f"{scene_id}: prediction shapes A={tuple(prediction_a.shape)}, "
                    f"B={tuple(prediction_b.shape)}, GT={tuple(gt.shape)}"
                )
            features, _ = build_gate_features_and_oracle(prediction_a, prediction_b, None)
            gate_alpha = gate(features)
            fixed_prediction = FIXED_ALPHA * prediction_a + (1.0 - FIXED_ALPHA) * prediction_b
            dynamic_prediction = apply_regional_fusion(prediction_a, prediction_b, gate_alpha)

        prediction_map = {
            "RegionalFull": prediction_a,
            "IndependentUpsampler": prediction_b,
            "FixedAlpha_0.800": fixed_prediction,
            "DynamicRegionalGate": dynamic_prediction,
        }
        gt_hwc = tensor_to_hwc(gt)
        for model_name, prediction in prediction_map.items():
            pred_hwc = tensor_to_hwc(prediction)
            if pred_hwc.shape != gt_hwc.shape:
                raise RuntimeError(f"{scene_id}/{model_name}: prediction/GT shape mismatch")
            metric_data = {"img": pred_hwc, "img2": gt_hwc}
            row: dict[str, Any] = {"scene_id": scene_id, "model": model_name}
            for metric_name in METRIC_NAMES:
                value = float(calculate_metric(metric_data, metrics_options[metric_name]))
                if not np.isfinite(value):
                    raise FloatingPointError(f"Non-finite {metric_name} for {scene_id}/{model_name}")
                row[metric_name] = value
            metric_rows.append(row)

        alpha_values = gate_alpha.squeeze(0).detach().cpu().numpy()
        gate_weight_rows.append({
            "scene_id": scene_id,
            **{REGION_NAMES[i]: float(alpha_values[i]) for i in range(4)},
        })
        print(
            f"[{index + 1}/4] {scene_id}: "
            + ", ".join(f"{name}={row[name]:.6f}" for name in METRIC_NAMES
                         for row in metric_rows[-4:] if row["model"] == "FixedAlpha_0.800"),
            flush=True,
        )
        del lq, lms, gt, prediction_a, prediction_b, fixed_prediction, dynamic_prediction, features, gate_alpha
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_rows = aggregate(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_dir / "test_metrics_per_scene.csv",
        ["scene_id", "model", *METRIC_NAMES], metric_rows,
    )
    write_csv(
        output_dir / "test_metrics_aggregate.csv",
        ["model", "scene_count", *[
            field for metric in METRIC_NAMES
            for field in (f"{metric}_mean", f"{metric}_std_across_scenes")
        ]], aggregate_rows,
    )
    write_csv(
        output_dir / "dynamic_gate_weights.csv",
        ["scene_id", *REGION_NAMES], gate_weight_rows,
    )

    by_name = {row["model"]: row for row in aggregate_rows}
    differences = {}
    for comparator in ("RegionalFull", "DynamicRegionalGate"):
        differences[f"FixedAlpha_0.800_minus_{comparator}"] = {
            metric: by_name["FixedAlpha_0.800"][f"{metric}_mean"]
                    - by_name[comparator][f"{metric}_mean"]
            for metric in METRIC_NAMES
        }

    metadata = {
        "experiment": "locked_fixed_alpha_080_fusion_test_evaluation",
        "fixed_alpha": FIXED_ALPHA,
        "fusion_definition": "0.8 * RegionalFull + 0.2 * IndependentUpsampler",
        "selection_basis": (
            "Alpha 0.80 selected as a representative point in the validation 0.75-0.85 plateau; "
            "cross-row validation selection chose 0.75 on row 3 and 0.80 on row 15."
        ),
        "test_exposure_warning": (
            "The four Chikusei test tiles were examined in prior development evaluations. "
            "This is not a pristine blind test and must be reported transparently."
        ),
        "test_dataset": str(test_root),
        "test_tile_count": len(test_dataset),
        "regional_model": meta_a,
        "independent_model": meta_b,
        "gate_checkpoint": str(gate_checkpoint_path),
        "gate_checkpoint_sha256": sha256_file(gate_checkpoint_path),
        "gate_hidden_dim": gate.hidden_dim,
        "gate_parameter_count": sum(p.numel() for p in gate.parameters()),
        "gate_target_definition": gate_meta.get("target_definition"),
        "metrics": list(METRIC_NAMES),
        "fixed_minus_comparator_differences": differences,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reproducibility_settings": reproducibility,
        "test_ground_truth_used_for_fusion_weights": False,
        "note": (
            "Ground truth was used only to calculate final metrics. Fixed alpha was locked from validation; "
            "no alpha selection or gate fitting was performed on test data."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nTEST AGGREGATE METRICS (mean across four tiles):", flush=True)
    for row in aggregate_rows:
        print(
            f"{row['model']:24s} "
            + " | ".join(f"{metric}={row[f'{metric}_mean']:.6f}" for metric in METRIC_NAMES),
            flush=True,
        )
    print("\nFixedAlpha_0.800 differences:", flush=True)
    for comparator, values in differences.items():
        print(f"  {comparator}", flush=True)
        for metric, value in values.items():
            print(f"    {metric}: {value:+.8f}", flush=True)
    print(f"\nReports saved to: {output_dir}", flush=True)
    print(
        "CAVEAT: Existing test tiles were used in earlier development comparisons; "
        "do not describe this as a pristine blind evaluation.", flush=True
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
