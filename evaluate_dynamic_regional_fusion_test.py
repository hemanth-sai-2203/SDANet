#!/usr/bin/env python3
"""Evaluate a trained dynamic regional fusion gate on Chikusei_test.

This script is inference/evaluation only. It does not fit or modify either
backbone or the fusion gate. It never uses test ground truth to calculate gate
weights; ground truth is used only for final metrics.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python evaluate_dynamic_regional_fusion_test.py

Default inputs:
  RegionalFull checkpoint/config: existing 66K model
  Independent checkpoint/config: existing 51K model
  Fusion gate: results/dynamic_regional_fusion/dynamic_regional_fusion_gate.pth
  Test data: Chikusei_test.mat, from the RegionalFull test YAML

Outputs are written to results/dynamic_regional_fusion_test_eval. The script
refuses to overwrite an existing output directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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
    build_gate_features_and_oracle,
    build_frozen_model,
    load_yaml,
    make_dataset,
    resolve_path,
    set_reproducible_settings,
    tensor_to_hwc,
    write_csv,
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


def sample_name(batch: dict[str, Any], index: int) -> str:
    value = batch.get("lq_path", f"test_img_{index}")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return Path(str(value)).stem or f"test_img_{index}"


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


def aggregate(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model_names = (
        "RegionalFull",
        "IndependentUpsampler",
        "Fixed50_50",
        "DynamicRegionalGate",
    )
    output: list[dict[str, Any]] = []
    for model_name in model_names:
        selected = [row for row in metric_rows if row["model"] == model_name]
        if not selected:
            continue
        row: dict[str, Any] = {"model": model_name, "sample_count": len(selected)}
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
        default=Path("results/dynamic_regional_fusion/dynamic_regional_fusion_gate.pth"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dynamic_regional_fusion_test_eval"),
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--check-only", action="store_true",
                        help="Load all checkpoints and run one test-scene forward pass; write no reports")
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
            "Choose a different --output-dir if you intend another run."
        )

    regional_options = load_yaml(regional_config_path)
    independent_options = load_yaml(independent_config_path)
    for config_name, options in (("RegionalFull", regional_options), ("Independent", independent_options)):
        if options.get("model_type") != "SDAModel":
            raise ValueError(f"{config_name} config must specify model_type: SDAModel")
        if options.get("scale") != 4:
            raise ValueError(f"{config_name} config must specify scale: 4")
        if "test_1" not in (options.get("datasets") or {}):
            raise KeyError(f"{config_name} config must contain datasets.test_1")

    metrics_options = regional_options.get("val", {}).get("metrics") or {}
    independent_metrics = independent_options.get("val", {}).get("metrics") or {}
    for metric_name in METRIC_NAMES:
        if metric_name not in metrics_options:
            raise KeyError(f"RegionalFull test config is missing metric {metric_name}")
        if metric_name not in independent_metrics:
            raise KeyError(f"Independent test config is missing metric {metric_name}")
    if metrics_options != independent_metrics:
        raise ValueError("RegionalFull and Independent configs use different metric definitions")

    reproducibility = set_reproducible_settings(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print("Loading frozen SDANet checkpoints...", flush=True)
    model_a, _, meta_a = build_frozen_model(regional_config_path, device, repo_root, "RegionalFull")
    model_b, _, meta_b = build_frozen_model(independent_config_path, device, repo_root, "IndependentUpsampler")
    print(f"RegionalFull parameters: {meta_a['parameter_count']:,}", flush=True)
    print(f"Independent parameters:  {meta_b['parameter_count']:,}", flush=True)

    gate, gate_meta = load_gate(gate_checkpoint_path, device)
    expected_input_dim = int(regional_options["network_g"].get("n_colors", 128)) * 6 + 4
    if gate.input_dim != expected_input_dim:
        raise ValueError(
            f"Gate input dimension mismatch: checkpoint={gate.input_dim}, "
            f"expected={expected_input_dim} from model spectral bands"
        )
    print(
        f"Dynamic gate loaded: input_dim={gate.input_dim}, hidden_dim={gate.hidden_dim}, "
        f"parameters={sum(p.numel() for p in gate.parameters()):,}",
        flush=True,
    )

    test_options = dict(regional_options["datasets"]["test_1"])
    test_dataset, test_root = make_dataset(test_options, "test", int(regional_options["scale"]), repo_root)
    if len(test_dataset) == 0:
        raise RuntimeError("Test dataset is empty")
    print(f"Test dataset: {test_root} ({len(test_dataset)} scenes)", flush=True)

    if args.check_only:
        sample = test_dataset[0]
        lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
        lms = sample["lms"].unsqueeze(0).to(device=device, dtype=torch.float32)
        gt = sample["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            pred_a = model_a(lq, lms)
            pred_b = model_b(lq, lms)
            if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
                raise RuntimeError(
                    f"Test preflight shape mismatch: A={tuple(pred_a.shape)}, "
                    f"B={tuple(pred_b.shape)}, GT={tuple(gt.shape)}"
                )
            features, _ = build_gate_features_and_oracle(pred_a, pred_b, None)
            alpha = gate(features)
            fused = apply_regional_fusion(pred_a, pred_b, alpha)
            if fused.shape != gt.shape or alpha.shape != (1, 4):
                raise RuntimeError(
                    f"Test preflight fusion mismatch: fused={tuple(fused.shape)}, "
                    f"alpha={tuple(alpha.shape)}"
                )
            if not torch.isfinite(fused).all() or not torch.isfinite(alpha).all():
                raise FloatingPointError("Non-finite test preflight fusion output")
        print(
            f"CHECK ONLY PASSED: test_scene={sample_name(sample, 0)}, "
            f"model_output={tuple(pred_a.shape)}, gate_features={tuple(features.shape)}, "
            f"alpha={alpha.detach().cpu().tolist()}. No reports written.",
            flush=True,
        )
        return 0

    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    model_a.eval()
    model_b.eval()
    gate.eval()
    print("\nEvaluating base models and both fusion variants on Chikusei_test...", flush=True)

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
                    f"{scene_id}: output/GT dimensions mismatch: "
                    f"Regional={tuple(prediction_a.shape)}, Independent={tuple(prediction_b.shape)}, "
                    f"GT={tuple(gt.shape)}"
                )
            features, _ = build_gate_features_and_oracle(prediction_a, prediction_b, None)
            alpha = gate(features)
            if alpha.shape != (1, 4) or not torch.isfinite(alpha).all():
                raise RuntimeError(f"{scene_id}: invalid dynamic gate output: {alpha}")
            fixed_prediction = 0.5 * prediction_a + 0.5 * prediction_b
            dynamic_prediction = apply_regional_fusion(prediction_a, prediction_b, alpha)

        prediction_map = {
            "RegionalFull": prediction_a,
            "IndependentUpsampler": prediction_b,
            "Fixed50_50": fixed_prediction,
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
                    raise FloatingPointError(f"Non-finite {metric_name} for {scene_id}/{model_name}: {value}")
                row[metric_name] = value
            metric_rows.append(row)

        alpha_values = alpha.squeeze(0).detach().cpu().numpy()
        weight_rows.append({
            "scene_id": scene_id,
            **{REGION_NAMES[i]: float(alpha_values[i]) for i in range(4)},
        })
        print(
            f"[{index + 1}/{len(test_dataset)}] {scene_id}: "
            + ", ".join(f"{REGION_NAMES[i]}={alpha_values[i]:.4f}" for i in range(4)),
            flush=True,
        )

        del lq, lms, gt, prediction_a, prediction_b, fixed_prediction, dynamic_prediction

    aggregate_rows = aggregate(metric_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(
        output_dir / "test_metrics_per_scene.csv",
        ["scene_id", "model", *METRIC_NAMES], metric_rows,
    )
    write_csv(
        output_dir / "test_metrics_aggregate.csv",
        ["model", "sample_count", *[
            field for metric in METRIC_NAMES
            for field in (f"{metric}_mean", f"{metric}_std_across_scenes")
        ]], aggregate_rows,
    )
    write_csv(
        output_dir / "predicted_regional_weights.csv",
        ["scene_id", *REGION_NAMES], weight_rows,
    )

    dynamic_row = next(row for row in aggregate_rows if row["model"] == "DynamicRegionalGate")
    regional_row = next(row for row in aggregate_rows if row["model"] == "RegionalFull")
    differences = {
        metric: dynamic_row[f"{metric}_mean"] - regional_row[f"{metric}_mean"]
        for metric in METRIC_NAMES
    }
    metadata = {
        "experiment": "dynamic_regional_fusion_final_test_evaluation",
        "warning": (
            "The four Chikusei test scenes were used in earlier model-development evaluations; "
            "treat these results as a final comparison on the existing test split, not as a pristine blind test."
        ),
        "test_dataset": str(test_root),
        "test_scene_count": len(test_dataset),
        "regional_model": meta_a,
        "independent_model": meta_b,
        "gate_checkpoint": str(gate_checkpoint_path),
        "gate_checkpoint_sha256": sha256_file(gate_checkpoint_path),
        "gate_input_dim": gate.input_dim,
        "gate_hidden_dim": gate.hidden_dim,
        "gate_parameter_count": sum(p.numel() for p in gate.parameters()),
        "gate_target_definition": gate_meta.get("target_definition"),
        "fusion_definition": gate_meta.get("fusion_definition"),
        "metrics": list(METRIC_NAMES),
        "dynamic_minus_regionalfull_metric_differences": differences,
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reproducibility_settings": reproducibility,
        "test_ground_truth_used_for_gate_weights": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nTEST AGGREGATE METRICS (mean across scenes):", flush=True)
    for row in aggregate_rows:
        print(
            f"{row['model']:24s} "
            + " | ".join(f"{metric}={row[f'{metric}_mean']:.6f}" for metric in METRIC_NAMES),
            flush=True,
        )
    print("\nDynamicRegionalGate minus RegionalFull:", flush=True)
    for metric, value in differences.items():
        print(f"  {metric}: {value:+.8f}", flush=True)
    print(f"\nReports saved to: {output_dir}", flush=True)
    print(
        "NOTE: Gate weights were computed from predictions only. Test GT was used only for final metrics.",
        flush=True,
    )
    print(
        "CAVEAT: These four test scenes were previously inspected during development; "
        "do not describe this as a pristine blind evaluation.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
