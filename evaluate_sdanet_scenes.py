#!/usr/bin/env python3
"""Evaluate one SDANet test YAML scene-by-scene without modifying BasicSR.

Run from the SDANet repository root, for example:
  PYTHONPATH="$PWD" .venv/bin/python evaluate_sdanet_scenes.py \
    --config options/test/HSI/test_SDANetFineGateControl5KChux4.yml

The script uses the repository's registered network, dataset and metric
implementations; loads the configured EMA checkpoint strictly; evaluates
samples one at a time; writes per-scene and aggregate CSV files; and saves
one float32 HWC prediction MAT file per scene.

It does not train, modify source files, or overwrite an existing output folder.
"""

# Set the cuBLAS workspace configuration before importing torch / initializing CUDA.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio
import torch
import yaml

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric

METRIC_NAMES = ("ssim", "mpsnr", "sam", "ergas", "CC")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def set_reproducible_settings(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings are applied consistently to every model evaluated with this script.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enforced": torch.are_deterministic_algorithms_enabled(),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return config


def choose_test_dataset(config: dict[str, Any]) -> dict[str, Any]:
    datasets = config.get("datasets") or {}
    candidates = [
        value for key, value in datasets.items()
        if str(key).split("_", 1)[0] == "test"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one test dataset entry (for example test_1); "
            f"found {len(candidates)}."
        )
    dataset_options = dict(candidates[0])
    dataset_options.setdefault("phase", "test")
    if "scale" in config:
        dataset_options.setdefault("scale", config["scale"])
    return dataset_options


def build_model_from_config(config: dict[str, Any], checkpoint_path: Path, device: torch.device):
    network_options = config.get("network_g")
    if not isinstance(network_options, dict):
        raise ValueError("The YAML must contain a network_g mapping.")

    path_options = config.get("path") or {}
    param_key = path_options.get("param_key_g", "params_ema")
    strict = path_options.get("strict_load_g", True)
    if strict is not True:
        raise ValueError("This evaluator requires strict_load_g: true.")

    model = build_network(network_options).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or param_key not in checkpoint:
        available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise KeyError(
            f"Checkpoint does not contain param_key_g={param_key!r}; available={available}"
        )

    incompat = model.load_state_dict(checkpoint[param_key], strict=True)
    # strict=True should produce no missing or unexpected keys.
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint load mismatch: {incompat}")
    model.eval()
    return model, param_key


def to_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Existing BasicSR test YAML")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: results/final_repro/<config name>)",
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=None, help="Override YAML manual_seed")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Test configuration not found: {config_path}")

    config = load_yaml(config_path)
    run_name = str(config.get("name", config_path.stem))
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (
        repo_root / "results" / "final_repro" / run_name
    )
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}\n"
            "Choose a fresh --output-dir to preserve previous results."
        )

    path_options = config.get("path") or {}
    checkpoint_value = path_options.get("pretrain_network_g")
    if not checkpoint_value:
        raise ValueError("The YAML path.pretrain_network_g is missing.")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_options = choose_test_dataset(config)
    dataset_path_value = dataset_options.get("dataroot_gt")
    if not dataset_path_value:
        raise ValueError("The test dataset dataroot_gt is missing.")
    dataset_path = Path(dataset_path_value).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Test MAT file not found: {dataset_path}")

    metric_options = config.get("val", {}).get("metrics") or {}
    missing_metrics = [name for name in METRIC_NAMES if name not in metric_options]
    if missing_metrics:
        raise ValueError(f"Test YAML is missing required metrics: {missing_metrics}")

    seed = int(args.seed if args.seed is not None else config.get("manual_seed", 10))
    backend_settings = set_reproducible_settings(seed)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    # Create output only after input/config validation succeeds.
    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_dir = output_dir / "predictions"
    predictions_dir.mkdir()

    print(f"Run: {run_name}", flush=True)
    print(f"Config: {config_path}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Dataset: {dataset_path}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Backend: {json.dumps(backend_settings, sort_keys=True)}", flush=True)

    model, param_key = build_model_from_config(config, checkpoint_path, device)
    dataset = build_dataset(dataset_options)
    if len(dataset) == 0:
        raise RuntimeError("The test dataset contains no scenes.")

    scene_rows: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        scene_id = str(sample.get("lq_path", f"test_img_{index}"))
        lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
        lms = sample["lms"].unsqueeze(0).to(device=device, dtype=torch.float32)
        gt_hwc = sample["gt"].permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)

        with torch.inference_mode():
            prediction = model(lq, lms)
        prediction_hwc = to_hwc(prediction)

        if prediction_hwc.shape != gt_hwc.shape:
            raise RuntimeError(
                f"Output shape mismatch for {scene_id}: "
                f"prediction={prediction_hwc.shape}, GT={gt_hwc.shape}"
            )
        if not np.isfinite(prediction_hwc).all():
            raise FloatingPointError(f"Non-finite prediction values found in {scene_id}.")

        metric_data = {"img": prediction_hwc, "img2": gt_hwc}
        row: dict[str, Any] = {
            "run": run_name,
            "scene_index": index,
            "scene_id": scene_id,
        }
        for metric_name in METRIC_NAMES:
            value = calculate_metric(metric_data, metric_options[metric_name])
            row[metric_name] = float(value)
        scene_rows.append(row)

        safe_scene_id = Path(scene_id).stem.replace("/", "_").replace("\\", "_")
        prediction_path = predictions_dir / f"{safe_scene_id}_prediction.mat"
        sio.savemat(prediction_path, {"sr_data": prediction_hwc})

        print(
            f"[{index + 1}/{len(dataset)}] {scene_id}: "
            + ", ".join(f"{name}={row[name]:.6f}" for name in METRIC_NAMES),
            flush=True,
        )
        del prediction, lq, lms
        if device.type == "cuda":
            torch.cuda.empty_cache()

    scene_fields = ["run", "scene_index", "scene_id", *METRIC_NAMES]
    write_csv(output_dir / "per_scene_metrics.csv", scene_fields, scene_rows)

    aggregate_rows = []
    for metric_name in METRIC_NAMES:
        values = np.asarray([row[metric_name] for row in scene_rows], dtype=np.float64)
        aggregate_rows.append({
            "run": run_name,
            "metric": metric_name,
            "mean_across_scenes": float(values.mean()),
            "population_std_across_scenes": float(values.std(ddof=0)),
            "minimum_scene_value": float(values.min()),
            "maximum_scene_value": float(values.max()),
            "scene_count": int(values.size),
        })
    write_csv(
        output_dir / "aggregate_metrics.csv",
        ["run", "metric", "mean_across_scenes", "population_std_across_scenes",
         "minimum_scene_value", "maximum_scene_value", "scene_count"],
        aggregate_rows,
    )

    metadata = {
        "run": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root_at_runtime": str(repo_root),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_parameter_key": param_key,
        "strict_checkpoint_load": True,
        "network_type": config.get("network_g", {}).get("type"),
        "fine_gate_enabled": config.get("network_g", {}).get("fine_gate_enabled", False),
        "dataset_type": dataset_options.get("type"),
        "dataset_name": dataset_options.get("name"),
        "dataset_path": str(dataset_path),
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "backend_settings": backend_settings,
        "metric_names": list(METRIC_NAMES),
        "scene_count": len(scene_rows),
        "prediction_format": "MAT, variable sr_data, float32 HWC",
        "note": (
            "Per-scene metrics are calculated with this repository's registered BasicSR metric "
            "functions. Aggregate metrics are the arithmetic mean of the scene-level values. "
            "This evaluation is deterministic-configured but is not a guarantee of bitwise "
            "reproducibility across different hardware/software stacks."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"\nEvaluation complete. Results saved to: {output_dir}", flush=True)
    print(f"Per-scene metrics: {output_dir / 'per_scene_metrics.csv'}", flush=True)
    print(f"Aggregate metrics: {output_dir / 'aggregate_metrics.csv'}", flush=True)
    print(f"Predictions: {predictions_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
