#!/usr/bin/env python3
"""Train a small, content-conditioned regional fusion gate for two frozen SDANet models.

This is a post-hoc ensemble experiment. It does NOT modify either SDANet model
or train either backbone. The gate is trained only on the configured training
split and selected using the configured validation/evaluation split. It never
loads or evaluates Chikusei_test.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python train_dynamic_regional_fusion.py

Default inputs:
  Regional model:     RegionalFull 66K checkpoint/config
  Independent model:  IndependentUpsamplerFinetune 51K checkpoint/config
  Gate training:      .../trains
  Gate validation:    .../evals

The gate predicts one blending coefficient for each 2x2 spatial quadrant.
Its input is derived only from the two model predictions (regional means and
standard deviations of each prediction and their absolute difference), plus a
quadrant identity code. Training targets are per-quadrant MSE-optimal blending
coefficients computed from training ground truth. At inference, the gate needs
only the two predictions; ground truth is never an input to the gate.

Outputs are written to a new directory and existing output directories are
never overwritten.
"""

from __future__ import annotations

# Must be set before importing torch / initializing CUDA.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

from basicsr.archs import build_network
from basicsr.data import build_dataset
from basicsr.metrics import calculate_metric

METRIC_NAMES = ("ssim", "mpsnr", "sam", "ergas", "CC")
REGION_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def set_reproducible_settings(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
    }


def choose_dataset_entry(config: dict[str, Any], key: str) -> dict[str, Any]:
    datasets = config.get("datasets") or {}
    if key not in datasets or not isinstance(datasets[key], dict):
        raise KeyError(f"Dataset entry {key!r} is missing from the training YAML")
    return copy.deepcopy(datasets[key])


def make_dataset(
    dataset_options: dict[str, Any], phase: str, scale: int, repo_root: Path
):
    options = copy.deepcopy(dataset_options)
    options["phase"] = phase
    options["scale"] = int(scale)
    root_value = options.get("dataroot_gt")
    if not root_value:
        raise ValueError(f"Dataset {options.get('name', '<unnamed>')} has no dataroot_gt")
    root = resolve_path(root_value, repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    options["dataroot_gt"] = str(root)
    return build_dataset(options), root


def build_frozen_model(
    test_config_path: Path,
    device: torch.device,
    repo_root: Path,
    label: str,
):
    config = load_yaml(test_config_path)
    network_options = config.get("network_g")
    path_options = config.get("path") or {}
    if not isinstance(network_options, dict):
        raise ValueError(f"{test_config_path} is missing network_g")

    checkpoint_value = path_options.get("pretrain_network_g")
    if not checkpoint_value:
        raise ValueError(f"{test_config_path} is missing path.pretrain_network_g")
    checkpoint_path = resolve_path(checkpoint_value, repo_root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{label} checkpoint not found: {checkpoint_path}")

    param_key = path_options.get("param_key_g", "params_ema")
    if path_options.get("strict_load_g", True) is not True:
        raise ValueError(f"{label} config must use strict_load_g: true")

    model = build_network(network_options).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or param_key not in checkpoint:
        available = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        raise KeyError(
            f"{label} checkpoint lacks {param_key!r}; available checkpoint entries: {available}"
        )

    incompat = model.load_state_dict(checkpoint[param_key], strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"{label} strict state-dict load failed: {incompat}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    metadata = {
        "label": label,
        "network_type": network_options.get("type"),
        "config_path": str(test_config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_parameter_key": param_key,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "scale": int(config.get("scale", 4)),
    }
    return model, config, metadata


def region_slices(height: int, width: int):
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(
            f"Regional fusion requires even output dimensions; got {height}x{width}"
        )
    half_h, half_w = height // 2, width // 2
    return (
        (slice(0, half_h), slice(0, half_w)),
        (slice(0, half_h), slice(half_w, width)),
        (slice(half_h, height), slice(0, half_w)),
        (slice(half_h, height), slice(half_w, width)),
    )


def build_gate_features_and_oracle(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    ground_truth: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return [B,4,6*C+4] prediction-only features and optional MSE oracle alpha."""
    if prediction_a.ndim != 4 or prediction_b.ndim != 4:
        raise ValueError("Both predictions must have BCHW shape")
    if prediction_a.shape != prediction_b.shape:
        raise ValueError(
            f"Model output shapes differ: {tuple(prediction_a.shape)} vs "
            f"{tuple(prediction_b.shape)}"
        )
    if ground_truth is not None and ground_truth.shape != prediction_a.shape:
        raise ValueError(
            f"Ground-truth shape {tuple(ground_truth.shape)} does not match "
            f"prediction shape {tuple(prediction_a.shape)}"
        )

    batch, channels, height, width = prediction_a.shape
    slices = region_slices(height, width)
    region_features: list[torch.Tensor] = []
    oracle_targets: list[torch.Tensor] = []
    abs_difference = (prediction_a - prediction_b).abs()

    for region_index, (ys, xs) in enumerate(slices):
        a_region = prediction_a[:, :, ys, xs]
        b_region = prediction_b[:, :, ys, xs]
        d_region = abs_difference[:, :, ys, xs]

        stats = []
        for region_tensor in (a_region, b_region, d_region):
            stats.append(region_tensor.mean(dim=(2, 3)))
            stats.append(region_tensor.std(dim=(2, 3), unbiased=False))

        region_code = prediction_a.new_zeros((batch, 4))
        region_code[:, region_index] = 1.0
        feature = torch.cat([*stats, region_code], dim=1)
        region_features.append(feature)

        if ground_truth is not None:
            gt_region = ground_truth[:, :, ys, xs]
            delta = a_region - b_region
            target_delta = gt_region - b_region
            numerator = (delta * target_delta).sum(dim=(1, 2, 3))
            denominator = delta.square().sum(dim=(1, 2, 3))
            safe_alpha = numerator / denominator.clamp_min(1e-12)
            alpha = torch.where(denominator > 1e-12, safe_alpha, torch.full_like(safe_alpha, 0.5))
            oracle_targets.append(alpha.clamp(0.0, 1.0))

    features = torch.stack(region_features, dim=1)
    targets = torch.stack(oracle_targets, dim=1) if ground_truth is not None else None
    expected_features = 6 * channels + 4
    if features.shape != (batch, 4, expected_features):
        raise RuntimeError(f"Unexpected gate feature shape: {tuple(features.shape)}")
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite values in gate features")
    if targets is not None and not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite values in oracle fusion weights")
    return features, targets


class DynamicRegionalFusionGate(nn.Module):
    """Shared MLP producing one content-conditioned alpha per spatial quadrant."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected [B,R,{self.input_dim}] gate features; got {tuple(features.shape)}"
            )
        return torch.sigmoid(self.network(features).squeeze(-1))


def apply_regional_fusion(
    prediction_a: torch.Tensor,
    prediction_b: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    if alpha.ndim != 2 or alpha.shape != (prediction_a.shape[0], 4):
        raise ValueError(f"Expected alpha shape [B,4], got {tuple(alpha.shape)}")
    if prediction_a.shape != prediction_b.shape:
        raise ValueError("Prediction shapes differ")
    output = torch.empty_like(prediction_a)
    slices = region_slices(prediction_a.shape[-2], prediction_a.shape[-1])
    for region_index, (ys, xs) in enumerate(slices):
        weight = alpha[:, region_index].view(-1, 1, 1, 1)
        output[:, :, ys, xs] = (
            weight * prediction_a[:, :, ys, xs]
            + (1.0 - weight) * prediction_b[:, :, ys, xs]
        )
    return output


def get_sample_name(batch: dict[str, Any], index: int) -> str:
    value = batch.get("lq_path", f"eval_{index}")
    if isinstance(value, (list, tuple)):
        value = value[0]
    return Path(str(value)).stem or f"eval_{index}"


def collect_gate_dataset(
    dataset,
    model_a: nn.Module,
    model_b: nn.Module,
    device: torch.device,
    batch_size: int,
    max_samples: int | None,
    seed: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    if len(dataset) == 0:
        raise RuntimeError(f"The {label} dataset is empty")

    if max_samples is not None and max_samples > 0 and len(dataset) > max_samples:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
        selected_dataset = Subset(dataset, indices)
    else:
        selected_dataset = dataset

    loader = DataLoader(
        selected_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    feature_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    samples_seen = 0
    model_a.eval()
    model_b.eval()

    for batch_index, batch in enumerate(loader):
        lq = batch["lq"].to(device=device, dtype=torch.float32, non_blocking=True)
        lms = batch["lms"].to(device=device, dtype=torch.float32, non_blocking=True)
        gt = batch["gt"].to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.inference_mode():
            prediction_a = model_a(lq, lms)
            prediction_b = model_b(lq, lms)
            if prediction_a.shape != gt.shape:
                raise RuntimeError(
                    f"{label} batch {batch_index}: RegionalFull output {tuple(prediction_a.shape)} "
                    f"does not match GT {tuple(gt.shape)}"
                )
            features, targets = build_gate_features_and_oracle(prediction_a, prediction_b, gt)

        assert targets is not None
        feature_chunks.append(features.reshape(-1, features.shape[-1]).cpu().numpy().astype(np.float32))
        target_chunks.append(targets.reshape(-1).cpu().numpy().astype(np.float32))
        samples_seen += int(gt.shape[0])

        if (batch_index + 1) % 25 == 0:
            print(f"  {label}: processed {samples_seen} samples", flush=True)

        del lq, lms, gt, prediction_a, prediction_b, features, targets

    x = np.concatenate(feature_chunks, axis=0)
    y = np.concatenate(target_chunks, axis=0)
    if x.shape[0] != 4 * samples_seen or y.shape[0] != 4 * samples_seen:
        raise RuntimeError(f"Unexpected {label} feature/target counts: X={x.shape}, y={y.shape}")
    print(
        f"{label} data ready: {samples_seen} samples, {x.shape[0]} regional targets, "
        f"feature dimension {x.shape[1]}",
        flush=True,
    )
    return x, y, samples_seen


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_gate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[DynamicRegionalFusionGate, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    input_dim = int(train_x.shape[1])
    gate = DynamicRegionalFusionGate(input_dim=input_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)

    train_tensor = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True, num_workers=0)
    val_features = torch.from_numpy(val_x).to(device=device, dtype=torch.float32)
    val_targets = torch.from_numpy(val_y).to(device=device, dtype=torch.float32)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        gate.train()
        train_losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_alpha = gate(features.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(predicted_alpha, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        gate.eval()
        with torch.inference_mode():
            val_pred = gate(val_features.unsqueeze(1)).squeeze(1)
            val_loss = float(F.smooth_l1_loss(val_pred, val_targets).cpu())
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        history.append({"epoch": epoch, "train_huber": train_loss, "val_huber": val_loss})
        print(
            f"Gate epoch {epoch:03d}/{epochs}: train_huber={train_loss:.7f}, "
            f"val_huber={val_loss:.7f}",
            flush=True,
        )

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no validation improvement for {patience} epochs.", flush=True)
                break

    if best_state is None:
        raise RuntimeError("Gate training did not produce a valid checkpoint")
    gate.load_state_dict(best_state, strict=True)
    gate.eval()
    print(f"Selected gate epoch {best_epoch} (validation Huber={best_val:.7f})", flush=True)
    return gate, history


def tensor_to_hwc(tensor: torch.Tensor) -> np.ndarray:
    if tensor.shape[0] != 1:
        raise ValueError("Metric conversion expects evaluation batch size 1")
    return tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)


def evaluate_on_validation(
    dataset,
    model_a: nn.Module,
    model_b: nn.Module,
    gate: DynamicRegionalFusionGate,
    metrics_options: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    model_names = ("RegionalFull", "IndependentUpsampler", "Fixed50_50", "DynamicRegionalGate")

    model_a.eval()
    model_b.eval()
    gate.eval()

    for index, batch in enumerate(loader):
        sample_id = get_sample_name(batch, index)
        lq = batch["lq"].to(device=device, dtype=torch.float32)
        lms = batch["lms"].to(device=device, dtype=torch.float32)
        gt = batch["gt"].to(device=device, dtype=torch.float32)

        with torch.inference_mode():
            prediction_a = model_a(lq, lms)
            prediction_b = model_b(lq, lms)
            if prediction_a.shape != gt.shape or prediction_b.shape != gt.shape:
                raise RuntimeError(
                    f"Validation sample {sample_id}: prediction shapes must match GT; "
                    f"A={tuple(prediction_a.shape)}, B={tuple(prediction_b.shape)}, "
                    f"GT={tuple(gt.shape)}"
                )
            features, _ = build_gate_features_and_oracle(prediction_a, prediction_b, None)
            alpha = gate(features)
            fixed_prediction = 0.5 * prediction_a + 0.5 * prediction_b
            dynamic_prediction = apply_regional_fusion(prediction_a, prediction_b, alpha)

        predictions = {
            "RegionalFull": prediction_a,
            "IndependentUpsampler": prediction_b,
            "Fixed50_50": fixed_prediction,
            "DynamicRegionalGate": dynamic_prediction,
        }
        gt_hwc = tensor_to_hwc(gt)
        for model_name in model_names:
            pred_hwc = tensor_to_hwc(predictions[model_name])
            if pred_hwc.shape != gt_hwc.shape:
                raise RuntimeError(f"{sample_id}/{model_name}: prediction/GT HWC shape mismatch")
            metric_data = {"img": pred_hwc, "img2": gt_hwc}
            row: dict[str, Any] = {"sample_id": sample_id, "model": model_name}
            for metric_name in METRIC_NAMES:
                metric_value = float(calculate_metric(metric_data, metrics_options[metric_name]))
                if not np.isfinite(metric_value):
                    raise FloatingPointError(
                        f"Non-finite {metric_name} for {sample_id}/{model_name}: {metric_value}"
                    )
                row[metric_name] = metric_value
            metric_rows.append(row)

        alpha_values = alpha.squeeze(0).detach().cpu().numpy()
        weight_row: dict[str, Any] = {"sample_id": sample_id}
        for region_index, region_name in enumerate(REGION_NAMES):
            weight_row[region_name] = float(alpha_values[region_index])
        weight_rows.append(weight_row)

        print(
            f"Validation [{index + 1}/{len(dataset)}] {sample_id}: "
            + ", ".join(f"{name}={alpha_values[i]:.3f}" for i, name in enumerate(REGION_NAMES)),
            flush=True,
        )
        del lq, lms, gt, prediction_a, prediction_b, fixed_prediction, dynamic_prediction

    return metric_rows, weight_rows


def aggregate_metrics(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in ("RegionalFull", "IndependentUpsampler", "Fixed50_50", "DynamicRegionalGate"):
        selected = [row for row in metric_rows if row["model"] == model_name]
        if not selected:
            continue
        row: dict[str, Any] = {"model": model_name, "sample_count": len(selected)}
        for metric_name in METRIC_NAMES:
            values = np.asarray([entry[metric_name] for entry in selected], dtype=np.float64)
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std_across_samples"] = float(values.std(ddof=0))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regional-config", type=Path,
                        default=Path("options/test/HSI/test_SDANetRegionalFullChux4.yml"))
    parser.add_argument("--independent-config", type=Path,
                        default=Path("options/test/HSI/test_SDANetIndependentUpsamplerFinetuneChux4.yml"))
    parser.add_argument("--data-config", type=Path,
                        default=Path("options/train/HSI/SDANetRegionalFullChux4.yml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dynamic_regional_fusion"))
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=256,
                        help="Maximum augmented training samples (0 means use all)")
    parser.add_argument("--inference-batch-size", type=int, default=2,
                        help="Batch size for frozen-model prediction generation")
    parser.add_argument("--gate-batch-size", type=int, default=256,
                        help="Batch size for the small gate MLP")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--check-only", action="store_true",
                        help="Validate both checkpoints and one training/validation sample without training the gate")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    if not (repo_root / "basicsr").is_dir():
        raise RuntimeError("Run this script from the SDANet repository root (the folder containing basicsr/).")

    for name, value in (("inference batch size", args.inference_batch_size),
                        ("gate batch size", args.gate_batch_size),
                        ("epochs", args.epochs), ("patience", args.patience)):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.max_train_samples < 0:
        raise ValueError("--max-train-samples must be >= 0")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")

    regional_config_path = resolve_path(args.regional_config, repo_root)
    independent_config_path = resolve_path(args.independent_config, repo_root)
    data_config_path = resolve_path(args.data_config, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)
    if not args.check_only and output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output directory: {output_dir}. "
            "Choose a new --output-dir if you intentionally want another run."
        )

    regional_data = load_yaml(regional_config_path)
    independent_data = load_yaml(independent_config_path)
    data_config = load_yaml(data_config_path)
    scales = {int(regional_data.get("scale", 4)), int(independent_data.get("scale", 4)),
              int(data_config.get("scale", 4))}
    if len(scales) != 1:
        raise ValueError(f"Model/data scale mismatch: {sorted(scales)}")
    scale = scales.pop()

    metrics_options = regional_data.get("val", {}).get("metrics") or {}
    independent_metrics_options = independent_data.get("val", {}).get("metrics") or {}
    for metric_name in METRIC_NAMES:
        if metric_name not in metrics_options:
            raise KeyError(f"Regional test config is missing metric {metric_name}")
        if metric_name not in independent_metrics_options:
            raise KeyError(f"Independent test config is missing metric {metric_name}")
    if metrics_options != independent_metrics_options:
        raise ValueError("Regional and Independent test configurations have different metric definitions")

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
    print(f"RegionalFull: {meta_a['checkpoint_path']} ({meta_a['parameter_count']:,} parameters)", flush=True)
    print(f"Independent:  {meta_b['checkpoint_path']} ({meta_b['parameter_count']:,} parameters)", flush=True)

    train_options = choose_dataset_entry(data_config, "train")
    val_key = next((key for key in data_config.get("datasets", {}) if key.split("_", 1)[0] == "val"), None)
    if val_key is None:
        raise KeyError("Data config must contain a validation dataset entry such as val_1")
    val_options = choose_dataset_entry(data_config, val_key)
    train_dataset, train_root = make_dataset(train_options, "train", scale, repo_root)
    val_dataset, val_root = make_dataset(val_options, "val", scale, repo_root)
    print(f"Training split:   {train_root} ({len(train_dataset)} augmented samples)", flush=True)
    print(f"Validation split: {val_root} ({len(val_dataset)} samples)", flush=True)
    if "test" in str(train_root).lower() or "test" in str(val_root).lower():
        raise RuntimeError("Safety check failed: gate data paths must not point to a test split")

    if args.check_only:
        print("\nRunning one-sample compatibility checks on train and validation splits...", flush=True)
        for split_name, dataset in (("train", train_dataset), ("validation", val_dataset)):
            if len(dataset) == 0:
                raise RuntimeError(f"{split_name} dataset is empty")
            sample = dataset[0]
            lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            lms = sample["lms"].unsqueeze(0).to(device=device, dtype=torch.float32)
            gt = sample["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                pred_a = model_a(lq, lms)
                pred_b = model_b(lq, lms)
                if pred_a.shape != gt.shape or pred_b.shape != gt.shape:
                    raise RuntimeError(
                        f"{split_name} compatibility failure: A={tuple(pred_a.shape)}, "
                        f"B={tuple(pred_b.shape)}, GT={tuple(gt.shape)}"
                    )
                features, oracle = build_gate_features_and_oracle(pred_a, pred_b, gt)
                if features.shape[1:] != (4, 6 * gt.shape[1] + 4) or oracle.shape != (1, 4):
                    raise RuntimeError(f"Unexpected {split_name} gate feature/target shapes")
            print(
                f"{split_name}: outputs={tuple(pred_a.shape)}, "
                f"gate_features={tuple(features.shape)}, oracle={tuple(oracle.shape)} — OK",
                flush=True,
            )
        print("\nCHECK ONLY PASSED. No training or output files were created.", flush=True)
        return 0

    train_limit = args.max_train_samples if args.max_train_samples > 0 else None
    print("\nGenerating frozen-model predictions for gate training...", flush=True)
    train_x, train_y, train_sample_count = collect_gate_dataset(
        train_dataset, model_a, model_b, device, args.inference_batch_size,
        train_limit, args.seed, "train",
    )
    print("\nGenerating validation targets for early stopping...", flush=True)
    val_x, val_y, val_sample_count = collect_gate_dataset(
        val_dataset, model_a, model_b, device, 1, None, args.seed + 1, "validation",
    )

    if train_x.shape[1] != val_x.shape[1]:
        raise RuntimeError(f"Train/validation gate feature sizes differ: {train_x.shape[1]} vs {val_x.shape[1]}")

    output_dir.mkdir(parents=True, exist_ok=False)
    print("\nTraining the dynamic regional gate (backbones remain frozen)...", flush=True)
    gate, history = train_gate(
        train_x, train_y, val_x, val_y, device,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.gate_batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    print("\nEvaluating both models and fusion methods on the validation split only...", flush=True)
    metric_rows, weight_rows = evaluate_on_validation(
        val_dataset, model_a, model_b, gate, metrics_options, device
    )
    aggregate_rows = aggregate_metrics(metric_rows)

    write_csv(output_dir / "gate_training_history.csv",
              ["epoch", "train_huber", "val_huber"], history)
    write_csv(output_dir / "validation_metrics_per_sample.csv",
              ["sample_id", "model", *METRIC_NAMES], metric_rows)
    write_csv(output_dir / "validation_metrics_aggregate.csv",
              ["model", "sample_count", *[field for metric in METRIC_NAMES
               for field in (f"{metric}_mean", f"{metric}_std_across_samples")]], aggregate_rows)
    write_csv(output_dir / "predicted_regional_weights.csv",
              ["sample_id", *REGION_NAMES], weight_rows)

    gate_checkpoint = {
        "state_dict": {key: value.detach().cpu() for key, value in gate.state_dict().items()},
        "input_dim": gate.input_dim,
        "hidden_dim": gate.hidden_dim,
        "region_names": list(REGION_NAMES),
        "feature_definition": "per-quadrant means/stds of prediction_A, prediction_B, abs(A-B), plus one-hot quadrant ID",
        "target_definition": "per-quadrant MSE-optimal scalar blend coefficient computed on training GT",
        "fusion_definition": "Y=alpha_r*A+(1-alpha_r)*B within quadrant r",
    }
    torch.save(gate_checkpoint, output_dir / "dynamic_regional_fusion_gate.pth")

    metadata = {
        "experiment": "post_hoc_dynamic_regional_fusion_gate",
        "note": "This is a two-checkpoint post-hoc ensemble, not a shared-backbone dual-decoder architecture.",
        "regional_model": meta_a,
        "independent_model": meta_b,
        "train_dataset": str(train_root),
        "validation_dataset": str(val_root),
        "train_sample_count_used": train_sample_count,
        "validation_sample_count": val_sample_count,
        "train_region_target_count": int(train_y.shape[0]),
        "validation_region_target_count": int(val_y.shape[0]),
        "seed": args.seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reproducibility_settings": reproducibility,
        "gate_input_dim": gate.input_dim,
        "gate_hidden_dim": gate.hidden_dim,
        "gate_selected_epochs": len(history),
        "gate_best_validation_huber": min(row["val_huber"] for row in history),
        "fusion_regions": list(REGION_NAMES),
        "held_out_test_split_used": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("\nValidation aggregate metrics:")
    for row in aggregate_rows:
        print(
            f"{row['model']:24s} "
            + " | ".join(f"{metric}={row[f'{metric}_mean']:.6f}" for metric in METRIC_NAMES)
        )
    print(f"\nGate checkpoint and reports saved to: {output_dir}", flush=True)
    print("IMPORTANT: These are validation-split results, not final held-out test results.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise
