#!/usr/bin/env python3
"""Read-only preflight for the matched global-attention SDANet control.

Run from the SDANet repository root:
  PYTHONPATH="$PWD" .venv/bin/python preflight_global_matched_control.py \
    --config options/train/HSI/SDANetGlobalMatchedContinue66KChux4.yml

This script does not train, save checkpoints, create experiment directories,
or open the Chikusei test MAT file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from basicsr.archs import build_network
from basicsr.data.paired_image_dataset_8x_aug_dataset import PairedMatDataset8xAug


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = args.config.expanduser().resolve()
    require(config_path.is_file(), f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    print("========== CONFIGURATION CHECK ==========")
    require(cfg.get("name") == "SDANetGlobalMatchedContinue66K_spatial_split",
            f"Unexpected experiment name: {cfg.get('name')}")
    require(cfg.get("network_g", {}).get("type") == "SDANET",
            "Control must use the original global-attention SDANET architecture.")
    require(cfg.get("scale") == 4, "Expected x4 scale.")

    train_cfg = cfg["train"]
    require(train_cfg["total_iter"] == 66000, "Expected 66,000 additional iterations.")
    require(train_cfg["scheduler"]["periods"] == [33000, 33000],
            "Expected two 33K cosine cycles.")
    require(train_cfg["scheduler"]["restart_weights"] == [1, 1],
            "Expected both cosine cycles to restart at full base LR.")
    require(train_cfg["optim_g"]["type"] == "Adam", "Expected Adam optimizer.")
    require(abs(float(train_cfg["optim_g"]["lr"]) - 1e-4) < 1e-12,
            "Unexpected base learning rate.")
    require(train_cfg["optim_g"]["betas"] == [0.9, 0.99], "Unexpected Adam betas.")
    require(abs(float(train_cfg["ema_decay"]) - 0.999) < 1e-12,
            "Unexpected EMA decay.")

    output_dir = repo_root / "experiments" / cfg["name"]
    require(not output_dir.exists(),
            f"Experiment output already exists; refusing to reuse it: {output_dir}")

    checkpoint_path = Path(cfg["path"]["pretrain_network_g"]).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (repo_root / checkpoint_path).resolve()
    require(checkpoint_path.is_file(), f"Initial checkpoint not found: {checkpoint_path}")

    print("Experiment:", cfg["name"])
    print("Architecture:", cfg["network_g"]["type"])
    print("Initial checkpoint:", checkpoint_path)
    print("Training iterations:", train_cfg["total_iter"])
    print("Cosine periods:", train_cfg["scheduler"]["periods"])
    print("Base LR:", train_cfg["optim_g"]["lr"])
    print("Output directory is absent:", output_dir)

    print("\n========== CHECKPOINT CHECK ==========")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint, dict), "Checkpoint must be a dictionary.")
    for key in ("params", "params_ema"):
        require(key in checkpoint, f"Checkpoint is missing '{key}'.")
        require(isinstance(checkpoint[key], dict), f"Checkpoint entry '{key}' is not a state dict.")
        print(f"{key}: {len(checkpoint[key])} tensors")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Preflight device:", device)

    print("\n========== STRICT NETWORK LOAD ==========")
    model = build_network(cfg["network_g"]).to(device)
    params_result = model.load_state_dict(checkpoint["params"], strict=True)
    require(not params_result.missing_keys and not params_result.unexpected_keys,
            f"Strict params load mismatch: {params_result}")
    print("params strict load: PASS")

    params_ema_result = model.load_state_dict(checkpoint["params_ema"], strict=True)
    require(not params_ema_result.missing_keys and not params_ema_result.unexpected_keys,
            f"Strict params_ema load mismatch: {params_ema_result}")
    print("params_ema strict load: PASS")
    model.eval()

    print("\n========== DATASET CHECK ==========")
    dataset_options = cfg["datasets"]
    train_opt = dict(dataset_options["train"])
    train_opt.update({"phase": "train", "scale": cfg["scale"]})
    val_opt = dict(dataset_options["val_1"])
    val_opt.update({"phase": "val", "scale": cfg["scale"]})

    # Resolve relative dataset paths from the repository root, as BasicSR is
    # expected to be launched from that directory.
    for opt in (train_opt, val_opt):
        root = Path(opt["dataroot_gt"]).expanduser()
        if not root.is_absolute():
            opt["dataroot_gt"] = str((repo_root / root).resolve())

    train_dataset = PairedMatDataset8xAug(train_opt)
    val_dataset = PairedMatDataset8xAug(val_opt)
    print(f"Training source MAT files: {len(train_dataset.mat_files)}")
    print(f"Training augmented entries: {len(train_dataset)}")
    print(f"Validation MAT files/entries: {len(val_dataset)}")
    require(len(train_dataset.mat_files) == 651, "Expected 651 training source MAT files.")
    require(len(train_dataset) == 5208, "Expected 5,208 augmented training entries.")
    require(len(val_dataset) == 62, "Expected 62 validation samples.")

    print("\n========== FORWARD PASS CHECK ==========")
    with torch.inference_mode():
        for label, dataset, index in (
            ("train", train_dataset, 0),
            ("validation", val_dataset, 0),
        ):
            sample = dataset[index]
            lq = sample["lq"].unsqueeze(0).to(device=device, dtype=torch.float32)
            lms = sample["lms"].unsqueeze(0).to(device=device, dtype=torch.float32)
            gt = sample["gt"]
            prediction = model(lq, lms)
            require(tuple(prediction.shape) == (1, 128, 128, 128),
                    f"Unexpected {label} output shape: {tuple(prediction.shape)}")
            require(tuple(gt.shape) == (128, 128, 128),
                    f"Unexpected {label} GT shape: {tuple(gt.shape)}")
            require(torch.isfinite(prediction).all().item(),
                    f"Non-finite prediction on {label} sample.")
            print(f"{label}: input={tuple(lq.shape)}, output={tuple(prediction.shape)}, finite=PASS")

    print("\n========== PREFLIGHT PASSED ==========")
    print("No optimizer was stepped. No checkpoint/config was modified.")
    print("No experiment directory was created. Test MAT data was not opened.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PREFLIGHT FAILED: {exc}", file=sys.stderr)
        raise
