#!/usr/bin/env python3
"""Create isolated, reproducible final-evaluation YAMLs for SDANet HSI-SR.

Run from the SDANet repository root. This script never edits the source configs
and refuses to overwrite any existing final-evaluation config.
"""
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "options/test/HSI/final_repro"

SPECS = [
    {
        "label": "baseline",
        "source": ROOT / "options/test/HSI/test_SDAnetChux4.yml",
        "filename": "test_SDANetBaseline33KFinalChux4.yml",
        "name": "SDANetBaseline33K_Final_Chikusei_x4_test",
        "checkpoint": ROOT / "experiments/SDANet_spatial_split_baseline/models/net_g_33000.pth",
        "fine_gate": None,
    },
    {
        "label": "regional_full",
        "source": ROOT / "options/test/HSI/test_SDANetRegionalFullChux4.yml",
        "filename": "test_SDANetRegionalFull66KFinalChux4.yml",
        "name": "SDANetRegionalFull66K_Final_Chikusei_x4_test",
        "checkpoint": ROOT / "experiments/SDANetRegionalContinue66K_spatial_split/models/net_g_66000.pth",
        "fine_gate": False,
    },
    {
        "label": "fine_gate_control",
        "source": ROOT / "options/test/HSI/test_SDANetFineGateControl5KChux4.yml",
        "filename": "test_SDANetFineGateControl71KFinalChux4.yml",
        "name": "SDANetFineGateControl71K_Final_Chikusei_x4_test",
        "checkpoint": ROOT / "experiments/SDANetRegionalFineGateControl5K_spatial_split/models/net_g_5000.pth",
        "fine_gate": False,
    },
    {
        "label": "fine_gate_treatment",
        "source": ROOT / "options/test/HSI/test_SDANetFineGateTreatment5KChux4.yml",
        "filename": "test_SDANetFineGateTreatment71KFinalChux4.yml",
        "name": "SDANetFineGateTreatment71K_Final_Chikusei_x4_test",
        "checkpoint": ROOT / "experiments/SDANetRegionalFineGateTreatment5K_spatial_split/models/net_g_5000.pth",
        "fine_gate": True,
    },
]

BACKEND = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_allow_tf32": False,
    "cuda_matmul_allow_tf32": False,
    "float32_matmul_precision": "highest",
}


def main() -> int:
    missing = []
    for spec in SPECS:
        for key in ("source", "checkpoint"):
            path = spec[key]
            if not path.is_file():
                missing.append(f"{spec['label']} {key} missing: {path}")
    if missing:
        print("Preflight failed:", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    planned = [OUT_DIR / spec["filename"] for spec in SPECS]
    existing = [path for path in planned if path.exists()]
    if existing:
        print("Refusing to overwrite existing configs:", file=sys.stderr)
        print("\n".join(str(path) for path in existing), file=sys.stderr)
        return 3

    prepared = []
    for spec, output_path in zip(SPECS, planned):
        with spec["source"].open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)

        config["name"] = spec["name"]
        config.setdefault("path", {})["pretrain_network_g"] = str(spec["checkpoint"])
        config["path"]["strict_load_g"] = True
        config["path"]["param_key_g"] = "params_ema"
        config.setdefault("val", {})["save_img"] = True
        config["val"]["save_per_image_metrics"] = True
        config["test_backend"] = dict(BACKEND)

        if spec["fine_gate"] is not None:
            if config.get("network_g", {}).get("type") != "SDANET_SADS":
                raise ValueError(
                    f"{spec['label']} must use SDANET_SADS for explicit fine-gate configuration"
                )
            config["network_g"]["fine_gate_enabled"] = spec["fine_gate"]

        with output_path.open("x", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        prepared.append((spec["label"], output_path, config))

    print("Created four isolated final-evaluation configurations:")
    for label, path, config in prepared:
        print(f"\n[{label}]")
        print(f"  config: {path.relative_to(ROOT)}")
        print(f"  checkpoint: {config['path']['pretrain_network_g']}")
        print(f"  architecture: {config['network_g']['type']}")
        print(f"  fine_gate_enabled: {config['network_g'].get('fine_gate_enabled', 'not applicable')}")
        print(f"  save_img: {config['val']['save_img']}")
        print(f"  save_per_image_metrics: {config['val']['save_per_image_metrics']}")
        print(f"  backend: {config['test_backend']}")
    print("\nConfig generation complete. No training or testing was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
