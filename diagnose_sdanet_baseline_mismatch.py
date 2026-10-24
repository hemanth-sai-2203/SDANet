import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import yaml

from basicsr.archs import build_network
from basicsr.models import build_model


def metrics(reference, candidate):
    reference64 = reference.astype(np.float64)
    candidate64 = candidate.astype(np.float64)
    difference = candidate64 - reference64
    absolute = np.abs(difference)
    denominator = np.maximum(np.abs(reference64), 1e-12)
    return {
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
        "relative_l2": float(
            np.linalg.norm(difference.ravel())
            / max(np.linalg.norm(reference64.ravel()), 1e-12)
        ),
        "mean_relative_abs": float(np.mean(absolute / denominator)),
    }


def log_record(log_file, record):
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def state_difference(first, second):
    first_state = first.state_dict()
    second_state = second.state_dict()
    if first_state.keys() != second_state.keys():
        return {"same_keys": False, "different_key_count": len(first_state.keys() ^ second_state.keys())}
    different = []
    maximum = 0.0
    for key in first_state:
        left = first_state[key].detach().cpu()
        right = second_state[key].detach().cpu()
        if not torch.equal(left, right):
            different.append(key)
            maximum = max(maximum, float((left - right).abs().max()))
    return {
        "same_keys": True,
        "different_tensor_count": len(different),
        "maximum_tensor_abs_difference": maximum,
        "different_tensors": different[:20],
    }


def chw_batch(array, contiguous, device):
    chw = np.transpose(array, (2, 0, 1))[None]
    if contiguous:
        chw = np.ascontiguousarray(chw)
    return torch.from_numpy(chw).float().to(device)


def make_basic_sr_options(config):
    return {
        "model_type": "SDAModel",
        "network_g": config["network_g"],
        "num_gpu": 1,
        "is_train": False,
        "dist": False,
        "path": {
            "pretrain_network_g": config["checkpoint"],
            "strict_load_g": True,
            "param_key_g": "params_ema",
        },
    }


def direct_model(config, device):
    model = build_network(config["network_g"]).to(device)
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params_ema"], strict=True)
    model.eval()
    return model


def forward_direct(model, lq, lms):
    model.eval()
    with torch.no_grad():
        return model(lq, lms).detach().cpu().numpy()[0].transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose BasicSR versus direct SDANet inference parity."
    )
    parser.add_argument(
        "--config", type=Path,
        default=Path("options/test/HSI/SDANetOCTTAPilot.yml")
    )
    parser.add_argument(
        "--reference-config", type=Path,
        default=Path("options/test/HSI/test_SDANetOCTTABaselineReference.yml")
    )
    parser.add_argument(
        "--reference-prediction", type=Path,
        default=None,
    )
    parser.add_argument(
        "--log", type=Path,
        default=Path("results/SDANetOCTTA_mismatch_diagnostics/scene_1.jsonl")
    )
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument(
        "--cudnn-benchmark", choices=("true", "false"), default="true"
    )
    parser.add_argument("--save-wrapper-output", type=Path, default=None)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    reference_config = yaml.safe_load(args.reference_config.read_text(encoding="utf-8"))
    if args.reference_prediction is None:
        args.reference_prediction = (
            Path(config["baseline_reference_dir"])
            / f"test_img_{args.scene}_baseline_reference.mat"
        )
    if args.log.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic log: {args.log}")
    if not args.reference_prediction.is_file():
        raise FileNotFoundError(args.reference_prediction)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == "true"
    checkpoint = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    reference_mat = sio.loadmat(
        config["data_path"], variable_names=["ms", "ms_bicubic"]
    )
    observed_lq_hwc = np.asarray(reference_mat["ms"][args.scene], dtype=np.float32)
    lms_hwc = np.asarray(reference_mat["ms_bicubic"][args.scene], dtype=np.float32)
    saved_mat = sio.loadmat(args.reference_prediction, variable_names=["sr_data"])
    saved_prediction = np.asarray(saved_mat["sr_data"], dtype=np.float32)

    with args.log.open("x", encoding="utf-8") as log_file:
        log_record(log_file, {
            "event": "diagnostic_start",
            "timestamp": datetime.now().astimezone().isoformat(),
            "scene_index": args.scene,
            "pilot_config": str(args.config.resolve()),
            "reference_config": str(args.reference_config.resolve()),
            "checkpoint": config["checkpoint"],
            "checkpoint_key": "params_ema",
            "checkpoint_keys": sorted(checkpoint.keys()),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "data_shapes_hwc": {
                "lr": observed_lq_hwc.shape,
                "lms": lms_hwc.shape,
                "saved_prediction": saved_prediction.shape,
            },
            "data_dtype": str(observed_lq_hwc.dtype),
            "input_normalization": "stored float32 values, passed unchanged",
            "reference_save": "BasicSR savemat({'sr_data': HWC float32})",
        })

        # Load both implementations from the same EMA state, then compare every state tensor.
        pilot_net = direct_model(config, device)
        basic_opt = make_basic_sr_options(config)
        basicsr_model = build_model(basic_opt)
        basicsr_model.net_g.eval()
        log_record(log_file, {
            "event": "checkpoint_state_comparison",
            **state_difference(pilot_net, basicsr_model.net_g),
        })
        del checkpoint

        # Match TestMatDatasetCHU exactly: HWC -> contiguous CHW float32 -> batch -> device.
        lq = chw_batch(observed_lq_hwc, contiguous=True, device=device)
        lms = chw_batch(lms_hwc, contiguous=True, device=device)
        basic_data = {"lq": lq.clone(), "lms": lms.clone()}
        log_record(log_file, {
            "event": "input_tensor_facts",
            "lq": {
                "shape": tuple(lq.shape), "dtype": str(lq.dtype),
                "device": str(lq.device), "stride": lq.stride(),
                "contiguous": lq.is_contiguous(),
                "max_abs_difference_from_mat": float(
                    np.max(np.abs(
                        lq[0].permute(1, 2, 0).cpu().numpy() - observed_lq_hwc
                    ))
                ),
            },
            "lms": {
                "shape": tuple(lms.shape), "dtype": str(lms.dtype),
                "device": str(lms.device), "stride": lms.stride(),
                "contiguous": lms.is_contiguous(),
            },
        })

        # BasicSR's actual feed_data/test path.
        basicsr_model.feed_data(basic_data)
        basicsr_model.test()
        wrapper_prediction = basicsr_model.output.detach().cpu().numpy()[0].transpose(1, 2, 0)
        if args.save_wrapper_output is not None:
            args.save_wrapper_output.parent.mkdir(parents=True, exist_ok=True)
            sio.savemat(
                args.save_wrapper_output,
                {"sr_data": wrapper_prediction.astype(np.float32, copy=False)},
            )
        log_record(log_file, {
            "event": "basic_sr_wrapper_vs_saved_mat",
            **metrics(saved_prediction, wrapper_prediction),
        })

        # Direct pilot-style call, with exactly the same model weights and input tensors.
        direct_prediction = forward_direct(pilot_net, lq, lms)
        log_record(log_file, {
            "event": "direct_vs_basic_sr_wrapper",
            **metrics(wrapper_prediction, direct_prediction),
        })
        log_record(log_file, {
            "event": "direct_vs_saved_mat",
            **metrics(saved_prediction, direct_prediction),
        })

        # One-factor input-layout check on the same direct model and backend settings.
        strided_lq = chw_batch(observed_lq_hwc, contiguous=False, device=device)
        strided_lms = chw_batch(lms_hwc, contiguous=False, device=device)
        strided_prediction = forward_direct(pilot_net, strided_lq, strided_lms)
        log_record(log_file, {
            "event": "input_layout_only_strided_vs_contiguous",
            "strided_lq_stride": strided_lq.stride(),
            "contiguous_lq_stride": lq.stride(),
            **metrics(direct_prediction, strided_prediction),
        })

        # One-factor backend checks, restoring the evaluator's benchmark setting each time.
        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_precision = torch.get_float32_matmul_precision()
        selected_benchmark = torch.backends.cudnn.benchmark
        backend_probes = (
            ("cudnn_tf32_only_disabled", False, original_matmul_tf32, original_precision),
            ("cuda_matmul_tf32_only_enabled", original_cudnn_tf32, True, original_precision),
            ("float32_precision_only_highest", original_cudnn_tf32, original_matmul_tf32, "highest"),
        )
        for name, cudnn_tf32, matmul_tf32, precision in backend_probes:
            torch.backends.cudnn.benchmark = selected_benchmark
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.set_float32_matmul_precision(precision)
            candidate = forward_direct(pilot_net, lq, lms)
            log_record(log_file, {
                "event": "backend_setting_only_vs_wrapper",
                "setting": name,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                **metrics(wrapper_prediction, candidate),
            })
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.set_float32_matmul_precision(original_precision)
        torch.backends.cudnn.benchmark = selected_benchmark

        # MAT round-trip check using the wrapper output and the same SciPy serialization path.
        roundtrip_path = args.log.parent / "diagnostic_roundtrip.mat"
        sio.savemat(roundtrip_path, {"sr_data": wrapper_prediction})
        roundtrip = np.asarray(
            sio.loadmat(roundtrip_path, variable_names=["sr_data"])["sr_data"],
            dtype=np.float32,
        )
        log_record(log_file, {
            "event": "mat_roundtrip_wrapper_vs_reloaded",
            "saved_dtype": str(wrapper_prediction.dtype),
            "reloaded_dtype": str(roundtrip.dtype),
            **metrics(wrapper_prediction, roundtrip),
        })

        # Repeat the exact direct inference under unchanged settings.
        repeated_prediction = forward_direct(pilot_net, lq, lms)
        log_record(log_file, {
            "event": "same_settings_repeat_direct_inference",
            **metrics(direct_prediction, repeated_prediction),
        })

        # Record the reference YAML's resolved inference-relevant fields.
        log_record(log_file, {
            "event": "reference_config_facts",
            "name": reference_config["name"],
            "network_g": reference_config["network_g"],
            "checkpoint": reference_config["path"]["pretrain_network_g"],
            "strict_load_g": reference_config["path"]["strict_load_g"],
            "param_key_g": reference_config["path"]["param_key_g"],
            "save_img": reference_config["val"]["save_img"],
            "metrics_used_for_reference_generation": True,
        })


if __name__ == "__main__":
    main()