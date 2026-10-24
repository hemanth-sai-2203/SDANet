import argparse
import copy
import inspect
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
import yaml

import basicsr.metrics
from basicsr.archs import build_network
from basicsr.archs.SDAnet_arch import LayerNorm
from basicsr.metrics import calculate_metric
from validate_chikusei_degradation import matlab_bicubic_downsample


def load_model(config, device):
    model = build_network(config["network_g"]).to(device)
    checkpoint = torch.load(
        config["checkpoint"], map_location="cpu", weights_only=False
    )
    state = checkpoint[config["checkpoint_key"]]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def select_norm_affine_parameters(model):
    selected = []
    selected_names = []
    for module_name, module in model.named_modules():
        if isinstance(module, LayerNorm):
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name)
                selected.append(parameter)
                selected_names.append(f"{module_name}.{parameter_name}")
    if not selected:
        raise RuntimeError(
            "The instantiated model has no custom LayerNorm affine parameters; "
            "refusing to unfreeze arbitrary parameters."
        )
    return selected, selected_names


def adaptation_objective(model, lq, lms, frozen_anchor,
                         observation_weight, anchor_weight):
    prediction = model(lq, lms)
    reprojected = matlab_bicubic_downsample(prediction, scale=0.25)
    observation_loss = F.l1_loss(reprojected, lq)
    anchor_loss = F.l1_loss(prediction, frozen_anchor)
    total_loss = observation_weight * observation_loss + anchor_weight * anchor_loss
    return prediction, observation_loss, anchor_loss, total_loss


def adapt_scene(model, lq, lms, frozen_anchor, settings):
    """Adapt using only LR, LMS, and a frozen original model prediction."""
    if "gt" in inspect.signature(adapt_scene).parameters:
        raise AssertionError("GT must not be an adaptation-function argument.")
    if frozen_anchor.requires_grad:
        raise AssertionError("The prediction anchor must be detached and frozen.")

    trainable_parameters, trainable_names = select_norm_affine_parameters(model)
    selected_ids = {id(parameter) for parameter in trainable_parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
    if any(
        parameter.requires_grad and id(parameter) not in selected_ids
        for parameter in model.parameters()
    ):
        raise AssertionError("A non-normalization parameter became trainable.")

    optimizer = torch.optim.Adam(
        trainable_parameters, lr=settings["learning_rate"], weight_decay=0
    )
    initial_parameter_snapshot = [
        parameter.detach().clone() for parameter in trainable_parameters
    ]
    with torch.no_grad():
        _, initial_observation, initial_anchor, initial_total = adaptation_objective(
            model, lq, lms, frozen_anchor,
            settings["observation_weight"], settings["anchor_weight"]
        )
    if float(initial_anchor) != 0.0:
        raise AssertionError("Initial anchor loss must be zero for the baseline output.")

    accepted = 0
    rejected = 0
    gradient_norms = []
    parameter_delta = 0.0
    current_objective = float(initial_total)

    for _ in range(settings["steps"]):
        optimizer.zero_grad(set_to_none=True)
        _, observation_loss, anchor_loss, total_loss = adaptation_objective(
            model, lq, lms, frozen_anchor,
            settings["observation_weight"], settings["anchor_weight"]
        )
        if not torch.isfinite(total_loss):
            rejected += 1
            break
        total_loss.backward()

        gradient_parameters = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if id(parameter) not in selected_ids:
                    raise AssertionError(f"Unexpected gradient reached {name}.")
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"Non-finite gradient in {name}.")
                gradient_parameters.append(parameter)
        if not gradient_parameters:
            raise RuntimeError("No selected normalization affine parameter received gradients.")

        grad_norm = torch.nn.utils.clip_grad_norm_(
            gradient_parameters, settings["max_grad_norm"]
        )
        gradient_norms.append(float(grad_norm))
        parameter_snapshot = [p.detach().clone() for p in trainable_parameters]
        optimizer_snapshot = copy.deepcopy(optimizer.state_dict())
        optimizer.step()

        parameters_finite = all(
            torch.isfinite(parameter).all() for parameter in trainable_parameters
        )
        if parameters_finite:
            with torch.no_grad():
                _, next_observation, next_anchor, next_total = adaptation_objective(
                    model, lq, lms, frozen_anchor,
                    settings["observation_weight"], settings["anchor_weight"]
                )
            candidate_objective = float(next_total)
            stable = (
                math.isfinite(candidate_objective)
                and candidate_objective
                <= current_objective * (1 + settings["maximum_objective_increase_ratio"])
            )
        else:
            stable = False

        if stable:
            accepted += 1
            current_objective = candidate_objective
            parameter_delta = math.sqrt(sum(
                float((parameter.detach() - original).square().sum())
                for parameter, original in zip(
                    trainable_parameters, initial_parameter_snapshot
                )
            ))
        else:
            with torch.no_grad():
                for parameter, original in zip(trainable_parameters, parameter_snapshot):
                    parameter.copy_(original)
            optimizer.load_state_dict(optimizer_snapshot)
            rejected += 1
            break

    with torch.no_grad():
        final_prediction, final_observation, final_anchor, final_total = adaptation_objective(
            model, lq, lms, frozen_anchor,
            settings["observation_weight"], settings["anchor_weight"]
        )

    if accepted and parameter_delta <= settings["minimum_parameter_delta"]:
        raise RuntimeError("Accepted adaptation steps did not change parameters measurably.")
    return final_prediction.detach(), {
        "initial_observation_loss": float(initial_observation),
        "final_observation_loss": float(final_observation),
        "initial_anchor_loss": float(initial_anchor),
        "final_anchor_loss": float(final_anchor),
        "initial_objective": float(initial_total),
        "final_objective": float(final_total),
        "gradient_norms_preclip": gradient_norms,
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "parameter_change_l2": parameter_delta,
        "trainable_parameter_names": trainable_names,
        "optimizer_state_is_scene_local": True,
        "model_state_is_scene_local": True,
    }


def run_zero_step_identity_test(device):
    class TinyNormModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = LayerNorm(2)

        def forward(self, lq, lms):
            upsampled = F.interpolate(lq, scale_factor=4, mode="nearest")
            return self.norm(upsampled + lms)

    model = TinyNormModel().to(device).eval()
    lq = torch.rand((1, 2, 2, 2), device=device)
    lms = torch.rand((1, 2, 8, 8), device=device)
    with torch.no_grad():
        baseline = model(lq, lms).detach().clone()
        observed_lq = matlab_bicubic_downsample(baseline, scale=0.25)
    result, record = adapt_scene(
        model,
        lq,
        lms,
        baseline,
        {
            "steps": 0,
            "learning_rate": 1e-5,
            "observation_weight": 1.0,
            "anchor_weight": 10.0,
            "max_grad_norm": 0.01,
            "maximum_objective_increase_ratio": 0.25,
            "minimum_parameter_delta": 1e-12,
        },
    )
    if not torch.equal(result, baseline):
        raise AssertionError("Zero adaptation steps did not reproduce the baseline.")
    if record["accepted_updates"] != 0 or record["rejected_updates"] != 0:
        raise AssertionError("Zero-step identity test unexpectedly updated parameters.")


def run_gradient_scope_preflight(model, lq, lms, frozen_anchor, settings):
    if "gt" in inspect.signature(adaptation_objective).parameters:
        raise AssertionError("Ground truth must not enter the adaptation objective.")
    trainable_parameters, trainable_names = select_norm_affine_parameters(model)
    selected_ids = {id(parameter) for parameter in trainable_parameters}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)

    model.zero_grad(set_to_none=True)
    prediction, observation_loss, anchor_loss, total_loss = adaptation_objective(
        model,
        lq,
        lms,
        frozen_anchor,
        settings["observation_weight"],
        settings["anchor_weight"],
    )
    if float(anchor_loss.detach()) != 0.0:
        raise AssertionError("Preflight anchor loss is not zero against baseline output.")
    total_loss.backward()

    received_gradients = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if id(parameter) not in selected_ids:
            raise AssertionError(f"Unexpected preflight gradient reached {name}.")
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Non-finite preflight gradient in {name}.")
        received_gradients.append((name, parameter.grad))
    if not received_gradients:
        raise RuntimeError("No selected normalization parameter received a gradient.")
    if not any(torch.count_nonzero(gradient) for _, gradient in received_gradients):
        raise RuntimeError("Selected normalization gradients are all zero.")
    if frozen_anchor.grad is not None:
        raise AssertionError("Gradient propagated into the frozen baseline anchor.")

    gradient_norm = torch.sqrt(sum(
        gradient.detach().square().sum()
        for _, gradient in received_gradients
    ))
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "event": "gradient_scope_preflight_passed",
        "selected_parameter_tensors": len(trainable_parameters),
        "parameters_with_gradients": [name for name, _ in received_gradients],
        "selected_parameter_names": trainable_names,
        "all_gradients_finite": True,
        "all_gradients_within_selected_norm_affine": True,
        "anchor_loss": float(anchor_loss.detach()),
        "anchor_has_no_gradient": frozen_anchor.grad is None,
        "gradient_norm": float(gradient_norm),
        "observation_loss": float(observation_loss.detach()),
        "objective_finite": bool(torch.isfinite(total_loss.detach())),
        "no_optimizer_step_performed": True,
        "prediction_shape": tuple(prediction.shape),
    }


def prediction_metrics(prediction, ground_truth, metrics_config):
    data = {"img": prediction, "img2": ground_truth}
    return {
        name: float(calculate_metric(data, config))
        for name, config in metrics_config.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Run a controlled SDANet OC-TTA pilot.")
    parser.add_argument(
        "--config", type=Path,
        default=Path("options/test/HSI/SDANetOCTTAPilot.yml")
    )
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    output_dir = Path(config["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir()
    log_path = output_dir / "pilot.jsonl"
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    run_zero_step_identity_test(device)

    mat = sio.loadmat(config["data_path"], variable_names=["ms", "ms_bicubic"])
    lr_scenes = np.asarray(mat["ms"], dtype=np.float32)
    lms_scenes = np.asarray(mat["ms_bicubic"], dtype=np.float32)
    if lr_scenes.shape[0] != 4 or lms_scenes.shape[0] != 4:
        raise ValueError("Expected four LR/LMS scenes.")
    if lr_scenes.shape != (4, 128, 128, 128):
        raise ValueError(f"Unexpected LR shape: {lr_scenes.shape}")
    if lms_scenes.shape != (4, 512, 512, 128):
        raise ValueError(f"Unexpected LMS shape: {lms_scenes.shape}")

    reference_dir = Path(config["baseline_reference_dir"])
    metric_results = []
    with log_path.open("x", encoding="utf-8") as log_file:
        def report(record):
            line = json.dumps(record, sort_keys=True)
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        report({
            "event": "start",
            "timestamp": datetime.now().astimezone().isoformat(),
            "config": str(args.config.resolve()),
            "checkpoint": config["checkpoint"],
            "checkpoint_key": config["checkpoint_key"],
            "ground_truth_loaded_during_adaptation": False,
            "settings": config["adaptation"],
        })

        baseline_predictions = []
        baseline_reference_errors = []
        for scene_index in range(4):
            lq = torch.from_numpy(
                lr_scenes[scene_index].transpose(2, 0, 1)[None]
            ).to(device)
            lms = torch.from_numpy(
                lms_scenes[scene_index].transpose(2, 0, 1)[None]
            ).to(device)

            baseline_model = load_model(config, device)
            with torch.no_grad():
                baseline_tensor = baseline_model(lq, lms).detach()
            baseline_prediction = baseline_tensor[0].permute(1, 2, 0).cpu().numpy()
            reference_path = reference_dir / f"test_img_{scene_index}_baseline_reference.mat"
            if not reference_path.is_file():
                raise FileNotFoundError(
                    f"Missing baseline reference prediction: {reference_path}; "
                    "run the configured BasicSR reference test first."
                )
            reference_data = sio.loadmat(reference_path, variable_names=["sr_data"])
            reference_prediction = np.asarray(reference_data["sr_data"], dtype=np.float32)
            max_baseline_error = float(np.max(np.abs(
                baseline_prediction.astype(np.float64)
                - reference_prediction.astype(np.float64)
            )))
            if max_baseline_error > config["baseline_prediction_tolerance"]:
                raise RuntimeError(
                    f"Scene {scene_index + 1} baseline differs from BasicSR output: "
                    f"max_abs={max_baseline_error:.9g}"
                )
            baseline_predictions.append(baseline_prediction)
            baseline_reference_errors.append(max_baseline_error)
            sio.savemat(
                prediction_dir / f"scene_{scene_index + 1}_baseline.mat",
                {"sr_data": baseline_prediction},
                do_compression=True,
            )

            del baseline_tensor, baseline_model, reference_data, lq, lms
            if device.type == "cuda":
                torch.cuda.empty_cache()

        report({
            "event": "baseline_gate_passed",
            "scene_max_abs_errors": baseline_reference_errors,
            "tolerance": config["baseline_prediction_tolerance"],
            "all_scenes_verified_before_adaptation": True,
        })

        preflight_lq = torch.from_numpy(
            lr_scenes[0].transpose(2, 0, 1)[None]
        ).to(device)
        preflight_lms = torch.from_numpy(
            lms_scenes[0].transpose(2, 0, 1)[None]
        ).to(device)
        preflight_model = load_model(config, device)
        preflight_anchor = torch.from_numpy(
            baseline_predictions[0].transpose(2, 0, 1)[None]
        ).to(device).detach()
        gradient_preflight = run_gradient_scope_preflight(
            preflight_model,
            preflight_lq,
            preflight_lms,
            preflight_anchor,
            config["adaptation"],
        )
        report(gradient_preflight)
        del preflight_model, preflight_lq, preflight_lms, preflight_anchor
        if device.type == "cuda":
            torch.cuda.empty_cache()

        adapted_predictions = []
        adaptation_records = []
        for scene_index in range(4):
            lq = torch.from_numpy(
                lr_scenes[scene_index].transpose(2, 0, 1)[None]
            ).to(device)
            lms = torch.from_numpy(
                lms_scenes[scene_index].transpose(2, 0, 1)[None]
            ).to(device)
            adapted_model = load_model(config, device)
            if not any(isinstance(module, LayerNorm) for module in adapted_model.modules()):
                raise AssertionError("Fresh adaptation model has no eligible LayerNorm.")
            frozen_anchor = torch.from_numpy(
                baseline_predictions[scene_index].transpose(2, 0, 1)[None]
            ).to(device).detach()
            adapted_tensor, adaptation_record = adapt_scene(
                adapted_model,
                lq,
                lms,
                frozen_anchor,
                config["adaptation"],
            )
            adapted_prediction = adapted_tensor[0].permute(1, 2, 0).cpu().numpy()
            adapted_predictions.append(adapted_prediction)
            adaptation_record.update({
                "event": "scene_adapted",
                "scene": scene_index + 1,
                "baseline_reference_max_abs_error": baseline_reference_errors[scene_index],
                "fresh_adaptation_model_loaded": True,
                "fresh_optimizer_created": True,
            })
            adaptation_records.append(adaptation_record)
            report(adaptation_record)
            sio.savemat(
                prediction_dir / f"scene_{scene_index + 1}_adapted.mat",
                {"sr_data": adapted_prediction},
                do_compression=True,
            )
            del adapted_model, adapted_tensor, frozen_anchor, lq, lms
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # HR ground truth is loaded only after all scene adaptation is complete.
        gt_mat = sio.loadmat(config["data_path"], variable_names=["gt"])
        gt_scenes = np.asarray(gt_mat["gt"], dtype=np.float32)
        for scene_index in range(4):
            baseline_scores = prediction_metrics(
                baseline_predictions[scene_index], gt_scenes[scene_index], config["metrics"]
            )
            adapted_scores = prediction_metrics(
                adapted_predictions[scene_index], gt_scenes[scene_index], config["metrics"]
            )
            result = {
                "event": "scene_metrics",
                "scene": scene_index + 1,
                "baseline": baseline_scores,
                "adapted": adapted_scores,
                "delta_adapted_minus_baseline": {
                    name: adapted_scores[name] - baseline_scores[name]
                    for name in baseline_scores
                },
            }
            metric_results.append(result)
            report(result)

        names = list(metric_results[0]["baseline"])
        mean_result = {
            "event": "mean_metrics",
            "baseline": {
                name: float(np.mean([item["baseline"][name] for item in metric_results]))
                for name in names
            },
            "adapted": {
                name: float(np.mean([item["adapted"][name] for item in metric_results]))
                for name in names
            },
        }
        mean_result["delta_adapted_minus_baseline"] = {
            name: mean_result["adapted"][name] - mean_result["baseline"][name]
            for name in names
        }
        expected_metrics = config["historical_baseline_metrics"]["expected"]
        baseline_tolerance = config["historical_baseline_metrics"]["tolerance"]
        baseline_metric_differences = {
            name: mean_result["baseline"][name] - expected_metrics[name]
            for name in expected_metrics
        }
        if any(
            abs(difference) > baseline_tolerance
            for difference in baseline_metric_differences.values()
        ):
            raise RuntimeError(
                "Fresh baseline mean metrics do not match the existing experiment "
                f"within {baseline_tolerance}: {baseline_metric_differences}"
            )
        mean_result["historical_baseline_metric_differences"] = (
            baseline_metric_differences
        )
        mean_result["historical_baseline_tolerance"] = baseline_tolerance
        report(mean_result)
        report({
            "event": "complete",
            "exploratory": True,
            "settings_tuned_on_spatial_eval": False,
            "scene_adaptation_state_reused": False,
            "lr_consistency_is_not_hr_quality_evidence": True,
        })


if __name__ == "__main__":
    main()