import argparse
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from basicsr.utils.img_util import imresize_np


DEFAULT_DATA = Path(
    "/home/aidslab/hemanth-ug4/datasets/Chikusei_x4_sspsr_spatial_split/"
    "Chikusei_test.mat"
)
DEFAULT_LOG = Path("results/degradation_operator_validation_20261024.log")


def cubic(x):
    abs_x = torch.abs(x)
    abs_x2 = abs_x.square()
    abs_x3 = abs_x2 * abs_x
    first = (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * (abs_x <= 1).to(x.dtype)
    second = (
        -0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2
    ) * ((abs_x > 1) & (abs_x <= 2)).to(x.dtype)
    return first + second


def calculate_weights_indices(in_length, out_length, scale, antialiasing):
    kernel_width = 4
    if scale < 1 and antialiasing:
        kernel_width /= scale

    output_coordinates = torch.linspace(1, out_length, out_length)
    input_coordinates = (
        output_coordinates / scale + 0.5 * (1 - 1 / scale)
    )
    left = torch.floor(input_coordinates - kernel_width / 2)
    tap_count = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1) + torch.arange(tap_count).view(1, tap_count)
    distances = input_coordinates.view(out_length, 1) - indices

    if scale < 1 and antialiasing:
        weights = scale * cubic(distances * scale)
    else:
        weights = cubic(distances)
    weights /= weights.sum(dim=1, keepdim=True)

    zero_columns = (weights == 0).sum(dim=0)
    if not math.isclose(float(zero_columns[0]), 0, rel_tol=1e-6):
        indices = indices[:, 1:tap_count - 1]
        weights = weights[:, 1:tap_count - 1]
    if not math.isclose(float(zero_columns[-1]), 0, rel_tol=1e-6):
        indices = indices[:, :tap_count - 2]
        weights = weights[:, :tap_count - 2]

    indices = indices.to(torch.int64).contiguous()
    weights = weights.contiguous()
    symmetric_start = int(-indices.min() + 1)
    symmetric_end = int(indices.max() - in_length)
    indices = indices + symmetric_start - 1
    return weights, indices, symmetric_start, symmetric_end


def symmetric_source_indices(extended_indices, symmetric_start, length):
    source_indices = extended_indices - symmetric_start
    folded = torch.remainder(source_indices, 2 * length)
    return torch.where(folded < length, folded, 2 * length - 1 - folded)


def resize_axis(tensor, axis, scale, antialiasing=True):
    input_length = tensor.shape[axis]
    output_length = math.ceil(input_length * scale)
    weights, indices, symmetric_start, _ = calculate_weights_indices(
        input_length, output_length, scale, antialiasing
    )
    source_indices = symmetric_source_indices(
        indices, symmetric_start, input_length
    ).to(tensor.device)
    weights = weights.to(device=tensor.device, dtype=tensor.dtype)

    output = None
    for tap in range(source_indices.shape[1]):
        sampled = tensor.index_select(axis, source_indices[:, tap])
        weight_shape = [1] * tensor.ndim
        weight_shape[axis] = output_length
        weighted = sampled * weights[:, tap].view(weight_shape)
        output = weighted if output is None else output + weighted
    if output is None:
        raise RuntimeError("Resize axis produced no interpolation taps.")
    return output


def matlab_bicubic_downsample(image_nchw, scale=0.25):
    """Differentiable NCHW equivalent of repository imresize(..., scale)."""
    resized = resize_axis(image_nchw, axis=2, scale=scale)
    return resize_axis(resized, axis=3, scale=scale)


def error_metrics(prediction, reference):
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    absolute = np.abs(difference)
    return {
        "max_abs": float(absolute.max()),
        "mean_abs": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(difference ** 2))),
    }


def format_metrics(metrics):
    return (
        f"max_abs={metrics['max_abs']:.9g}, "
        f"mean_abs={metrics['mean_abs']:.9g}, "
        f"rmse={metrics['rmse']:.9g}"
    )


def run_gradient_test(device):
    torch.manual_seed(20261024)
    image = torch.rand((1, 3, 24, 28), device=device, requires_grad=True)
    output = matlab_bicubic_downsample(image, scale=0.25)
    loss = output.square().mean()
    loss.backward()
    gradient = image.grad
    if gradient is None:
        raise RuntimeError("Autograd did not populate the input gradient.")
    return {
        "output_shape": tuple(output.shape),
        "requires_grad": output.requires_grad,
        "grad_finite": bool(torch.isfinite(gradient).all()),
        "grad_nonzero_count": int(torch.count_nonzero(gradient)),
        "grad_element_count": gradient.numel(),
        "grad_l1": float(gradient.abs().sum()),
        "grad_max_abs": float(gradient.abs().max()),
        "loss": float(loss.detach()),
    }


def make_edge_test_image():
    height, width = 20, 28
    image = np.zeros((height, width, 1), dtype=np.float32)
    image[0, 0, 0] = 1.0
    image[0, -1, 0] = 0.65
    image[-1, 0, 0] = 0.8
    image[-1, -1, 0] = 0.45
    image[:, 0, 0] += np.linspace(0.1, 0.9, height, dtype=np.float32)
    image[:, -1, 0] -= np.linspace(0.05, 0.35, height, dtype=np.float32)
    image[6:10, 9:13, 0] += 0.3
    return image


def run_boundary_test(device):
    edge_image = make_edge_test_image()
    reference = imresize_np(edge_image.copy(), 0.25, antialiasing=True)
    tensor = torch.from_numpy(edge_image.transpose(2, 0, 1)[None]).to(device)
    actual = matlab_bicubic_downsample(tensor, scale=0.25)
    actual_hwc = actual[0].permute(1, 2, 0).cpu().numpy()
    return error_metrics(actual_hwc, reference), actual_hwc.shape


def main():
    parser = argparse.ArgumentParser(
        description="Validate differentiable SDANet bicubic downsampling."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(f"Test MAT file not found: {args.data}")
    if args.log.exists():
        raise FileExistsError(f"Refusing to overwrite existing log: {args.log}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.log.open("x", encoding="utf-8") as log_file:
        def report(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        report("SDANet differentiable degradation validation")
        report(f"timestamp={datetime.now().astimezone().isoformat()}")
        report(f"data={args.data.resolve()}")
        report(f"device={device}")
        report("reference=basicsr.utils.img_util.imresize_np, scale=0.25, antialiasing=True")
        report("input=stored float32 GT; comparison=stored float32 ms; no model loaded")

        data = sio.loadmat(args.data, variable_names=["gt", "ms"])
        ground_truth = np.asarray(data["gt"], dtype=np.float32)
        observations = np.asarray(data["ms"], dtype=np.float32)
        if ground_truth.ndim != 4 or observations.ndim != 4:
            raise ValueError(
                f"Expected N,H,W,C arrays; got GT={ground_truth.shape}, "
                f"LR={observations.shape}"
            )
        if ground_truth.shape[0] != 4:
            raise ValueError(f"Expected four held-out scenes, got {ground_truth.shape[0]}")
        if ground_truth.shape[0] != observations.shape[0]:
            raise ValueError("GT and LR scene counts differ.")

        report(f"gt_shape={ground_truth.shape}; lr_shape={observations.shape}")
        for scene_index in range(ground_truth.shape[0]):
            gt_hwc = ground_truth[scene_index]
            lr_hwc = observations[scene_index]
            if math.ceil(gt_hwc.shape[0] * 0.25) != lr_hwc.shape[0] or math.ceil(
                gt_hwc.shape[1] * 0.25
            ) != lr_hwc.shape[1]:
                raise ValueError(
                    f"Scene {scene_index + 1} shape mismatch: "
                    f"GT={gt_hwc.shape}, LR={lr_hwc.shape}"
                )

            channel_predictions = []
            for channel_start in range(0, gt_hwc.shape[2], 8):
                channel_end = min(channel_start + 8, gt_hwc.shape[2])
                gt_chunk = torch.from_numpy(
                    gt_hwc[:, :, channel_start:channel_end].transpose(2, 0, 1)[None]
                ).to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    lr_chunk = matlab_bicubic_downsample(gt_chunk, scale=0.25)
                channel_predictions.append(
                    lr_chunk[0].permute(1, 2, 0).cpu().numpy()
                )
                del gt_chunk, lr_chunk

            prediction = np.concatenate(channel_predictions, axis=2)
            if prediction.shape != lr_hwc.shape:
                raise ValueError(
                    f"Scene {scene_index + 1} output shape {prediction.shape} "
                    f"does not match LR {lr_hwc.shape}"
                )
            metrics = error_metrics(prediction, lr_hwc)
            results.append(metrics)
            report(f"scene_{scene_index + 1}: {format_metrics(metrics)}")

        aggregate_prediction_metrics = {
            "max_abs": max(item["max_abs"] for item in results),
            "mean_abs": float(np.mean([item["mean_abs"] for item in results])),
            "rmse": float(
                np.sqrt(np.mean([item["rmse"] ** 2 for item in results]))
            ),
        }
        report(f"aggregate_all_pixels: {format_metrics(aggregate_prediction_metrics)}")

        gradient = run_gradient_test(device)
        report(
            "gradient_test: "
            f"output_shape={gradient['output_shape']}, "
            f"requires_grad={gradient['requires_grad']}, "
            f"finite={gradient['grad_finite']}, "
            f"nonzero={gradient['grad_nonzero_count']}/{gradient['grad_element_count']}, "
            f"grad_l1={gradient['grad_l1']:.9g}, "
            f"grad_max_abs={gradient['grad_max_abs']:.9g}, "
            f"loss={gradient['loss']:.9g}"
        )

        boundary_metrics, boundary_shape = run_boundary_test(device)
        report(
            f"boundary_test: output_shape={boundary_shape}, "
            f"{format_metrics(boundary_metrics)}"
        )
        report("tolerance=1e-5 for scene and boundary max_abs error")
        passed = (
            aggregate_prediction_metrics["max_abs"] <= 1e-5
            and boundary_metrics["max_abs"] <= 1e-5
            and gradient["requires_grad"]
            and gradient["grad_finite"]
            and gradient["grad_nonzero_count"] > 0
        )
        report(f"overall_status={'PASS' if passed else 'FAIL'}")

    if not passed:
        raise SystemExit(f"Validation failed; see {args.log}")


if __name__ == "__main__":
    main()