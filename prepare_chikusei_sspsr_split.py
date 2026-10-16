
from pathlib import Path
import h5py
import numpy as np
from scipy.io import savemat, loadmat
import shutil

from basicsr.utils.img_util import imresize_np

# -------------------- PATHS --------------------
RAW = (
    Path.home()
    / "hemanth-ug4/datasets/Chikusei_raw_reextracted/"
      "Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat"
)
OUT = Path.home() / "hemanth-ug4/datasets/Chikusei_x4_sspsr_split"

PATCH = 128
STRIDE = 64
SCALE = 4
SEED = 42

# -------------------- SAFETY CHECKS --------------------
if not RAW.exists():
    raise FileNotFoundError(f"Raw dataset not found: {RAW}")

if OUT.exists() and any(OUT.iterdir()):
    raise FileExistsError(
        f"Output directory is not empty: {OUT}\n"
        "Do not overwrite it. Choose a new output directory."
    )

train_dir = OUT / "trains"
eval_dir = OUT / "evals"
test_dir = OUT

train_dir.mkdir(parents=True, exist_ok=True)
eval_dir.mkdir(parents=True, exist_ok=True)

# -------------------- LOAD RAW CUBE --------------------
print("Loading Chikusei cube...", flush=True)

with h5py.File(RAW, "r") as f:
    cube = np.asarray(f["chikusei"][:], dtype=np.float32)     

# MATLAB source layout: bands x rows x columns
# Convert to rows x columns x bands.
if cube.ndim != 3 or cube.shape[0] != 128:
    raise ValueError(f"Unexpected raw cube shape: {cube.shape}")

cube = cube.transpose(2, 1, 0)

# MATLAB crop: rows 107:2410, columns 144:2191
cube = cube[106:2410, 143:2191, :].copy()

# Global maximum normalization, as in the SSPSR preprocessing code.
maximum = float(np.max(cube))
if not np.isfinite(maximum) or maximum <= 0:
    raise ValueError(f"Invalid cube maximum: {maximum}")

cube /= maximum

if not np.isfinite(cube).all():
    raise ValueError("Cube contains NaN or infinite values.")

print(f"Cropped cube: {cube.shape}", flush=True)
print(f"Normalized range: {cube.min():.6f} to {cube.max():.6f}",
      flush=True)

# -------------------- FOUR HELD-OUT TEST SCENES --------------------
print("Preparing four test scenes...", flush=True)

test_scenes = []

for i, x in enumerate((0, 512, 1024, 1536)):
    gt = np.ascontiguousarray(cube[:512, x:x + 512, :])
    lr = imresize_np(gt, 1.0 / SCALE)
    lms = imresize_np(lr, float(SCALE))

    if gt.shape != (512, 512, 128):
        raise ValueError(f"Unexpected test GT shape: {gt.shape}")
    if lr.shape != (128, 128, 128):
        raise ValueError(f"Unexpected test LR shape: {lr.shape}")
    if lms.shape != gt.shape:
        raise ValueError(f"Unexpected test LMS shape: {lms.shape}")

    test_scenes.append((gt, lr, lms))
    print(f"Test scene {i + 1}/4 prepared", flush=True)

test_path = OUT / "Chikusei_test.mat"
savemat(
    str(test_path),
    {
        "gt": np.stack([item[0] for item in test_scenes]).astype(np.float32),
        "ms": np.stack([item[1] for item in test_scenes]).astype(np.float32),
        "ms_bicubic": np.stack([item[2] for item in test_scenes]).astype(np.float32),
    },
    do_compression=True,
)

del test_scenes

# -------------------- PATCH-LEVEL 90/10 SPLIT --------------------
# Use the full area below the four test scenes.
region = cube[512:, :, :]
del cube

print(f"Patch source region: {region.shape}", flush=True)

# Resize the full region before cropping patches.
# This keeps the LR/LMS coordinates aligned with the HR region.
lr_region = imresize_np(region, 1.0 / SCALE).astype(np.float32)
lms_region = imresize_np(lr_region, float(SCALE)).astype(np.float32)

if lms_region.shape != region.shape:
    raise ValueError(
        f"HR/LMS shape mismatch: {region.shape} vs {lms_region.shape}"
    )

coordinates = [
    (y, x)
    for y in range(0, region.shape[0] - PATCH + 1, STRIDE)
    for x in range(0, region.shape[1] - PATCH + 1, STRIDE)
]

rng = np.random.default_rng(SEED)
permutation = rng.permutation(len(coordinates))

n_eval = int(np.ceil(0.10 * len(coordinates)))
eval_indices = set(permutation[:n_eval].tolist())

print(
    f"Total patches: {len(coordinates)} | "
    f"Train: {len(coordinates) - n_eval} | Eval: {n_eval}",
    flush=True,
)

for i, (y, x) in enumerate(coordinates):
    gt = np.ascontiguousarray(region[y:y + PATCH, x:x + PATCH, :])
    lms = np.ascontiguousarray(lms_region[y:y + PATCH, x:x + PATCH, :])

    ly, lx = y // SCALE, x // SCALE
    lr = np.ascontiguousarray(
        lr_region[
            ly:ly + PATCH // SCALE,
            lx:lx + PATCH // SCALE,
            :
        ]
    )

    if gt.shape != (PATCH, PATCH, 128):
        raise ValueError(f"Bad GT patch shape at {(y, x)}: {gt.shape}")
    if lr.shape != (PATCH // SCALE, PATCH // SCALE, 128):
        raise ValueError(f"Bad LR patch shape at {(y, x)}: {lr.shape}")
    if lms.shape != gt.shape:
        raise ValueError(f"Bad LMS patch shape at {(y, x)}: {lms.shape}")

    is_eval = i in eval_indices
    destination = eval_dir if is_eval else train_dir
    prefix = "eval" if is_eval else "train"
    patch_path = destination / f"{prefix}_{i:06d}.mat"

    savemat(
        str(patch_path),
        {
            "gt": gt.astype(np.float32),
            "ms": lr.astype(np.float32),
            "ms_bicubic": lms.astype(np.float32),
        },
        do_compression=True,
    )

    if (i + 1) % 50 == 0 or i + 1 == len(coordinates):
        print(f"Saved {i + 1}/{len(coordinates)} patches", flush=True)

# -------------------- VALIDATE OUTPUT --------------------
print("Validating generated dataset...", flush=True)

train_files = sorted(train_dir.glob("*.mat"))
eval_files = sorted(eval_dir.glob("*.mat"))

if not train_files or not eval_files:
    raise RuntimeError("Training or evaluation folder is empty.")

for path in (train_files[0], eval_files[0]):
    data = loadmat(path)
    for key in ("gt", "ms", "ms_bicubic"):
        if key not in data:
            raise RuntimeError(f"{path.name} is missing key {key}")

    assert data["gt"].shape == (128, 128, 128)
    assert data["ms"].shape == (32, 32, 128)
    assert data["ms_bicubic"].shape == (128, 128, 128)

test_data = loadmat(test_path)
assert test_data["gt"].shape == (4, 512, 512, 128)
assert test_data["ms"].shape == (4, 128, 128, 128)
assert test_data["ms_bicubic"].shape == (4, 512, 512, 128)

print("\nDATASET BUILD COMPLETE", flush=True)
print(f"Train patches: {len(train_files)}", flush=True)
print(f"Eval patches:  {len(eval_files)}", flush=True)
print(f"Test file:     {test_path}", flush=True)
print(f"Output:        {OUT}", flush=True)
