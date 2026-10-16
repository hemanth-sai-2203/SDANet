from pathlib import Path
import math
import h5py
import numpy as np
import scipy.io as sio
from basicsr.utils.img_util import imresize_np

RAW = Path.home() / (
    "hemanth-ug4/datasets/Chikusei_raw_reextracted/"
    "Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat"
)
OUT = Path.home() / "hemanth-ug4/datasets/Chikusei_x4_rebuilt"
PATCH, STRIDE, SCALE = 128, 64, 4
SEED = 42

if not RAW.exists():
    raise FileNotFoundError(RAW)

# Refuse to mix with an incomplete prior run.
if OUT.exists() and any(OUT.iterdir()):
    raise RuntimeError(
        f"{OUT} is not empty. Preserve it and choose a fresh output "
        "directory before running."
    )
(OUT / "trains").mkdir(parents=True, exist_ok=True)
(OUT / "evals").mkdir(parents=True, exist_ok=True)

print("Loading raw cube...", flush=True)
with h5py.File(RAW, "r") as f:
    cube = np.asarray(f["chikusei"], dtype=np.float32)

if cube.shape[0] == 128:
    cube = cube.transpose(2, 1, 0)
if cube.shape != (2517, 2335, 128):
    raise ValueError(f"Unexpected cube shape: {cube.shape}")

# SSPSR spatial crop and global-max normalization.
cube = cube[106:2410, 143:2191, :]
cube /= float(cube.max())
cube = np.clip(cube, 0, 1).astype(np.float32)
assert cube.shape == (2304, 2048, 128)
print("Cropped cube:", cube.shape, flush=True)

def resize_pair(scene):
    # SDANet repo implementation of MATLAB-style bicubic imresize.
    lr = imresize_np(scene, 1.0 / SCALE, antialiasing=True)
    lms = imresize_np(lr, float(SCALE), antialiasing=True)
    return lr.astype(np.float32), lms.astype(np.float32)

# Test scenes: first 512 rows, four 512x512 spatial tiles.
test_file = OUT / "Chikusei_test.mat"
test_gt, test_lr, test_lms = [], [], []
for j, left in enumerate((0, 512, 1024, 1536), start=1):
    gt = cube[:512, left:left+512, :].copy()
    lr, lms = resize_pair(gt)
    test_gt.append(gt)
    test_lr.append(lr)
    test_lms.append(lms[:512, :512, :])
    print(f"Test scene {j}/4 resized", flush=True)

sio.savemat(test_file, {
    "gt": np.stack(test_gt),
    "ms": np.stack(test_lr),
    "ms_bicubic": np.stack(test_lms),
}, do_compression=True)

# Remaining spatial area is split by non-overlapping source rows:
# this avoids train/eval overlap. This is a documented adaptation,
# not the exact SSPSR "manually move 10%" patch procedure.
region = cube[512:, :, :]
del cube

# Use 90/10 contiguous spatial split, with a 128-pixel guard band.
eval_height = math.ceil(region.shape[0] * 0.10)
eval_height = math.ceil(eval_height / STRIDE) * STRIDE
guard = PATCH
eval_start = region.shape[0] - eval_height
train_end = eval_start - guard
if train_end < PATCH or eval_height < PATCH:
    raise RuntimeError("Train/eval spatial regions are too small.")

train_gt = region[:train_end, :, :]
eval_gt = region[eval_start:, :, :]
del region

def save_patches(area, folder, prefix):
    h, w, _ = area.shape
    coords = [
        (y, x)
        for y in range(0, h - PATCH + 1, STRIDE)
        for x in range(0, w - PATCH + 1, STRIDE)
    ]
    print(f"{prefix}: {len(coords)} patches from {area.shape}", flush=True)
    lr_area, lms_area = resize_pair(area)
    count = 0
    for n, (y, x) in enumerate(coords):
        path = folder / f"patch_{n:06d}.mat"
        gt = area[y:y+PATCH, x:x+PATCH, :]
        ly, lx, ls = y // SCALE, x // SCALE, PATCH // SCALE
        lr = lr_area[ly:ly+ls, lx:lx+ls, :]
        lms = lms_area[y:y+PATCH, x:x+PATCH, :]
        if gt.shape != (PATCH, PATCH, 128):
            raise RuntimeError(f"Bad GT shape at {path}: {gt.shape}")
        if lr.shape != (ls, ls, 128) or lms.shape != gt.shape:
            raise RuntimeError(
                f"Bad paired shapes: GT={gt.shape}, LR={lr.shape}, LMS={lms.shape}"
            )
        tmp = path.with_suffix(".tmp.mat")
        sio.savemat(tmp, {"gt": gt, "ms": lr, "ms_bicubic": lms},
                    do_compression=True)
        tmp.replace(path)
        count += 1
        if count % 25 == 0 or count == len(coords):
            print(f"{prefix}: {count}/{len(coords)}", flush=True)
    return count

n_train = save_patches(train_gt, OUT / "trains", "train")
n_eval = save_patches(eval_gt, OUT / "evals", "eval")

# Basic post-write validation.
for folder, expected in (("trains", n_train), ("evals", n_eval)):
    files = sorted((OUT / folder).glob("*.mat"))
    if len(files) != expected or not files:
        raise RuntimeError(f"{folder}: expected {expected}, found {len(files)}")
    d = sio.loadmat(files[0])
    assert d["gt"].shape == (PATCH, PATCH, 128)
    assert d["ms"].shape == (PATCH//SCALE, PATCH//SCALE, 128)
    assert d["ms_bicubic"].shape == (PATCH, PATCH, 128)

d = sio.loadmat(test_file)
assert d["gt"].shape == (4, 512, 512, 128)
assert d["ms"].shape == (4, 128, 128, 128)
assert d["ms_bicubic"].shape == (4, 512, 512, 128)

print("DATASET BUILD COMPLETE")
print("Train patches:", n_train)
print("Eval patches:", n_eval)
print("Test scenes:", d["gt"].shape[0])
print("Output:", OUT)
