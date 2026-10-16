from pathlib import Path
import h5py
import numpy as np
import scipy.io as sio
from scipy.ndimage import zoom

RAW = Path.home() / "hemanth-ug4/datasets/Chikusei_raw/Chikusei_MATLAB/HyperspecVNIR_Chikusei_20140729.mat"
OUT = Path.home() / "hemanth-ug4/datasets/Chikusei_x4"
PATCH, STRIDE, SCALE = 128, 64, 4

if not RAW.exists():
    raise FileNotFoundError(f"Missing raw cube: {RAW}")

print("Loading Chikusei cube...")
with h5py.File(RAW, "r") as f:
    cube = np.asarray(f["chikusei"], dtype=np.float32)

# HDF5 stores the MATLAB v7.3 dimensions in reversed order.
if cube.shape[0] == 128:
    cube = np.transpose(cube, (2, 1, 0))
elif cube.shape[-1] != 128:
    raise ValueError(f"Unexpected cube shape: {cube.shape}")

print("Original HWC shape:", cube.shape)
# Follow the spatial crop used by the SSPSR Chikusei preparation.
cube = cube[106:2410, 143:2191, :]
if cube.shape != (2304, 2048, 128):
    raise ValueError(f"Unexpected cropped shape: {cube.shape}")

# Global scaling, not per-band normalization.
cube /= float(cube.max())
cube = np.clip(cube, 0, 1).astype(np.float32)

# Hold out the first 512-pixel-high strip for testing.
test_scenes = [cube[:512, x:x+512, :] for x in (0, 512, 1024, 1536)]

# Reserve the next strip for evaluation; train on the remaining area.
eval_area = cube[512:1024, :, :]
train_area = cube[1024:, :, :]

def make_pair(gt):
    # Bicubic downsampling; upsample back to GT size for the LMS input.
    h, w, c = gt.shape
    lr = zoom(gt, (1 / SCALE, 1 / SCALE, 1), order=3)
    lr = np.clip(lr, 0, 1).astype(np.float32)
    lms = zoom(lr, (SCALE, SCALE, 1), order=3)
    lms = np.clip(lms[:h, :w, :], 0, 1).astype(np.float32)
    return gt.astype(np.float32), lr, lms

def save_patches(area, folder):
    folder.mkdir(parents=True, exist_ok=True)
    count = 0
    h, w, _ = area.shape
    for top in range(0, h - PATCH + 1, STRIDE):
        for left in range(0, w - PATCH + 1, STRIDE):
            gt = area[top:top+PATCH, left:left+PATCH, :]
            gt, lr, lms = make_pair(gt)
            sio.savemat(folder / f"patch_{count:06d}.mat",
                        {"gt": gt, "ms": lr, "ms_bicubic": lms},
                        do_compression=True)
            count += 1
    return count

print("Creating training patches...")
n_train = save_patches(train_area, OUT / "trains")
print("Creating evaluation patches...")
n_eval = save_patches(eval_area, OUT / "evals")

print("Creating held-out test file...")
gt_list, lr_list, lms_list = [], [], []
for scene in test_scenes:
    gt, lr, lms = make_pair(scene)
    gt_list.append(gt)
    lr_list.append(lr)
    lms_list.append(lms)

sio.savemat(OUT / "Chikusei_test.mat", {
    "gt": np.stack(gt_list),
    "ms": np.stack(lr_list),
    "ms_bicubic": np.stack(lms_list),
}, do_compression=True)

print(f"Done. Train patches: {n_train}; eval patches: {n_eval}; test scenes: {len(test_scenes)}")
print("Output:", OUT)
