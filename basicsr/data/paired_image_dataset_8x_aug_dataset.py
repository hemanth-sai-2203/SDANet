
import os

import numpy as np
import scipy.io as sio
import torch
from torch.utils import data

from basicsr.utils.registry import DATASET_REGISTRY


def data_augmentation(image, mode=0):
    """Apply one of eight spatial augmentations to an HWC image."""
    if mode == 0:
        return image
    elif mode == 1:
        return np.flipud(image)
    elif mode == 2:
        return np.rot90(image)
    elif mode == 3:
        return np.flipud(np.rot90(image))
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(image, k=2))
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(image, k=3))

    raise ValueError(f"Invalid augmentation mode: {mode}")


@DATASET_REGISTRY.register()
class PairedMatDataset8xAug(data.Dataset):
    """Load paired hyperspectral GT, LR and bicubic-upsampled LMS .mat data.

    Expected .mat keys:
        gt:         HWC ground-truth hyperspectral image
        ms:         HWC low-resolution image
        ms_bicubic: HWC bicubic-upsampled image

    Training uses eight augmentation modes and random aligned crops.
    Validation uses every file exactly once without augmentation or cropping.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.phase = opt["phase"]
        self.scale = int(opt["scale"])

        if self.scale <= 0:
            raise ValueError(f"Scale must be positive, got {self.scale}")

        self.mat_folder = opt["dataroot_gt"]

        if not os.path.isdir(self.mat_folder):
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.mat_folder}"
            )

        self.mat_files = sorted(
            os.path.join(self.mat_folder, filename)
            for filename in os.listdir(self.mat_folder)
            if filename.lower().endswith(".mat")
        )

        if not self.mat_files:
            raise RuntimeError(
                f"No .mat files found in: {self.mat_folder}"
            )

        if self.phase == "train":
            self.gt_size = int(opt["gt_size"])

            if self.gt_size <= 0 or self.gt_size % self.scale != 0:
                raise ValueError(
                    f"gt_size ({self.gt_size}) must be positive and "
                    f"divisible by scale ({self.scale})."
                )

    def __getitem__(self, index):
        if self.phase == "train":
            original_index = index // 8
            augmentation_mode = index % 8
        else:
            original_index = index
            augmentation_mode = 0

        mat_path = self.mat_files[original_index]

        try:
            mat_data = sio.loadmat(mat_path)
            img_gt = np.asarray(mat_data["gt"], dtype=np.float32)
            img_lq = np.asarray(mat_data["ms"], dtype=np.float32)
            img_lms = np.asarray(mat_data["ms_bicubic"], dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load valid GT/LR/LMS data from {mat_path}"
            ) from exc

        if (
            img_gt.ndim != 3
            or img_lq.ndim != 3
            or img_lms.ndim != 3
        ):
            raise ValueError(
                f"Expected HWC 3D arrays in {mat_path}; got "
                f"gt={img_gt.shape}, ms={img_lq.shape}, "
                f"ms_bicubic={img_lms.shape}"
            )

        if img_gt.shape[2] != img_lq.shape[2] or (
            img_gt.shape[2] != img_lms.shape[2]
        ):
            raise ValueError(
                f"Spectral-band mismatch in {mat_path}: "
                f"gt={img_gt.shape}, ms={img_lq.shape}, "
                f"ms_bicubic={img_lms.shape}"
            )

        if img_gt.shape[:2] != img_lms.shape[:2]:
            raise ValueError(
                f"GT/LMS spatial mismatch in {mat_path}: "
                f"gt={img_gt.shape}, ms_bicubic={img_lms.shape}"
            )

        expected_lq_shape = (
            img_gt.shape[0] // self.scale,
            img_gt.shape[1] // self.scale,
        )
        if img_gt.shape[0] % self.scale or img_gt.shape[1] % self.scale:
            raise ValueError(
                f"GT dimensions must be divisible by scale={self.scale}: "
                f"{img_gt.shape} in {mat_path}"
            )

        if img_lq.shape[:2] != expected_lq_shape:
            raise ValueError(
                f"GT/LR dimensions do not match scale={self.scale} in "
                f"{mat_path}: gt={img_gt.shape}, ms={img_lq.shape}, "
                f"expected LR spatial size={expected_lq_shape}"
            )

        if self.phase == "train":
            gt_size = self.gt_size
            h_gt, w_gt, _ = img_gt.shape

            if h_gt < gt_size or w_gt < gt_size:
                raise ValueError(
                    f"GT image {mat_path} is smaller than crop size "
                    f"{gt_size}: spatial size={(h_gt, w_gt)}"
                )

            # Choose GT crop coordinates aligned to the LR grid.
            top = (
                np.random.randint(0, (h_gt - gt_size) // self.scale + 1)
                * self.scale
            )
            left = (
                np.random.randint(0, (w_gt - gt_size) // self.scale + 1)
                * self.scale
            )

            img_gt = img_gt[
                top:top + gt_size,
                left:left + gt_size,
                :
            ]
            img_lms = img_lms[
                top:top + gt_size,
                left:left + gt_size,
                :
            ]

            top_lq = top // self.scale
            left_lq = left // self.scale
            lq_size = gt_size // self.scale

            img_lq = img_lq[
                top_lq:top_lq + lq_size,
                left_lq:left_lq + lq_size,
                :
            ]

            img_gt = data_augmentation(img_gt, augmentation_mode)
            img_lq = data_augmentation(img_lq, augmentation_mode)
            img_lms = data_augmentation(img_lms, augmentation_mode)

        # Convert HWC NumPy arrays to contiguous CHW float32 tensors.
        img_gt = torch.from_numpy(
            np.ascontiguousarray(img_gt.transpose(2, 0, 1))
        ).float()

        img_lq = torch.from_numpy(
            np.ascontiguousarray(img_lq.transpose(2, 0, 1))
        ).float()

        img_lms = torch.from_numpy(
            np.ascontiguousarray(img_lms.transpose(2, 0, 1))
        ).float()

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lms": img_lms,
            "lq_path": mat_path,
            "gt_path": mat_path,
        }

    def __len__(self):
        if self.phase == "train":
            return len(self.mat_files) * 8
        return len(self.mat_files)
