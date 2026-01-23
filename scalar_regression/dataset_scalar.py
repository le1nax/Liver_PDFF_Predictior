"""
Dataset and collate utilities for scalar fat fraction regression.
"""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import binary_erosion, generate_binary_structure, rotate


class LiverFatScalarDataset(Dataset):
    """
    Dataset that maps a 3D T2 volume to a single scalar target in [0, 1].

    Target is computed as mean fat fraction within the liver mask.
    """

    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        t2_suffix: str = "_t2_aligned",
        ff_suffix: str = "_ff_normalized",
        mask_suffix: str = "_segmentation",
        input_mask_suffix: str = "_t2_original_segmentation",
        use_subdirs: bool = False,
        use_patient_subdirs: bool = False,
        t2_subdir: str = "t2_images",
        ff_subdir: str = "fat_fraction_maps",
        mask_subdir: Optional[str] = "liver_masks",
        normalize_t2: bool = True,
        log_t2: bool = False,
        normalize_ff: bool = True,
        augment: bool = False,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.2,
        rotate_angle_min: float = 1.0,
        rotate_angle_max: float = 15.0,
        cache_data: bool = False,
        clip_ff_range: Tuple[float, float] = (0.0, 1.0),
        mask_erosion: int = 3,
        transform: Optional[Callable] = None,
        validate_files: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.use_subdirs = use_subdirs
        self.use_patient_subdirs = use_patient_subdirs
        self.t2_suffix = t2_suffix
        self.ff_suffix = ff_suffix
        self.mask_suffix = mask_suffix
        self.input_mask_suffix = input_mask_suffix
        self.normalize_t2 = normalize_t2
        self.log_t2 = log_t2
        self.normalize_ff = normalize_ff
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_angle_min = rotate_angle_min
        self.rotate_angle_max = rotate_angle_max
        self.cache_data = cache_data
        self.clip_ff_range = clip_ff_range
        self.mask_erosion = mask_erosion
        self.transform = transform

        if use_subdirs:
            self.t2_dir = self.data_dir / t2_subdir
            self.ff_dir = self.data_dir / ff_subdir
            self.mask_dir = self.data_dir / mask_subdir if mask_subdir else None
        elif use_patient_subdirs:
            self.t2_dir = self.data_dir
            self.ff_dir = self.data_dir
            self.mask_dir = self.data_dir
        else:
            self.t2_dir = self.data_dir
            self.ff_dir = self.data_dir
            self.mask_dir = self.data_dir

        if patient_ids is None:
            self.patient_ids = self._discover_patient_ids()
        else:
            self.patient_ids = patient_ids

        if validate_files:
            self._validate_files()

        self.cache = {} if cache_data else None

    def _discover_patient_ids(self) -> List[str]:
        if self.use_subdirs:
            t2_files = sorted(list(self.t2_dir.glob("*.nii.gz")) + list(self.t2_dir.glob("*.nii")))
            return [f.stem.replace(".nii", "") for f in t2_files]
        if self.use_patient_subdirs:
            patient_ids = []
            for subdir in sorted(self.data_dir.iterdir()):
                if subdir.is_dir():
                    t2_file = list(subdir.glob(f"*{self.t2_suffix}.nii.gz"))
                    if t2_file:
                        patient_ids.append(subdir.name)
            return patient_ids

        t2_files = sorted(
            list(self.t2_dir.glob(f"*{self.t2_suffix}.nii.gz"))
            + list(self.t2_dir.glob(f"*{self.t2_suffix}.nii"))
        )
        patient_ids = []
        for f in t2_files:
            filename = f.stem.replace(".nii", "")
            if filename.endswith(self.t2_suffix):
                patient_ids.append(filename[: -len(self.t2_suffix)])
        return patient_ids

    def _get_file_path(self, directory: Path, patient_id: str, suffix: str = "") -> Path:
        if self.use_patient_subdirs:
            filename = f"{patient_id}{suffix}"
            path_gz = directory / patient_id / f"{filename}.nii.gz"
            path_nii = directory / patient_id / f"{filename}.nii"
        else:
            filename = f"{patient_id}{suffix}"
            path_gz = directory / f"{filename}.nii.gz"
            path_nii = directory / f"{filename}.nii"

        if path_gz.exists():
            return path_gz
        if path_nii.exists():
            return path_nii
        return path_gz

    def _validate_files(self) -> None:
        valid_patients = []
        invalid_patients = []
        min_size = 8

        for pid in self.patient_ids:
            t2_path = self._get_file_path(self.t2_dir, pid, self.t2_suffix if not self.use_subdirs else "")
            ff_path = self._get_file_path(self.ff_dir, pid, self.ff_suffix if not self.use_subdirs else "")

            missing_or_invalid = []
            if not t2_path.exists():
                missing_or_invalid.append("T2 missing")
            else:
                try:
                    t2_img = nib.load(str(t2_path))
                    shape = t2_img.shape
                    if len(shape) < 3 or any(dim < min_size for dim in shape):
                        missing_or_invalid.append(f"T2 too small: {shape}")
                except Exception as exc:
                    missing_or_invalid.append(f"T2 load error: {exc}")

            if not ff_path.exists():
                missing_or_invalid.append("FF missing")

            if self.mask_dir:
                mask_path = self._get_file_path(
                    self.mask_dir, pid, self.mask_suffix if not self.use_subdirs else ""
                )
                if not mask_path.exists():
                    missing_or_invalid.append("Mask missing")
            if self.input_mask_suffix:
                input_mask_path = self._get_file_path(
                    self.t2_dir, pid, self.input_mask_suffix if not self.use_subdirs else ""
                )
                if not input_mask_path.exists():
                    missing_or_invalid.append("Input mask missing")

            if missing_or_invalid:
                invalid_patients.append((pid, missing_or_invalid))
            else:
                valid_patients.append(pid)

        if invalid_patients:
            print(f"\nWarning: Skipping {len(invalid_patients)} patients with missing files:")
            for pid, missing in invalid_patients[:5]:
                print(f"  - {pid}: {', '.join(missing)}")
            if len(invalid_patients) > 5:
                print(f"  ... and {len(invalid_patients) - 5} more")
            print(f"\nUsing {len(valid_patients)} valid patients (removed {len(invalid_patients)} incomplete)")

        self.patient_ids = valid_patients
        if len(self.patient_ids) == 0:
            raise ValueError("No valid patients found! All patients have missing files.")

    def _load_nifti(self, file_path: Path) -> np.ndarray:
        nii = nib.load(str(file_path))
        return nii.get_fdata().astype(np.float32)

    def _normalize_t2(self, t2_data: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(t2_data, [1, 99])
        t2_norm = np.clip(t2_data, p1, p99)
        return (t2_norm - p1) / (p99 - p1 + 1e-8)

    def _normalize_ff(self, ff_data: np.ndarray) -> np.ndarray:
        if ff_data.max() > 1.5:
            ff_norm = ff_data / 100.0
        else:
            ff_norm = ff_data
        return np.clip(ff_norm, self.clip_ff_range[0], self.clip_ff_range[1])

    def _apply_log_t2(self, t2_data: np.ndarray) -> np.ndarray:
        return np.log(np.clip(t2_data, a_min=0, a_max=None) + 1e-6)

    def _apply_augmentations(
        self,
        t2_data: np.ndarray,
        input_mask: Optional[np.ndarray],
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if not self.augment:
            return t2_data, input_mask

        # Random flips along each axis
        for axis in (0, 1, 2):
            if np.random.rand() < self.flip_prob:
                t2_data = np.flip(t2_data, axis=axis)
                if input_mask is not None:
                    input_mask = np.flip(input_mask, axis=axis)

        # Random in-plane rotations (H, W) by small angles
        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(self.rotate_angle_min, self.rotate_angle_max)
            if np.random.rand() < 0.5:
                angle = -angle
            t2_data = rotate(t2_data, angle=angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            if input_mask is not None:
                input_mask = rotate(input_mask, angle=angle, axes=(1, 2), reshape=False, order=0, mode="nearest")

        return t2_data, input_mask

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        patient_id = self.patient_ids[idx]

        if self.cache is not None and patient_id in self.cache:
            return self.cache[patient_id]

        t2_path = self._get_file_path(self.t2_dir, patient_id, self.t2_suffix if not self.use_subdirs else "")
        ff_path = self._get_file_path(self.ff_dir, patient_id, self.ff_suffix if not self.use_subdirs else "")

        t2_data = self._load_nifti(t2_path)
        ff_data = self._load_nifti(ff_path)

        if self.log_t2:
            t2_data = self._apply_log_t2(t2_data)

        mask_data = None
        if self.mask_dir:
            mask_path = self._get_file_path(
                self.mask_dir, patient_id, self.mask_suffix if not self.use_subdirs else ""
            )
            mask_data = self._load_nifti(mask_path)
            mask_data = (mask_data > 0).astype(np.float32)
            if self.mask_erosion > 0:
                struct = generate_binary_structure(3, 1)
                mask_data = binary_erosion(mask_data, structure=struct, iterations=self.mask_erosion).astype(
                    np.float32
                )

        input_mask = None
        if self.input_mask_suffix:
            input_mask_path = self._get_file_path(
                self.t2_dir, patient_id, self.input_mask_suffix if not self.use_subdirs else ""
            )
            if input_mask_path.exists():
                input_mask = self._load_nifti(input_mask_path)
                input_mask = (input_mask > 0).astype(np.float32)

        t2_data, input_mask = self._apply_augmentations(t2_data, input_mask)

        if self.normalize_t2:
            t2_data = self._normalize_t2(t2_data)
        if self.normalize_ff:
            ff_data = self._normalize_ff(ff_data)

        if mask_data is None or mask_data.sum() == 0:
            target_value = float(np.median(ff_data))
        else:
            target_value = float(np.median(ff_data[mask_data > 0]))

        sample = {
            "t2": torch.from_numpy(t2_data.astype(np.float32)).unsqueeze(0),
            "target": torch.tensor([target_value], dtype=torch.float32),
            "patient_id": patient_id,
        }

        if input_mask is not None:
            input_mask_tensor = torch.from_numpy(input_mask.copy()).unsqueeze(0)
            sample["t2"] = sample["t2"] * input_mask_tensor

        if self.transform:
            sample = self.transform(sample)

        if self.cache is not None:
            self.cache[patient_id] = sample

        return sample


class ScalarInferenceDataset(Dataset):
    """Dataset for scalar inference that only loads T2 volumes."""

    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        t2_suffix: str = "_t2_aligned",
        use_subdirs: bool = False,
        use_patient_subdirs: bool = False,
        t2_subdir: str = "t2_images",
        normalize_t2: bool = True,
        log_t2: bool = False,
        transform: Optional[Callable] = None,
        validate_files: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.use_subdirs = use_subdirs
        self.use_patient_subdirs = use_patient_subdirs
        self.t2_suffix = t2_suffix
        self.normalize_t2 = normalize_t2
        self.log_t2 = log_t2
        self.transform = transform

        if use_subdirs:
            self.t2_dir = self.data_dir / t2_subdir
        else:
            self.t2_dir = self.data_dir

        if patient_ids is None:
            self.patient_ids = self._discover_patient_ids()
        else:
            self.patient_ids = patient_ids

        if validate_files:
            self._validate_files()

    def _discover_patient_ids(self) -> List[str]:
        if self.use_subdirs:
            t2_files = sorted(list(self.t2_dir.glob("*.nii.gz")) + list(self.t2_dir.glob("*.nii")))
            return [f.stem.replace(".nii", "") for f in t2_files]
        if self.use_patient_subdirs:
            patient_ids = []
            for subdir in sorted(self.data_dir.iterdir()):
                if subdir.is_dir():
                    t2_file = list(subdir.glob(f"*{self.t2_suffix}.nii.gz"))
                    if t2_file:
                        patient_ids.append(subdir.name)
            return patient_ids

        t2_files = sorted(
            list(self.t2_dir.glob(f"*{self.t2_suffix}.nii.gz"))
            + list(self.t2_dir.glob(f"*{self.t2_suffix}.nii"))
        )
        patient_ids = []
        for f in t2_files:
            filename = f.stem.replace(".nii", "")
            if filename.endswith(self.t2_suffix):
                patient_ids.append(filename[: -len(self.t2_suffix)])
        return patient_ids

    def _get_file_path(self, directory: Path, patient_id: str, suffix: str = "") -> Path:
        if self.use_patient_subdirs:
            filename = f"{patient_id}{suffix}"
            path_gz = directory / patient_id / f"{filename}.nii.gz"
            path_nii = directory / patient_id / f"{filename}.nii"
        else:
            filename = f"{patient_id}{suffix}"
            path_gz = directory / f"{filename}.nii.gz"
            path_nii = directory / f"{filename}.nii"

        if path_gz.exists():
            return path_gz
        if path_nii.exists():
            return path_nii
        return path_gz

    def _validate_files(self) -> None:
        valid_patients = []
        invalid_patients = []
        min_size = 8

        for pid in self.patient_ids:
            t2_path = self._get_file_path(self.t2_dir, pid, self.t2_suffix if not self.use_subdirs else "")
            missing_or_invalid = []
            if not t2_path.exists():
                missing_or_invalid.append("T2 missing")
            else:
                try:
                    t2_img = nib.load(str(t2_path))
                    shape = t2_img.shape
                    if len(shape) < 3 or any(dim < min_size for dim in shape):
                        missing_or_invalid.append(f"T2 too small: {shape}")
                except Exception as exc:
                    missing_or_invalid.append(f"T2 load error: {exc}")

            if missing_or_invalid:
                invalid_patients.append((pid, missing_or_invalid))
            else:
                valid_patients.append(pid)

        if invalid_patients:
            print(f"\nWarning: Skipping {len(invalid_patients)} patients with missing files:")
            for pid, missing in invalid_patients[:5]:
                print(f"  - {pid}: {', '.join(missing)}")
            if len(invalid_patients) > 5:
                print(f"  ... and {len(invalid_patients) - 5} more")
            print(f"\nUsing {len(valid_patients)} valid patients (removed {len(invalid_patients)} incomplete)")

        self.patient_ids = valid_patients
        if len(self.patient_ids) == 0:
            raise ValueError("No valid patients found! All patients have missing files.")

    def _load_nifti(self, file_path: Path) -> np.ndarray:
        nii = nib.load(str(file_path))
        return nii.get_fdata().astype(np.float32)

    def _normalize_t2(self, t2_data: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(t2_data, [1, 99])
        t2_norm = np.clip(t2_data, p1, p99)
        return (t2_norm - p1) / (p99 - p1 + 1e-8)

    def _apply_log_t2(self, t2_data: np.ndarray) -> np.ndarray:
        return np.log(np.clip(t2_data, a_min=0, a_max=None) + 1e-6)

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        patient_id = self.patient_ids[idx]
        t2_path = self._get_file_path(self.t2_dir, patient_id, self.t2_suffix if not self.use_subdirs else "")
        t2_data = self._load_nifti(t2_path)
        if self.log_t2:
            t2_data = self._apply_log_t2(t2_data)
        if self.normalize_t2:
            t2_data = self._normalize_t2(t2_data)

        sample = {
            "t2": torch.from_numpy(t2_data.astype(np.float32)).unsqueeze(0),
            "patient_id": patient_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def pad_collate_scalar(batch: List[dict]) -> dict:
    """Pad variable-size volumes to the max size in the batch."""
    max_d = max(item["t2"].shape[1] for item in batch)
    max_h = max(item["t2"].shape[2] for item in batch)
    max_w = max(item["t2"].shape[3] for item in batch)

    batch_size = len(batch)
    t2_batch = torch.zeros((batch_size, 1, max_d, max_h, max_w), dtype=torch.float32)
    targets = []
    patient_ids = []

    for i, item in enumerate(batch):
        t2 = item["t2"]
        d, h, w = t2.shape[1:]
        t2_batch[i, :, :d, :h, :w] = t2
        patient_ids.append(item["patient_id"])
        if "target" in item:
            targets.append(item["target"])

    out = {
        "t2": t2_batch,
        "patient_id": patient_ids,
    }
    if targets:
        out["target"] = torch.stack(targets, dim=0)
    return out
