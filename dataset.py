"""
Dataset and DataLoader for T2 MRI to fat fraction prediction.
Handles NIfTI files with optional liver segmentation masks.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from pathlib import Path
from typing import Optional, Tuple, List, Callable
import json
from scipy.ndimage import binary_erosion, generate_binary_structure, rotate


class LiverFatDataset(Dataset):
    """
    Dataset for T2 MRI and fat fraction pairs.

    Supports three directory structures:

    Option 1 - Single directory with all files (RECOMMENDED):
        data_dir/
            patient_001_t2_aligned.nii.gz
            patient_001_ff_normalized.nii.gz
            patient_001_segmentation.nii.gz
            patient_002_t2_aligned.nii.gz
            ...

    Option 2 - Patient subdirectories (direct from preprocessing):
        data_dir/
            patient_001/
                patient_001_t2_aligned.nii.gz
                patient_001_ff_normalized.nii.gz
                patient_001_segmentation.nii.gz
            patient_002/
                patient_002_t2_aligned.nii.gz
                ...

    Option 3 - Separate type subdirectories:
        data_dir/
            t2_images/
                patient_001.nii.gz
                patient_002.nii.gz
            fat_fraction_maps/
                patient_001.nii.gz
                ...
    """

    def __init__(
        self,
        data_dir: str,
        patient_ids: Optional[List[str]] = None,
        t2_suffix: str = '_t2_aligned',
        ff_suffix: str = '_ff_normalized',
        mask_suffix: str = '_segmentation',
        use_subdirs: bool = False,
        use_patient_subdirs: bool = False,
        t2_subdir: str = 't2_images',
        ff_subdir: str = 'fat_fraction_maps',
        mask_subdir: Optional[str] = 'liver_masks',
        transform: Optional[Callable] = None,
        augment: bool = False,
        flip_prob: float = 0.5,
        rotate_prob: float = 0.2,
        rotate_angle_min: float = 1.0,
        rotate_angle_max: float = 15.0,
        normalize_t2: bool = True,
        log_t2: bool = False,
        normalize_ff: bool = True,
        cache_data: bool = False,
        clip_ff_range: Tuple[float, float] = (0.0, 1.0),
        validate_files: bool = True,
        mask_erosion: int = 0
    ):
        """
        Args:
            data_dir: Root directory
            patient_ids: List of patient IDs to include (None = all patients)
            use_patient_subdirs: If True, files are in patient_id/ subdirectories (Option 2)
            t2_suffix: Suffix for T2 files (for single dir mode)
            ff_suffix: Suffix for fat fraction files (for single dir mode)
            mask_suffix: Suffix for mask files (for single dir mode)
            use_subdirs: If True, use subdirectory structure (Option 2)
            t2_subdir: Subdirectory name for T2 images (if use_subdirs=True)
            ff_subdir: Subdirectory name for fat fraction maps (if use_subdirs=True)
            mask_subdir: Subdirectory name for liver masks (if use_subdirs=True)
            transform: Optional transform to apply
            normalize_t2: Whether to normalize T2 images to [0, 1]
            normalize_ff: Whether to normalize FF to [0, 1] (assuming input is percentage)
            cache_data: Cache all data in memory (use only if dataset fits in RAM)
            clip_ff_range: Clip fat fraction values to this range
            mask_erosion: Number of iterations to erode the mask (0 = no erosion)
        """
        self.data_dir = Path(data_dir)
        self.use_subdirs = use_subdirs
        self.use_patient_subdirs = use_patient_subdirs
        self.t2_suffix = t2_suffix
        self.ff_suffix = ff_suffix
        self.mask_suffix = mask_suffix

        if use_subdirs:
            # Option 3: type-based subdirectories
            self.t2_dir = self.data_dir / t2_subdir
            self.ff_dir = self.data_dir / ff_subdir
            self.mask_dir = self.data_dir / mask_subdir if mask_subdir else None
        elif use_patient_subdirs:
            # Option 2: patient subdirectories - handled in _get_file_path
            self.t2_dir = self.data_dir
            self.ff_dir = self.data_dir
            self.mask_dir = self.data_dir
        else:
            # Option 1: single flat directory
            self.t2_dir = self.data_dir
            self.ff_dir = self.data_dir
            self.mask_dir = self.data_dir

        self.transform = transform
        self.augment = augment
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_angle_min = rotate_angle_min
        self.rotate_angle_max = rotate_angle_max
        self.normalize_t2 = normalize_t2
        self.log_t2 = log_t2
        self.normalize_ff = normalize_ff
        self.cache_data = cache_data
        self.clip_ff_range = clip_ff_range
        self.mask_erosion = mask_erosion

        # Get patient IDs
        if patient_ids is None:
            self.patient_ids = self._discover_patient_ids()
        else:
            self.patient_ids = patient_ids

        # Validate that all files exist (only if validate_files=True)
        if validate_files:
            self._validate_files()

        # Cache for storing loaded data
        self.cache = {} if cache_data else None

        print(f"Initialized dataset with {len(self.patient_ids)} patients")
        if self.mask_dir:
            print(f"Using liver segmentation masks")

    def _discover_patient_ids(self) -> List[str]:
        """Automatically discover patient IDs from files"""
        if self.use_subdirs:
            # Option 3: Look for files in t2_subdir
            t2_files = sorted(list(self.t2_dir.glob('*.nii.gz')) + list(self.t2_dir.glob('*.nii')))
            return [f.stem.replace('.nii', '') for f in t2_files]
        elif self.use_patient_subdirs:
            # Option 2: Patient subdirectories - return directory names
            patient_ids = []
            for subdir in sorted(self.data_dir.iterdir()):
                if subdir.is_dir():
                    # Check if it contains expected files
                    t2_file = list(subdir.glob(f'*{self.t2_suffix}.nii.gz'))
                    if t2_file:
                        patient_ids.append(subdir.name)
            return patient_ids
        else:
            # Option 1: Look for files with t2_suffix in main directory
            t2_files = sorted(list(self.data_dir.glob(f'*{self.t2_suffix}.nii.gz')) +
                            list(self.data_dir.glob(f'*{self.t2_suffix}.nii')))
            # Extract patient ID by removing suffix
            patient_ids = []
            for f in t2_files:
                filename = f.stem.replace('.nii', '')
                if filename.endswith(self.t2_suffix):
                    patient_id = filename[:-len(self.t2_suffix)]
                    patient_ids.append(patient_id)
            return patient_ids

    def _validate_files(self):
        """Check that all required files exist and have valid dimensions, remove invalid patients"""
        valid_patients = []
        invalid_patients = []
        min_size = 8  # Minimum size for any dimension (depth=3 U-Net needs 2^3=8)

        for pid in self.patient_ids:
            t2_path = self._get_file_path(self.t2_dir, pid, self.t2_suffix if not self.use_subdirs else '')
            ff_path = self._get_file_path(self.ff_dir, pid, self.ff_suffix if not self.use_subdirs else '')

            # Check if required files exist
            missing_or_invalid = []
            if not t2_path.exists():
                missing_or_invalid.append(f"T2 missing")
            else:
                # Check dimensions
                try:
                    t2_img = nib.load(str(t2_path))
                    shape = t2_img.shape
                    if len(shape) < 3 or any(dim < min_size for dim in shape):
                        missing_or_invalid.append(f"T2 too small: {shape} (min {min_size}x{min_size}x{min_size})")
                except Exception as e:
                    missing_or_invalid.append(f"T2 load error: {str(e)}")

            if not ff_path.exists():
                missing_or_invalid.append(f"FF missing")

            if self.mask_dir:
                mask_path = self._get_file_path(self.mask_dir, pid, self.mask_suffix if not self.use_subdirs else '')
                if not mask_path.exists():
                    missing_or_invalid.append(f"Mask missing")

            # Only keep patients with all required files and valid dimensions
            if missing_or_invalid:
                invalid_patients.append((pid, missing_or_invalid))
            else:
                valid_patients.append(pid)

        # Update patient_ids to only include valid patients
        if invalid_patients:
            print(f"\nWarning: Skipping {len(invalid_patients)} patients with missing files:")
            for pid, missing in invalid_patients[:5]:  # Show first 5
                print(f"  - {pid}: {', '.join(missing)}")
            if len(invalid_patients) > 5:
                print(f"  ... and {len(invalid_patients) - 5} more")
            print(f"\nUsing {len(valid_patients)} valid patients (removed {len(invalid_patients)} incomplete)")

        self.patient_ids = valid_patients

        if len(self.patient_ids) == 0:
            raise ValueError("No valid patients found! All patients have missing files.")

    def _get_file_path(self, directory: Path, patient_id: str, suffix: str = '') -> Path:
        """Get file path, checking both .nii.gz and .nii extensions"""
        if self.use_patient_subdirs:
            # Files are in patient_id/ subdirectory
            filename = f"{patient_id}{suffix}"
            path_gz = directory / patient_id / f"{filename}.nii.gz"
            path_nii = directory / patient_id / f"{filename}.nii"
        else:
            # Files are directly in directory
            filename = f"{patient_id}{suffix}"
            path_gz = directory / f"{filename}.nii.gz"
            path_nii = directory / f"{filename}.nii"

        if path_gz.exists():
            return path_gz
        elif path_nii.exists():
            return path_nii
        else:
            return path_gz  # Return .nii.gz for error message

    def _load_nifti(self, file_path: Path) -> np.ndarray:
        """Load NIfTI file and return data as numpy array"""
        nii = nib.load(str(file_path))
        data = nii.get_fdata()
        return data.astype(np.float32)

    def _normalize_t2(self, t2_data: np.ndarray) -> np.ndarray:
        """Normalize T2 image using percentile-based normalization"""
        # Use percentile normalization to be robust to outliers
        p1, p99 = np.percentile(t2_data, [1, 99])
        t2_norm = np.clip(t2_data, p1, p99)
        t2_norm = (t2_norm - p1) / (p99 - p1 + 1e-8)
        return t2_norm

    def _normalize_ff(self, ff_data: np.ndarray) -> np.ndarray:
        """Normalize fat fraction to [0, 1] range"""
        # Assuming input FF is in percentage (0-100) or already (0-1)
        if ff_data.max() > 1.5:  # Likely in percentage
            ff_norm = ff_data / 100.0
        else:
            ff_norm = ff_data

        # Clip to valid range
        ff_norm = np.clip(ff_norm, self.clip_ff_range[0], self.clip_ff_range[1])
        return ff_norm

    def _apply_log_t2(self, t2_data: np.ndarray) -> np.ndarray:
        return np.log(np.clip(t2_data, a_min=0, a_max=None) + 1e-6)

    def _apply_augmentations(
        self,
        t2_data: np.ndarray,
        ff_data: np.ndarray,
        mask_data: Optional[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not self.augment:
            return t2_data, ff_data, mask_data

        for axis in (0, 1, 2):
            if np.random.rand() < self.flip_prob:
                t2_data = np.flip(t2_data, axis=axis)
                ff_data = np.flip(ff_data, axis=axis)
                if mask_data is not None:
                    mask_data = np.flip(mask_data, axis=axis)

        if np.random.rand() < self.rotate_prob:
            angle = np.random.uniform(self.rotate_angle_min, self.rotate_angle_max)
            if np.random.rand() < 0.5:
                angle = -angle
            t2_data = rotate(t2_data, angle=angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            ff_data = rotate(ff_data, angle=angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            if mask_data is not None:
                mask_data = rotate(mask_data, angle=angle, axes=(1, 2), reshape=False, order=0, mode="nearest")

        return t2_data, ff_data, mask_data

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - 't2': T2 image tensor [1, D, H, W]
                - 'fat_fraction': Fat fraction map [1, D, H, W]
                - 'mask': Liver mask [1, D, H, W] (if available)
                - 'patient_id': Patient identifier
        """
        patient_id = self.patient_ids[idx]

        # Check cache first
        if self.cache is not None and patient_id in self.cache:
            return self.cache[patient_id]

        # Load T2 image
        t2_path = self._get_file_path(self.t2_dir, patient_id, self.t2_suffix if not self.use_subdirs else '')
        t2_data = self._load_nifti(t2_path)

        # Load fat fraction map
        ff_path = self._get_file_path(self.ff_dir, patient_id, self.ff_suffix if not self.use_subdirs else '')
        ff_data = self._load_nifti(ff_path)

        if self.log_t2:
            t2_data = self._apply_log_t2(t2_data)

        # Load liver mask if available
        mask_data = None
        if self.mask_dir:
            mask_path = self._get_file_path(self.mask_dir, patient_id, self.mask_suffix if not self.use_subdirs else '')
            mask_data = self._load_nifti(mask_path)
            mask_data = (mask_data > 0).astype(np.float32)  # Binarize

        if self.normalize_t2:
            t2_data = self._normalize_t2(t2_data)
        if self.normalize_ff:
            ff_data = self._normalize_ff(ff_data)

        # Apply augmentations before erosion so erosion matches augmented masks
        t2_data, ff_data, mask_data = self._apply_augmentations(t2_data, ff_data, mask_data)

        # Apply erosion if requested
        if mask_data is not None and self.mask_erosion > 0:
            # Use 3D structuring element for isotropic erosion
            struct = generate_binary_structure(3, 1)  # 3D cross/connectivity=1
            mask_data = binary_erosion(mask_data, structure=struct, iterations=self.mask_erosion).astype(np.float32)

        # Convert to tensors and add channel dimension (ensure float32)
        t2_tensor = torch.from_numpy(t2_data.astype(np.float32)).unsqueeze(0)  # [1, D, H, W]
        ff_tensor = torch.from_numpy(ff_data.astype(np.float32)).unsqueeze(0)

        sample = {
            't2': t2_tensor,
            'fat_fraction': ff_tensor,
            'patient_id': patient_id
        }

        if mask_data is not None:
            mask_tensor = torch.from_numpy(mask_data).unsqueeze(0)
            sample['mask'] = mask_tensor
        else:
            # Create dummy mask (all ones)
            sample['mask'] = torch.ones_like(t2_tensor)

        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)

        # Cache if enabled
        if self.cache is not None:
            self.cache[patient_id] = sample

        return sample


def create_data_splits(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    save_splits: bool = True,
    use_subdirs: bool = False,
    use_patient_subdirs: bool = False,
    t2_suffix: str = '_t2_aligned'
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits and optionally save them.

    Args:
        data_dir: Root directory containing data
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        random_seed: Random seed for reproducibility
        save_splits: Whether to save splits to JSON file
        use_subdirs: Whether data is in type-based subdirectories (t2_images/, etc.)
        use_patient_subdirs: Whether data is in patient subdirectories (patient_001/, etc.)
        t2_suffix: Suffix for T2 files (for flat or patient subdir modes)

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    data_path = Path(data_dir)

    # Get all patient IDs based on directory structure
    if use_patient_subdirs:
        # Option 1: Patient subdirectories - get directory names
        all_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        all_ids = sorted([d.name for d in all_dirs])
        print(f"Found {len(all_ids)} patient directories")
    elif use_subdirs:
        # Option 3: Type-based subdirectories
        t2_dir = data_path / 't2_images'
        t2_files = sorted(list(t2_dir.glob('*.nii.gz')) + list(t2_dir.glob('*.nii')))
        all_ids = [f.stem.replace('.nii', '') for f in t2_files]
        print(f"Found {len(all_ids)} patients in t2_images/")
    else:
        # Option 2: Single flat directory - look for T2 files with suffix
        t2_files = sorted(list(data_path.glob(f'*{t2_suffix}.nii.gz')) + list(data_path.glob(f'*{t2_suffix}.nii')))
        all_ids = []
        for f in t2_files:
            filename = f.stem.replace('.nii', '')
            if filename.endswith(t2_suffix):
                patient_id = filename[:-len(t2_suffix)]
                all_ids.append(patient_id)
        print(f"Found {len(all_ids)} patients with {t2_suffix} suffix")

    # Shuffle with seed
    np.random.seed(random_seed)
    shuffled_ids = np.random.permutation(all_ids).tolist()

    # Split
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    print(f"Created splits: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

    # Save splits
    if save_splits:
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'random_seed': random_seed
        }
        split_file = Path(data_dir) / 'data_splits.json'
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Saved splits to {split_file}")

    return train_ids, val_ids, test_ids


def load_data_splits(data_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Load previously saved data splits"""
    split_file = Path(data_dir) / 'data_splits.json'
    with open(split_file, 'r') as f:
        splits = json.load(f)
    return splits['train'], splits['val'], splits['test']


def _discover_patient_ids_for_splits(
    data_dir: str,
    use_subdirs: bool = False,
    use_patient_subdirs: bool = False,
    t2_suffix: str = '_t2_aligned'
) -> List[str]:
    data_path = Path(data_dir)
    if use_patient_subdirs:
        all_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        return sorted([d.name for d in all_dirs])
    if use_subdirs:
        t2_dir = data_path / 't2_images'
        t2_files = sorted(list(t2_dir.glob('*.nii.gz')) + list(t2_dir.glob('*.nii')))
        return [f.stem.replace('.nii', '') for f in t2_files]
    t2_files = sorted(list(data_path.glob(f'*{t2_suffix}.nii.gz')) + list(data_path.glob(f'*{t2_suffix}.nii')))
    patient_ids = []
    for f in t2_files:
        filename = f.stem.replace('.nii', '')
        if filename.endswith(t2_suffix):
            patient_ids.append(filename[:-len(t2_suffix)])
    return patient_ids


def _get_file_path_static(
    directory: Path,
    patient_id: str,
    suffix: str,
    use_patient_subdirs: bool
) -> Path:
    filename = f"{patient_id}{suffix}"
    if use_patient_subdirs:
        path_gz = directory / patient_id / f"{filename}.nii.gz"
        path_nii = directory / patient_id / f"{filename}.nii"
    else:
        path_gz = directory / f"{filename}.nii.gz"
        path_nii = directory / f"{filename}.nii"
    if path_gz.exists():
        return path_gz
    if path_nii.exists():
        return path_nii
    return path_gz


def _normalize_ff_static(ff_data: np.ndarray) -> np.ndarray:
    if ff_data.max() > 1.5:
        ff_data = ff_data / 100.0
    return np.clip(ff_data, 0.0, 1.0)


def create_fat_stratified_splits(
    data_dir: str,
    bin_edges: List[float],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    save_splits: bool = True,
    split_filename: str = 'data_splits_fatassigned.json',
    use_subdirs: bool = False,
    use_patient_subdirs: bool = False,
    t2_suffix: str = '_t2_aligned',
    ff_suffix: str = '_ff_normalized',
    mask_suffix: str = '_segmentation',
    mask_erosion: int = 3
) -> Tuple[List[str], List[str], List[str]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    if len(bin_edges) < 2:
        raise ValueError("bin_edges must contain at least two values")

    data_path = Path(data_dir)
    patient_ids = _discover_patient_ids_for_splits(
        data_dir,
        use_subdirs=use_subdirs,
        use_patient_subdirs=use_patient_subdirs,
        t2_suffix=t2_suffix
    )

    struct = generate_binary_structure(3, 1)
    rng = np.random.RandomState(random_seed)
    by_bin: Dict[int, List[str]] = {}
    median_map: Dict[str, float] = {}

    for pid in patient_ids:
        ff_path = _get_file_path_static(data_path, pid, ff_suffix, use_patient_subdirs)
        mask_path = _get_file_path_static(data_path, pid, mask_suffix, use_patient_subdirs)
        if not ff_path.exists() or not mask_path.exists():
            continue
        ff_data = nib.load(str(ff_path)).get_fdata().astype(np.float32)
        mask_data = nib.load(str(mask_path)).get_fdata().astype(np.float32)
        ff_data = _normalize_ff_static(ff_data)
        mask = (mask_data > 0).astype(np.float32)
        if mask_erosion > 0:
            mask = binary_erosion(mask, structure=struct, iterations=mask_erosion).astype(np.float32)
        masked_values = ff_data[mask > 0]
        if masked_values.size == 0:
            continue
        median_ff = float(np.median(masked_values))
        median_map[pid] = median_ff
        bin_idx = len(bin_edges) - 2
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= median_ff < bin_edges[i + 1]:
                bin_idx = i
                break
        by_bin.setdefault(bin_idx, []).append(pid)

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for _, ids in sorted(by_bin.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n_total = len(ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train:n_train + n_val])
        test_ids.extend(ids[n_train + n_val:])

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_ids)

    if save_splits:
        splits = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'random_seed': random_seed,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'bin_edges': bin_edges,
            'median_ff': median_map
        }
        split_file = Path(data_dir) / split_filename
        with open(split_file, 'w') as f:
            json.dump(splits, f, indent=2)
        print(f"Saved fat-stratified splits to {split_file}")

    return train_ids, val_ids, test_ids


def create_inference_yaml(
    data_dir: str,
    test_ids: List[str],
    output_path: str,
    use_patient_subdirs: bool = False,
    t2_suffix: str = '_t2_aligned',
    ff_suffix: str = '_ff_normalized',
    mask_suffix: str = '_segmentation'
):
    """
    Create inference YAML file that maps T2 images to ground truth Dixon images.

    Args:
        data_dir: Root directory containing data
        test_ids: List of test patient IDs
        output_path: Path to save the inference YAML file
        use_patient_subdirs: Whether data is in patient subdirectories
        t2_suffix: Suffix for T2 files
        ff_suffix: Suffix for fat fraction (Dixon) files
        mask_suffix: Suffix for mask files
    """
    import yaml

    data_path = Path(data_dir)
    inference_data = {
        'test_cases': []
    }

    for patient_id in test_ids:
        if use_patient_subdirs:
            # Patient subdirectory structure
            patient_dir = data_path / patient_id
            t2_path = patient_dir / f"{patient_id}{t2_suffix}.nii.gz"
            gt_path = patient_dir / f"{patient_id}{ff_suffix}.nii.gz"
            mask_path = patient_dir / f"{patient_id}{mask_suffix}.nii.gz"
        else:
            # Flat directory structure
            t2_path = data_path / f"{patient_id}{t2_suffix}.nii.gz"
            gt_path = data_path / f"{patient_id}{ff_suffix}.nii.gz"
            mask_path = data_path / f"{patient_id}{mask_suffix}.nii.gz"

        # Only include if all files exist
        if t2_path.exists() and gt_path.exists():
            case = {
                'patient_id': patient_id,
                't2_image': str(t2_path.absolute()),
                'ground_truth': str(gt_path.absolute())
            }

            # Add mask if it exists
            if mask_path.exists():
                case['mask'] = str(mask_path.absolute())

            inference_data['test_cases'].append(case)

    # Save to YAML
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Use custom representer to force all patient IDs to be quoted strings
    def str_representer(dumper, data):
        if data.isdigit():
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    yaml.add_representer(str, str_representer)

    with open(output_file, 'w') as f:
        yaml.dump(inference_data, f, default_flow_style=False, sort_keys=False)

    print(f"Created inference YAML with {len(inference_data['test_cases'])} test cases: {output_file}")

    return output_file


def pad_collate_fn(batch):
    """
    Custom collate function to handle variable-sized volumes.
    Pads all volumes in the batch to match the largest dimensions.
    """
    import torch.nn.functional as F

    # Find maximum dimensions in this batch
    max_d = max([item['t2'].shape[1] for item in batch])
    max_h = max([item['t2'].shape[2] for item in batch])
    max_w = max([item['t2'].shape[3] for item in batch])

    # Pad each item to max size
    padded_batch = []
    for item in batch:
        # Get current dimensions
        _, d, h, w = item['t2'].shape

        # Calculate padding (pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
        pad_w = max_w - w
        pad_h = max_h - h
        pad_d = max_d - d

        # Pad format: (left, right, top, bottom, front, back)
        padding = (0, pad_w, 0, pad_h, 0, pad_d)

        # Pad t2, fat_fraction, and mask
        padded_item = {
            't2': F.pad(item['t2'], padding, mode='constant', value=0),
            'fat_fraction': F.pad(item['fat_fraction'], padding, mode='constant', value=0),
            'mask': F.pad(item['mask'], padding, mode='constant', value=0),
            'patient_id': item['patient_id']
        }
        padded_batch.append(padded_item)

    # Now stack the padded tensors
    return {
        't2': torch.stack([item['t2'] for item in padded_batch]),
        'fat_fraction': torch.stack([item['fat_fraction'] for item in padded_batch]),
        'mask': torch.stack([item['mask'] for item in padded_batch]),
        'patient_id': [item['patient_id'] for item in padded_batch]
    }


def get_dataloaders(
    data_dir: str,
    batch_size: int = 2,
    num_workers: int = 4,
    train_ids: Optional[List[str]] = None,
    val_ids: Optional[List[str]] = None,
    test_ids: Optional[List[str]] = None,
    train_sampler: Optional[torch.utils.data.Sampler] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Root directory containing data
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        train_ids: List of training patient IDs
        val_ids: List of validation patient IDs
        test_ids: List of test patient IDs
        train_sampler: Optional sampler for training dataset
        **dataset_kwargs: Additional arguments for LiverFatDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets (disable validation since IDs come from splits that were already validated)
    train_kwargs = dict(dataset_kwargs)
    val_kwargs = dict(dataset_kwargs)
    test_kwargs = dict(dataset_kwargs)
    val_kwargs["augment"] = False
    test_kwargs["augment"] = False

    train_dataset = LiverFatDataset(data_dir, patient_ids=train_ids, validate_files=False, **train_kwargs)
    val_dataset = LiverFatDataset(data_dir, patient_ids=val_ids, validate_files=False, **val_kwargs)
    test_dataset = LiverFatDataset(data_dir, patient_ids=test_ids, validate_files=False, **test_kwargs)

    collate_fn = pad_collate_fn

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    print("Dataset module - example usage:")
    print("\n# Create data splits:")
    print("train_ids, val_ids, test_ids = create_data_splits('/path/to/data')")
    print("\n# Create dataloaders:")
    print("train_loader, val_loader, test_loader = get_dataloaders(")
    print("    data_dir='/path/to/data',")
    print("    batch_size=2,")
    print("    train_ids=train_ids,")
    print("    val_ids=val_ids,")
    print("    test_ids=test_ids")
    print(")")
