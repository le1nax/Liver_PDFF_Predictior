"""
Dataset and collate utilities for liver fat fraction classification.

Wraps LiverFatScalarDataset via composition, converting continuous FF targets
into discrete steatosis-grade labels.
"""

from typing import List, Optional

import torch
from torch.utils.data import Dataset

from scalar_regression.dataset_scalar import LiverFatScalarDataset


CLASS_NAMES = ["Healthy", "Slight", "Mild", "Strong"]
DEFAULT_THRESHOLDS = [0.05, 0.15, 0.25]


def ff_to_class(ff_value: float, thresholds: List[float] = None) -> int:
    """Convert a fat fraction value in [0, 1] to a class index.

    Default thresholds: Healthy <5%, Slight 5-15%, Mild 15-25%, Strong >=25%.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    for i, t in enumerate(thresholds):
        if ff_value < t:
            return i
    return len(thresholds)


class LiverFatClassificationDataset(Dataset):
    """
    Dataset that wraps LiverFatScalarDataset and converts the continuous FF
    target to a discrete class label.

    Returns dict with keys:
        t2         : (1, D, H, W) float tensor
        mask       : (1, D, H, W) float tensor
        target     : () long tensor (class index)
        ff_value   : () float tensor (original continuous FF)
        patient_id : str
    """

    def __init__(
        self,
        scalar_dataset: LiverFatScalarDataset,
        thresholds: Optional[List[float]] = None,
    ):
        self.scalar_dataset = scalar_dataset
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    @property
    def patient_ids(self) -> List[str]:
        return self.scalar_dataset.patient_ids

    def __len__(self) -> int:
        return len(self.scalar_dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.scalar_dataset[idx]
        ff_value = sample["target"].item()
        class_idx = ff_to_class(ff_value, self.thresholds)

        return {
            "t2": sample["t2"],
            "mask": sample["mask"],
            "target": torch.tensor(class_idx, dtype=torch.long),
            "ff_value": sample["target"].squeeze(),
            "patient_id": sample["patient_id"],
        }


def pad_collate_classification(batch: List[dict]) -> dict:
    """Pad variable-size volumes to the max size in the batch (classification variant)."""
    max_d = max(item["t2"].shape[1] for item in batch)
    max_h = max(item["t2"].shape[2] for item in batch)
    max_w = max(item["t2"].shape[3] for item in batch)

    batch_size = len(batch)
    t2_batch = torch.zeros((batch_size, 1, max_d, max_h, max_w), dtype=torch.float32)
    mask_batch = torch.zeros((batch_size, 1, max_d, max_h, max_w), dtype=torch.float32)
    targets = []
    ff_values = []
    patient_ids = []

    for i, item in enumerate(batch):
        t2 = item["t2"]
        d, h, w = t2.shape[1:]
        t2_batch[i, :, :d, :h, :w] = t2
        if "mask" in item and item["mask"] is not None:
            mask_batch[i, :, :d, :h, :w] = item["mask"]
        else:
            mask_batch[i, :, :d, :h, :w] = 1.0
        targets.append(item["target"])
        ff_values.append(item["ff_value"])
        patient_ids.append(item["patient_id"])

    return {
        "t2": t2_batch,
        "mask": mask_batch,
        "target": torch.stack(targets, dim=0),
        "ff_value": torch.stack(ff_values, dim=0),
        "patient_id": patient_ids,
    }


def compute_class_weights(dataset: LiverFatClassificationDataset,
                          num_classes: int = 4) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Returns a float tensor of shape (num_classes,).
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    for i in range(len(dataset)):
        label = dataset[i]["target"].item()
        counts[label] += 1

    # Avoid division by zero for empty classes
    counts = counts.clamp(min=1.0)
    weights = len(dataset) / (num_classes * counts)
    return weights.float()
