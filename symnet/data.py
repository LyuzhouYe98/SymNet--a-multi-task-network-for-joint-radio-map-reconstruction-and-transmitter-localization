from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
from torch.utils.data import Dataset


@lru_cache(maxsize=64)
def _load_environment(building_path: str, cutoff: float):
    building = _load_gray(building_path)
    distance = distance_transform_edt(building < 0.5).astype(np.float32)
    dnb = 1.0 - np.clip(distance, 0.0, cutoff) / cutoff
    return building, dnb.astype(np.float32)


@lru_cache(maxsize=256)
def _load_gray(path: str):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _localization_heatmap(antenna, sigma=17):
    # Match the archived training script, including Gaussian boundary handling.
    heatmap = gaussian_filter((antenna > 0).astype(np.float32), sigma=sigma, mode="reflect")
    maximum = float(heatmap.max())
    if maximum > 0:
        heatmap /= maximum
    return heatmap.astype(np.float32)


@lru_cache(maxsize=256)
def _load_localization_heatmap(path: str, sigma: float):
    return _localization_heatmap(_load_gray(path), sigma=sigma)


class SymNetDataset(Dataset):
    """Loader for the compact coordinate-mask release format."""

    def __init__(self, data_root, split="train", sample_count=None, dnb_cutoff=20.0,
                 positive_ratio_bin=None, localization_sigma=17.0):
        self.data_root = Path(data_root)
        self.split = split
        self.dnb_cutoff = float(dnb_cutoff)
        self.localization_sigma = float(localization_sigma)
        if self.dnb_cutoff <= 0 or self.localization_sigma <= 0:
            raise ValueError("DNB cutoff and localization Gaussian sigma must be positive")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        if split != "test":
            if sample_count is not None or positive_ratio_bin is not None:
                raise ValueError("Sampling protocol selectors apply only to the test split")
            mask_name = split
        else:
            if (sample_count is None) == (positive_ratio_bin is None):
                raise ValueError("Select exactly one of sample_count or positive_ratio_bin")
            if positive_ratio_bin is not None:
                if positive_ratio_bin not in range(10, 101, 10):
                    raise ValueError("positive_ratio_bin must be a nominal upper bound: 10, 20, ..., 100")
                mask_name = f"test_pr_{positive_ratio_bin - 10}_{positive_ratio_bin}"
            else:
                if sample_count not in (20, 40, 60, 80, 100):
                    raise ValueError("sample_count must be 20, 40, 60, 80, or 100")
                mask_name = f"test_{sample_count}"
        self.mask_name = mask_name
        mask_root = self.data_root / "packed_masks" / mask_name
        if not (mask_root / "complete.json").exists():
            raise FileNotFoundError(f"Missing packed mask set: {mask_root}")
        self.records = np.load(mask_root / "records.npy", mmap_mode="r")
        self.counts = np.load(mask_root / "counts.npy", mmap_mode="r")
        self.coords = np.load(mask_root / "coords_yx.npy", mmap_mode="r")
        self.positive_counts = (
            np.load(mask_root / "positive_counts.npy", mmap_mode="r")
            if positive_ratio_bin is not None else None
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        building_id, transmitter_id = (int(value) for value in self.records[index, :2])
        sample_key = f"{building_id}_{transmitter_id}"
        split_root = self.data_root / "maps" / self.split
        signal = _load_gray(str(split_root / "dpm" / f"{sample_key}.png"))
        antenna_path = str(split_root / "antennas" / f"{sample_key}.png")
        antenna = _load_gray(antenna_path)
        building, dnb = _load_environment(
            str(split_root / "buildings" / f"{building_id}.png"), self.dnb_cutoff
        )

        mask = np.zeros(signal.shape, dtype=np.float32)
        count = int(self.counts[index])
        points = np.asarray(self.coords[index, :count], dtype=np.int64)
        mask[points[:, 0], points[:, 1]] = 1.0
        free_space = 1.0 - building
        masked_signal = signal * mask * free_space
        environment = mask - building
        model_input = np.stack((masked_signal, environment, dnb), axis=0)

        sample = {
            "input": torch.from_numpy(model_input.copy()),
            "radio_map": torch.from_numpy((signal * free_space).copy()),
            "localization_heatmap": torch.from_numpy(
                _load_localization_heatmap(antenna_path, self.localization_sigma).copy()
            ),
            "antenna_map": torch.from_numpy(antenna.copy()),
            "free_space": torch.from_numpy(free_space.copy()),
            "sample_mask": torch.from_numpy(mask),
            "building_id": building_id,
            "transmitter_id": transmitter_id,
        }
        if self.positive_counts is not None:
            sample["sample_count"] = count
            sample["positive_count"] = int(self.positive_counts[index])
        return sample
