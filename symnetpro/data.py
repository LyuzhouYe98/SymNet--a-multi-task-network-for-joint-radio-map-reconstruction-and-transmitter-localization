"""Multi-source inputs, observed-source targets, and transmitter-drop augmentation."""

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import Dataset

from .graph import PatchGraphBuilder

P_MIN = -120.0
P_MAX = -5.041252136230469


def normalize(dbm):
    return ((dbm - P_MIN) / (P_MAX - P_MIN)).astype(np.float32)


def read_image(path):
    with Image.open(path) as image:
        return np.array(image.convert("L"), dtype=np.float32) / 255.0


def compose(dbms, masks, keep, fill_dbm):
    shape = dbms.shape[-2:]
    if not keep:
        return np.full(shape, P_MIN, dtype=np.float32)
    mw = np.zeros(shape, dtype=np.float32)
    total_mask = np.zeros(shape, dtype=np.uint8)
    for i in keep:
        valid = masks[i].astype(np.float32)
        mw += 10.0 ** (dbms[i] / 10.0) * valid
        total_mask |= (valid > 0).astype(np.uint8)
    result = np.full(shape, fill_dbm, dtype=np.float32)
    valid = total_mask.astype(bool)
    result[valid] = 10.0 * np.log10(mw[valid])
    return result


def antenna_heatmap(ants, shape, sigma=17):
    heatmap = np.zeros(shape, dtype=np.float32)
    if len(ants):
        ys, xs = np.nonzero(np.sum(ants > 0, axis=0) > 0)
        for y, x in zip(ys, xs):
            point = np.zeros(shape, dtype=np.float32)
            point[y, x] = 1.0
            heatmap = np.maximum(
                gaussian_filter(point, sigma=sigma, mode="constant"), heatmap
            )
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
    return heatmap


class SymNetProDataset(Dataset):
    def __init__(
        self,
        root,
        split="train",
        sampling_points=100,
        patch_topk=16,
        augment=False,
        los_root=None,
        max_samples=None,
        noise_db=0.0,
        noise_seed=42,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.noise_db = float(noise_db)
        self.noise_seed = int(noise_seed)
        if augment and noise_db:
            raise ValueError(
                "The released main model uses transmitter-drop, not noisy training."
            )
        self.folder = (
            f"test_multi_{sampling_points}"
            if split == "test"
            else f"masks_multi_{sampling_points}"
        )
        masks = self.root / split / self.folder
        if not masks.is_dir():
            raise FileNotFoundError(f"Missing sampling masks: {masks}")
        self.list_images = sorted(path.name for path in masks.glob("*.png"))
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            self.list_images = self.list_images[:max_samples]
        if not self.list_images:
            raise ValueError(f"No PNG masks in {masks}")
        self.graph = PatchGraphBuilder(root, split, patch_topk, los_root)

    def __len__(self):
        return len(self.list_images)

    def __getitem__(self, index):
        name = self.list_images[index]
        parts = Path(name).stem.split("_")
        building_id = parts[0]
        scene = (
            f"{parts[0]}_k1_{parts[0]}_{parts[2]}.npz"
            if parts[1] == "k1"
            else "_".join(parts[:3]) + ".npz"
        )
        path = self.root / self.split / "multi_transmitter_pairs_npz_with_mask" / scene
        if not path.is_file():
            path = (
                self.root
                / self.split
                / self.split
                / "multi_transmitter_pairs_npz_with_mask"
                / scene
            )
        with np.load(path, allow_pickle=False) as data:
            pairs = data["pairs"].astype(np.float32)
            valid = data["per_tx_mask"].astype(np.uint8)
            fill_dbm = float(data["fill_dbm"]) if "fill_dbm" in data else P_MIN
        dbms, ants = pairs[:, 0], pairs[:, 1]
        building = read_image(
            self.root / self.split / "buildings" / f"{building_id}.png"
        )
        mask = read_image(self.root / self.split / self.folder / name)
        region = (mask > 0) & ((1.0 - building) > 0)
        observed = [
            i for i in range(len(dbms)) if (normalize(dbms[i])[region] > 0).any()
        ]
        selections = {"full": observed}
        if self.augment:
            dropped = int(np.random.choice(observed)) if len(observed) > 1 else None
            selections["drop"] = [i for i in observed if i != dropped]
        pack = self.graph.load_los(building_id)
        result = {"name": name, "nominal_k": int(parts[1][1:])}
        for branch, keep in selections.items():
            clean_dbm = compose(dbms, valid, keep, fill_dbm)
            clean = (
                normalize(clean_dbm)
                if keep
                else np.zeros(building.shape, dtype=np.float32)
            )
            input_map = clean
            if self.noise_db > 0 and keep:
                noisy_dbm = clean_dbm.copy()
                rng = np.random.default_rng(self.noise_seed + index)
                noisy_dbm[region] += rng.normal(
                    0.0, self.noise_db, int(region.sum())
                ).astype(np.float32)
                input_map = normalize(noisy_dbm)
            result[f"data_{branch}"] = np.stack(
                [input_map * mask, mask - building]
            ).astype(np.float32)
            result[f"target_{branch}"] = (clean * (1.0 - building)).astype(np.float32)
            result[f"atts_{branch}"] = antenna_heatmap(ants[keep], building.shape)
            result[f"tx_count_{branch}"] = np.float32(len(keep))
            result[f"{branch}_flag"] = np.float32(
                bool(observed) if branch == "full" else len(observed) > 1
            )
            result[f"graph_{branch}"] = self.graph._build_patch_graph(
                mask, building, input_map, pack
            )
        centers = []
        for ant in ants[observed]:
            ys, xs = np.nonzero(ant > 0)
            if len(xs):
                centers.append([float(xs.mean()), float(ys.mean())])
        result["gt_centers"] = np.asarray(centers, dtype=np.float32).reshape(-1, 2)
        return result


def collate(batch):
    out = {
        "names": [sample["name"] for sample in batch],
        "gt_centers": [sample["gt_centers"] for sample in batch],
        "nominal_k": [sample["nominal_k"] for sample in batch],
    }
    for branch in ("full", "drop"):
        if f"data_{branch}" not in batch[0]:
            continue
        for key in (
            f"data_{branch}",
            f"target_{branch}",
            f"atts_{branch}",
            f"tx_count_{branch}",
            f"{branch}_flag",
        ):
            out[key] = torch.stack([torch.as_tensor(sample[key]) for sample in batch])
        out[f"graphs_{branch}"] = []
        for sample in batch:
            graph = {}
            for key, value in sample[f"graph_{branch}"].items():
                tensor = torch.from_numpy(value)
                graph[key] = (
                    tensor.long()
                    if np.issubdtype(value.dtype, np.integer)
                    else tensor.float()
                )
            out[f"graphs_{branch}"].append(graph)
    return out
