#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from PIL import Image
from tqdm import tqdm


EXPECTED = {
    "train": {"maps": 27400, "masks": 274000, "max_points": 100},
    "val": {"maps": 5900, "masks": 236000, "max_points": 100},
    "test_20": {"maps": 5900, "masks": 590000, "max_points": 20},
    "test_40": {"maps": 5900, "masks": 590000, "max_points": 40},
    "test_60": {"maps": 5900, "masks": 590000, "max_points": 60},
    "test_80": {"maps": 5900, "masks": 590000, "max_points": 80},
    "test_100": {"maps": 5900, "masks": 590000, "max_points": 100},
    **{
        f"test_pr_{upper - 10}_{upper}": {
            "maps": 5900,
            "masks": 590000,
            "max_points": 100,
            "nominal_positive_ratio_percent": [upper - 10, upper],
        }
        for upper in range(10, 101, 10)
    },
}


@lru_cache(maxsize=32)
def positive_signal_pixels(split_root, building_id, transmitter_id):
    with Image.open(split_root / "dpm" / f"{building_id}_{transmitter_id}.png") as image:
        signal = np.asarray(image.convert("L"))
    with Image.open(split_root / "buildings" / f"{building_id}.png") as image:
        building = np.asarray(image.convert("L"))
    return (signal > 0) & (building < 255)


def positive_ratio_summary(counts, positive_counts, lower, upper):
    counts = np.asarray(counts, dtype=np.int32)
    positive_counts = np.asarray(positive_counts, dtype=np.int32)
    percent = 100.0 * positive_counts / counts
    return {
        "actual_percent_min": float(percent.min()),
        "actual_percent_max": float(percent.max()),
        "actual_percent_mean": float(percent.mean()),
        "below_nominal_lower": int(np.count_nonzero(100 * positive_counts < lower * counts)),
        "above_nominal_upper": int(np.count_nonzero(
            100 * positive_counts > upper * counts if upper == 100
            else 100 * positive_counts >= upper * counts
        )),
    }


def numeric_key(name):
    return tuple(int(part) for part in Path(name).stem.split("_"))


def read_mask(path):
    fields = numeric_key(path.name)
    with Image.open(path) as handle:
        image = np.asarray(handle.convert("L"))
    if image.shape != (256, 256) or not np.all((image == 0) | (image == 255)):
        raise ValueError(f"Expected a binary 256x256 mask: {path}")
    coords = np.argwhere(image > 0).astype(np.uint8)
    positive_count = None
    if path.parent.name in {f"masks_{upper}" for upper in range(10, 101, 10)}:
        positive = positive_signal_pixels(path.parent.parent, fields[0], fields[1])
        positive_count = int(positive[coords[:, 0], coords[:, 1]].sum())
    return fields, coords, positive_count


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_base_maps(source_root, output_root):
    for split in ("train", "val", "test"):
        for category in ("buildings", "dpm", "antennas"):
            source_dir = source_root / split / category
            destination_dir = output_root / "maps" / split / category
            files = sorted(source_dir.glob("*.png"), key=lambda path: numeric_key(path.name))
            for source in tqdm(files, desc=f"link {split}/{category}"):
                link_or_copy(source, destination_dir / source.name)


def pack_mask_set(source_dir, destination_dir, expected_count, max_points, workers, ratio_bin=None):
    complete = destination_dir / "complete.json"
    if complete.exists():
        metadata = json.loads(complete.read_text())
        if metadata["source_directory"] != source_dir.name:
            raise ValueError(f"Existing packed masks came from a different source: {destination_dir}")
        print(f"[skip] {destination_dir}")
        return
    names = [entry.name for entry in os.scandir(source_dir) if entry.name.endswith(".png")]
    names.sort(key=numeric_key)
    if len(names) != expected_count:
        raise RuntimeError(f"{source_dir}: expected {expected_count} masks, found {len(names)}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    records = open_memmap(destination_dir / "records.npy", mode="w+", dtype=np.uint16, shape=(len(names), 4))
    counts = open_memmap(destination_dir / "counts.npy", mode="w+", dtype=np.uint8, shape=(len(names),))
    coords = open_memmap(
        destination_dir / "coords_yx.npy", mode="w+", dtype=np.uint8,
        shape=(len(names), max_points, 2),
    )
    records[:] = np.iinfo(np.uint16).max
    coords[:] = np.iinfo(np.uint8).max
    positive_counts = None
    if ratio_bin is not None:
        positive_counts = open_memmap(
            destination_dir / "positive_counts.npy", mode="w+", dtype=np.uint8, shape=(len(names),)
        )
    paths = (source_dir / name for name in names)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(read_mask, paths, chunksize=128)
        for index, (fields, points, positive_count) in enumerate(
            tqdm(results, total=len(names), desc=source_dir.name, mininterval=2)
        ):
            if not 20 <= len(points) <= max_points:
                raise RuntimeError(f"{names[index]} has {len(points)} points; limit is {max_points}")
            if source_dir.name.startswith("masks_smpl_") and len(points) != max_points:
                raise RuntimeError(f"{names[index]} must contain exactly {max_points} points")
            records[index, : len(fields)] = fields
            counts[index] = len(points)
            coords[index, : len(points)] = points
            if positive_counts is not None:
                if positive_count is None:
                    raise RuntimeError(f"Missing positive count: {names[index]}")
                positive_counts[index] = positive_count
    records.flush()
    counts.flush()
    coords.flush()
    metadata = {
        "source_directory": source_dir.name,
        "num_masks": len(names),
        "max_points": max_points,
        "record_columns": ["building_id", "transmitter_id", "condition_or_repeat", "repeat_or_unused"],
        "coordinate_order": "row_y,column_x",
    }
    if positive_counts is not None:
        positive_counts.flush()
        metadata["nominal_positive_ratio_percent"] = list(ratio_bin)
        metadata["positive_count_definition"] = "sampled pixels with nonzero signal outside buildings"
        metadata["positive_ratio_audit"] = positive_ratio_summary(counts, positive_counts, *ratio_bin)
    complete.write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    parser.add_argument("--skip-base-maps", action="store_true")
    args = parser.parse_args()

    if not args.skip_base_maps:
        copy_base_maps(args.source, args.output)
    sources = {
        "train": args.source / "train" / "masks",
        "val": args.source / "val" / "masks",
        **{
            # These are the fixed-cardinality masks used for Table 3. The
            # similarly named masks_<n> folders stratify positive ratio and
            # do not contain a fixed number of sampled points.
            f"test_{count}": args.source / "test" / f"masks_smpl_{count}_random_pos_rate"
            for count in (20, 40, 60, 80, 100)
        },
        **{
            f"test_pr_{upper - 10}_{upper}": args.source / "test" / f"masks_{upper}"
            for upper in range(10, 101, 10)
        },
    }
    for name, source_dir in sources.items():
        settings = EXPECTED[name]
        pack_mask_set(
            source_dir,
            args.output / "packed_masks" / name,
            settings["masks"],
            settings["max_points"],
            args.workers,
            ratio_bin=settings.get("nominal_positive_ratio_percent"),
        )
    (args.output / "dataset_manifest.json").write_text(json.dumps(EXPECTED, indent=2) + "\n")


if __name__ == "__main__":
    main()
