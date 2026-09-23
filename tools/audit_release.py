#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from symnet.data import SymNetDataset
from symnet.model import load_symnet_checkpoint
from tools.prepare_release_data import EXPECTED, positive_ratio_summary


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/symnet_arxiv2608_00087.ckpt"))
    parser.add_argument("--source", type=Path, help="Optionally compare packed masks to the original PNGs")
    parser.add_argument("--samples-per-set", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples_per_set <= 0:
        parser.error("--samples-per-set must be positive")
    rng = np.random.default_rng(42)
    report = {"sets": {}}
    for name, settings in EXPECTED.items():
        map_count, mask_count = settings["maps"], settings["masks"]
        split = name.split("_")[0]
        dpm_count = len(list((args.data_root / "maps" / split / "dpm").glob("*.png")))
        packed = args.data_root / "packed_masks" / name
        metadata = json.loads((packed / "complete.json").read_text())
        records = np.load(packed / "records.npy", mmap_mode="r")
        counts = np.load(packed / "counts.npy", mmap_mode="r")
        coords = np.load(packed / "coords_yx.npy", mmap_mode="r")
        assert dpm_count == map_count, (name, dpm_count, map_count)
        assert len(records) == len(counts) == mask_count, (name, len(records), mask_count)
        assert coords.shape == (mask_count, settings["max_points"], 2), name
        map_ids, repeats = np.unique(records[:, :2], axis=0, return_counts=True)
        assert len(map_ids) == map_count and np.all(repeats == mask_count // map_count), name
        entry = {"maps": dpm_count, "masks": len(records)}
        ratio_bin = settings.get("nominal_positive_ratio_percent")
        positive_counts = None
        if ratio_bin is not None:
            positive_counts = np.load(packed / "positive_counts.npy", mmap_mode="r")
            assert positive_counts.shape == counts.shape and np.all(positive_counts <= counts), name
            assert int(counts.min()) == 20 and int(counts.max()) == 100, name
            entry["nominal_positive_ratio_percent"] = ratio_bin
            entry["positive_ratio_audit"] = positive_ratio_summary(counts, positive_counts, *ratio_bin)
            assert entry["positive_ratio_audit"] == metadata["positive_ratio_audit"], name
        elif name.startswith("test_"):
            expected_points = int(name.split("_")[1])
            assert np.all(counts == expected_points), (name, "non-fixed mask cardinality")
        else:
            assert int(counts.min()) == 20 and int(counts.max()) == 100
        if args.source is not None:
            sample_indices = rng.choice(mask_count, min(args.samples_per_set, mask_count), replace=False)
            source_dir = args.source / split / metadata["source_directory"]
            dataset = (
                SymNetDataset(args.data_root, "test", positive_ratio_bin=ratio_bin[1])
                if ratio_bin is not None else None
            )
            for index in sample_indices:
                fields = [str(int(value)) for value in records[index] if value != 65535]
                source_file = source_dir / ("_".join(fields) + ".png")
                with Image.open(source_file) as image:
                    original = np.asarray(image.convert("L"))
                restored = np.zeros((256, 256), dtype=np.uint8)
                points = coords[index, :int(counts[index])]
                restored[points[:, 0], points[:, 1]] = 255
                assert np.array_equal(original, restored), (name, source_file.name)
                if positive_counts is not None:
                    sample = dataset[int(index)]
                    assert int((sample["input"][0] > 0).sum()) == int(positive_counts[index]), name
            entry["original_png_roundtrips"] = len(sample_indices)
        report["sets"][name] = entry

    sample = SymNetDataset(args.data_root, split="test", sample_count=20)[0]
    assert int(sample["sample_mask"].sum()) == 20
    model = load_symnet_checkpoint(args.checkpoint)
    report["checkpoint"] = {
        "sha256": sha256(args.checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
