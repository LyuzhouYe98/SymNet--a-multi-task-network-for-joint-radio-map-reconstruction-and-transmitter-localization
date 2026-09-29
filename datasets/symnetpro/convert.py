"""Build SymNetPro data from the published SymNet maps and fixed scene indices."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


P_MIN = -120.0
SOURCE_P_MAX = -4.588664114340091
MASK_ALGORITHM = "sha256-pcg64-choice-v1"


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_scene_metadata(root, split, building_id):
    with zipfile.ZipFile(root / "scene_index.zip") as archive:
        with archive.open(f"{split}/{building_id}.npz") as member:
            return load_npz(member)


def read_gray(path):
    with Image.open(path) as image:
        result = np.asarray(image.convert("L")).copy()
    if result.shape != (256, 256):
        raise ValueError(f"Expected a 256x256 map: {path}, got {result.shape}")
    return result


def load_sources(maps_root, split, metadata):
    pairs, valid = [], []
    for name in metadata["source_names"]:
        name = str(name)
        signal = read_gray(maps_root / split / "dpm" / name).astype(np.float32) / 255.0
        antenna = (
            read_gray(maps_root / split / "antennas" / name).astype(np.float32) / 255.0
        )
        # This decodes the source PNG; mixed-map model normalization is separate.
        dbm = signal * (SOURCE_P_MAX - P_MIN) + P_MIN
        pairs.append(np.stack((dbm, antenna)))
        valid.append((dbm > P_MIN).astype(np.uint8))
    return np.stack(pairs), np.stack(valid)


def scene_payload(metadata, pairs, valid, index):
    count = int(metadata["scene_counts"][index])
    ids = metadata["scene_source_indices"][index, :count].astype(np.int64)
    if count not in range(1, 6) or (ids < 0).any() or (ids >= len(pairs)).any():
        raise ValueError("Invalid scene source indices")
    width = int(metadata["scene_filename_chars"][index])
    return {
        "pairs": pairs[ids],
        "filenames": np.asarray(metadata["source_names"][ids], dtype=f"<U{width}"),
        "map_id": np.asarray(str(metadata["building_id"])),
        "per_tx_mask": valid[ids],
        "fill_dbm": np.asarray(metadata["scene_fill_dbm"][index], dtype=np.float32),
    }


def mask_stem(scene_name):
    parts = Path(scene_name).stem.split("_")
    return f"{parts[0]}_k1_{parts[3]}" if parts[1] == "k1" else Path(scene_name).stem


def generate_mask(free_pixels, shape, seed, split, scene_name, points, repeat):
    # Seed each mask by identity, not process order or Python's randomized hash().
    key = json.dumps(
        [MASK_ALGORITHM, seed, split, scene_name, points, repeat], separators=(",", ":")
    ).encode("ascii")
    entropy = int.from_bytes(hashlib.sha256(key).digest(), "little")
    rng = np.random.Generator(np.random.PCG64(entropy))
    chosen = rng.choice(free_pixels, size=points, replace=False)
    pixels = np.zeros(shape, dtype=np.uint8)
    pixels.flat[chosen] = 255
    return pixels


def visible_rays(blocked, start, end):
    """Vectorized integer Bresenham, including both endpoints."""
    x, y = start[:, 0].copy(), start[:, 1].copy()
    x1, y1 = end[:, 0].copy(), end[:, 1].copy()
    dx, dy = np.abs(x1 - x), -np.abs(y1 - y)
    sx, sy = np.where(x < x1, 1, -1), np.where(y < y1, 1, -1)
    error = dx + dy
    active = np.arange(len(start))
    visible = np.ones(len(start), dtype=bool)
    while active.size:
        hit = blocked[y, x]
        visible[active[hit]] = False
        keep = ~hit & ((x != x1) | (y != y1))
        active = active[keep]
        if not active.size:
            break
        x, y, x1, y1 = x[keep], y[keep], x1[keep], y1[keep]
        dx, dy, sx, sy = dx[keep], dy[keep], sx[keep], sy[keep]
        error = error[keep]
        twice = 2 * error
        step_x, step_y = twice >= dy, twice <= dx
        error += np.where(step_x, dy, 0) + np.where(step_y, dx, 0)
        x += np.where(step_x, sx, 0)
        y += np.where(step_y, sy, 0)
    return visible


def build_patch_los(building):
    patch_size = 8
    height, width = building.shape
    hp, wp = height // patch_size, width // patch_size
    blocked = building > 127
    reps = np.full((hp * wp, 2), -1, dtype=np.int32)
    valid = np.zeros(hp * wp, dtype=bool)
    for row in range(hp):
        for col in range(wp):
            y0, x0 = row * patch_size, col * patch_size
            ys, xs = np.where(~blocked[y0 : y0 + patch_size, x0 : x0 + patch_size])
            if not len(xs):
                continue
            nearest = np.argmin((xs - 4) ** 2 + (ys - 4) ** 2)
            index = row * wp + col
            reps[index] = [x0 + xs[nearest], y0 + ys[nearest]]
            valid[index] = True
    los = np.diag(valid.astype(np.uint8))
    src, dst = np.triu_indices(len(reps), k=1)
    keep = valid[src] & valid[dst]
    src, dst = src[keep], dst[keep]
    # Match the stored table: trace lower patch ID -> higher ID, then mirror.
    for offset in range(0, len(src), 65536):
        i, j = src[offset : offset + 65536], dst[offset : offset + 65536]
        values = visible_rays(blocked, reps[i], reps[j])
        los[i, j] = values
        los[j, i] = values
    return {
        "patch_size": np.asarray(patch_size, dtype=np.int64),
        "patch_hw": np.asarray([hp, wp], dtype=np.int64),
        "patch_rep_xy": reps,
        "patch_valid_mask": valid,
        "patch_los_matrix": los,
    }


def convert_building(task):
    root, maps_root, output, split, building_id, counts, repeats, seed, with_los = task
    metadata = load_scene_metadata(root, split, building_id)
    building_path = maps_root / split / "buildings" / f"{building_id}.png"
    building = read_gray(building_path)
    free_pixels = np.flatnonzero(building != 255)
    if max(counts) > len(free_pixels):
        raise ValueError(f"Not enough non-building pixels: {split}/{building_id}")
    pairs, valid = load_sources(maps_root, split, metadata)
    target = output / split
    buildings = target / "buildings"
    buildings.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(building_path, buildings / building_path.name)
    distance = distance_transform_edt(building != 255).astype(np.float32)
    np.save(buildings / f"{building_id}.npy", distance)
    if with_los:
        folder = output / "precomputed_patch_los_ps8" / split
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / f"{building_id}.npz", **build_patch_los(building))
    scene_dir = target / "multi_transmitter_pairs_npz_with_mask"
    scene_dir.mkdir(parents=True, exist_ok=True)
    mask_dirs = {}
    for points in counts:
        name = f"test_multi_{points}" if split == "test" else f"masks_multi_{points}"
        mask_dirs[points] = target / name
        mask_dirs[points].mkdir(parents=True, exist_ok=True)
    for index, scene_name in enumerate(metadata["scene_names"]):
        scene_name = str(scene_name)
        np.savez_compressed(
            scene_dir / scene_name, **scene_payload(metadata, pairs, valid, index)
        )
        stem = mask_stem(scene_name)
        for points in counts:
            for repeat in range(repeats):
                pixels = generate_mask(
                    free_pixels, building.shape, seed, split, scene_name, points, repeat
                )
                Image.fromarray(pixels).save(mask_dirs[points] / f"{stem}_{repeat}.png")
    scenes = len(metadata["scene_names"])
    return scenes, scenes * len(counts) * repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--symnet-root",
        type=Path,
        required=True,
        help="Unpacked SymNet release root, or its data/maps directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Directional_Dataset"),
        help="New or empty directory; existing data is never overwritten",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument(
        "--sampling-points",
        nargs="+",
        type=int,
        help="Generate only these counts where available in each split",
    )
    parser.add_argument(
        "--buildings", nargs="+", type=int, help="Optional subset of building IDs"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--with-los", action="store_true", help="Also generate patch-size-8 LOS tables"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.test_repeats < 1:
        parser.error("--workers and --test-repeats must be positive")
    root = Path(__file__).resolve().parent
    with (root / "dataset.json").open() as source:
        info = json.load(source)
    maps_root = args.symnet_root.resolve()
    if (maps_root / "data" / "maps").is_dir():
        maps_root = maps_root / "data" / "maps"
    splits = list(dict.fromkeys(args.splits))
    requested = set(args.sampling_points) if args.sampling_points is not None else None
    available = {
        n for split in splits for n in info["splits"][split]["sampling_points"]
    }
    if requested is not None and requested - available:
        parser.error(f"Available sampling counts: {sorted(available)}")
    buildings = set(map(str, args.buildings)) if args.buildings is not None else None
    all_buildings = {
        b for split in splits for b in info["splits"][split]["building_ids"]
    }
    if buildings is not None and buildings - all_buildings:
        parser.error(
            f"Unknown building IDs in selected splits: {sorted(buildings - all_buildings)}"
        )
    output = args.output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"Output must be a new or empty directory: {output}")
    tasks = []
    for split in splits:
        spec = info["splits"][split]
        counts = [
            n for n in spec["sampling_points"] if requested is None or n in requested
        ]
        if not counts:
            continue
        repeats = args.test_repeats if split == "test" else spec["repeats"]
        for bid in spec["building_ids"]:
            if buildings is not None and bid not in buildings:
                continue
            metadata = load_scene_metadata(root, split, bid)
            for folder, names in (
                ("buildings", [f"{bid}.png"]),
                ("dpm", metadata["source_names"]),
                ("antennas", metadata["source_names"]),
            ):
                for name in names:
                    path = maps_root / split / folder / str(name)
                    if not path.is_file():
                        parser.error(f"Missing SymNet source map: {path}")
            tasks.append(
                (
                    root,
                    maps_root,
                    output,
                    split,
                    bid,
                    counts,
                    repeats,
                    args.seed,
                    args.with_los,
                )
            )
    if not tasks:
        parser.error("No buildings match the requested sampling counts and splits")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "dataset": info["dataset"],
        "mask_algorithm": MASK_ALGORITHM,
        "seed": args.seed,
        "numpy_version": np.__version__,
        "with_los": args.with_los,
        "status": "incomplete",
        "splits": {},
    }
    for _, _, _, split, bid, counts, repeats, _, _ in tasks:
        spec = record["splits"].setdefault(
            split, {"buildings": [], "sampling_points": counts, "repeats": repeats}
        )
        spec["buildings"].append(bid)
    record_path = output / "conversion.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    scenes, masks = 0, 0
    if args.workers == 1:
        for n, m in tqdm(
            map(convert_building, tasks), total=len(tasks), desc="Buildings"
        ):
            scenes += n
            masks += m
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(convert_building, task) for task in tasks]
            for future in tqdm(
                as_completed(futures), total=len(tasks), desc="Buildings"
            ):
                n, m = future.result()
                scenes += n
                masks += m
    record.update(status="complete", scenes=scenes, masks=masks)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Created {scenes:,} scenes and {masks:,} regenerated masks in {output}")


if __name__ == "__main__":
    main()
