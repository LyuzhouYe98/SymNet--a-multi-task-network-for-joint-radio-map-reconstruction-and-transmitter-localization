"""Evaluate SymNetPro, optionally saving predicted radio/localization maps."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import SymNetProDataset, collate
from .metrics import postprocess
from .model import load_model


def summarize(records):
    result = {"samples": len(records)}
    for name in ("ospa", "mle", "mdr", "far", "miss_count", "fa_count", "radio_rmse"):
        values = [row[name] for row in records if row[name] is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("Directional_Dataset"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--sampling-points", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--postprocess-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--noise-db", type=float, default=0.0)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/symnetpro.json"))
    parser.add_argument(
        "--save-predictions", type=Path, help="Optional per-sample NPZ output directory"
    )
    args = parser.parse_args()
    if (
        min(args.batch_size, args.postprocess_workers, args.cpu_threads) < 1
        or args.num_workers < 0
        or args.noise_db < 0
    ):
        parser.error(
            "Batch/thread counts must be positive; workers/noise must be nonnegative"
        )
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    model, config = load_model(args.checkpoint, args.device)
    threshold = (
        config["localization_threshold"] if args.threshold is None else args.threshold
    )
    dataset = SymNetProDataset(
        args.data_root,
        args.split,
        args.sampling_points,
        config["patch_topk"],
        max_samples=args.max_samples,
        noise_db=args.noise_db,
        noise_seed=args.noise_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        persistent_workers=False,
        pin_memory=str(args.device).startswith("cuda"),
    )
    if args.save_predictions:
        args.save_predictions.mkdir(parents=True, exist_ok=True)
    records = []
    with torch.inference_mode(), ThreadPoolExecutor(
        max_workers=args.postprocess_workers
    ) as pool:
        for batch in tqdm(
            loader, desc=f"{args.split} / {args.sampling_points} points", unit="batch"
        ):
            _, _, localization, radio = model(
                batch["data_full"].to(args.device), batch["graphs_full"]
            )
            localization = localization.squeeze(1).float().cpu().numpy()
            radio = radio.squeeze(1).float().cpu().numpy()
            targets = batch["target_full"].numpy()
            outside = batch["data_full"][:, 1].numpy() > -0.5
            tasks = [
                (
                    heatmap,
                    gt,
                    threshold,
                    config["nms_size"],
                    config["ospa_cutoff"],
                    config["ospa_order"],
                )
                for heatmap, gt in zip(localization, batch["gt_centers"])
            ]
            for i, (points, metrics) in enumerate(pool.map(postprocess, tasks)):
                metrics.update(
                    name=batch["names"][i],
                    nominal_k=batch["nominal_k"][i],
                    radio_rmse=float(
                        np.sqrt(
                            np.sum(((radio[i] - targets[i]) ** 2) * outside[i])
                            / max(1, int(outside[i].sum()))
                        )
                    ),
                )
                records.append(metrics)
                if args.save_predictions:
                    np.savez_compressed(
                        args.save_predictions / (Path(batch["names"][i]).stem + ".npz"),
                        input=batch["data_full"][i].numpy(),
                        radio_map=radio[i],
                        localization_map=localization[i],
                        predicted_xy=points,
                        target_radio_map=targets[i],
                        target_xy=batch["gt_centers"][i],
                    )
    report = {
        "patch_topk": config["patch_topk"],
        "threshold": threshold,
        "noise_db": args.noise_db,
        "sampling_points": args.sampling_points,
        "summary": summarize(records),
        "by_observed_source_count": {
            str(k): summarize([r for r in records if r["num_gt"] == k])
            for k in sorted({r["num_gt"] for r in records})
        },
        "samples": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
