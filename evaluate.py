#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from symnet.data import SymNetDataset
from symnet.metrics import localization_error, masked_rmse, masked_ssim
from symnet.model import load_symnet_checkpoint

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Evaluate SymNet on Table 3 or Table 4 test masks.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/symnet_arxiv2608_00087.ckpt")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    protocol = parser.add_mutually_exclusive_group(required=True)
    protocol.add_argument("--sample-count", type=int, choices=(20, 40, 60, 80, 100))
    protocol.add_argument(
        "--positive-ratio-bin", type=int, choices=range(10, 101, 10),
        help="Table 4 historical stratum; specify its nominal upper bound in percent",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))

    dataset = SymNetDataset(
        args.data_root, split="test", sample_count=args.sample_count,
        positive_ratio_bin=args.positive_ratio_bin,
    )
    mask_name = dataset.mask_name
    if args.max_samples is not None:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    model = load_symnet_checkpoint(args.checkpoint).to(device).eval()
    totals = {"rmse": 0.0, "ssim": 0.0, "localization_error_px": 0.0}
    count = 0
    ratio_sum = 0.0
    below_nominal = 0
    above_nominal = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=mask_name):
            inputs = batch["input"].to(device, non_blocking=True)
            target = batch["radio_map"].to(device, non_blocking=True)
            antenna = batch["antenna_map"].to(device, non_blocking=True)
            free_space = batch["free_space"].to(device, non_blocking=True)
            predicted_heatmap, predicted_map = model(inputs)
            values = {
                "rmse": masked_rmse(predicted_map, target, free_space),
                "ssim": masked_ssim(predicted_map, target, free_space),
                "localization_error_px": localization_error(predicted_heatmap, antenna),
            }
            batch_size = inputs.shape[0]
            for name, value in values.items():
                totals[name] += value.sum().item()
            count += batch_size
            if args.positive_ratio_bin is not None:
                n = batch["sample_count"].long()
                positives = batch["positive_count"].long()
                lower, upper = args.positive_ratio_bin - 10, args.positive_ratio_bin
                ratio_sum += (100.0 * positives.double() / n).sum().item()
                below_nominal += int((100 * positives < lower * n).sum())
                above_nominal += int((100 * positives >= upper * n).sum()) if upper < 100 else 0
    result = {"mask_set": mask_name, "num_examples": count}
    if args.positive_ratio_bin is None:
        result["sample_count"] = args.sample_count
    else:
        result["nominal_positive_ratio_percent"] = [args.positive_ratio_bin - 10, args.positive_ratio_bin]
        result["actual_positive_ratio_percent_mean"] = ratio_sum / count
        result["below_nominal_lower"] = below_nominal
        result["above_nominal_upper"] = above_nominal
    result.update({name: total / count for name, total in totals.items()})
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
