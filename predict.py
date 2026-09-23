#!/usr/bin/env python3
"""Run the pretrained model on normalized [B, 3, 256, 256] input arrays."""

import argparse
from pathlib import Path

import numpy as np
import torch

from symnet.metrics import center_of_mass
from symnet.model import load_symnet_checkpoint

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/legacy_reference.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/symnet_arxiv2608_00087.ckpt")
    parser.add_argument("--output", type=Path, default=Path("results/prediction.npz"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    with np.load(args.input, allow_pickle=False) as pack:
        inputs = torch.from_numpy(pack["inputs"].astype(np.float32))
    if inputs.ndim != 4 or inputs.shape[1:] != (3, 256, 256) or len(inputs) == 0:
        parser.error("The NPZ 'inputs' array must have shape [B, 3, 256, 256] with B > 0")
    if not torch.isfinite(inputs).all():
        parser.error("Input contains non-finite values")
    model = load_symnet_checkpoint(args.checkpoint).to(device).eval()
    localizations, maps, coordinates = [], [], []
    with torch.inference_mode():
        for sample in inputs.split(1):
            localization, radio_map = model(sample.to(device))
            localizations.append(localization.cpu().numpy())
            maps.append(radio_map.cpu().numpy())
            coordinates.append(center_of_mass(localization).cpu().numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        localization_heatmap=np.concatenate(localizations),
        radio_map=np.concatenate(maps),
        transmitter_yx=np.concatenate(coordinates),
    )
    print(f"Saved {len(inputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
