"""Compare the portable checkpoint against frozen outputs of the original model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from symnet.model import load_symnet_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/symnet_arxiv2608_00087.ckpt")
    parser.add_argument("--reference", type=Path, default=ROOT / "examples/legacy_reference.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(8, torch.get_num_threads()))
    model = load_symnet_checkpoint(args.checkpoint).to(device).eval()
    reference_suffix = "_cpu" if device.type == "cpu" else ""
    report = {"device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "samples": []}
    with np.load(args.reference, allow_pickle=False) as reference, torch.inference_mode():
        for index, array in enumerate(reference["inputs"]):
            outputs = model(torch.from_numpy(array[None]).to(device))
            errors = {}
            for key, actual in zip(("localization_heatmap", "radio_map"), outputs):
                expected = torch.from_numpy(reference[key + reference_suffix][index:index + 1]).to(device)
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                errors[key + "_max_abs_difference"] = (actual - expected).abs().max().item()
            report["samples"].append(errors)
    report["passed"] = True
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
