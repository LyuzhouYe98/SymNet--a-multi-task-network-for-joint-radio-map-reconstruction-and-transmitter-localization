#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from symnet.data import SymNetDataset
from symnet.model import SymNet

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train SymNet with the paper configuration.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=Path("runs/symnet"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--devices", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="Full training checkpoint, including optimizer state")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one training and one validation batch")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["cross_attention_layers"] != 1:
        parser.error("The checkpoint-compatible architecture has exactly one cross-attention layer")
    args.batch_size = config["batch_size"] if args.batch_size is None else args.batch_size
    args.epochs = config["epochs"] if args.epochs is None else args.epochs
    args.seed = config["seed"] if args.seed is None else args.seed
    L.seed_everything(args.seed, workers=True)
    model = SymNet(
        img_size=config["image_size"], patch_size=config["patch_size"],
        embed_dim=config["embedding_width"], num_heads=config["attention_heads"],
        depth=config["self_attention_depth"], mlp_ratio=config["mlp_ratio"],
        skipnet_learning_rate=config["skipnet_learning_rate"],
        other_learning_rate=config["other_learning_rate"],
        optimizer_name=config["optimizer"], weight_decay=config["weight_decay"],
        radio_map_loss_weight=config["radio_map_loss_weight"],
        localization_loss_weight=config["localization_loss_weight"],
    )

    dataset_args = dict(
        dnb_cutoff=config["dnb_cutoff_pixels"],
        localization_sigma=config["localization_gaussian_sigma_pixels"],
    )
    train_set = SymNetDataset(args.data_root, split="train", **dataset_args)
    val_set = SymNetDataset(args.data_root, split="val", **dataset_args)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="symnet-{epoch:03d}-{val_loss:.5f}",
        monitor=config["checkpoint_monitor"],
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp" if len(args.devices) > 1 else "auto",
        max_epochs=args.epochs,
        callbacks=[checkpoint],
        default_root_dir=args.output,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
