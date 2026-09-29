"""Train SymNetPro or resume a complete, configuration-labelled checkpoint."""

import argparse
import json
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader

from .data import SymNetProDataset, collate
from .model import MODEL_CONFIG, build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("Directional_Dataset"))
    parser.add_argument("--output", type=Path, default=Path("runs/symnetpro"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Batch size per device")
    parser.add_argument("--patch-topk", type=int)
    parser.add_argument("--sampling-points", type=int)
    parser.add_argument("--devices", type=int, nargs="+", default=[0])
    parser.add_argument("--accelerator", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--val-batches",
        type=int,
        default=0,
        help="0 disables validation; e.g. 100 enables it",
    )
    parser.add_argument(
        "--limit-train-batches", type=int, help="Optional smoke-test limit"
    )
    parser.add_argument(
        "--max-samples", type=int, help="Optional dataset subset for a smoke test"
    )
    args = parser.parse_args()
    if args.num_workers < 0 or args.val_batches < 0 or args.cpu_threads < 1:
        parser.error(
            "Worker/validation counts must be nonnegative; CPU threads must be positive"
        )
    config_path = args.config or Path(__file__).with_name("config.json")
    config = json.loads(config_path.read_text())
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved = checkpoint.get("symnetpro_config")
        if saved is None or not checkpoint.get("optimizer_states"):
            parser.error(
                "--resume needs a packaged/full training checkpoint, not weights alone"
            )
        if args.config and config != saved:
            parser.error(
                "Resume with the checkpoint's configuration, without --config overrides"
            )
        config = saved.copy()
    for arg, key in (
        (args.patch_topk, "patch_topk"),
        (args.sampling_points, "sampling_points"),
        (args.max_epochs, "max_epochs"),
        (args.batch_size, "batch_size_per_device"),
    ):
        if arg is not None:
            if arg < 1:
                parser.error(f"{key} must be positive")
            if (
                checkpoint
                and key in ("patch_topk", "sampling_points")
                and arg != config[key]
            ):
                parser.error(f"Cannot change {key} while resuming this checkpoint")
            config[key] = arg
    if config["model"] != MODEL_CONFIG:
        parser.error("Unsupported model architecture")
    if checkpoint and config["max_epochs"] <= int(checkpoint["epoch"]) + 1:
        parser.error(
            f"Checkpoint completed epoch {checkpoint['epoch']}; increase --max-epochs"
        )
    if args.limit_train_batches is not None and args.limit_train_batches < 1:
        parser.error("--limit-train-batches must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config["seed"], workers=True)
    use_gpu = args.accelerator == "gpu" or (
        args.accelerator == "auto" and torch.cuda.is_available()
    )
    if not use_gpu and len(args.devices) != 1:
        parser.error("CPU mode supports a single process; use --devices 0")
    options = dict(
        batch_size=config["batch_size_per_device"],
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=use_gpu,
        persistent_workers=False,
    )
    dataset = SymNetProDataset(
        args.data_root,
        "train",
        config["sampling_points"],
        config["patch_topk"],
        augment=True,
        max_samples=args.max_samples,
    )
    train_loader = DataLoader(dataset, shuffle=True, **options)
    val_loader = None
    if args.val_batches:
        validation = SymNetProDataset(
            args.data_root,
            "val",
            config["sampling_points"],
            config["patch_topk"],
            max_samples=args.max_samples,
        )
        val_loader = DataLoader(validation, shuffle=False, **options)
    model = build_model()
    model.run_config = config
    # Keep the historical parameter-group ordering, including its empty group.
    for key, expected in (
        ("optimizer", "Adam"),
        ("skip_lr", 0.00025),
        ("other_lr", 0.0005),
        ("weight_decay", 0.0),
        ("scheduler", None),
    ):
        if config[key] != expected:
            parser.error(f"This reproduction uses {key}={expected!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    callback = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="epoch-{epoch:02d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        save_last=True,
        every_n_epochs=1,
        save_on_train_epoch_end=True,
    )
    trainer = L.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=args.devices if use_gpu else 1,
        strategy="ddp" if use_gpu and len(args.devices) > 1 else "auto",
        max_epochs=config["max_epochs"],
        precision="32-true",
        num_sanity_val_steps=0,
        limit_train_batches=args.limit_train_batches or 1.0,
        limit_val_batches=args.val_batches,
        callbacks=[callback],
        logger=CSVLogger(args.output, name="logs"),
        log_every_n_steps=10,
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
