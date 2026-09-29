# SymNetPro

Multi-transmitter radio-map reconstruction and localization with a sparse
LOS/distance attention bias and transmitter-drop training augmentation.
This module is separate from the original single-transmitter `symnet/` model.
Run the commands below from the repository root after installing the main
`requirements.txt`.

## Data and Checkpoint

Reuse the existing SymNet dataset release and run the
[multi-source converter](../datasets/symnetpro/README.md) with `--with-los`:

```bash
python datasets/symnetpro/convert.py --symnet-root . \
  --output Directional_Dataset --workers 8 --seed 42 --with-los
```

The all-in-one archive `symnet_with_symnetpro_code_checkpoint_20260928.zip`
contains this code and `checkpoints/symnetpro/topk16_epoch06.ckpt`. If you
already cloned the code, extract only the checkpoint from that archive:

```bash
unzip symnet_with_symnetpro_code_checkpoint_20260928.zip 'checkpoints/symnetpro/*'
```

This checkpoint is the original **topk=16, epoch 6 (zero-based), step 149849**
model used by the clean evaluation script, not the later topk=32 model or a
noise-training ablation. It includes model parameters, Adam state, training-loop
state and the matching configuration. Obsolete callback paths are removed; all
model and optimizer tensors are retained. Loading uses PyTorch's
`weights_only=True`; no global unsafe checkpoint-loading override is needed.

The converter preserves source maps and combinations but regenerates masks.
Therefore the converted data support rerunning the protocol, not exact
reproduction of the original published scores or shuffled training sequence.

## Inference and Evaluation

```bash
# A small inference run that also saves radio maps, localization maps and XY peaks.
python -m symnetpro.evaluate \
  --checkpoint checkpoints/symnetpro/topk16_epoch06.ckpt \
  --data-root Directional_Dataset --sampling-points 100 \
  --max-samples 5 --save-predictions results/symnetpro_predictions

# Evaluate a full sampling-count test set. Repeat with the other available counts.
python -m symnetpro.evaluate \
  --checkpoint checkpoints/symnetpro/topk16_epoch06.ckpt \
  --data-root Directional_Dataset --sampling-points 500 \
  --output results/symnetpro_500.json
```

Use `--device cpu --num-workers 0` for a CPU smoke test. Progress is displayed
per batch, with threaded post-processing (`--postprocess-workers`, default 8).
`--noise-db 4` adds Gaussian noise only to sampled measurements; clean targets
are unchanged. `--split val` evaluates validation masks. `--threshold` overrides
the checkpoint's default 0.25 only when explicitly requested.

The decoder applies an 11x11 local maximum filter and represents each connected
peak plateau by its centroid. It acts on the raw localization output, without
an added sigmoid. OSPA uses c=20, p=2; mLE is Hungarian-matched mean Euclidean
error. MDR/FAR are count deficits/excesses divided by the observed GT count,
not distance-gated match rates. They are undefined for empty GT scenes. Results
include per-sample metrics and breakdowns by observed source count.

## Training and Resume

```bash
# Reference global batch size: four devices x 16 samples per device.
python -m symnetpro.train --data-root Directional_Dataset \
  --devices 0 1 2 3 --batch-size 16 --max-epochs 7 --output runs/symnetpro

# CPU smoke test, including one validation batch.
python -m symnetpro.train --data-root Directional_Dataset \
  --accelerator cpu --batch-size 1 --num-workers 0 --max-samples 2 \
  --limit-train-batches 2 --val-batches 1 --max-epochs 1 --output runs/pro_smoke

# Continue the released checkpoint for one additional epoch.
python -m symnetpro.train --data-root Directional_Dataset \
  --resume checkpoints/symnetpro/topk16_epoch06.ckpt \
  --devices 0 1 2 3 --max-epochs 8 --output runs/symnetpro_resume
```

The default device is one GPU when available, otherwise CPU. Batch size is
**per device**; a single device with batch size 16 is not the reference global
batch size of 64. Multi-GPU execution uses Lightning DDP. Training does not
automatically resume: `--resume` restores the optimizer and loop state, not just
weights. `--max-epochs` is the total target, including completed epochs.
Use epoch-boundary checkpoints; exact mid-batch replay is not provided.

Every epoch is saved, including `last.ckpt`, with optimizer state. Validation
is off by default as in the final training script; `--val-batches 100` enables
a 100-batch validation pass. It does not change which epoch files are saved.
Loaders use non-persistent workers. There is no learning-rate scheduler.

## Configuration

`config.json` records the released model's settings. Network dimensions and
the optimizer recipe are fixed to this checkpoint-compatible implementation.
`--patch-topk` and `--sampling-points` can select a new training experiment,
but cannot change the corresponding settings during a resume. Inference takes
topk directly from the checkpoint metadata.

| Setting | Value |
|---|---|
| Input / patches / embedding | 2 x 256 x 256 / 8 x 8 / 192 |
| Transformer depth / heads | 6 / 16 |
| Normalization in the network | DynamicTanh |
| Training samples / seed | 100 / 42 |
| Adam LR: SkipNets / other parameters | 0.00025 / 0.0005 |
| Weight decay / scheduler | 0 / none |
| Neighbors / confidence | K=16 / min(1, sampled pixels per patch / 4) |
| Candidate distance buckets | <50, 50-100, >=100 pixels; retain 12, 12, 8 candidates |
| Candidate weighting | confidence * exp(-d * (1 + (1-LOS)) / 0.12) |
| Attention-bias logits / temperature | logits = -2d - (1-LOS); temperature = 2 |

Distance d is divided by sqrt(H^2 + W^2) + 1e-6. Candidate selection and
attention-bias softmax use separate temperatures. Bias is row-centered and
masked on building-only patches, as in the trained implementation.

Training uses 0.25 * [(signal_MSE_full + 0.4 * localization_MSE_full) +
(signal_MSE_drop + localization_MSE_drop)]. Each branch is masked by its
validity flag and averaged over the full batch. The drop branch is valid only
when at least two sources were observed; targets are formed after removing
one randomly selected source. Localization targets use sigma=17 Gaussian maps.
Validation follows the original heatmap-weighted localization loss.

Model code is in `model.py`, source composition and augmentation in `data.py`,
neighbor selection in `graph.py`, and plateau decoding/metrics in `metrics.py`.
The unchanged shared SkipNet layers are imported from `symnet.layers`.
