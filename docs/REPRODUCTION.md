# Implementation Details

## Inputs and Targets

Each input is a float32 array of shape `[3, 256, 256]`:

| Channel | Definition |
| --- | --- |
| Sampled RSS | `radio_map * sampling_mask * free_space` |
| Sampling/building map | `sampling_mask - building_map` |
| DNB | `1 - clip(distance_transform_edt(free_space), 0, 20) / 20` |

The supplied radio-map PNGs are normalized by dividing their pixel values by
255. Buildings are 1 in `building_map`; free space is `1 - building_map`.
The DNB cutoff is **20 pixels**.

The localization target is a normalized Gaussian heatmap centered on the
transmitter, with `sigma=17` pixels and SciPy's `reflect` boundary mode.
The radio-map target is masked to free space. The two task losses have equal
weight:

```text
loss = MSE(predicted_radio_map, target_radio_map)
     + MSE(predicted_localization_heatmap, target_localization_heatmap)
```

## Architecture

The Transformer uses `8x8` patches, embedding width 192, 16 attention heads,
and MLP ratio 4. Each branch has six self-attention blocks before and six after
the bidirectional cross-attention layer. LayerNorm is replaced by DynamicTanh.

There are three attention-gated SkipNets. Each uses nine `3x3` encoder
convolutions at width 27, three average-pooling stages, a four-channel
bottleneck, bilinear decoder upsampling, and attention-gated skip connections.
The model returns `(localization_heatmap, radio_map)`, each shaped
`[B, 1, 256, 256]`.

## Dataset

| Split or protocol | Radio maps | Masks |
| --- | ---: | ---: |
| Training | 27,400 from 274 buildings | 274,000 |
| Validation | 5,900 from 59 buildings | 236,000 |
| Fixed-count test | 5,900 from 59 buildings | 590,000 per sample count |
| Positive-ratio test | 5,900 from 59 buildings | 590,000 per nominal bin |

Fixed counts are 20, 40, 60, 80, and 100. Positive-ratio bins have nominal upper
bounds 10, 20, ..., 100 percent and contain 20--100 points per mask. Positive
ratio is the fraction of sampled pixels with nonzero RSS outside buildings.
Some masks fall slightly below a nominal boundary, consistent with integer
rounding of positive sample counts; the original bin assignments are used.

`symnet/data.py` reads the maps under `data/maps/` and the ready-to-use masks
under `data/packed_masks/`. Each mask is stored as sampled coordinates.
No preprocessing or mask-generation script is needed.

## Training

`train.py` reads `config.json`. Its defaults are batch size 32, 100 epochs,
AdamW with weight decay 0.01, learning rate `5e-4` for SkipNets and `1e-3` for
other modules. Command-line options override batch size, epochs, and seed.

**Correction:** the original training used Adam with zero weight decay,
rather than the AdamW listed in the paper. To use that optimizer, set these
entries in `config.json`:

```json
"optimizer": "Adam",
"weight_decay": 0.0
```

The released checkpoint is from epoch 122 (zero-based), beyond the paper's
listed 100-epoch setting. It can be loaded directly for inference.

Checkpoints are selected by validation argmax localization error. A short
training check and resume command are:

```bash
python train.py --fast-dev-run --batch-size 2 --num-workers 2 --devices 0
python train.py --resume runs/symnet/checkpoints/last.ckpt --devices 0
```

Resume requires a checkpoint created by `train.py`, which includes optimizer
state. The downloadable pretrained checkpoint contains inference weights only.
Use `--devices 0 1` for multi-GPU training.

## Evaluation and Custom Measurements

Evaluation uses free-space RMSE and SSIM for the radio map, and the
nonnegative heatmap centroid for localization. Coordinates and localization
errors are in pixels; physical units require the map's pixel-to-meter scale.

For real measurements, align RSS samples with the building map, compute DNB
at that resolution, and calibrate the RSS normalization. Fine-tuning on measured
data is recommended; keep training and test locations separate.
