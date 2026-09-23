# SymNet Reproduction Package

This guide describes the standalone release archives. For installation into a
GitHub checkout, follow the [repository README](../README.md); its extraction
commands install only data and checkpoints without replacing checked-out code.

Minimal single-transmitter artifact for [SymNet: A Multi-Task Network for Joint Radio Map Reconstruction and Transmitter Localization](https://arxiv.org/abs/2608.00087).

This package intentionally excludes later multi-transmitter experiments, alternative sampling-mask generations, cached LOS graphs, visualization outputs, and unrelated checkpoints. It does not modify the source dataset.

The propagation maps originate from the directional dataset released with ViT-RefineNet. The surviving artifact contains the maps and sampling masks but not the upstream radio-propagation simulator; this package therefore reproduces SymNet input construction and training, not re-simulation of the propagation maps.

## Contents

- `symnet/model.py`: network forward pass and Lightning training hooks, with strict checkpoint loading.
- `symnet/layers.py`: DynamicTanh, gated SkipNet, and cross-attention building blocks.
- `symnet/data.py` and `symnet/metrics.py`: compact-mask loader and evaluation metrics.
- `data/maps/`: the single-transmitter building, DPM, and antenna PNGs.
- `data/packed_masks/`: the archived experimental masks stored as sampled coordinates instead of 9.36 million tiny PNG files.
- `checkpoints/symnet_arxiv2608_00087.ckpt`: weights-only checkpoint compatible with the cleaned model.
- `train.py` and `evaluate.py`: paper-configured training and fixed-mask evaluation.
- `predict.py`: standalone checkpoint inference on an NPZ input, without the full dataset.
- `examples/legacy_reference.npz`: three inputs and their outputs from the original model.
- `tools/verify_checkpoint.py`: reproducible comparison against those original outputs.
- `tools/prepare_release_data.py`: reproducible conversion from the original directory.
- `tools/audit_release.py`: mask counts, positive ratios, optional original-PNG comparison, and checkpoint SHA-256 audit.
- `verification/`: recorded data audits and checkpoint compatibility checks.

The fixed-cardinality test masks come from the historical
`masks_smpl_<N>_random_pos_rate` directories. The older `masks_<N>` directories
are positive-ratio strata whose masks actually contain between 20 and 100
points; treating their suffix as sample count would reproduce the wrong test.

Both test protocols are included:

| Experiment | Original directory | Release directory | Masks per set |
| --- | --- | --- | --- |
| Training | `train/masks` | `packed_masks/train` | 274,000 |
| Validation | `val/masks` | `packed_masks/val` | 236,000 |
| Table 3: fixed sample count | `test/masks_smpl_<N>_random_pos_rate` | `packed_masks/test_<N>` | 590,000 for each N=20,40,60,80,100 |
| Table 4: positive-ratio stratum | `test/masks_<U>` | `packed_masks/test_pr_<U-10>_<U>` | 590,000 for each U=10,20,...,100 |

For example, `masks_10` is stored as `test_pr_0_10`, and `masks_100` as
`test_pr_90_100`. Each Table 4 stratum contains 100 masks for each of 5,900
single-transmitter maps, with 20--100 total sampled points per mask.

The positive ratio is `100 * positive_count / sample_count`, where a positive
point has a nonzero DPM value outside buildings. Each Table 4 set includes
`positive_counts.npy` and a `complete.json` summary of actual ratios. The historical
directory labels are nominal bins: some masks lie below the lower percentage
boundary, consistent with integer rounding of positive sample counts. The
release preserves these masks and reports boundary deviations; it does not
filter or relabel them. Use the historical strata to repeat the archived tests.

## Experimental Configuration

The input has three channels: sampled signal, sampling/building channel, and distance-to-nearest-building (DNB). The DNB channel is generated from free-space Euclidean distance as

```text
DNB = 1 - clip(distance_transform_edt(free_space), 0, 20) / 20
```

Thus, the DNB cutoff is **20 pixels**. Localization targets are normalized Gaussian heatmaps with `sigma=17` pixels and SciPy's `reflect` boundary mode, matching the archived training script. The two task losses are unweighted MSE terms:

```text
loss = MSE(predicted radio map, radio map) + MSE(predicted heatmap, heatmap)
```

The model uses `256x256` inputs, `8x8` patches, width 192, 16 heads, MLP ratio 4, and one bidirectional cross-attention layer. Each branch has six self-attention blocks before and six after cross-attention (12 per branch). Each of the three attention-gated SkipNets uses nine `3x3` encoder convolutions at width 27, three average-pooling stages, a four-channel bottleneck, bilinear decoder upsampling, and skip attention gates.

Training defaults follow the paper: batch size 32, 100 epochs, AdamW, learning rate `5e-4` for SkipNet modules and `1e-3` elsewhere. Training maps have 10 fixed masks each, validation maps have 40, and testing uses 100 masks per map at 20, 40, 60, 80, and 100 sampled points.

Training reads `config.json`; command-line batch size, epochs, and seed override
that file. Checkpoints are selected by validation argmax localization error,
as in the historical training script. Evaluation uses nonnegative heatmap
centroids and the raw antenna-map GT coordinates, as in the historical test
script. Localization errors are reported in pixels; conversion to meters
requires the map's physical scale.

## Setup and Use

Extract either archive and enter `symnet_release_arxiv2608_00087/`. The full
archive includes the dataset; the code-and-checkpoint archive contains the same
code and weights plus standalone examples. On Linux, extract with
`tar --zstd -xf <archive>.tar.zst` (requires `zstd`).

```bash
sha256sum -c SHA256SUMS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m tools.verify_checkpoint --device cpu
python predict.py --device cpu --output results/prediction.npz
```

The above commands also work with the smaller code-and-checkpoint archive,
which includes three example inputs but omits the full dataset. For dataset
evaluation and training, use the full archive (or supply `--data-root`):

```bash
python -m tools.audit_release
python evaluate.py --sample-count 20 --max-samples 1000
python evaluate.py --positive-ratio-bin 10 --max-samples 1000
python train.py --devices 0
```

The dependencies are pinned to the tested versions (Python 3.12, PyTorch 2.7.0).
GPU inference can be selected with `--device cuda:0`; a CPU works for the
standalone examples. Inference loads no original scripts or external model
weights. The checkpoint's layer names are intentionally preserved.

For a short training check, run
`python train.py --fast-dev-run --batch-size 2 --num-workers 2 --devices 0`.
To resume your own training, pass `--resume runs/symnet/checkpoints/last.ckpt`;
Lightning training checkpoints contain optimizer state. The supplied portable
checkpoint contains inference weights and is not a training-resume checkpoint.

Remove `--max-samples` to evaluate all 590,000 examples for one sampling count
or one positive-ratio stratum. The selectors are mutually exclusive.
`--positive-ratio-bin 20` selects the nominal `[10,20)%` stratum; `100` selects
the nominal `[90,100]%` stratum. Evaluation JSON records the selected mask set,
actual mean positive ratio, and the number of masks outside its nominal bounds.

Run all ten Table 4 strata with:

```bash
for upper in 10 20 30 40 50 60 70 80 90 100; do
  python evaluate.py --positive-ratio-bin "$upper" --output "results/pr_${upper}.json"
done
```

To compare sampled packed masks with the original PNG files:

```bash
python -m tools.audit_release --source /path/to/Directional_Dataset
```

To recreate the compact data from the original directory:

```bash
python tools/prepare_release_data.py \
  --source /path/to/Directional_Dataset \
  --output data \
  --workers 64
```

## Artifact Notes

- The paper reports 27,600 training maps and 276,000 training pairs. The surviving source directory and its original ZIP archive both contain 27,400 maps from 274 buildings and 274,000 pairs. Validation and test counts match the paper: 5,900 maps from 59 buildings, 236,000 validation pairs, and 590,000 test pairs per sampling count.
- The paper specifies 100 epochs. The surviving pretrained checkpoint is labeled epoch 122 (zero-based), so it reflects additional training beyond the paper schedule.
- The paper specifies AdamW. The surviving checkpoint's archived optimizer state has zero weight decay and is consistent with the historical Adam training script. `train.py` uses AdamW with its default weight decay of 0.01 (weight decay is unspecified in the paper). To use the historical optimizer instead, set `optimizer` to `Adam` and `weight_decay` to `0.0` in `config.json`. The supplied checkpoint preserves the surviving trained weights.
- This package is for the paper's single-transmitter task. Multi-transmitter files found beside the dataset belong to later experiments and are deliberately excluded.
- Checkpoint compatibility is tested against the original `bnd.py` on three examples covering fixed-count and positive-ratio tests. On each tested backend (CPU and H100), both prediction maps are exactly equal before and after cleanup. CPU and GPU results can differ numerically, so separate frozen references are included. This verifies code compatibility, not a fresh rerun of every published result.

## Real-World RSS Adaptation

Use the same fixed global normalization learned from the target receiver hardware, replace simulated building rasters with aligned floor-plan occupancy maps, and compute DNB at the real map resolution. Fine-tune first on synthetic-to-real mixtures and then on measured maps while keeping train/validation locations or buildings disjoint. If antenna labels are unavailable, train radio reconstruction first and calibrate localization with a small labeled subset; do not assume that simulated RSS dynamic range or pixel-to-meter scale transfers unchanged.

No redistribution license is asserted by this reconstruction package. Select and add the intended code and dataset licenses before public distribution.
