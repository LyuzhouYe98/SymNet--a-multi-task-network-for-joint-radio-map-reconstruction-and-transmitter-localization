# SymNetPro Dataset Extension

Build directional multi-transmitter scenes from the existing SymNet source maps.
This folder contains one converter, the original scene-combination indices in
`scene_index.zip`, protocol metadata in `dataset.json`, and lightweight
dependencies. It contains no radio maps, precomputed masks, model or checkpoint.

## Quick Start

Run commands from the **repository root**. First download and extract the full
dataset following the [main README](../../README.md#dataset-and-checkpoint).
If `data/maps/` is already present, no additional map download is needed.
The checkpoint-only download and GitHub's automatic source archives do not
contain the required maps. Do not unpack `scene_index.zip`; the converter reads
it directly.

```bash
# Only needed if you have not installed the repository's dependencies.
python -m pip install -r datasets/symnetpro/requirements.txt

# Build the full dataset, including LOS tables.
python datasets/symnetpro/convert.py --symnet-root . \
  --output Directional_Dataset --workers 8 --seed 42 --with-los
```

For a small test before converting everything:

```bash
python datasets/symnetpro/convert.py --symnet-root . \
  --output symnetpro_smoke --splits test --buildings 1 \
  --sampling-points 100 --workers 1 --with-los
```

Use Python 3.10 or newer. `--symnet-root` also accepts the unpacked original
release directory or its `data/maps` directory directly. The output must be new
or empty; original data are never overwritten. No GPU is needed, and progress
is shown per building. Conversion expands into many NPZ/PNG files; the small
index size is not the required output disk space. A failed run leaves
`conversion.json` marked `incomplete`; retry in a fresh directory.

## Protocols

| Split | Buildings | Scenes | Sample counts | Masks per scene/count |
|---|---:|---:|---|---:|
| Train | 274 | 137,000 | 100, 3300 | 10 |
| Validation | 59 | 29,500 | 100, 3300 | 10 |
| Test | 59 | 29,500 | 100, 200, 300, 400, 500, 660, 1320, 1980, 2640, 3300 | 1 |

The original building splits, transmitter combinations, source order and
per-source radio-map arrays are preserved. Each building has 100 combinations
for each source count K=1,...,5. The default output contains 196,000 scenes and
3,625,000 masks. Training/validation retain all scenes and all 10 sampling
realizations per count, but the mask pixels are newly generated.

Masks uniformly sample distinct pixels outside buildings (`building != 255`),
without selecting for positive RSS. Each split/scene/count/repeat receives its
own SHA-256-derived PCG64 stream. With the pinned NumPy version and the same
seed, worker count, processing order and subset selection do not change a mask.
Different sampling counts are independent, not nested. `--test-repeats N`
generates N new test masks per scene/count. `--sampling-points` and `--splits`
restrict conversion to the needed protocol.

These are reproducible new masks, not the original paper realizations. Changing
masks can also change which transmitters are observed at sampled pixels, so
exact published scores are not expected.

## Output and Loading

```text
Directional_Dataset/
  conversion.json
  train/
    buildings/                              # PNG + distance-transform NPY
    multi_transmitter_pairs_npz_with_mask/   # per-source dBm/antenna pairs
    masks_multi_100/
    masks_multi_3300/
  val/                                      # same layout as train
  test/
    buildings/
    multi_transmitter_pairs_npz_with_mask/
    test_multi_100/                          # likewise for the other counts
  precomputed_patch_los_ps8/                 # only with --with-los
    train/
    val/
    test/
```

Scene NPZ files retain `pairs`, `filenames`, `map_id`, `per_tx_mask`, and
`fill_dbm`. The SymNetPro loader combines selected per-source maps in linear
power and constructs its full/drop targets online. No separate combined-map
archive is required. Source PNG decoding uses the original fixed range
[-120, -4.588664114340091] dBm. This is separate from SymNetPro's mixed-map model
normalization `(dBm + 120) / (120 - 5.041252136230469)`.

`--with-los` generates patch-size-8 LOS tables from building PNGs. Include it
when using the precomputed-LOS SymNetPro loader; omit it for dataset-only use
that does not need LOS. Building distance maps are always generated.

This folder prepares data only. Use the separate [SymNetPro module](../../symnetpro/README.md)
for multi-transmitter training and inference. The root-level `train.py`,
`evaluate.py` and original SymNet checkpoint remain single-transmitter tools.
Source-data terms remain unchanged.
