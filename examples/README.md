# Standalone Inference Examples

`legacy_reference.npz` contains three examples drawn from `test_20`,
`test_pr_0_10`, and `test_pr_90_100`. Their record indices and the hashes of
the original source and released checkpoint are in `legacy_reference.json`.

Arrays in the NPZ:

- `inputs`: float32 `[3, 3, 256, 256]`, ordered as sampled normalized RSS,
  sampling/building channel, and DNB.
- `localization_heatmap`: float32 `[3, 1, 256, 256]`, original GPU model output.
- `radio_map`: float32 `[3, 1, 256, 256]`, original GPU model output.
- `localization_heatmap_cpu` and `radio_map_cpu`: the corresponding original
  model outputs on CPU. Verification selects the reference for its backend;
  CPU and GPU floating-point results need not be identical.

The sampling/building channel is `mask - building`: sampled free-space pixels
are 1, unobserved free space is 0, and buildings are -1. RSS images from the
dataset are divided by 255. DNB is `1 - clip(distance, 0, 20) / 20`.

Run from the package root:

```bash
python -m tools.verify_checkpoint --device cpu
python predict.py --input examples/legacy_reference.npz --device cpu
```

`predict.py` writes the two output maps and centroid coordinates under
`transmitter_yx` (row, column, in pixels). Outputs are the raw trained model
predictions; they are not thresholded or clipped. For your own inputs, provide
an NPZ with the `inputs` key and the same channel order and normalization.
