# Example Inputs

`legacy_reference.npz` includes three example inputs for standalone inference.
The `inputs` array is float32 with shape `[3, 3, 256, 256]`.

Channels are ordered as:

1. Sampled normalized RSS.
2. Sampling mask minus building map: sampled free-space pixels are 1, unobserved free space is 0, and buildings are -1.
3. DNB: `1 - clip(distance, 0, 20) / 20`, with distance in pixels.

From the repository root, after downloading the checkpoint:

```bash
python predict.py --device cpu
python predict.py --input your_inputs.npz --device cuda:0 --output results/prediction.npz
```

Custom NPZ files must contain an `inputs` array of shape `[B, 3, 256, 256]`.
The output NPZ contains `localization_heatmap`, `radio_map`, and
`transmitter_yx` (centroid row and column, in pixels).
