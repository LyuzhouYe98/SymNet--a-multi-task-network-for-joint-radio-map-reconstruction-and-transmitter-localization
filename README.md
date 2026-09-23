# SymNet

Code and pretrained model for **SymNet: A Multi-Task Network for Joint Radio Map
Reconstruction and Transmitter Localization**.

Lyuzhou Ye, Thanh Dat Le, and Yan Huang.

[Paper](https://ieeexplore.ieee.org/document/11492179) |
[Dataset and checkpoint](https://github.com/LyuzhouYe98/SymNet--a-multi-task-network-for-joint-radio-map-reconstruction-and-transmitter-localization/releases/tag/Dataset) |
[Implementation details](docs/REPRODUCTION.md)

SymNet jointly predicts a radio map and a transmitter-localization heatmap from
sampled RSS, a sampling/building map, and a distance-to-nearest-building (DNB)
map. This repository implements the single-transmitter task.

## Installation

Clone this repository, enter its root, and use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset and Checkpoint

Download from the [Dataset release](https://github.com/LyuzhouYe98/SymNet--a-multi-task-network-for-joint-radio-map-reconstruction-and-transmitter-localization/releases/tag/Dataset):

- **Full package (1.44 GiB):** `symnet_release_arxiv2608_00087.tar.zst`, containing the dataset and checkpoint.
- **Checkpoint only for a quick start (43 MiB):** `symnet_code_checkpoint_arxiv2608_00087.tar.zst`, containing the checkpoint and standalone examples.

For training and evaluation, run the following from the repository root
(requires `curl`, GNU `tar`, and `zstd`):

```bash
BASE=https://github.com/LyuzhouYe98/SymNet--a-multi-task-network-for-joint-radio-map-reconstruction-and-transmitter-localization/releases/download/Dataset
curl -fL "$BASE/symnet_release_arxiv2608_00087.tar.zst" -o symnet_data.tar.zst
tar --zstd -xf symnet_data.tar.zst --strip-components=1 \
  symnet_release_arxiv2608_00087/data \
  symnet_release_arxiv2608_00087/checkpoints
```

This installs `data/` and `checkpoints/` without replacing the repository's code.
The full package already includes the checkpoint, so only one archive is needed.
For the smaller quick-start archive, extract just its checkpoint:

```bash
tar --zstd -xf symnet_code_checkpoint_arxiv2608_00087.tar.zst --strip-components=1 \
  symnet_release_arxiv2608_00087/checkpoints
```

The propagation maps come from the ViT-RefineNet directional dataset. The package
includes training/validation masks, five fixed-count test sets, and ten
positive-ratio test sets. Data are ready to load; no dataset-generation step is
needed. The upstream propagation simulator is not included.

## Run SymNet

```bash
# Predict on the included example inputs.
python predict.py --device cuda:0

# Evaluate 1,000 examples with exactly 20 sampled points.
python evaluate.py --sample-count 20 --max-samples 1000 --output results/test20.json

# Evaluate the nominal [10,20)% positive-ratio test set.
python evaluate.py --positive-ratio-bin 20 --max-samples 1000 --output results/pr20.json

# Train using config.json.
python train.py --devices 0
```

Use `--device cpu` for CPU inference or evaluation. Remove `--max-samples` to
evaluate all 590,000 examples in a selected test set. Fixed sample counts are
`20,40,60,80,100`; positive-ratio bin upper bounds are `10,20,...,100`.
Use `--data-root /path/to/data` if the dataset is stored elsewhere.

`predict.py` saves the predicted radio map, localization heatmap, and
transmitter coordinates. See [example input format](examples/README.md) for
custom inputs and [implementation details](docs/REPRODUCTION.md) for DNB, loss,
SkipNet, training settings, and resume options.

## Configuration Corrections

The experiments used 274 training buildings (27,400 maps and 274,000 training
pairs) and Adam with zero weight decay. These correct the training-set size and
optimizer listed in the paper. The released checkpoint is from epoch 122
(zero-based). The current `config.json` defaults to AdamW; set `optimizer` to
`Adam` and `weight_decay` to `0.0` to use the original training optimizer.

## Citation

```bibtex
@INPROCEEDINGS{SymNet,
  author={Ye, Lyuzhou and Le, Thanh Dat and Huang, Yan},
  booktitle={2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)}, 
  title={SymNet: A Multi-Task Network for Joint Radio Map Reconstruction and Transmitter Localization}, 
  year={2026},
  volume={},
  number={},
  pages={150-159},
  keywords={Antennas;Feeds;Antennas and propagation;Directional antennas;Radio networks;Radio broadcasting;Filtering;Filters;MIMICs;Millimeter wave integrated circuits;transmitter localization;joint prediction;multi-task learning;wireless propagation;radio map reconstruction;directional transmitter},
  doi={10.1109/WACV61042.2026.00023}}
@inproceedings{ViT-RefineNet,
author = {Ye, Lyuzhou and Le, Thanh Dat and Huang, Yan},
title = {ViT-RefineNet for Directional Signal Radio Map Reconstruction from Sparse Samples},
year = {2025},
isbn = {9798400720864},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3748636.3762750},
doi = {10.1145/3748636.3762750},
abstract = {Accurately predicting directional radio maps is crucial for various wireless applications; however, existing methods that primarily rely on Convolutional Neural Networks (CNNs) or U-Nets designed for omnidirectional signals struggle to capture the long-range dependencies and angular sensitivities inherent in directional signal propagation (especially when influenced by environmental factors like building occlusions and reflections). To overcome these limitations, we introduce a novel framework, denoted as "ViT-RefineNet," that is specifically designed for predicting directional radio maps from sparse signal measurements. Our approach integrates a Vision Transformer (ViT)-based model to capture global dependencies with multiple U-Net-like modules for localized refinement at different processing stages. By effectively combining global context understanding with local detail enhancement, ViT-RefineNet offers significant advantages over traditional CNN-based and U-Net-based methods in predicting directional radio maps from sparse directional signal measurements. Experimental results demonstrate the superior performance of our proposed ViT-RefineNet compared to state-of-the-art models.1},
booktitle = {Proceedings of the 33rd ACM International Conference on Advances in Geographic Information Systems},
pages = {396–406},
numpages = {11},
keywords = {radio map estimation, directional transmitter, deep learning, visiontransformer},
location = {The Graduate Hotel Minneapolis, Minneapolis, MN, USA},
series = {SIGSPATIAL '25}
}
```
