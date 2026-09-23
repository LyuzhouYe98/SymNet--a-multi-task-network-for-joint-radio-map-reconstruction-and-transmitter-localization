# SymNet

Reproduction code for **SymNet: A Multi-Task Network for Joint Radio Map
Reconstruction and Transmitter Localization**.

Lyuzhou Ye, Thanh Dat Le, and Yan Huang.
[Paper](https://arxiv.org/abs/2608.00087) |
[Downloads](https://github.com/LyuzhouYe98/SymNet-Directional-Transmitter-Dataset/releases) |
[Detailed reproduction guide](docs/REPRODUCTION.md) |
[Verification records](verification/README.md)

SymNet jointly predicts a radio map and a transmitter-localization heatmap from
three input channels: sampled normalized RSS, sampling/building information,
and distance to the nearest building (DNB). This release covers the original
**single-transmitter** task, not later SymNetPro multi-transmitter experiments.

## Installation

Clone this repository, enter its root, and use Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sha256sum -c SHA256SUMS
```

Dependencies are pinned to the tested environment, including PyTorch 2.7.0.
The source repository includes three reference examples but excludes the full
dataset and checkpoint. Download those from the repository's **Releases** page,
not GitHub's automatically generated "Source code" archives.

## Download the Checkpoint

The commands below use `curl`, GNU `tar`, and `zstd`. The `v1.0.0` release
must be published with the listed attachments before these URLs work.

```bash
REPO=LyuzhouYe98/SymNet-Directional-Transmitter-Dataset
TAG=v1.0.0
ASSET=symnet_code_checkpoint_arxiv2608_00087.tar.zst
mkdir -p downloads
curl -fL --retry 3 "https://github.com/$REPO/releases/download/$TAG/$ASSET" -o "downloads/$ASSET"
curl -fL --retry 3 "https://github.com/$REPO/releases/download/$TAG/$ASSET.sha256" -o "downloads/$ASSET.sha256"
(cd downloads && sha256sum -c "$ASSET.sha256") && \
  tar --zstd -xf "downloads/$ASSET" --strip-components=1 \
  symnet_release_arxiv2608_00087/checkpoints
```

This downloads the approximately 44 MiB quick-start archive and installs only
`checkpoints/`; it does not replace your checked-out source. Alternatively,
download the two files in your browser and use the same verification/extraction
commands after placing them in `downloads/`.

```bash
python -m tools.verify_checkpoint --device cpu
python predict.py --device cpu --output results/prediction.npz
```

The predictions NPZ contains `localization_heatmap`, `radio_map`, and
`transmitter_yx` (row, column, in pixels). Use `--device cuda:0` for GPU
inference. See [input format and examples](examples/README.md) for custom data.

## Download the Dataset

For training and evaluation, use the approximately 1.5 GiB full archive:

```bash
REPO=LyuzhouYe98/SymNet-Directional-Transmitter-Dataset
TAG=v1.0.0
ASSET=symnet_release_arxiv2608_00087.tar.zst
mkdir -p downloads
curl -fL --retry 3 "https://github.com/$REPO/releases/download/$TAG/$ASSET" -o "downloads/$ASSET"
curl -fL --retry 3 "https://github.com/$REPO/releases/download/$TAG/$ASSET.sha256" -o "downloads/$ASSET.sha256"
(cd downloads && sha256sum -c "$ASSET.sha256") && \
  tar --zstd -xf "downloads/$ASSET" --strip-components=1 \
  symnet_release_arxiv2608_00087/data \
  symnet_release_arxiv2608_00087/checkpoints
python -m tools.audit_release
```

The full archive includes the checkpoint, so downloading both archives is not
required. Data are installed into `data/`; neither data nor checkpoints should
be committed to Git. Alternative dataset locations can be selected with
`--data-root`.

## Evaluation and Training

```bash
# Table 3: exactly 20 sampled points.
python evaluate.py --sample-count 20 --max-samples 1000 --output results/test20.json

# Table 4: historical nominal positive-ratio bin [10,20)%.
python evaluate.py --positive-ratio-bin 20 --max-samples 1000 --output results/pr20.json

# One training batch and one validation batch.
python train.py --fast-dev-run --batch-size 2 --num-workers 2 --devices 0

# Full training using config.json.
python train.py --devices 0
```

Remove `--max-samples` for the full 590,000 examples in a selected test set.
Fixed counts are `20,40,60,80,100`. Positive-ratio selectors are the nominal
upper bounds `10,20,...,100`. The two selectors are mutually exclusive.

DNB cutoff is 20 pixels; Gaussian localization targets use sigma 17 pixels;
the default loss is radio-map MSE plus localization-heatmap MSE, with equal
weights. Full architecture, mask counts, normalization, evaluation definitions,
and training settings are documented in the [reproduction guide](docs/REPRODUCTION.md).

## Reproducibility Scope

Checkpoint loading is strict and requires no legacy scripts or fixed server
paths. The cleaned and original models matched exactly on three examples on
each tested backend (CPU and H100). Separate backend references are included.
These checks verify compatibility, not a new full run of every paper result.

Historical differences are documented rather than hidden: the preserved
training split contains 274 buildings, the checkpoint is from epoch 122
(zero-based), and its historical optimizer differs from the paper's AdamW
setting. See [artifact notes](docs/REPRODUCTION.md#artifact-notes).

## Citation and License

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

Author approval of the code license and upstream dataset redistribution terms
is pending in this staging copy. See [provenance and license status](docs/PROVENANCE.md).
