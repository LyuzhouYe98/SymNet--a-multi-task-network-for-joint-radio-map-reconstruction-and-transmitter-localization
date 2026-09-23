# Provenance and License Status

## SymNet Code and Weights

The model is a cleanup of the authors' archived SymNet implementation. Module
attribute names are retained to load the preserved checkpoint strictly. The
original-source hash, checkpoint hash, and before/after output comparisons
are recorded in `examples/legacy_reference.json`.

No code license has been selected in this staging copy. The author must choose
and approve the intended license before describing the project as an
open-source release. This document is a provenance note, not a license grant.

## Data

The archived propagation maps come from the directional dataset used with
ViT-RefineNet. This package adds the historical SymNet sampling protocols and
converts binary sampling masks to compact coordinate arrays without changing
the sampled pixels. DNB is recomputed from the supplied building maps.

The upstream propagation simulator and its data-generation scripts were not
found in the archived material. The dataset is not presented as a newly
simulated dataset. The dataset's upstream distribution URL, attribution, and
redistribution terms must be confirmed before public upload; a code license
must not be assumed to cover those data. The three reference examples also
contain inputs derived from these maps.

## Dependencies

The code imports PyTorch, torchvision, timm, Lightning, NumPy, SciPy, Pillow,
and tqdm. They are installed via `requirements.txt`, not bundled here. Their
respective licenses remain applicable. In particular, Transformer blocks,
patch embedding, and sinusoidal position embedding are imported from timm.

## Paper

Lyuzhou Ye, Thanh Dat Le, and Yan Huang. "SymNet: A Multi-Task Network for Joint
Radio Map Reconstruction and Transmitter Localization." arXiv:2608.00087, 2026.
https://doi.org/10.48550/arXiv.2608.00087
