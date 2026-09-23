# SymNet Reproduction Artifacts

Single-transmitter reproduction materials for
[SymNet: A Multi-Task Network for Joint Radio Map Reconstruction and Transmitter Localization](https://arxiv.org/abs/2608.00087).

## Downloads

- **Quick start, approximately 44 MiB:**
  `symnet_code_checkpoint_arxiv2608_00087.tar.zst` contains code, the pretrained
  checkpoint, and three reference examples. No full dataset is required to run
  the example predictions.
- **Full reproduction package, approximately 1.5 GiB:**
  `symnet_release_arxiv2608_00087.tar.zst` additionally contains the archived
  single-transmitter maps and 9,360,000 masks in compact coordinate format.
- Download the corresponding `.sha256` file to verify either archive.

The full package includes training/validation masks, five fixed-sample-count
test sets, and ten positive-ratio test strata. The GitHub repository's README
explains how to extract only the data and checkpoint into a source checkout.

## Verification and Scope

The cleaned and original models produced exactly equal outputs on three
reference examples when compared on the same tested backend (CPU or H100).
The standalone inference and evaluation entry points were exercised, and the
training entry point passed a one-training-batch/one-validation-batch smoke
test. This is not a new full training run or a rerun of every published table.

The preserved checkpoint is from epoch 122 (zero-based). The surviving training
split has 274 buildings rather than the paper's reported 276. The paper's
AdamW configuration and the historical Adam configuration are both explained
in the repository's reproduction guide. Historical positive-ratio bins are
preserved, including documented integer-rounding boundary deviations.

The upstream propagation simulator is not included. Later multi-transmitter
experiments and unrelated checkpoints are excluded. Consult the repository's
license and provenance information before reusing or redistributing artifacts.
