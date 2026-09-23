# Verification Scope

- `data_audit.json`: all 17 mask sets were checked for expected counts, map
  coverage, repetitions per map, and sampling protocol. Each set also passed
  20 randomly selected round trips to the original PNG masks (340 total).
  Positive-ratio summaries cover every mask in all ten strata.
- `checkpoint_cpu.json` and `checkpoint_cuda.json`: released checkpoint outputs
  compared with the original model's frozen outputs for three examples.
- `../examples/legacy_reference.json`: original-versus-cleaned input, target,
  and prediction comparisons on both CPU and GPU, plus source/checkpoint hashes.

The training entry point also passed one training batch and one validation
batch on an H100 using `--fast-dev-run --batch-size 2 --num-workers 2`.
Fixed-count and positive-ratio evaluation entry points passed small smoke tests.
These checks do not constitute retraining or a full rerun of the paper's tables.

`SHA256SUMS` in the source repository covers code, configuration, documentation,
examples, and verification records, but not separately downloaded files.
The external archive SHA-256 covers the checkpoint and, for the full archive,
the dataset. Attachment names, sizes, and hashes are in
`../docs/release_assets.json`.
