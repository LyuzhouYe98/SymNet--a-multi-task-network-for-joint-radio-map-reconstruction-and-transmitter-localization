# Publishing Checklist

This directory is a source-only staging copy. Preparing it does not create a
remote repository or publish any files. The intended repository is
https://github.com/LyuzhouYe98/SymNet-Directional-Transmitter-Dataset.
The code license still needs to be selected by the author.

## Before Publication

1. Select the license for the authors' code and add the approved `LICENSE`.
   Confirm the upstream dataset's redistribution terms separately. Do not apply
   the code license automatically to propagation maps or pretrained weights.
2. Review the intended GitHub repository's existing contents and visibility.
3. Check that the main README's owner/repo and planned `v1.0.0` tag match
   the destination release.
4. Review `git status --short` before committing. Do not force-add ignored
   datasets, checkpoints, environments, or archives.
5. Tag the reviewed source commit as `v1.0.0`, and create a draft GitHub Release
   for that tag. Use `RELEASE_NOTES_v1.0.0.md` for its description.
6. Attach the four files listed below. Verify their hashes against
   `release_assets.json`, then publish only when the metadata and permissions
   have been reviewed.

## Release Attachments

- `symnet_code_checkpoint_arxiv2608_00087.tar.zst`
- `symnet_code_checkpoint_arxiv2608_00087.tar.zst.sha256`
- `symnet_release_arxiv2608_00087.tar.zst`
- `symnet_release_arxiv2608_00087.tar.zst.sha256`

The first archive contains the original standalone reproduction package,
checkpoint, and three examples. The full archive additionally contains the
dataset. Source files used for model loading, training, evaluation, and mask
preparation are unchanged between those archives and this staging copy.
GitHub-facing documentation and `.gitignore` are specific to this copy.

The README extracts only `checkpoints/` or `data/` from attachments, so it does
not overwrite the checked-out source or documentation with archived copies.

The upload manifest contains attachment hashes and sizes, not local absolute
paths. The checksum file at this repository's root covers only files shipped
in this source repository; attachment checksums cover downloaded archives.

GitHub currently requires each release asset to be smaller than 2 GiB. These
archives were checked against that limit during preparation. See the
[GitHub release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
