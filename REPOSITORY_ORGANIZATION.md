# Repository organization

Audit date: 2026-10-08
Scope: `/vol/gpudata/yz3522-llmtest/MResProject`, branch `main`
Remote: `Yng314/MResProject`

## Intended boundary

The `main` branch is the lightweight source repository. It keeps Python and shell source, tests, configuration, dependency manifests, and concise protocols. It excludes datasets, virtual environments, vendor copies, result tables, training logs, OOF arrays, generated figures, scratch notebooks, checkpoints, and machine-specific links. The broader dataset and result map is in [DATA_LAYOUT.md](DATA_LAYOUT.md).

## Changes in this pass

- Stopped tracking 99 generated or scratch files: 36 CSVs, 48 logs, two OOF arrays, one Excel workbook, nine PNGs, two scratch/reference notebooks, and one exported transcript.
- Kept every original file at its existing GPU Data path. The full output-bearing toy notebook was copied to the parent workspace's ignored `.repo_backups/MResProject_cleanup_20261008/` before its generated cell outputs were cleared; the notebook's code remains tracked.
- Copied 15 synthetic known-truth toy CSV/PNG outputs to `/vol/bitbucket/yz3522/NoiseRate_results_archive/repo_artifacts/toy_synthetic_20261008/` and verified every copy by SHA-256; the destination includes a path/hash manifest.
- Added generated-artifact ignore rules and a `NoiseRate/requirements.txt` dependency manifest. No virtual environment was added.
- Did not copy MIMIC-CXR/REFLACX data or their derived experiment outputs to Bitbucket. Their permitted storage location still needs confirmation; see [DATA_LAYOUT.md](DATA_LAYOUT.md).

## Current classification

| Class | Repository treatment |
| --- | --- |
| Python/shell source, tests, configuration | Keep in Git |
| Dependency manifests and lightweight protocols | Keep in Git |
| Dataset files, predictions, result tables, logs, plots, checkpoints | Ignore; retain originals outside Git |
| Scratch/reference notebooks and transcripts | Ignore; retain originals locally |
| Virtual environments, caches, vendored packages, external symlinks | Ignore |
| MIMIC-CXR and REFLACX source data | Keep in the current local dataset directory until an approved storage destination is established |

## Known history and branch scope

On 2026-10-08, the public `main` history was rewritten to remove 111 reviewed paths that included generated/scratch material and selected case-level files. The rewrite left the source tree hash unchanged at `8aed17114f5ed49612c14090135a25905d1a64da`; a follow-up documentation commit then updated the repository notes. The rewritten branch had 22 commits at `2c337ce`.

The obsolete remote `llmtest-workspace` branch, which contained a combined workspace snapshot including thesis/defence material, was deleted after explicit authorization. The remote now exposes only `main`. A mirror of the pre-rewrite remote refs is kept locally at the parent workspace's ignored `.repo_backups/MResProject_remote_before_rewrite_20261008.git`; do not push that backup. Rewriting the remote cannot remove earlier copies already held in independent clones, forks, or caches.

## Validation

The staged allowlist was checked before publication. `git diff --check`, notebook JSON validation, 176 Python-file parses, and `bash -n` on 154 shell files passed. The full pytest run cannot collect `test_xrv_utils.py` because this machine has no `torch`; excluding that file gives 4 passes and 1 pre-existing failure because `validate_cleanlab_format()` returns `None` after reporting success. No test or experiment code changed. After the history rewrite, all 111 reviewed paths are absent from `main` history; the remote exposes only `main`, and the local source checkout matches it. All 99 index-removed files remain locally and match their saved SHA-256 values.
