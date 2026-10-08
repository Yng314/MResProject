# Repository organization

Audit date: 2026-10-08
Scope: `/vol/gpudata/yz3522-llmtest/MResProject`, branch `main`
Remote: `Yng314/MResProject`

## Intended boundary

The `main` branch is the lightweight source repository. It keeps Python and shell source, tests, configuration, dependency manifests, and concise protocols. It excludes datasets, virtual environments, vendor copies, result tables, training logs, OOF arrays, generated figures, scratch notebooks, checkpoints, and machine-specific links. The broader dataset and result map is in [DATA_LAYOUT.md](DATA_LAYOUT.md).

## Changes in this pass

- Stopped tracking 99 generated or scratch files: 36 CSVs, 48 logs, two OOF arrays, one Excel workbook, nine PNGs, two scratch/reference notebooks, and one exported transcript.
- Kept every original file at its existing GPU Data path. The full output-bearing toy notebook was copied to the parent workspace's ignored `.repo_backups/MResProject_cleanup_20261008/` before its generated cell outputs were cleared; the notebook's code remains tracked.
- Added generated-artifact ignore rules and a `NoiseRate/requirements.txt` dependency manifest. No virtual environment was added.
- Did not copy MIMIC-CXR/REFLACX data or their case-level derivatives to Bitbucket. Their permitted storage location still needs confirmation; see [DATA_LAYOUT.md](DATA_LAYOUT.md).

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

The current `main` tree no longer tracks the generated files listed above. Earlier public commits still contain previously tracked outputs and some case-level material. A normal commit cannot remove those old Git objects; history rewriting would require a separate, explicit decision.

This audit targets `main` only. The separate public `llmtest-workspace` branch is an older combined workspace snapshot that includes thesis/defence material. It is left unchanged because that material was explicitly excluded from this cleanup. Do not treat that branch as the source-only checkout.

## Validation

The staged allowlist was checked before publication. `git diff --check`, notebook JSON validation, 176 Python-file parses, and `bash -n` on 154 shell files passed. The full pytest run cannot collect `test_xrv_utils.py` because this machine has no `torch`; excluding that file gives 4 passes and 1 pre-existing failure because `validate_cleanlab_format()` returns `None` after reporting success. No test or experiment code changed. Commit `d8f4e04` was pushed, then `HEAD` and `origin/main` matched and the worktree was clean. All 99 index-removed files remain locally and match their saved SHA-256 values.
