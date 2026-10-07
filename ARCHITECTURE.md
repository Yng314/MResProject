# Repository architecture

`MResProject` is a source repository with two research lines and an external artifact store.

## Modules

- `NoiseRate/`: active label-noise research code.
- `NoiseRate/cxr_real_experiment/`: experiment-specific Python programs, Slurm runners, evaluation scripts, protocol notes, and presentation builders. It reads datasets and writes results to the external archive rather than to GitHub.
- `NoiseRate/cxr_toy_experiment/`: small known-truth experiments used to sanity-check noise-rate and DQS calculations.
- `NoiseRate/utils/`, `NoiseRate/models/`, `NoiseRate/config/`, `NoiseRate/tests/`: reusable utilities, model definitions, configuration, and unit tests.
- `MedSoul/`: earlier weakly supervised medical-image classification pipeline, including data preparation, training, evaluation, and API helpers.
- `MedSoul/slurm_jobs/run_*.sh`: three historical cluster entrypoints for training, noise estimation and their combined pipeline; original script contents are preserved and their old paths need adaptation before execution.
- `reference/`: research notes and exploratory notebooks.
- `NoiseRate/local_notes/`: local-only literature, meeting, and audit notes grouped away from executable experiment code.
- `/vol/bitbucket/yz3522/NoiseRate_results_archive`: cluster-side result store for checkpoints, run transactions, logs, large tables, and generated figures/decks.

## Call flow

```text
cluster datasets
  -> NoiseRate experiment driver
  -> OOF / selection / review / evaluation stages
  -> external Bitbucket result archive
  -> small reviewed aggregate summaries
```

The source checkout may expose selected archive folders through symbolic links for local analysis. Those links are ignored because they are machine-specific and can accidentally expose large or sensitive artifacts in Git history.

## Design decisions

1. Keep code, protocols, tests, and reproducibility notes in GitHub.
2. Keep datasets, checkpoints, raw review responses, patient-level tables, Slurm logs, and generated delivery packages outside the public repository.
3. Keep the existing experiment paths stable for now. A path-wide refactor would break Slurm entrypoints and archived manifests; classification is therefore enforced first through documentation and ignore rules.
4. Review every aggregate CSV or notebook before publication because a small file can still contain subject, study, DICOM, report, or image-path fields.
5. Case-level tables and their meeting drafts remain at the original analysis paths without being Git inputs. The explicit ignore list and removal from Git tracking separate local availability from publication; no source entrypoint or aggregate table is removed.
6. The public `llmtest-workspace` branch contains a snapshot of both active projects and the root entrypoint. Their source commits are recorded in `SOURCE_VERSIONS.json`; the existing project repositories retain their independent history.

## Verification boundary

Before pushing, inspect `git status --short --untracked-files=all`, audit the exact candidate list, run `git diff --check`, and confirm that no ignored symlink or generated result path is staged. A normal commit cannot remove data already present in old public commits; a history rewrite would be a separate, explicit operation.
