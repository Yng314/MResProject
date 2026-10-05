# Repository organization audit

Audit date: 2026-10-05
Checkout: `/vol/gpudata/yz3522-llmtest/MResProject`
Remote: `git@github-yng314:Yng314/MResProject.git` (`https://github.com/Yng314/MResProject`)
Remote branch: `main` at `cf504cb`

The safe source/protocol cleanup was pushed as commit `5331f88`, the local-note grouping as `89fb6c8`, and the final context records as `cf504cb`; `HEAD` and `origin/main` now match.

The parent GPU Data checkout also contains separate repositories, `MResAIML_thesis` and `CheXGPT`; this audit targets `MResProject` because it is the repository connected to the noisy experiment tree and the remote above.

## Current state

- The GitHub repository is public.
- The current public tree has 517 tracked paths after the allowlist push; its older history still contains artifacts that are not part of the new boundary.
- At the start of the audit the local tree had two modified tracked files and 343 changed or untracked `NoiseRate` entries, plus two local output symlinks under `MedSoul`; after the new ignore rules and review split, the source/protocol candidate contains 289 files.
- The local result links point to `/vol/bitbucket/yz3522/NoiseRate_results_archive`; the Bitbucket archive contains the large MIMIC-CXR, VinDr-CXR, MedSoul, Slurm, checkpoint, and presentation outputs.
- The local checkout also contains a vendored DICOM/JPEG dependency under `NoiseRate/.vendor/`; this is an environment artifact, not project source.

## Classification

| Class | Examples | Action for GitHub |
| --- | --- | --- |
| Core source | `NoiseRate/*.py`, `NoiseRate/utils/`, `NoiseRate/models/`, `MedSoul/` source | Keep after a source-level review |
| Experiment entrypoints | `NoiseRate/cxr_real_experiment/run_*.sh`, `submit_*.sh`, drivers, evaluators | Keep; these define reproducible cluster workflows |
| Protocol and state docs | `*_protocol*.md`, `*_validation*.md`, `CONTEXT.md`, architecture notes | Keep after checking paths and claims; retain the verbose `ASSIGNMENT.md` job log locally |
| Tests | `NoiseRate/tests/`, `test_*.py` | Keep and run the relevant tests |
| Toy outputs and safe aggregates | known-truth toy summaries and metric tables without identifiers | Keep selectively |
| Raw or patient-level tables | rows with subject/study/DICOM IDs, report text, image paths, or API review explanations | Do not add; review already tracked copies for removal |
| Generated delivery files | PPTX/PDF/PNG bundles, deck asset folders, temporary slide packages | Keep external; regenerate from source when needed |
| Run artifacts | `results_*`, `outputs`, `slurm_logs`, checkpoints, `.job_complete` markers | Keep in the Bitbucket archive |
| Environment/vendor files | `.vendor/`, caches, local virtual environments | Ignore and leave local |
| Machine-specific links | symlinks from `MResProject` into `/vol/bitbucket/...` | Ignore; do not commit |
| Local notes | `NoiseRate/local_notes/literature_review/`, `meeting_materials/`, `analysis_audits/` | Keep grouped locally; publish only the directory README |

## Public-data issue found

Several files already in the public remote tree contain patient-level or exam-level fields. Examples include the old top-5 LLM summary and meeting-follow-up tables with `subject_id`, `study_id`, `dicom_id`, image paths, and report text. A normal new commit can stop additional exposure, but it cannot erase those old blobs from public Git history.

The current-tree files that need a privacy review first include:

- `NoiseRate/cxr_real_experiment/top5_llm_judgment_summary.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/densenet_sample_issue_review_full.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/densenet_sample_issue_review_top100.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/report_false_alarm_scan/high_suspicion_entries_with_report_support.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/report_false_alarm_scan/likely_false_alarm_candidates_from_reports.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/xrv_linear_baseline_top_noisy_samples.csv`
- the accompanying case-note Markdown files in that meeting-follow-up directory

The top-5 summary, the two report-scan CSVs, the XRV baseline table, and the accompanying case notes are already in the remote history or current tracked tree. The two `densenet_sample_issue_review_*.csv` files are local material that the new ignore rules keep out of future commits.

The safe sequence is:

1. Freeze the candidate source/document allowlist.
2. Remove patient-level files from the current tree and add redacted aggregate replacements only where needed.
3. Run a secret/identifier scan and inspect the exact staged list. The source scan completed here with no private-key or hard-coded API-token match; code references to environment variables and dataset column names are expected and still require normal review.
4. The safe current-tree commits are already pushed (`5331f88`, `89fb6c8`, `cf504cb`).
5. Separately decide whether to rewrite the public history and coordinate that force-push; it is a destructive remote operation.

## Proposed repository boundary

```text
GitHub (public): source, tests, configs, protocols, safe aggregate summaries
Bitbucket archive: raw results, logs, checkpoints, generated decks, review tables,
                   datasets, and all patient-level or report-level artifacts
```

The first cleanup pass deliberately keeps existing paths stable. Moving hundreds of experiment files would invalidate archived command lines and manifests; path refactoring can follow after the publication boundary is safe.

## Validation performed in this pass

- Python AST parse: 106 local Python files parsed successfully.
- Shell syntax: all current `NoiseRate/cxr_real_experiment/*.sh` files passed `bash -n`.
- Existing lightweight test: 4 tests passed and 1 existing test failed because `validate_cleanlab_format()` returns `None` despite printing a successful validation message.
- Full `NoiseRate/tests` collection is blocked by the environment because `torch` is not installed for `test_xrv_utils.py`.
- `git diff --check` passed.
