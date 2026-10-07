# Repository organization audit

Audit date: 2026-10-07
Checkout: `/vol/gpudata/yz3522-llmtest/MResProject`
Remote: `git@github-yng314:Yng314/MResProject.git` (`https://github.com/Yng314/MResProject`)
Remote branch: `main` (local `HEAD` matched `origin/main` at the end of the audit)

The safe source/protocol cleanup was pushed as commit `5331f88`, the local-note grouping as `89fb6c8`, and the cleanup documentation was refreshed afterward; `HEAD` and `origin/main` matched at the end of the audit.

The parent GPU Data checkout also contains separate repositories, `MResAIML_thesis` and `CheXGPT`; this audit targets `MResProject` because it is the repository connected to the noisy experiment tree and the remote above.

## Current state

- The GitHub repository is public.
- The initial allowlist tree had 517 tracked paths; this cleanup stops tracking ten case-level files, leaving 507 paths. Older history still contains artifacts outside the current boundary.
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

The identified case-level files include:

- `NoiseRate/cxr_real_experiment/top5_llm_judgment_summary.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/densenet_sample_issue_review_full.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/densenet_sample_issue_review_top100.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/report_false_alarm_scan/high_suspicion_entries_with_report_support.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/report_false_alarm_scan/likely_false_alarm_candidates_from_reports.csv`
- `NoiseRate/cxr_real_experiment/meeting_followup_20260520/xrv_linear_baseline_top_noisy_samples.csv`
- the accompanying case-note Markdown files in that meeting-follow-up directory

The top-5 summary, the two report-scan CSVs, the XRV baseline table, and the accompanying case notes are already in the remote history or current tracked tree. The two `densenet_sample_issue_review_*.csv` files are local material that the new ignore rules keep out of future commits.

### Case-level exclusion on 2026-10-07

The cleanup stops tracking the four CSVs above and six meeting drafts: `MRes_Meeting_20260604.md`, `noisy_sample_case_notes.md`, `supplemental_case_notes_20260603.md`, `weighted_and_entry_trend_script.md`, and the two corresponding draft copies in `ppt_bundle_20260528/`. Five drafts contain identifier-like values; the sixth (`weighted_and_entry_trend_script_local.md`) repeats report cases without their identifiers. The ten originals stay at the same paths and have hash-verified recovery copies under the parent workspace's `.repo_backups/`.

The CSV filenames are generated outputs of the report scan and follow-up analysis. No executable input reference was found in the inspected Python, shell, JSON, YAML, and Markdown source search; references between the local meeting drafts remain valid. Aggregate CSVs, plotting source, experiment programs, and binary artifacts are unchanged. The meeting area's Git index contains no tracked PNG/JPG/PDF/PPTX/ZIP artifact.

The ten originals are excluded from the resulting Git tree; earlier commits still contain them because history was not rewritten. The post-cleanup scan checked all 36 retained CSV headers and all tracked Markdown for identifier-like values, plus the meeting draft's repeated report cases; it does not certify every binary artifact or historical Git object.

The publication sequence is:

1. Freeze the candidate source/document allowlist.
2. Stop tracking the identified patient-level files while retaining local originals and aggregate summaries.
3. Run a secret/identifier scan and inspect the exact staged list. The source scan completed here with no private-key or hard-coded API-token match; code references to environment variables and dataset column names are expected and still require normal review.
4. Publish this ten-path exclusion separately from the earlier source/organization changes.
5. Separately decide whether to rewrite the public history and coordinate that force-push; it is a destructive remote operation.

## Proposed repository boundary

```text
GitHub (public): source, tests, configs, protocols, safe aggregate summaries
Bitbucket archive: raw results, logs, checkpoints, generated decks, review tables,
                   datasets, and all patient-level or report-level artifacts
```

## Largest local result area

`NoiseRate/cxr_real_experiment/meeting_followup_20260520/` is the largest dated local follow-up area in this checkout (about 74 MB). It contains a 69 MB ignored sample-level review table, report-support CSVs, case images, meeting notes, exploratory figures, and a small plotting script. The folder is being indexed before any path move: scripts and protocol notes are source candidates, aggregate figures are regeneration candidates, and sample-level tables, report text, case notes, and presentation bundles remain local/archive candidates.

The first cleanup pass deliberately keeps existing paths stable. Moving hundreds of experiment files would invalidate archived command lines and manifests; path refactoring can follow after the publication boundary is safe.

## Validation performed in this pass

- Python AST parse: 106 local Python files parsed successfully.
- Shell syntax: all current `NoiseRate/cxr_real_experiment/*.sh` files passed `bash -n`.
- Existing lightweight test: 4 tests passed and 1 existing test failed because `validate_cleanlab_format()` returns `None` despite printing a successful validation message.
- Full `NoiseRate/tests` collection is blocked by the environment because `torch` is not installed for `test_xrv_utils.py`.
- `git diff --check` passed.
