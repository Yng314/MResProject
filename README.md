# MResProject

This repository contains the source code and protocol documents for the MRes medical-image label-quality projects. The main active line is `NoiseRate`, which studies label-noise detection, selective review, and iterative refinement for chest X-ray labels. `MedSoul` is the earlier weakly supervised classification project retained for provenance.

## Repository boundary

GitHub stores source code, configuration, tests, protocol documents, and small aggregate summaries that do not contain patient-level records. Datasets, model checkpoints, Slurm logs, generated slide packages, result trees, external-API response tables, and local links to result archives stay outside GitHub.

The large experiment archive is on the cluster at:

```text
/vol/bitbucket/yz3522/NoiseRate_results_archive
```

The working checkout is on GPU Data at:

```text
/vol/gpudata/yz3522-llmtest/MResProject
```

Some local result paths are symbolic links into the Bitbucket archive. They are intentionally ignored and must not be committed as links containing machine-specific paths.

Four case-level CSVs and six meeting drafts listed explicitly in `.gitignore` are local-only. Git no longer tracks them, and the original files remain at their existing paths so local analysis can continue to read them. The meeting bundle's local-path draft repeats the report cases even though it omits their identifiers, so it follows the same boundary. Aggregate tables and plotting source remain tracked. Earlier public commits still contain the removed versions; this cleanup does not rewrite history.

## Main areas

| Area | Role | GitHub status |
| --- | --- | --- |
| `NoiseRate/` | Current label-noise analysis implementation | Source, tests, protocols, and selected safe summaries |
| `NoiseRate/cxr_real_experiment/` | MIMIC-CXR and VinDr experiment drivers, evaluators, and Slurm entrypoints | Scripts and protocol documents; generated results remain external |
| `NoiseRate/cxr_toy_experiment/` | Small known-truth validation examples | Reproducible code and toy outputs only |
| `NoiseRate/utils/`, `models/`, `config/`, `tests/` | Shared utilities, models, configuration, and tests | Tracked source |
| `MedSoul/` | Earlier weakly supervised classification pipeline | Tracked source and documentation; data and outputs remain external |
| `reference/` | Literature and exploratory notebooks | Review individually before publishing |
| `NoiseRate/local_notes/` | Local literature, meeting, and audit notes | Kept on the cluster; only the directory README is public |

## Local use

Create an environment from the project requirements, then run the unit tests from `NoiseRate/`:

```bash
cd /vol/gpudata/yz3522-llmtest/MResProject
python -m pytest NoiseRate/tests
```

GPU experiments are submitted from `NoiseRate/cxr_real_experiment/` with the relevant `run_*.sh` or `submit_*.sh` entrypoint. Those scripts expect cluster datasets and the external result archive; they are not local-only demos.

The three historical `MedSoul/slurm_jobs/run_*.sh` entrypoints are also tracked. Their hard-coded project path predates the move into `MResProject/`; consult the Slurm README and adapt paths before using them. The workspace entrypoint is [llmtest-workspace](https://github.com/Yng314/MResProject/tree/llmtest-workspace), which publishes a source snapshot of this repository alongside the thesis. Large-data locations and long-term storage recommendations are documented there.

## Deployment and publication

There is no application deployment. Publication means committing the reviewed source/doc subset to `origin/main`. Before a push, check the candidate file list, scan for secrets and patient-level fields, run `git diff --check`, and verify that generated outputs and external symlinks are ignored.

## Search record

This organization pass inspected the local Git history and the public GitHub metadata for `Yng314/MResProject`; no external `skills.sh` search was used. The public `main` branch was checked against the local `HEAD` at the end of the pass.

## Completed and pending

- Completed: separated the GPU Data checkout from the Bitbucket result archive in the repository policy; added ignore rules for external links, vendor copies, output trees, logs, and generated presentation directories; recorded the current classification and privacy review in `REPOSITORY_ORGANIZATION.md`.
- Completed in this pass: prepared and pushed the 289-file source/protocol candidate as `5331f88`; grouped 21 literature, meeting, and audit notes under ignored `NoiseRate/local_notes/` in `89fb6c8`; recorded and refreshed the cleanup documentation.
- Completed: ignore and stop tracking four case-level CSVs and six meeting drafts, retaining the originals and a hash-verified recovery copy.
- Pending: decide whether the existing public history must be rewritten to remove already-published patient-level artifacts. Earlier commits still contain the previously published copies; history rewriting remains a separate operation.
