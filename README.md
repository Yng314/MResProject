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

## Deployment and publication

There is no application deployment. Publication means committing the reviewed source/doc subset to `origin/main`. Before a push, check the candidate file list, scan for secrets and patient-level fields, run `git diff --check`, and verify that generated outputs and external symlinks are ignored.

## Search record

This organization pass inspected the local Git history and the public GitHub metadata for `Yng314/MResProject`; no external `skills.sh` search was used. The public `main` branch now points to the cleanup commits ending at `cf504cb`.

## Completed and pending

- Completed: separated the GPU Data checkout from the Bitbucket result archive in the repository policy; added ignore rules for external links, vendor copies, output trees, logs, and generated presentation directories; recorded the current classification and privacy review in `REPOSITORY_ORGANIZATION.md`.
- Completed in this pass: prepared and pushed the 289-file source/protocol candidate as `5331f88`; grouped 21 literature, meeting, and audit notes under ignored `NoiseRate/local_notes/` in `89fb6c8`; recorded the final cleanup state in `cf504cb`.
- Pending: decide whether the existing public history must be rewritten to remove already-published patient-level artifacts. The safe current-tree boundary is already pushed; history rewriting remains a separate destructive operation.
