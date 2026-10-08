# MResProject

Research source code for chest X-ray label-noise analysis and weakly supervised classification. The repository contains programs, tests, configuration, protocols, and dependency manifests; generated experiment outputs and datasets are kept outside Git.

## Project areas

- `NoiseRate/` contains the active label-noise analysis tools, experiment drivers, evaluation code, tests, configuration, and research protocols.
- `MedSoul/` contains the earlier weakly supervised classification pipeline and its own dependency manifest.
- `reference/` contains lightweight method notes and source material.

The module relationships and data flow are described in [ARCHITECTURE.md](ARCHITECTURE.md). Current dataset and result locations, including items that still need storage approval, are listed in [DATA_LAYOUT.md](DATA_LAYOUT.md).

## Environments and local use

No virtual environment, package cache, model weights, or vendored dependency is part of Git. `NoiseRate/requirements.txt` lists the Python packages used by the active research code. Install the PyTorch build appropriate for the machine first, then install the remaining requirements:

```bash
python -m pip install -r NoiseRate/requirements.txt
python -m pytest NoiseRate/tests
```

`MedSoul/requirements.txt` is retained for the earlier pipeline. Its environment can be prepared separately. GPU experiments run on the Imperial cluster and require authorized datasets plus writable output locations; they are not local demo commands.

## Storage boundary

GitHub is for source and lightweight reproducibility documents. Datasets, logs, predictions, result tables, checkpoints, generated figures, and presentation exports are excluded by `.gitignore`. See [DATA_LAYOUT.md](DATA_LAYOUT.md) before locating or moving research data. Bitbucket is not a backup service, and restricted datasets must only be placed on storage approved for their data-use terms.

## Deployment and checks

There is no application deployment. Review an explicit file list before committing, then run:

```bash
git diff --check
python -m pytest NoiseRate/tests
```

The repository was inspected locally; no `skills.sh` search was needed. GitHub's public `main` branch and this checkout were compared during the repository-boundary audit.

## Status

- Completed: source, generated-output, environment, dataset, and external-storage boundaries are documented; current result artifacts are being excluded from the Git tree without deleting their local copies.
- Pending: confirm an approved storage location for MIMIC-CXR/REFLACX and their restricted derivatives; decide separately whether to clean older public Git history, which contains prior versions of files now excluded from the current tree.
