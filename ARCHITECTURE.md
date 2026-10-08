# Repository architecture

## Modules

- `NoiseRate/`: active label-noise research source, experiment programs, protocols, tests, and shared configuration.
- `NoiseRate/cxr_real_experiment/`: dataset-specific experiment drivers, evaluators, Slurm entrypoints, and protocols. Drivers read authorized dataset locations and write generated artifacts to configured external result locations.
- `NoiseRate/cxr_toy_experiment/` and `NoiseRate/toy_experiment/`: small known-truth validation code; generated tables and plots are excluded from Git.
- `NoiseRate/utils/`, `NoiseRate/models/`, `NoiseRate/config/`, and `NoiseRate/tests/`: reusable analysis functions, model definitions, configuration, and tests.
- `MedSoul/`: earlier weakly supervised classification pipeline with an independent `requirements.txt`.
- `reference/`: lightweight method notes; scratch notebooks are kept out of Git.
- `DATA_LAYOUT.md`: index of the actual dataset and result locations and their current storage status.

## Data and call flow

```text
authorized dataset location
  -> NoiseRate or MedSoul experiment code
  -> training / OOF / review / evaluation steps
  -> configured external result location
```

Git contains source, tests, configuration, dependency manifests, and lightweight protocol documents. Runtime environments and generated experiment artifacts stay out of Git. Machine-specific result links are ignored so a checkout cannot accidentally publish a local storage path or archive contents.

## Design decisions

1. Keep the project repository source-oriented. CSV/Excel result tables, logs, arrays, generated figures, checkpoints, and presentation exports are ignored; the original local files remain available at their existing paths.
2. Keep dependency declarations in `NoiseRate/requirements.txt` and `MedSoul/requirements.txt`; install a machine-appropriate PyTorch build separately before installing the listed packages.
3. Preserve established experiment paths until their consumers and archived run manifests have been checked. This cleanup changes Git tracking rules, not experiment code or result paths.
4. Treat dataset movement as a separate storage decision. MIMIC-CXR and REFLACX are not copied by this cleanup; consult their access terms and Imperial storage guidance first.
5. A normal commit changes the current Git tree only. It does not remove previous public Git objects; history rewriting requires a separate decision.

## Verification

Before publishing, inspect the exact staged file list, run `git diff --check` and the available tests, and confirm that no environment directory, dataset, generated output, or external-storage symlink is staged.
