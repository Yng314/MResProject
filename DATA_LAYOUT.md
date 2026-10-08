# Dataset and experiment-artifact locations

Last checked: 2026-10-08. This is a location index, not a data-use approval.

| Material | Current location | Git status | Notes |
| --- | --- | --- | --- |
| MIMIC-CXR-JPG and REFLACX working data | `MResProject/MedSoul/datasets/` under `/vol/gpudata/yz3522-llmtest` | Ignored; not in Git | Keep local until Imperial/PhysioNet-compatible storage is confirmed. Full recursive size was not established in the previous inventory. |
| VinDr-CXR working data | `/vol/bitbucket/yz3522/datasets/vindr-cxr` | Outside Git | Existing location; Bitbucket is temporary storage and is not backed up. |
| Large NoiseRate and MedSoul experiment archive | `/vol/bitbucket/yz3522/NoiseRate_results_archive` | Outside Git | Contains experiment outputs, logs, checkpoints, and generated materials. It is not a backup copy. |
| MedSoul outputs and Slurm logs | `MResProject/MedSoul/outputs/` and `MResProject/MedSoul/slurm_logs/` | Ignored symlinks to the archive above | Local links resolve into the `MedSoul/` subtree of the archive. |
| Synthetic known-truth toy outputs | `/vol/bitbucket/yz3522/NoiseRate_results_archive/repo_artifacts/toy_synthetic_20261008/` | Outside Git | 15 CSV/PNG files copied and SHA-256 checked; `MANIFEST.tsv` records the mapping. These outputs are regenerable and contain synthetic data. |
| Other small legacy outputs removed from Git | Existing `NoiseRate/` paths in the GPU Data checkout | Ignored; originals remain at those paths | Their archive copies have not been individually verified. MIMIC/REFLACX-derived materials were not copied while their storage status remains unresolved. |
| Python environments, caches, vendored packages | Local workspace paths | Ignored; not in Git | Recreate from `NoiseRate/requirements.txt` or `MedSoul/requirements.txt`; install a platform-specific PyTorch build separately. |

## Storage boundary

Imperial CSG describes `/vol/bitbucket` as temporary space for regenerable material and states it is not backed up. The CSG guide also warns against placing copyright-restricted material there. MIMIC-CXR requires credentialed access and a data-use agreement that prohibits sharing access to the data. Therefore the presence of MIMIC/REFLACX or their case-level derivatives in an existing folder does not by itself establish that Bitbucket is an approved archive destination. Confirm the permitted storage location with the supervisor or data administrator before moving or copying them.

Do not put restricted source data, reports, case-level tables, or API review payloads in public GitHub. The repository ignores generated result files by default. A regular Git commit only updates the current tree; older public commits may still contain earlier versions.

## Related guidance

- [Imperial CSG storage quota and Bitbucket guidance](https://www.imperial.ac.uk/computing/people/csg/guides/file-storage/quota/)
- [PhysioNet MIMIC-CXR access and DUA](https://physionet.org/content/mimic-cxr/2.1.0/)
