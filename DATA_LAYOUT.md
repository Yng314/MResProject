# Dataset and experiment-artifact locations

Last checked: 2026-10-08. This is a location index, not a data-use approval.

| Material | Current location | Git status | Notes |
| --- | --- | --- | --- |
| MIMIC-CXR-JPG and REFLACX working data | `/vol/bitbucket/yz3522/datasets/MResProject/MedSoul/datasets/` | Outside Git | Copied from `/vol/gpudata/yz3522-llmtest/MResProject/MedSoul/datasets/` on 2026-10-08; rsync completed successfully, total size reported as 6.03G. User confirms supervisor approval. GPU Data source is retained. Transfer note: `/vol/bitbucket/yz3522/datasets/MResProject/TRANSFER_MANIFEST.md`. |
| VinDr-CXR working data | `/vol/bitbucket/yz3522/datasets/vindr-cxr` | Outside Git | Existing location; Bitbucket is temporary storage and is not backed up. |
| Large NoiseRate and MedSoul experiment archive | `/vol/bitbucket/yz3522/NoiseRate_results_archive` | Outside Git | Contains experiment outputs, logs, checkpoints, and generated materials. It is not a backup copy. |
| MedSoul outputs and Slurm logs | `MResProject/MedSoul/outputs/` and `MResProject/MedSoul/slurm_logs/` | Ignored symlinks to the archive above | Local links resolve into the `MedSoul/` subtree of the archive. |
| Synthetic known-truth toy outputs | `/vol/bitbucket/yz3522/NoiseRate_results_archive/repo_artifacts/toy_synthetic_20261008/` | Outside Git | 15 CSV/PNG files copied and SHA-256 checked; `MANIFEST.tsv` records the mapping. These outputs are regenerable and contain synthetic data. |
| Small legacy outputs removed from Git | `/vol/bitbucket/yz3522/NoiseRate_results_archive/repo_artifacts/repository_cleanup_20261008/` | Outside Git | 99 files (6,600,837 bytes) copied and SHA-256 verified against the pre-removal manifest; GPU Data originals remain in place. |
| Python environments, caches, vendored packages | Local workspace paths | Ignored; not in Git | Recreate from `NoiseRate/requirements.txt` or `MedSoul/requirements.txt`; install a platform-specific PyTorch build separately. |

## Storage boundary

The user reports that the supervisor approved storing the MIMIC-CXR-JPG and REFLACX working data in the listed personal Bitbucket project directory; the copy completed on 2026-10-08 and the original remains on GPU Data. This index records location and transfer status, not data-use authorization. Keep the dataset under its existing access and data-use terms and out of public GitHub. Imperial describes `/vol/bitbucket` as temporary space that is not backed up, so it should not be treated as the sole long-term copy.

For durable retention, use an Imperial-maintained storage service approved for the project's data classification and provider terms. Keep a separate approved copy if these files must be retained beyond Bitbucket's temporary-storage period.

Do not put restricted source data, reports, case-level tables, or API review payloads in public GitHub. The repository ignores generated result files by default. A regular Git commit only updates the current tree; older public commits may still contain earlier versions.

## Related guidance

- [Imperial CSG storage quota and Bitbucket guidance](https://www.imperial.ac.uk/computing/people/csg/guides/file-storage/quota/)
- [PhysioNet MIMIC-CXR access and DUA](https://physionet.org/content/mimic-cxr/2.1.0/)
- [Imperial Research Computing Service access and data-classification terms](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/get-access/)
- [Imperial Secure Research Services](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/service-offering/secure-research-service/)
