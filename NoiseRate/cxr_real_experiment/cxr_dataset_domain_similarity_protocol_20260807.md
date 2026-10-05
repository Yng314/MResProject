# VinDr-CXR vs MIMIC-CXR Image-Domain Comparison

## Question

How large is the image-domain gap between the controlled VinDr-CXR benchmark and the MIMIC-CXR training data used by the main project?

## Locked design

- Compare all 3,000 verified VinDr-CXR test images with 3,000 MIMIC-CXR train AP/PA images.
- Select at most one MIMIC image per subject with fixed seed `20260807`.
- Use identical 224x224 grayscale loading and TorchXRayVision normalization.
- Freeze three DenseNet-121 encoders: `all`, `nih`, and `pc`.
- Extract the 1,024-dimensional pooled penultimate representation.
- Fit pooled standardization and 128-dimensional PCA separately for each encoder.

## Outcomes

1. Five-fold classifier two-sample test AUROC with a sample bootstrap interval.
2. RBF-MMD with a pooled median bandwidth, permutation p-value, and random within-dataset split references.
3. Frechet feature distance with matched-size repeated cross-dataset and within-dataset references.
4. PCA projections for descriptive visualization only.

The `all` encoder contains MIMIC-CXR in its pretraining mixture and is therefore a project-consistent but not independent representation. The `nih` and `pc` encoders exclude both MIMIC-CXR and VinDr-CXR from pretraining and serve as the independence sensitivity analysis.

## Interpretation

No universal cutoff defines two datasets as similar. Cross-dataset values are interpreted relative to the same encoder's within-dataset random-split reference. This experiment quantifies a domain gap; it does not prove that benchmark conclusions transfer without qualification.
