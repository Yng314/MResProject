# Noisy Sample Case Notes

## Case 1: Likely true label issue in CheXpert-derived label

- `sample suspicious rank`: 1
- `subject_id`: 17080143
- `study_id`: 53342449
- `dicom_id`: `029d87ab-8eec033b-51211e87-fcc13973-c67dfa35`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17080143/s53342449/029d87ab-8eec033b-51211e87-fcc13973-c67dfa35.jpg`
- `flagged label`: `Pleural Effusion`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.999601`
- `entry_quality_self_confidence`: `0.000399`
- `sample_quality_self_confidence`: `0.000399`

### Report snippet

> Bilateral effusions are again seen, unchanged since the prior chest x-ray.
> IMPRESSION: No change, bilateral effusions.

### Interpretation

This case is a strong example of a likely true label issue in the automatically derived CheXpert label. The report explicitly states bilateral pleural effusions, while the structured label for `Pleural Effusion` is `0`. The model also predicts `Pleural Effusion` with extremely high confidence, and cleanlab ranks this sample as the most suspicious one. Taken together, the report text, model prediction, and cleanlab score are all consistent with the conclusion that the CheXpert-derived label is likely incorrect for this sample.

### Supporting visualizations

- Image: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17080143/s53342449/029d87ab-8eec033b-51211e87-fcc13973-c67dfa35.jpg`
- Grad-CAM: `/vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/meeting_followup_20260520/gradcam_rank1_pleural_effusion.png`

## Case 2: Another likely true label issue in CheXpert-derived label

- `sample suspicious rank`: 4
- `subject_id`: 17143033
- `study_id`: 55070921
- `dicom_id`: `060a94d1-983a6aa4-4b52bb1e-c0abe86e-969229a7`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17143033/s55070921/060a94d1-983a6aa4-4b52bb1e-c0abe86e-969229a7.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.994902`
- `entry_quality_self_confidence`: `0.005098`
- `sample_quality_self_confidence`: `0.005098`

### Report snippet

> The heart is moderately enlarged.
> IMPRESSION: Moderate cardiomegaly, otherwise unremarkable.

### Interpretation

This case also looks like a likely true label issue rather than a model false alarm. Even though the image style initially made cardiomegaly overcalling seem plausible, the report explicitly states moderate cardiomegaly. The structured label for `Cardiomegaly` is nevertheless `0`, while the model prediction is extremely high and cleanlab ranks the sample among the most suspicious ones. This makes the automatically derived CheXpert label likely incorrect for this sample as well.

### Supporting visualizations

- Image: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17143033/s55070921/060a94d1-983a6aa4-4b52bb1e-c0abe86e-969229a7.jpg`
- Grad-CAM: `/vol/gpudata/yz3522-llmtest/MRes/NoiseRate/cxr_real_experiment/meeting_followup_20260520/gradcam_rank4_cardiomegaly.png`

## Case 3: Likely false alarm from the model

- `sample suspicious rank`: 12
- `subject_id`: 18402151
- `study_id`: 55113475
- `dicom_id`: `da878459-5a7451c5-668eefe3-a241198e-c23fc5b2`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p18/p18402151/s55113475/da878459-5a7451c5-668eefe3-a241198e-c23fc5b2.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.992731`
- `entry_quality_self_confidence`: `0.007269`

### Report snippet

> Heart size and mediastinum are overall unremarkable.

### Interpretation

This case looks like a likely false alarm from the model rather than a true label issue. The CheXpert-derived label is `0`, and the report text does not support cardiomegaly. In fact, the report explicitly describes the heart size as unremarkable. However, the model still predicts cardiomegaly with extremely high confidence, and cleanlab ranks the sample among the most suspicious ones. This suggests that, at least for this case, the model may be overcalling cardiomegaly despite both the structured label and the report text indicating otherwise.

## Case 4: Ambiguous case due to report section inconsistency

- `entry suspicious rank`: 1
- `subject_id`: `15561274`
- `study_id`: `53070008`
- `dicom_id`: `2ebd1fb4-58af802a-a2b7e57f-5f573cb1-2a125f7c`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p15/p15561274/s53070008/2ebd1fb4-58af802a-a2b7e57f-5f573cb1-2a125f7c.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.997660`
- `entry_quality_self_confidence`: `0.002340`

### Report snippet

> The heart is moderately enlarged, though appears grossly stable from prior exam.
> IMPRESSION: No cardiomegaly, consolidation or mass seen.

### Interpretation

This case is best treated as ambiguous rather than a clean true issue or a clean false alarm. The model predicts cardiomegaly with extremely high confidence, and the findings section of the report also states that the heart is moderately enlarged. However, the impression section then says “No cardiomegaly,” which directly conflicts with the findings. According to the MIMIC-CXR-JPG documentation, the released CheXpert-derived labels are based on the impression section when it is available, and only fall back to findings when impression is absent. Therefore, the structured label of `0` is understandable under the dataset’s section-priority rule, even though the full report is internally inconsistent. This makes the sample a useful example of section-level report inconsistency rather than a straightforward labeling mistake or model false alarm.
