# Supplemental Case Notes (2026-06-03)

These notes extend the initial case review in `noisy_sample_case_notes.md`.
All cases below come from the `XRV linear-head` baseline candidate pool, where:

- the structured CheXpert-derived label is `0`
- the model predicted probability for the flagged label is very high
- cleanlab ranked the corresponding entry as suspicious

The goal here is not to force every disagreement into a single explanation.
Instead, the goal is to collect representative cases for discussion, especially
for the tricky boundary between likely label issues, likely model false alarms,
and report-dependent ambiguous cases.

## Case summary table

| Case | Rank | Label | CheXpert-derived label | Pred prob | Report-based reading | Current interpretation |
|---|---:|---|---:|---:|---|---|
| Case A | entry rank 1, sample rank 13034 | Lung Lesion | 0 | 0.9951 | report repeatedly recommends follow-up to exclude underlying mass/adenopathy | likely true issue |
| Case B | entry rank 2, sample rank 1783 | Cardiomegaly | 0 | 0.9971 | report says stable enlargement of the cardiac silhouette | likely true issue, but regex miss in report scan |
| Case C | entry rank 4, sample rank 15187 | Cardiomegaly | 0 | 0.9961 | report explicitly says stable moderate to severe cardiomegaly | task-dependent / stable finding |
| Case D | entry rank 14, sample rank 508 | Pleural Effusion | 0 | 0.9955 | findings mention pleural effusions, impression says no effusion | ambiguous report inconsistency |
| Case E | entry rank 8, sample rank 35305 | Pleural Effusion | 0 | 0.9970 | report explicitly says no pleural effusions | likely false alarm |
| Case F | entry rank 11, sample rank 8 | Cardiomegaly | 0 | 0.9928 | report says top normal heart size | borderline / likely model overcall |

## Case A: Likely true label issue for Lung Lesion

- `entry suspicious rank`: 1
- `sample suspicious rank`: 13034
- `subject_id`: 17419105
- `study_id`: 57941247
- `dicom_id`: `001f47ed-b65bea40-5f8ca25c-d82cf022-9518c94b`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17419105/s57941247/001f47ed-b65bea40-5f8ca25c-d82cf022-9518c94b.jpg`
- `flagged label`: `Lung Lesion`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.995131`
- `entry_quality_self_confidence`: `0.004869`
- `sample_quality_self_confidence`: `0.498002`
- `report_relation`: `supported`
- `report_match_phrase`: `mass`

### Full report text

```text
FINAL REPORT
EXAMINATION: CHEST (PORTABLE AP)

INDICATION: ___ year old woman with NGT placement // evaluation of NGT
placement

TECHNIQUE: Chest single view

COMPARISON: ___ 05:49

FINDINGS:

Enteric tube tip in the mid stomach. Endotracheal tube tip in good position.
Right lung has largely re-expanded since prior exam. Persistent right mid
lung, right basilar patchy opacities. Right perihilar opacity, indeterminate,
this area was obscured secondary to patient positioning on ___, and
was normal on ___. Follow-up chest PA and lateral recommended to
document resolution and exclude underlying mass or adenopathy. Trace right
pleural effusion or thickening. Left lung is clear. Thoracolumbar
degenerative changes, mild curve convex to the right.

IMPRESSION:

Right lung is largely re-expanded, persistent patchy right mid lung, basilar
opacities.
Persistent right perihilar opacity, recommend follow-up chest PA and lateral
to document resolution, exclude adenopathy or mass.
```

### Interpretation

This is a strong supplemental example of a likely true issue in the structured
label. The report does not merely describe a nonspecific opacity; it explicitly
raises concern for an underlying `mass` or `adenopathy`, which is much closer
to the semantics of `Lung Lesion` than a routine pneumonia-only reading. The
model prediction is also extremely high, and cleanlab ranks this entry as the
most suspicious `Lung Lesion` negative in the candidate pool.

What makes this case useful is that it is not a top sample-level case. It is a
very strong entry-level conflict but only a middling sample-level conflict,
which also helps explain why entry-level and sample-level cleaning can behave
differently.

## Case B: Likely true issue for Cardiomegaly, but missed by simple report rule

- `entry suspicious rank`: 2
- `sample suspicious rank`: 1783
- `subject_id`: 11630519
- `study_id`: 57464780
- `dicom_id`: `638b72db-c3784dff-78bae056-d69548d6-e1fd957d`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p11/p11630519/s57464780/638b72db-c3784dff-78bae056-d69548d6-e1fd957d.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.997064`
- `entry_quality_self_confidence`: `0.002936`
- `sample_quality_self_confidence`: `0.180606`
- `report_relation`: `unclear`

### Full report text

```text
FINAL REPORT
EXAMINATION: CHEST (PORTABLE AP)

INDICATION: ___M with afib, CHF, DM, HTN, h/o CVA initially admitted ___ with
mechanical fall on anticoagulation with intracranial bleeding. Now awaiting___
rehab placement, receiving inpatient stroke rehab, seized on ___ c/b
lactic acidosis. On ___ more tachypneic s/p 1L fluids. Please assess for
pulmonary edema. // r/o pulmonary edema r/o pulmonary edema

IMPRESSION:

In comparison with the study of ___, there is stable enlargement of the
cardiac silhouette with little or no elevation in pulmonary venous pressure.
This discordance raises the possibility of underlying cardiomyopathy or
pericardial effusion. No evidence of acute focal pneumonia.
There is again an impression on the right of the lower cervical trachea,
suggesting thyroid enlargement.
```

### Interpretation

This case is interesting because the report scanner marked it as `unclear`, but
the full report is actually fairly supportive of cardiomegaly-like anatomy. The
key phrase is `stable enlargement of the cardiac silhouette`. That wording does
not match the simple regex rule used in the report scan, so it becomes a nice
example where the weak text rule under-calls support even though the report
seems to describe an enlarged heart silhouette.

For meeting discussion, this case is useful because it separates two issues:

- the structured label may be problematic
- the lightweight report-pattern scan may also miss clinically relevant support

So this case looks more like a likely true issue than a model false alarm, even
though the automatic report-support tag says `unclear`.

## Case C: Stable cardiomegaly as a task-dependent case

- `entry suspicious rank`: 4
- `sample suspicious rank`: 15187
- `subject_id`: 12363835
- `study_id`: 55275749
- `dicom_id`: `8c387e81-569ce48f-089284ac-106ec236-4d01fcfd`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p12/p12363835/s55275749/8c387e81-569ce48f-089284ac-106ec236-4d01fcfd.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.996133`
- `entry_quality_self_confidence`: `0.003867`
- `sample_quality_self_confidence`: `0.522104`
- `report_relation`: `supported`
- `report_match_phrase`: `cardiomegaly`

### Full report text

```text
FINAL REPORT
HISTORY: Throat pain.

COMPARISON: ___.

TECHNIQUE: PA and lateral views of the chest.

FINDINGS: Cardiomegaly, moderate to severe, is stable. The aorta is tortuous
and the knob is calcified. Trachea is slightly deviated to the right. There
is flattening of the hemidiaphragm suggestive of volume overload. Bibasal
atelectasis is present. Small right pleural effusion is stable over multiple
prior studies. There are no focal opacities concerning for pneumonia.

IMPRESSION: Stable radiograph with moderate to severe cardiomegaly but no
focal opacities concerning for pneumonia.
```

### Interpretation

This is an especially good case for the `tricky` discussion. On the surface, it
looks like a straightforward likely true issue: the report explicitly says
`moderate to severe cardiomegaly`, while the structured label is `0` and the
model prediction is very high.

But it is also exactly the kind of stable, chronic finding that raises the
task-definition question. If the intended label is supposed to track any
present radiographic finding, then this looks like a likely label issue. If the
implicit labeling behavior is closer to highlighting new or clinically active
findings, then one could argue this belongs in a more task-dependent category.

So I would not treat this as a clean false alarm. I would carry it into the
meeting as a strong example of a stable finding that forces the group to define
what the label is supposed to represent.

## Case D: Pleural Effusion with findings-impression inconsistency

- `entry suspicious rank`: 14
- `sample suspicious rank`: 508
- `subject_id`: 10573359
- `study_id`: 51592822
- `dicom_id`: `f23fb804-91cda364-877ce281-ae865e2f-d8d4d508`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p10/p10573359/s51592822/f23fb804-91cda364-877ce281-ae865e2f-d8d4d508.jpg`
- `flagged label`: `Pleural Effusion`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.995476`
- `entry_quality_self_confidence`: `0.004524`
- `sample_quality_self_confidence`: `0.080710`
- `report_relation`: `negated`
- `report_match_phrase`: `no effusion`

### Full report text

```text
FINAL REPORT
EXAMINATION: CHEST (PORTABLE AP)

INDICATION: ___ year old woman post op pna now with hernia recurrence with
continuous O2 requirements s/p chest tube placement // evaluate parapneumonic
effusion

TECHNIQUE: Portable AP film was obtained

COMPARISON: ___

FINDINGS:

There is cardiomegaly in this patient status post sternotomy. A right-sided
pleural effusion. Remains substantially resolved but there is now evidence of
a small right apical pneumothorax. Right-sided PICC line is in good anatomical
position. Retained barium is identified in the left retrocardiac area
suggesting that much of the opacity here may reflect a a hernia. Substantial
left-sided effusion with extensive atelectasis is suggested.

There remains significant increased attenuation throughout the lung parenchyma
consistent with a diffuse ground-glass opacification is observed recent CT.
This is presumptively represent of infection as per the CT though pulmonary
edema or other causes need to be considered.

IMPRESSION:

new right-sided pneumothorax. No effusion. Stable diffuse bilateral
parenchymal changes.

NOTIFICATION: The findings were discussed by Dr. ___ with Dr. ___ on
the telephone on ___ at 9:00 AM, 15 minutes after discovery of the
findings.
```

### Interpretation

This is not a clean false alarm even though the simple report scan marks it as
`negated`. The reason is that the report itself is internally inconsistent:
the findings text mentions both a right-sided pleural effusion history and a
suggested substantial left-sided effusion, while the impression ends with
`No effusion`.

This makes it a useful ambiguity case rather than a straightforward model
mistake. It also mirrors the earlier cardiomegaly findings-versus-impression
conflict and supports the broader point that some suspicious cases come from
report-level inconsistency rather than obvious parser failure or obvious model
failure.

## Case E: Likely false alarm for Pleural Effusion

- `entry suspicious rank`: 8
- `sample suspicious rank`: 35305
- `subject_id`: 17582273
- `study_id`: 58025392
- `dicom_id`: `0c638ace-057b45a2-a82a9487-53533dc7-05a58507`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17582273/s58025392/0c638ace-057b45a2-a82a9487-53533dc7-05a58507.jpg`
- `flagged label`: `Pleural Effusion`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.997009`
- `entry_quality_self_confidence`: `0.002991`
- `sample_quality_self_confidence`: `0.654740`
- `report_relation`: `negated`
- `report_match_phrase`: `no pleural effusions`

### Full report text

```text
FINAL REPORT
EXAMINATION: CHEST (PORTABLE AP)

INDICATION: ___ year old woman with subglottic stenosis s/p dilation,
intubated for acute hypoxia // interval change interval change

IMPRESSION:

Comparison to ___. The lung volumes have decreased. As a
consequence, the bilateral basal parenchymal opacities appear denser than on
the previous image. The monitoring and support devices are constant.
Moderate cardiomegaly. No pleural effusions.
```

### Interpretation

This is a much cleaner likely false-alarm case. The report explicitly states
`No pleural effusions`, and unlike Case D there is no offsetting findings text
that clearly reintroduces an effusion elsewhere in the report. The structured
label and the report are aligned, but the model still predicts pleural
effusion with extremely high probability.

That makes this case useful as a counterweight: not all high-suspicion entries
are likely label issues. Some really do look like model overcalls.

## Case F: Borderline cardiomegaly overcall

- `entry suspicious rank`: 11
- `sample suspicious rank`: 8
- `subject_id`: 12907170
- `study_id`: 57457256
- `dicom_id`: `47c121ca-9987725b-3bdf20f7-5c9aee85-b5323b8a`
- `image_path`: `/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p12/p12907170/s57457256/47c121ca-9987725b-3bdf20f7-5c9aee85-b5323b8a.jpg`
- `flagged label`: `Cardiomegaly`
- `CheXpert-derived label`: `0`
- `pred_prob`: `0.992759`
- `entry_quality_self_confidence`: `0.007241`
- `sample_quality_self_confidence`: `0.007241`
- `report_relation`: `unclear`

### Full report text

```text
FINAL REPORT
CHEST RADIOGRAPH PERFORMED ON ___

COMPARISON: Prior exam from earlier today.

CLINICAL HISTORY: Central venous catheter, assess position.

FINDINGS: Portable AP upright chest radiograph was provided. There is a left
IJ central venous catheter in place with its tip in the region of the mid SVC.
The heart size appears top normal. The lung volumes are low with
bronchovascular crowding and presumed atelectasis in the lower lungs. There
is no definite sign of pneumonia, large effusion, or pneumothorax. The
mediastinal contour appears stable.

IMPRESSION: Appropriately positioned left IJ central venous catheter. Top
normal heart size.
```

### Interpretation

This looks more like a borderline model overcall than a likely true label issue.
The report does talk about the heart, but the wording is specifically `top
normal heart size`, which is weaker than cardiomegaly and closer to a negative
or borderline-negative interpretation. The model probability is nevertheless
very high, and this is also a top sample-level suspicious case.

This case is useful because it is not a full negation like `no cardiomegaly`,
but it still does not really support a positive cardiomegaly label. So it
helps carve out an intermediate bucket between clean negation and clean support:
borderline wording that may still drive strong model disagreement.

## Take-home pattern

Across these supplemental examples, the suspicious cases again split into
multiple types rather than one:

- likely true issues in the structured label
- likely model false alarms
- stable or chronic findings that depend on task definition
- report-level inconsistencies between findings and impression
- borderline report wording that is neither a clean positive nor a clean negative

That is exactly why these cases are worth discussing directly in the meeting.
They are not just examples of high conflict; they are examples of different
reasons for high conflict.
