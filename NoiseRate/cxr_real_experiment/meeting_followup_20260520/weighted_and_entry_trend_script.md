# Weighted Metric and Entry-Level Trend

## Weighted metric

The first thing I wanted to check was whether the improvement we saw with confident learning was only a macro-average effect. Macro AUROC gives every label the same weight, so it does not account for class imbalance. To address that, I also computed a weighted AUROC, where each label is weighted by its number of valid test studies.

The main conclusion did not change. The strongest model is still the XRV linear-head model, and the confident-learning settings, especially the sample-level top-k settings, still outperform the corresponding baselines under both macro and weighted AUROC. So the improvement is not just coming from a few small labels dominating the macro average.

| Setting | Study Macro AUROC | Study Weighted AUROC |
|---|---:|---:|
| DenseNet baseline | 0.768 | 0.827 |
| DenseNet top05 entry | 0.821 | 0.832 |
| DenseNet top10 entry | 0.820 | 0.840 |
| DenseNet top20 entry | 0.807 | 0.832 |
| DenseNet top10 sample | 0.827 | 0.843 |
| XRV linear-head baseline | 0.816 | 0.833 |
| XRV linear-head top05 entry | 0.824 | 0.837 |
| XRV linear-head top10 entry | 0.785 | 0.820 |
| XRV linear-head top20 entry | 0.788 | 0.820 |
| XRV linear-head top10 sample | 0.838 | 0.847 |
| XRV linear-head top20 sample | 0.839 | 0.851 |

So the take-home message for this part is that the weighted metric supports the same story as the macro metric. XRV is still the strongest model, and confident learning, especially sample-level top-k cleaning, is still better than baseline.

## Why does entry-level cleaning sometimes get worse?

Before discussing the trend, I think it is important to define what top-k means in this setting. Here, top-k does not mean the top-k fraction of all labels, and it does not mean the top-k fraction of the whole dataset. Instead, confident learning first identifies a subset of suspicious entries, and then top-k refers to the top-k fraction within those suspicious entries.

I first looked at what sample-level and entry-level cleaning were actually removing. In the table below, each cell shows the AUROC for that label, and in brackets I show the percentage of positive and negative supervision removed in training. These percentages are computed with respect to all valid positive or negative supervision for that label in the training set. The last column shows the number of positive and negative cases in the test set.

| Label | Baseline | Top05 Sample | Top10 Sample | Top20 Sample | Top05 Entry | Top10 Entry | Top20 Entry | Test set |
|---|---:|---|---|---|---|---|---|---|
| Atelectasis | 0.899 | 0.899 (0.0%, 7.6%) | 0.943 (0.0%, 13.5%) | 0.969 (0.4%, 25.9%) | 0.844 (0.0%, 38.0%) | 0.802 (0.0%, 45.5%) | 0.882 (0.0%, 45.5%) | 212 / 2 |
| Lung Lesion | 0.765 | 0.745 (0.0%, 4.6%) | 0.892 (0.1%, 6.2%) | 0.814 (1.0%, 13.4%) | 0.775 (0.0%, 8.8%) | 0.598 (0.0%, 16.5%) | 0.618 (0.0%, 27.6%) | 51 / 2 |
| Lung Opacity | 0.711 | 0.736 (0.0%, 4.5%) | 0.718 (0.0%, 10.5%) | 0.729 (0.4%, 21.7%) | 0.697 (0.0%, 17.1%) | 0.658 (0.0%, 32.4%) | 0.656 (0.0%, 39.9%) | 148 / 18 |
| Pleural Other | 1.000 | 0.864 (0.0%, 5.7%) | 1.000 (0.0%, 8.9%) | 1.000 (0.9%, 18.7%) | 0.955 (0.0%, 22.0%) | 0.818 (0.0%, 30.1%) | 0.932 (0.0%, 40.7%) | 22 / 1 |

This table already shows one important pattern. Entry-level cleaning often removes more negative supervision than sample-level cleaning. At the same time, for several of these representative labels, the number of negative cases in the test set is extremely small. Lung Lesion has only 2 negative test cases, Pleural Other has only 1, and Atelectasis also has only 2. In that situation, once training removes too much negative supervision, the ranking of only a handful of negative cases can have a very large effect on AUROC.

However, this is still not the full explanation, because it only tells us what entry-level cleaning removes. It does not tell us why global entry-level top-k can become so unstable. To understand that, I then looked at how the global top-k budget is allocated across labels.

The next table compares the effective cleaning strength under global top-k and per-label top-k. Here I focus on the negative supervision removal rate under the top10 setting, because that is where the difference is most informative. In other words, instead of asking what fraction of issue rows is selected, I ask what fraction of all negative supervision for that label is actually removed.

| Label | Global Top10 negative removal | Per-label Top10 negative removal | Test set Pos / Neg |
|---|---:|---:|---:|
| Atelectasis | 45.5% | 4.5% | 212 / 2 |
| Lung Lesion | 16.5% | 5.1% | 51 / 2 |
| Lung Opacity | 32.4% | 4.1% | 148 / 18 |
| Pleural Other | 30.1% | 4.9% | 22 / 1 |

This is important because the global top-k strategy does not apply the same effective cleaning strength to every label. Some labels are barely touched, while others are cleaned very aggressively. So the problem is not only that entry-level cleaning removes more negative supervision. The problem is also that the original global top-k strategy allocates the cleaning budget very unevenly across labels, which can over-clean some labels.

Once this happens for labels that already have very few negative test cases, their AUROC becomes especially unstable. That is why labels such as Lung Lesion, Pleural Other, and Atelectasis show such large drops under the global entry-level strategy.

To test whether this interpretation is meaningful, I also changed the cleaning strategy from global entry top-k to per-label entry top-k. The idea is simple: instead of allowing all labels to compete in a single global issue pool, each label is allowed to keep its own top-k budget. If the problem really comes from global over-allocation, then per-label top-k should reduce the performance drop.

That is exactly what we see below.

| Setting | Study Macro AUROC | Study Weighted AUROC |
|---|---:|---:|
| Baseline | 0.8158 | 0.8330 |
| Global Top05 Entry | 0.8243 | 0.8373 |
| Per-label Top05 Entry | 0.8160 | 0.8384 |
| Global Top10 Entry | 0.7848 | 0.8197 |
| Per-label Top10 Entry | 0.8174 | 0.8392 |
| Global Top20 Entry | 0.7879 | 0.8204 |
| Per-label Top20 Entry | 0.8049 | 0.8301 |

The most important comparison is the top10 setting. Under global top10 entry cleaning, the study macro AUROC drops to 0.7848, whereas under per-label top10 entry cleaning it recovers to 0.8174. The same recovery is also visible in weighted AUROC, which increases from 0.8197 to 0.8392.

So my current interpretation is the following. Entry-level cleaning can become worse not only because it removes more negative supervision, but also because the original global entry top-k strategy allocates the cleaning budget very unevenly across labels. This can over-clean specific labels, and when those labels also have very few negative cases in the test set, the AUROC becomes especially unstable. The recovery we see with per-label top-k supports this explanation.

## Initial sample-level review

For the third part, I did a small initial review of high-ranking suspicious samples. Here I do not want to rely too much on the chest X-ray images alone, because without specialist radiology expertise it is difficult to make a reliable judgment from the image itself. Instead, I mainly use the images as context and then compare the structured CheXpert-derived label, the model prediction, and the original report text.

I would show three representative examples here: one likely true issue, one likely false alarm, and one ambiguous case.

### Example images

Likely true issue:

![Case 1 image](/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p17/p17080143/s53342449/029d87ab-8eec033b-51211e87-fcc13973-c67dfa35.jpg)

Likely false alarm:

![Case 3 image](/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p18/p18402151/s55113475/da878459-5a7451c5-668eefe3-a241198e-c23fc5b2.jpg)

Ambiguous case:

![Case 4 image](/vol/gpudata/yz3522-llmtest/MedSoul/datasets/mimic-cxr-jpg-224/p15/p15561274/s53070008/2ebd1fb4-58af802a-a2b7e57f-5f573cb1-2a125f7c.jpg)

### Case summary table

| Case | Rank | Label | CheXpert-derived label | Pred prob | Report-based reading | Current interpretation |
|---|---:|---|---:|---:|---|---|
| Case 1 | sample rank 1 | Pleural Effusion | 0 | 0.9996 | report explicitly says bilateral effusions | likely true issue |
| Case 3 | sample rank 12 | Cardiomegaly | 0 | 0.9927 | report says heart size is unremarkable | likely false alarm |
| Case 4 | entry rank 1 | Cardiomegaly | 0 | 0.9977 | findings say enlarged heart, impression says no cardiomegaly | ambiguous |

### Case 1: likely true issue

This sample is the highest-ranked suspicious sample. The flagged label is `Pleural Effusion`, the CheXpert-derived label is `0`, and the model prediction is `0.9996`. The report text explicitly says:

> Bilateral effusions are again seen.
> IMPRESSION: No change, bilateral effusions.

So in this case, the report directly supports pleural effusion, while the structured label is `0`. This makes the sample a likely true label issue in the CheXpert-derived label.

### Case 3: likely false alarm

This sample has `sample suspicious rank = 12`. The flagged label is `Cardiomegaly`, the CheXpert-derived label is `0`, and the model prediction is `0.9927`. The report text says:

> Heart size and mediastinum are overall unremarkable.

So in this case, both the structured label and the report text do not support cardiomegaly, but the model still predicts it with extremely high confidence. This makes the sample a likely false alarm from the model.

### Case 4: ambiguous case

This sample has `entry suspicious rank = 1`. The flagged label is `Cardiomegaly`, the CheXpert-derived label is `0`, and the model prediction is `0.9977`. The report text says:

> The heart is moderately enlarged, though appears grossly stable from prior exam.
> IMPRESSION: No cardiomegaly, consolidation or mass seen.

So this report is internally inconsistent. The findings section supports cardiomegaly, while the impression section says there is no cardiomegaly. According to the MIMIC-CXR-JPG release documentation, the released CheXpert-derived labels are based on the impression section when it is available. So this sample is better treated as an ambiguous case, rather than a clean example of either a true issue or a false alarm.

At this stage, I would not make a strong general claim from this review. I would only say that, once we look at high-ranking suspicious samples together with the report text, we can already see at least three different types of cases: likely true issues, likely false alarms, and ambiguous report-level inconsistencies.
