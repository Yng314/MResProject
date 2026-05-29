# Weighted Metric Summary

## Main message

We added a weighted metric to address the class-imbalance concern.

The main conclusion does not change after adding the weighted metric:

- the strongest model is still the **XRV linear-head** setting
- the best overall result is still obtained after applying **confident learning with sample-level top-k cleaning**
- therefore, the improvement is **not just an artifact of macro averaging**

## What the weighted metric shows

Previously, we mainly reported **macro AUROC**, which gives equal weight to each pathology.
Because the dataset is imbalanced, a natural concern is whether the observed gain is driven mainly by a few small classes.

To address this, we computed a **weighted AUROC**, where each label is weighted by its number of valid test studies.

This gives more influence to labels with larger support and reduces the effect of very small classes.

## Key result

Even under the weighted metric, the ranking is still broadly consistent with the macro result.

In particular:

- **XRV linear-head baseline** already performs strongly
- after applying **confident learning**, the **sample-level top-k cleaned model** still performs better than the baseline
- the best weighted result is still achieved by **XRV linear-head + sample top20%**

This means:

**Confident learning still improves performance even when class imbalance is explicitly taken into account.**

## Interpretation

This is important because it makes the result more convincing.

If the gain disappeared after switching from macro to weighted evaluation, then we might worry that the improvement was mainly caused by a few unstable small classes.

But that is not what we observe.

Instead, we see that:

- the weighted metric still favors the cleaned model over the baseline
- the best-performing family is still XRV
- the conclusion that confident learning is useful remains valid

## Short version for presentation

After adding a weighted metric to account for label imbalance, the overall conclusion remains the same.
The strongest family is still the XRV linear-head model, and the confident-learning cleaned version still performs better than the baseline.
So the improvement is not only visible under macro AUROC, but also remains when class imbalance is taken into account.

---

# Sample-Level vs Entry-Level Cleaning

## Main observation

We observed a clear difference between **sample-level top-k cleaning** and **entry-level top-k cleaning**.

Broadly:

- when **sample-level top-k** increases, performance often improves
- when **entry-level top-k** increases, performance often decreases

So the two strategies show an opposite trend.

## What the two strategies do

The difference comes from what is actually being removed.

### Sample-level cleaning

If a sample is identified as unreliable, the **whole sample is removed** from the training set.

So this strategy removes:

- the image
- all labels attached to that image

This is a relatively coarse but strong form of cleaning.

### Entry-level cleaning

If one label entry is identified as unreliable, the **sample is kept**, but that specific label is masked during training.

So this strategy removes:

- only the supervision for that label
- but keeps the image and the other labels

This is a more fine-grained cleaning strategy.

## Why sample-level cleaning can help

Our analysis suggests that sample-level top-k often removes **globally suspicious samples**.

In practice, this means the removed samples are often those where:

- the model prediction and the label are in strong conflict
- the sample may be globally inconsistent
- the negative label side often looks especially unreliable

For several representative labels, sample-level top-k removes very few positive samples, but removes a larger fraction of suspicious negative samples.

For example in the XRV linear-head experiments:

- **Lung Lesion**
  - top10 sample removes about **0.14%** of positive samples
  - but removes about **6.22%** of negative samples
- **Atelectasis**
  - top10 sample removes about **0.00%** of positive samples
  - but removes about **13.47%** of negative samples
- **Lung Opacity**
  - top10 sample removes about **0.02%** of positive samples
  - but removes about **10.51%** of negative samples

This suggests that sample-level cleaning is not mainly deleting rare positive evidence.
Instead, it is more often trimming suspicious negative samples, which can improve ranking performance.

## Why entry-level cleaning can hurt

Entry-level cleaning behaves differently because it does not remove the full sample.
Instead, it removes the supervision signal for a specific label.

When top-k becomes larger, this can lead to **over-masking useful supervision**.

The most important effect we found is that, for several labels, entry-level top-k removes a substantial fraction of **negative supervision**.

This is especially problematic for labels where the test set contains very few negative studies.

## Concrete example: Lung Lesion

`Lung Lesion` is a very clear example.

In the test set:

- valid studies = **53**
- positive studies = **51**
- negative studies = **2**

So its AUROC is extremely sensitive to the ranking of only two negative examples.

In the XRV linear-head baseline:

- entry-level top-k removes **no positive Lung Lesion entries**
- but it removes an increasing fraction of **negative Lung Lesion entries**

Specifically:

- top05 entry removes about **8.78%** of negative supervision
- top10 entry removes about **16.46%** of negative supervision
- top20 entry removes about **27.56%** of negative supervision

This makes it harder for the model to learn when `Lung Lesion` should be negative.

As a result, the AUROC drops sharply:

- baseline = **0.765**
- top05 entry = **0.775**
- top10 entry = **0.598**
- top20 entry = **0.618**

So the drop is not because we removed many positive rare samples.
It is more because we weakened the negative supervision for a label that already has extremely few negatives in the test set.

## Similar labels

This pattern is not unique to `Lung Lesion`.

Other labels that show a similar issue include:

- **Pleural Other**
- **Atelectasis**
- **Fracture**
- **Lung Opacity**

These labels either:

- have very small negative counts in the test set
- or become sensitive when too much negative supervision is masked

So entry-level top-k can disproportionately hurt their AUROC.

## Interpretation

The overall interpretation is:

- **sample-level cleaning** tends to remove globally unreliable samples and can improve several labels at once
- **entry-level cleaning** can become too aggressive and may remove useful supervision, especially negative supervision for hard or rare labels

That is why increasing top-k can help for sample-level cleaning but hurt for entry-level cleaning.

## Short version for presentation

Sample-level and entry-level cleaning behave differently because they remove different things.
Sample-level cleaning removes the entire suspicious sample, and in practice it often trims globally unreliable negatives without greatly reducing positive evidence.
Entry-level cleaning only masks one label at a time, but when the cleaning strength increases it can over-remove useful supervision, especially negative supervision for labels with very small negative test counts, such as Lung Lesion.
This explains why sample-level top-k tends to improve performance while entry-level top-k can degrade it.
