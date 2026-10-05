# MIMIC-CXR full issue-pool refinement pilot

## Question

Can iterative confident-learning selection clean MIMIC-CXR efficiently when every
currently flagged entry is reviewed, while avoiding repeated LLM review work?

## Locked pilot

- Dataset and binary projection: the same MIMIC-CXR AP/PA, 12-label U-Ones setup
  used by the existing own-top-20 experiment.
- Training seed: 13.
- Evidence model: MobileNetV3-small trained from scratch with four-fold OOF
  predictions, 100 epochs per fold, and batch size 32.
- Downstream evaluation: an independently trained MobileNetV3-small with a
  maximum of 100 epochs, batch size 32, early-stopping patience 10, and recovery
  of the checkpoint with the lowest validation loss. This common downstream
  protocol is applied to the no-clean baseline and every available refined state.
- Reviewer: GPT-5.4, temperature 0, using the existing entry-level prompt and
  binary relabel/mask/keep action mapping.
- Duration: eight cleaning loops.

## Selection and history

At each loop, every entry in the current confident-learning issue pool is selected.
An entry is excluded only if it has a successful LLM result in an earlier loop of
this pilot. API failures are not counted as reviewed and remain eligible for retry.
Reviews from older, separate experiments are not imported into this pilot.

Loop 1 reuses only the locked seed-13 baseline OOF evidence from the prior
MobileNetV3 experiment. It does not reuse that experiment's top-20 selection or LLM
decisions. After each completed review, relabel and mask actions are accumulated,
the OOF models are retrained, and the issue pool is recomputed from the refined
dataset.

## Completion rule

The pipeline does not train the next model while the current review pool still has
API errors. A scheduler timeout or residual API error is handled by resubmitting the
same run root; `--resume` skips successful responses and retries only unfinished
entries.

## Outputs

Each loop records the current issue-pool size, newly reviewed entries, cumulative
relabel/mask actions, DQS inputs, and held-out AUROC. The primary descriptive
comparison is against the existing seed-13 top-20 trajectory under the same model
and evaluation settings.
