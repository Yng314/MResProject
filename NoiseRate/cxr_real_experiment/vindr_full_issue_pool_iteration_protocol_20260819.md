# VinDr full-issue-pool iteration pilot

## Material Passport

- Experiment ID: `vindr_full_issue_pool_iteration_v1`
- Status: single-seed descriptive pilot locked before outcomes
- Seed: `11003`
- Condition: Scratch DenseNet121, `30%` hard-instance noise
- Correction: selected-only oracle correction

## Question

After every entry in the initial confident-learning issue pool has been reviewed, can retraining and rerunning confident learning expose additional true errors that were missed by the first issue pool?

## Design

- Reuse the locked 2,400-image action set, 600-image training-excluded sentinel set, corruption, folds, and Loop-0 evidence from the completed stress experiment.
- Loop 1 reviews and oracle-corrects every entry in the initial Loop-0 issue pool. This is the one-shot Frozen endpoint.
- Loops 2-5 retrain the model, recompute OOF/CL evidence, and review every current issue entry that has never previously been reviewed.
- Previously reviewed entries cannot re-enter later review pools.
- No fixed per-loop review budget is imposed; each loop processes its complete new issue pool.

## Interpretation

- Loop 1 versus Loop 0 measures the effect of one complete targeted cleaning pass.
- Loops 2-5 versus Loop 1 measure the additional yield and cost of iterative cleaning.
- This pilot does not make a same-budget dynamic-versus-frozen claim and does not support population-level inference from one seed.

## Outcomes

- New issues reviewed and true errors corrected per loop.
- New issue-pool precision and cumulative review count.
- Known true dataset quality and remaining error count.
- Raw entry DQS.
- Action-set clean-reference OOF macro AUROC.
- Training-excluded sentinel clean-reference macro AUROC and error-detection AUPRC/AUROC.

## Success criterion

Iteration provides descriptive additional value if Loops 2-5 discover a material number of true errors not included in the initial issue pool, with a review yield high enough to justify the additional reviews. AUROC and DQS are reported as supporting trajectories rather than required monotonic endpoints.
