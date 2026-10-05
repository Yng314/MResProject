# VinDr full-issue-pool iteration multi-seed extension

## Material passport

- Experiment ID: `vindr_full_issue_pool_iteration_multiseed_v1`
- Analysis role: fixed seven-seed extension of the completed seed-11003 pilot
- Condition: Scratch DenseNet121, `30%` hard-instance noise
- Seeds: `11003, 13007, 17011, 19001, 23003, 27011, 31013, 37003`
- Newly run seeds: `13007, 17011, 19001, 23003, 27011, 31013, 37003`
- Correction: selected-only oracle correction

## Question

After the complete initial confident-learning issue pool has been reviewed, does retraining and rerunning confident learning consistently expose additional true errors across the previously locked stress-experiment seeds?

## Locked design

- Reuse each seed's locked 2,400-image action set, 600-image training-excluded sentinel set, hard-instance corruption, folds, and Loop-0 evidence from the completed stress experiment.
- Seed `11003` is reused from the completed pilot without rerunning or altering its output.
- Run all seven remaining locked seeds exactly once and include every result regardless of direction.
- Loop 1 reviews and oracle-corrects every entry in the seed-specific initial Loop-0 issue pool. This is the one-shot Frozen endpoint.
- Loops 2-5 retrain the model, recompute OOF/CL evidence, and review every current issue entry that has never previously been reviewed.
- Previously reviewed entries cannot re-enter later review pools. No fixed per-loop review budget is imposed.
- At most three array tasks may use GPUs concurrently.

## Outcomes

- Primary mechanism outcomes: additional true errors corrected and additional known-quality gain in Loops 2-5 relative to the Loop-1 one-shot endpoint.
- Review-efficiency outcomes: additional reviews, per-loop issue-pool precision, and cumulative reviews.
- Supporting trajectories: raw entry DQS, action-set clean-reference OOF macro AUROC, and fully training-excluded sentinel clean-reference macro AUROC.
- Report all eight seed-level outcomes, the mean trajectory, positive/negative seed counts, and exact paired sign-flip tests where defined.

## Interpretation

- Consistent additional true-error discovery after Loop 1 supports the mechanism that retraining and reranking expose errors absent from the initial issue pool.
- This is not a same-budget comparison: the iterative extension spends additional reviews after the one-shot endpoint.
- Because seed `11003` was observed before this extension was launched, the combined eight-seed result is reported as a fixed replication extension rather than a fully outcome-blind confirmatory experiment.
