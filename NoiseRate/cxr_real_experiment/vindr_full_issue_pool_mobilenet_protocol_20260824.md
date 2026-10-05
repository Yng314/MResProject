# VinDr MobileNet full-issue-pool iteration

## Material Passport

- Experiment ID: `vindr_full_issue_pool_mobilenet_v1`
- Status: protocol locked before MobileNet outcomes
- Origin: matched architecture backfill of `vindr_full_issue_pool_iteration_v1`
- Execution date: 2026-08-24

## Question

After every entry in the initial MobileNet confident-learning issue pool has
been reviewed, can retraining the same MobileNet and rerunning confident
learning expose additional true errors and improve known label quality and
held-out performance?

## Locked design

- Reuse the eight frozen `hard_r30` VinDr preparations from the completed
  self-optimization stress experiment: seeds `11003`, `13007`, `17011`,
  `19001`, `23003`, `27011`, `31013`, and `37003`.
- Preserve every seed's 2,400-image action set, 600-image training-excluded
  sentinel set, injected errors, outer folds, and fixed inner-split reference.
- Replace only the evidence model with a scratch, one-channel, six-output
  MobileNetV3-small.
- Generate fresh MobileNet Loop-0 action OOF and sentinel evidence for every
  seed. No DenseNet evidence or result is reused.
- Use four outer folds, maximum 50 epochs, early-stopping patience 8,
  unweighted BCE-with-logits, Adam learning rate `1e-3`, and batch size 32.

## Refinement procedure

- Loop 1 reviews and oracle-corrects every entry in the initial MobileNet issue
  pool. This is the one-shot full-pool endpoint.
- Loops 2-5 retrain MobileNet from scratch, recompute OOF and sentinel evidence,
  and review every current issue entry that has never previously been reviewed.
- Reviewed entries cannot re-enter a later pool. No fixed per-loop review
  budget is imposed.
- The locked endpoint is Loop 5. A run with no new unreviewed issue before that
  endpoint fails closed and is inspected before any further execution.

## Outcomes

- Newly reviewed issues and true errors corrected per loop.
- Cumulative reviews, known action-set quality, remaining errors, and raw DQS.
- Clean-reference action OOF macro AUROC.
- Clean-reference training-excluded sentinel macro AUROC.
- Sentinel error-detection AUPRC and AUROC.
- Paired Loop-1-to-final changes across all eight seeds, with direction counts
  and exact sign-flip tests.

## Execution

- Seed `11003` is the locked first run and is admitted to the final eight-seed
  analysis when all postflight checks pass.
- The remaining seven seeds run as a Slurm array with maximum concurrency one,
  so the experiment occupies at most one GPU.
- The aggregate runs only after every admitted seed completes.
