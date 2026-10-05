# Seed42 Unseen-Review Extension Protocol

## Material Passport

- Workflow: Academic Research Suite / experiment-agent
- Mode: run with pre-execution protocol lock
- Status: LOCKED_BEFORE_EXECUTION
- Seed: `42`, chosen as the smallest current analysis seed other than the already
  extended seed `13`; selection is independent of Loop9-15 outcomes
- Source state: corrected binary-label own-top20 refinement, completed Loop8
- Planned range: Loop9 through Loop15
- Output filesystem: temporary `/vol/gpudata` root because `/vol/bitbucket` is full

## Research Question

Does replacing repeated LLM reviews with lower-ranked but previously unseen
suspicious entries increase later-loop action yield and improve the post-Loop8
trajectory for one exploratory seed?

## Fixed Intervention

1. Preserve seed42's exact cumulative relabel/mask state after Loop8.
2. Recompute OOF evidence after each new cleaning loop exactly as in the original
   corrected-binary v5 protocol.
3. Rank suspicious **samples** with the original ordering and keep the original
   budget `round(number of issue samples x 0.20)`.
4. Define an entry by stable `pool_row_id::label_index`.
5. Before each Loop9-15 selection, build history from every entry exported for
   LLM review in all earlier loops, including entries ending in API errors.
6. Within each candidate sample, remove every historically reviewed entry.
7. Skip candidates with no unseen suspicious entry and scan farther down the
   unchanged sample ranking until the full sample budget is restored.
8. Abort rather than silently shrink the budget if too few unseen candidates
   remain.
9. Assert zero overlap between the current expanded review entries and the full
   prior-loop history before any LLM request.

The exclusion unit is a disease-label **entry**, not an entire image. A sample
seen previously for one label may return only if a different currently
suspicious label entry has never been reviewed.

## Locked Components

- U-Ones projection: raw `-1 -> binary 1`
- Model: MobileNetV3 small from scratch
- OOF: four folds, seed42, same training settings
- Full train/eval: 50 epochs, no early stopping recovery
- LLM: `gpt-5.4`, temperature `0`
- Cleaning action builder and cumulative state merge
- Held-out test set and Loop15 maximum

## Required Audit Outputs

Each loop must save:

- number of prior unique reviewed entries;
- original issue-sample count and 20% target;
- number of ranked samples scanned to fill the budget;
- filtered historical entries and selected unseen entries;
- SHA-256 of the ordered selected entry keys;
- a hard assertion that history overlap is zero;
- normal LLM action, DQS, and held-out model outputs.

## Interpretation Boundary

This is an exploratory single-seed intervention. Comparing its outcome with the
standard seed13 extension confounds method and seed, so it cannot by itself
establish that unseen-only selection is superior. The defensible first readout
is the within-seed42 post-Loop8 trajectory plus action-yield and novelty
diagnostics. A causal method comparison would require standard and unseen-only
continuations from the same Loop8 state across paired seeds.
