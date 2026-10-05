# VinDr OOF AUM-style screen

## Purpose

This is a two-seed screening experiment, not the final detector comparison. It asks whether training dynamics add useful known-error ranking evidence beyond the final OOF probability before the method is expanded to all six seeds and corruption levels.

## Locked setting

- VinDr exact 20% global symmetric entry-noise cohort.
- Seeds `13` and `211`, representing one original and one new-replication seed.
- The same scratch MobileNetV3-small, four outer folds, nested inner early stopping, optimiser and image preprocessing as the completed benchmark.
- Maximum 100 epochs, patience 10, batch size 32 and learning rate 0.001.

## Score

At each epoch, the model trained without the outer fold predicts logits for that outer fold. For binary entry label `y` and logit `z`, the assigned-label margin is `(2y-1)z`. The mean margin over all epochs reached before early stopping is computed for every outer-fold entry. Lower mean margin indicates greater suspicion, so the ranking score is its negative.

This is an outcome-blind OOF adaptation of the Area Under the Margin training-dynamics idea. It should be described as `OOF AUM-style` rather than a literal reproduction of training-example AUM.

## Gate

The private injected-error reference is read only after blind probabilities and margins have been written and hashed. The screen compares OOF AUM-style ranking with OOF label incompatibility and Active Label Cleaning using AUPRC and error recall at a 3,600-entry review budget. Expansion to all six seeds is justified only if the AUM-style score adds a meaningful ranking improvement in this screen.
