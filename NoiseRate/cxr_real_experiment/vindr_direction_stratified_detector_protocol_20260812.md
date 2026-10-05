# VinDr direction-stratified detector exploratory protocol

## Status

This is a post-hoc exploratory experiment motivated by the completed detector-alternatives benchmark. It reuses locked blind evidence and must not be described as confirmatory evidence.

## Question

Can a two-queue detector preserve OOF/CL detection of observed-positive errors while using frozen-feature neighbourhood evidence to recover observed-negative errors at the same total review budget?

## Frozen inputs

- The verified MobileNet direction-sensitivity benchmark with 3,000 VinDr images, six labels, six seeds and 60 clean/balanced/fp-only/fn-only scenario runs.
- The existing outcome-blind MobileNet OOF evidence for every run.
- The existing 3,000 x 1,024 frozen XRV feature matrix.
- No model retraining and no LLM calls.

## Methods

1. Global OOF/CL-compatible ranking.
2. Global Active Label Cleaning ranking.
3. Global SimiFeat-style k=50 ranking.
4. Direction-stratified ranking: observed-positive entries use OOF/CL; observed-negative entries use SimiFeat-style k=50.
5. Fixed observed-negative budget quotas from 0% to 100% define a diagnostic trade-off curve only.
6. Two outcome-blind automatic quotas allocate budget according to summed OOF alternative-label probability or summed SimiFeat neighbourhood disagreement probability.

## Outcome isolation

All scores and automatic quotas are written before private references are opened. Injected-error outcomes are used only by the evaluation stage. The fixed-quota curve and any best quota selected from it are explicitly post-hoc diagnostics.

## Metrics

- Total error recall and precision at a total review budget equal to the injected-error count.
- `0 -> 1` and `1 -> 0` recall.
- Macro and minimum directional recall for balanced scenarios.
- Paired six-seed descriptive contrasts against global OOF, averaged across rates.

## Interpretation boundary

This experiment tests complementarity and allocation trade-offs. It cannot validate a quota chosen after viewing these outcomes. A future confirmatory experiment must freeze one outcome-blind allocation rule and evaluate it on new corruption draws or held-out runs.
