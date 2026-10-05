# Med-PaLM External-Suspicion Binary Reviewer Benchmark

## Question

Given a specific suspicious report-finding entry supplied by an external method,
can the existing LLM refinement stage improve agreement with an adjudicated
expert binary reference without discarding excessive supervision?

## Locked scope

- Selection source: the 1,378 Med-PaLM 2 versus CheXpert disagreement entries
  released with expert adjudication.
- Current-method compatibility: retain only entries in the existing XRV12 label
  schema whose raw CheXpert value is `-1`, `0`, or `1`.
- Expected analysis set: 498 entries from 457 studies and 179 subjects.
- Exclusions: 142 Support Devices entries and 738 current-schema entries whose
  raw CheXpert label is missing.
- The benchmark is conditional on external hard-case selection. It does not
  estimate population noise or validate confident-learning detection.

## Blinded reviewer input

Each API request represents one specific suspicious entry and contains:

- target finding;
- current U-Ones binary label;
- raw CheXpert label as provenance;
- the complete original MIMIC-CXR report.

The reviewer does not receive the Med-PaLM output, expert reference,
reader-agreement field, or a model/OOF probability. The prompt and action schema
otherwise preserve the current entry-level refinement logic.

## Binary reference and actions

- Expert `1` and `-1` map to binary `1`.
- Expert `0` maps to binary `0`.
- Expert missing is invalid under this protocol.
- `keep` retains the current binary label.
- `relabel` flips `0 <-> 1`.
- `mask` abstains and removes that entry from binary supervision.
- Conflicting response fields are masked, matching the existing action gate.

Before any benchmark outcome is observed, the expected binary reference contains
408 positives and 90 negatives. The current-label baseline has 371 correct
entries and 127 incorrect entries.

## Locked endpoints

Primary:

1. retained binary accuracy;
2. retained-entry coverage;
3. coverage-adjusted correct mass, defined as correct non-masked entries divided
   by all 498 entries;
4. change in coverage-adjusted correct mass from the unchanged-label baseline.

Supporting:

- binary action accuracy;
- relabel precision and recall;
- harmful flip rate among initially correct entries;
- mask rate overall and within expected keep/relabel groups;
- keep/relabel/mask confusion table;
- per-finding descriptive summaries;
- sensitivity analysis restricted to three-reader unanimous entries.

Uncertainty uses a fixed-seed percentile bootstrap that resamples subjects and
retains all entries belonging to each sampled subject.

## Interpretation boundary

A positive result supports the LLM correction stage on externally selected hard
cases. It does not establish that confident learning would find these entries,
that the result generalizes to all MIMIC-CXR entries, or that Med-PaLM and the
current detector select equivalent error distributions.
