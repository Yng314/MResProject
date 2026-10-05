# Locked Follow-up: CL Candidate Prioritization Policies

## Status

This protocol is locked before computing any outcome for the three alternative
policies, but after the current whole-study-mean policy was evaluated. It is a
post-result follow-up validation, not a prospective preregistration. The frozen
CL scores and both expert-reference files already exist.

## Research question

Given the same CL-hard candidate pool, can an entry-aligned prioritization rule
identify more expert-confirmed current-label issues than the existing policy,
without increasing the relevant review budget?

This experiment changes only prioritization. It performs no model training, no
new CL estimation, no LLM review, no label correction, and no iterative loops.

## Frozen inputs and references

- Frozen outcome-blind CL scores: four baseline estimators (`13`, `42`, `97`,
  `123`) and their probability ensemble over 3,050 official-test AP/PA studies.
- Primary reference: MIMIC-CXR-JPG 2.1 radiologist panel (expected 1,796 valid
  entries and 109 current-label issues).
- Secondary reference: Med-PaLM hard-case panel (expected 498 entries and 127
  current-label issues).
- The ensemble is primary. Individual seeds are robustness analyses.

## Candidate pool

For each estimator, the entry pool contains every entry with
`cl_hard_issue = 1`. The suspicious-study pool contains every study with at
least one such entry. Neither pool uses an expert reference.

## Locked policies

The current policy is retained as the comparator. Lower quality means higher
priority for every policy.

1. `current_whole_study_mean`: rank suspicious studies by the mean
   self-confidence across all valid entries in the study; expand all CL-hard
   entries from the selected studies.
2. `worst_entry`: rank suspicious studies by the minimum self-confidence among
   all valid entries in the study; expand all CL-hard entries from the selected
   studies.
3. `hard_entry_mean`: rank suspicious studies by the mean self-confidence among
   CL-hard entries only; expand all CL-hard entries from selected studies.
4. `entry_direct`: rank CL-hard entries directly by entry self-confidence and
   select entries without a second study aggregation.

Deterministic tie breaking is fixed before reference access. Study policies use
descending CL-hard-entry count and then ascending `study_id`. Entry ranking uses
descending suspiciousness percentile, ascending label name, and ascending
`entry_key` after entry self-confidence.

## Budget matching

- All study policies select exactly the same number of studies as
  `round(20% * suspicious studies)`, reproducing the current top-20% study
  budget. Their expanded entry counts may differ and must be reported.
- `entry_direct` selects exactly the number of entries expanded by
  `current_whole_study_mean` for that estimator. This matches the current LLM
  entry-review workload rather than the number of studies.
- Cross-policy claims must report both selected studies and selected entries.
  A larger issue count obtained with a larger entry workload is not sufficient
  evidence of a better policy.

## Outcome isolation

The program has separate `select` and `evaluate` stages.

1. `select` may read only the frozen score file and its outcome-blind manifest.
2. It writes complete study and entry rankings, selected entries, counts, hashes,
   policy definitions, and a completion marker.
3. `evaluate` verifies all blind-stage hashes before reading either reference.
4. No policy, tie breaker, budget, endpoint, or cohort may change after reference
   access.

Blind artifacts must contain no reader label, reference label, true-issue flag,
expected action, or reader agreement.

## Endpoints

For each reference, estimator, and policy:

- full-pool selected studies and selected entries;
- expert-reference entries covered;
- expert-confirmed issues captured;
- precision among covered entries, recall, reference coverage, and enrichment;
- 2,000 subject-cluster bootstrap intervals;
- paired subject-cluster bootstrap differences versus the current policy;
- 10,000 matched-random replicates.

Study-policy random comparators choose the same number of studies from the same
CL-suspicious study pool and expand their hard entries. The entry-policy random
comparator chooses the same number of entries from the same CL-hard entry pool.
Random reference coverage, issue capture, precision, and one-sided empirical
tail probabilities are all reported.

Per-finding results are descriptive because outcome support is sparse.

## Primary decision rule

An alternative is called `promising` only if, on the primary MIMIC-CXR 2.1
reference and ensemble estimator, it:

1. captures more known issues than the current policy;
2. has precision no lower than the current policy;
3. captures more issues than its matched-random mean with empirical
   `p_capture < 0.05`; and
4. shows issue-capture improvement versus current in at least three of four
   individual seeds.

This rule is a transparent engineering decision rule, not a proof of population
superiority. Effect sizes, intervals, workload, and reference coverage remain
primary evidence. No policy is selected from Med-PaLM alone.

## Interpretation boundaries

- The primary reference is sparse and single-reader; the secondary reference is
  externally selected for difficult disagreements.
- Precision and recall apply only to each labelled reference panel, not the full
  MIMIC-CXR training population.
- The experiment can validate prioritization of frozen CL evidence. It cannot
  show that iterative cleaning converges or improves downstream AUROC.
- Because the alternatives were motivated by the current policy result, any
  apparent winner requires later validation on a new dense reference or a new
  independently annotated sample.

## Reproducibility

- Freeze program, protocol, frozen scores, and imported helper program by
  SHA-256 before formal execution.
- Formal outputs become immutable after `.evaluation_complete`.
- Verify expected estimator/reference counts, uniqueness, policy budgets,
  hard-entry membership, replicate counts, and all output hashes.
