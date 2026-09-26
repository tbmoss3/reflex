# The Labeling Discipline (read before contributing datasets)

Reflex trains on **hindsight outcomes**, not on anyone's guess about what a model "should" say.
This document defines what counts as an honest label. PRs that violate it get rejected.

## The three honest label sources

1. **Outcome labels (strongest):** what actually happened after a decision — did the routed
   handler complete the task, did the user re-ask or correct, did the escalated case get
   resolved. Observable, timestamped, joins to the ledger.

2. **Counterfactual labels (CSP):** for "did this request need the full agent?" there is NO
   honest label in incumbent logs, because the incumbent handles everything. The only source
   is the Controlled-Skip Protocol: actually skip, observe, grade. See
   `docs/controlled_skip_protocol.md`.

3. **Behavioral labels (derived, weaker):** computable from traces (turns answered with ≤1
   tool call = cheap; 4 identical consecutive tool calls = stuck; message referenced later =
   keep). Honest about what happened, but measures *incumbent behavior*, not what a better
   system would do. Acceptable for pre-training a head; never for final evaluation.

## The forbidden labels (we rejected these ourselves — learn from it)

- **Deterministic labels** (which channel a message arrived in, whether a cron "completed"):
  a regex already answers it; training a model on it wastes an epoch. We shipped two datasets
  built this way and deleted both.
- **Incumbent-behavior labels** as ground truth: teaching the gate to replicate the current
  system's routing teaches it the current system's costs. Fine for bootstrapping, dishonest
  as "truth."
- **Invented thresholds**: any number in a spec that didn't come from a ledger measurement is
  a placeholder and must be marked as one.

## Frozen evals

Every task ships with a held-out eval set that is NEVER trained on, versioned alongside the
adapter. Eval sets train-test overlap = the bug we caught in our own v0.4 (a "100%/ECE 0.000"
result that turned out to be 83/152 eval rows inside the training file). If a number looks
too good, it is.

## No-regression deploy gate

A candidate adapter deploys only if it beats the incumbent on EVERY task metric, not on
average. The nightly trainer enforces this mechanically; no human override without a written
justification in the deploy log.
