# Controlled-Skip Protocol (CSP) v1
## The counterfactual label factory for the L1 needs-full-agent head

**Problem it solves:** the noul head can't learn "when would the cheap path suffice" from
logs, because the incumbent system (L3 full agent) handles everything. Only *actual skips
with observed outcomes* produce that label. CSP manufactures them safely.

## The protocol (strictly ordered)

### Phase 0 — Deterministic class only (first 20 skips)
- **Eligible requests:** single-intent, read-only lookups matched by high-confidence L1
  first_tool in {read_files, find_files, web_research, recall_history} — no writes anywhere.
- **Skip action:** answer the request with a DIRECT L2 handler call (one Qwen-4B completion
  with the request + the tool result it needs), NOT the full agent loop.
- **Blast radius:** zero — read-only, single turn, no state changes, no user-visible process.
- **Hindsight grade (24h later):** satisfied (no follow-up/correction) = TRUE NEGATIVE
  ("full agent was NOT needed" — the gold label); corrected/re-ask = false negative
  → the request goes back to L3 handling + logged as model-degrading example.

### Phase 1 — Gate expansion (after 20 skips, ≥90% satisfied)
- Expand eligibility to first_tool=run_code and memory_ops (still reversible).
- Skip rate stays capped at 10% of eligible traffic per day (measured, not guessed).

### Phase 2 — Authority (after 100 cumulative skips, ≥95% satisfied, zero corrected-skips on safety-adjacent classes)
- L1 skip verdict becomes BINDING for eligible classes; L3 never wakes for them.
- The never-gates list is unaffected and can never be skipped: money/pay/banking/payroll/
  posts/irreversible actions — L0 code, permanent.

### Safety invariants (all phases)
1. Skips are READ-ONLY through Phase 1; reversible-writes only from Phase 2 onward, and only
   via the Operator execute path (grants + snapshots).
2. Every skip is ledger-logged: request, L1 verdict+confidence, handler output, hindsight grade.
3. A single corrected-skip on a misrouted class pauses that class (not the protocol).
4. Escalation is always one message away: if the user's next message indicates the skip failed,
   the request is immediately re-queued to L3 with the failed-skip context attached.
5. The protocol never touches the safety spine.

### Why 20/100 thresholds: statistical honesty
20 skips at ≥90% gives a one-sided 95% CI lower bound of ~70% — enough to justify expansion,
not enough to claim authority. 100 at ≥95% gives LB ~89% — deployment-grade for reversible-only.

### What this produces
The ONLY dataset in existence of "requests a full agent did NOT need, proven by outcome" —
the gold standard L1 training set, system-relative by construction. Retraining the noul head
on these + the (larger) full-agent set fixes the yes-bias with honest counterfactuals.
