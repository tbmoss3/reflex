# Ship Checklist — Reflex OSS release (v0.1.x)

## ✅ Done
- [x] MIT license, README with honest eval numbers + baselines
- [x] Installable package (pyproject, pip install, python -m reflex entrypoints)
- [x] L1 scorer (pod-served, logprob confidence extraction, env-driven URLs)
- [x] L2 four-agent harness (Research/Judge/Report/Operator, alignment check)
- [x] Operator execute path (reversible-only, grants, snapshots, revert, audit trail)
- [x] Ledger + hindsight labeling + frozen-eval discipline (docs/LABELING.md)
- [x] Controlled-Skip Protocol (docs/controlled_skip_protocol.md)
- [x] Self-debugging ops tick (per-step error capture, tunnel/server self-heal, exit codes)
- [x] Adapters: Hermes (full, gate()), Claude Code (hooks), Codex (thin), LangChain (router/judge)
- [x] Demo GIF generator (scripts/make_demo.py — real calls, reproducible)
- [x] Pod spend tracker (runtime/pod_spend.py — daily $ line)
- [x] Adapter-sync invariant (adapter_sync.sh — latest adapter always local)
- [x] Code review pass (compileall clean, syntax clean, env-var audit done)

## ⏳ Before public launch (Show HN / X)
- [ ] HuggingFace weights release: reflex-v05mt adapter + model card with eval tables
      (needs a live pod export or use the local backup directly — backup is canonical)
- [ ] One clean end-to-end run on the fresh pod (L1 score + L2 harness + skip demo) to regenerate
      the demo GIF with current models — GIF shows v06 which was lost; regenerate with v05mt
- [ ] Launch post drafts (Show HN text + X thread) — Benton review before posting
- [ ] GitHub topic tags + repo description (done) + demo GIF in README
- [ ] Migrate repo to teloplex org when created (gh repo transfer)

## 🔲 Post-launch (v0.2)
- [ ] Claude Code adapter: live test in a real Claude Code session
- [ ] HF Spaces demo (one-click try)
- [ ] More harness adapters (PydanticAI, smolagents) on request/PR
- [ ] CSP Phase 1 expansion (20+ skips) -> first counterfactual noul retrain
- [ ] Managed-training waitlist page (Tier 1 funnel)