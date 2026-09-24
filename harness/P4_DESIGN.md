# P4 Four-Agent Harness — Design + Build Notes (v0.1, 2026-09-21)

## Status: IN BUILD

## Architecture (per approved Grep.ai-style taxonomy)

```
                 ┌──────────────────────────────────────┐
   request ──►   │ L1 Reflex gate (skill/needs-agent)   │── low conf ──► L3 frontier agent
                 └──────────┬───────────────────────────┘
                            ▼ (needs structured work)
                 ┌──────────────────────────────────────┐
                 │ HARNESS (this build)                 │
                 │  Research ─► Judge ─► Report         │
                 │     │           │                    │
                 │     └────►──────┴──► Operator        │
                 └──────────────────────────────────────┘
```

## Agent roles & runtime
| Role | Model | Tools | Grants |
|------|-------|-------|--------|
| Research | Qwen3-4B-Instruct (local, pod vllm) | web_search, web_extract, read_file, search_files | READ-ONLY |
| Judge | L3 API (GLM/Opus class) or local 4B for easy cases | read-only evidence access | decides; escalates hard cases to human/L3 |
| Report | Qwen3-4B-Instruct | render templates only | NEVER re-decides; renders facts to deliverable |
| Operator | Qwen3-4B-Instruct + strict schema | write_file, patch, terminal (whitelisted cmds only) | ONLY agent with writes; grants exclude pay/banking/payroll/posts — L0-enforced, trainer can NEVER modify grants |

## L1 gates at each handoff (once Reflex v0.5 backed up + serving locally)
- research_sufficient: noul — before Research→Judge ("is this evidence adequate?")
- risk_check: noul — before every Operator action ("reversible/safe?") — risk-tiered, irreversible always human
- output_quality: score — after Report renders

## Build order (v0.1 = single-file harness, JSON I/O, testable end-to-end)
1. `harness.py` — orchestrator: task JSON in → agent calls → deliverable out. Each agent = prompt template + model call + schema-checked output.
2. Model layer: `LocalModel` (vllm on pod / transformers local) + `APIModel` (openrouter for Judge-v1) behind one interface — swap without touching agents.
3. Grants: `grants.json` — static, L0-owned file the harness READS but never writes; harness refuses any tool not in grants.
4. Ledger: every agent call logged to the SAME reflex ledger (task=p4_research/judge/report/operator) with confidence + outcome — one flywheel.
5. Test case: "read repo X, summarize its architecture, propose a change" — full four-agent path, human reviews output before any Operator write.

## Safety spine (non-negotiable)
- Operator dry-run default: v0.1 Operator only PROPOSES actions (diff output), never executes. Execution requires explicit `--execute` + L0 grant check.
- Judge escalation to human for: anything touching external comms, money, client data beyond our own repos.

## What runs where
- v0.1: single process, agents call models via vllm server on reflex pod (Qwen3-4B) + openrouter (Judge). Tests on local repos.
- v0.2: package as the client-brain "harness" module (P6 packaging).
