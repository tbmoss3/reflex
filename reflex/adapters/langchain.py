#!/usr/bin/env python3
"""LangChain/LangSmith adapter — Reflex as a pre-router for chains and as a judge.
Use 1 (routing): wrap your chain entry:
    from reflex.adapters.langchain import route
    tier = route(query)   # -> "l0" | "l2" | "l3"
Use 2 (judge): score agent traces against rubric questions for cheap evals at scale."""
from reflex.adapters.hermes import gate

def route(query: str) -> str:
    v = gate(query)
    if v["action"] == "skip_to_handler": return "l0"     # direct/cheap
    if v.get("needs_full_agent_p", 1) < 0.5: return "l2"  # local structured work
    return "l3"                                           # frontier
