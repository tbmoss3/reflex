#!/usr/bin/env python3
"""Codex CLI adapter — score turns before Codex spends a frontier call.
Codex supports exec_policy hooks; wire `gate` into your wrapper script:
    from reflex.adapters.codex import gate
    verdict = gate(user_prompt)
    if verdict["action"] == "skip_to_handler": run cheap handler instead.
Same contract as the Hermes adapter; never raises."""
from reflex.adapters.hermes import gate
__all__ = ["gate"]
