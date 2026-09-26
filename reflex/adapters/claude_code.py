#!/usr/bin/env python3
"""Claude Code adapter — gate tool calls with Reflex L1 via Claude Code hooks.
Setup: in .claude/settings.json add to hooks.PreToolUse:
  {"matcher": "Bash|Read|Write|Edit", "hooks": [{"type": "command",
   "command": "python -m reflex.adapters.claude_code"}]}
The hook reads {tool_name, tool_input} on stdin, scores the request, and can block
with a permission decision. Shadow mode default (logs only); REFLEX_LIVE=1 enables
decisioning for reversible tools only."""
import sys, json, os
from reflex.adapters.hermes import gate

REVERSIBLE = {"Read", "Grep", "Glob", "WebSearch"}

def main():
    try:
        event = json.loads(sys.stdin.read())
        tool = event.get("tool_name", "?")
        inp = json.dumps(event.get("tool_input", {}))[:600]
        verdict = gate(f"Tool request: {tool}({inp})")
        # map reflex verdict to claude-code hook decision
        if os.environ.get("REFLEX_LIVE") == "1" and tool in REVERSIBLE and \
           verdict["action"] == "skip_to_handler":
            print(json.dumps({"decision": "approve"}))  # cheap: auto-approve
        # shadow mode / escalations: no output = fall through to normal policy
        sys.exit(0)
    except Exception:
        sys.exit(0)  # never block on adapter failure

if __name__ == "__main__":
    main()
