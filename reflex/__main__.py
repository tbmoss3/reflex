import sys, json, runpy

USAGE = """reflex — the intelligence layer for agent harnesses
usage:
  python -m reflex score          # score a request from stdin against the L1 decision model
  python -m reflex harness --task examples/task.json   # run the four-agent L2 harness
  python -m reflex shadow         # L2 shadow pass over new captured turns
  python -m reflex capture         # capture new user turns to the ledger
  python -m reflex hindsight      # label captured turns with hindsight outcomes
"""
def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE); return
    cmd = sys.argv[1]
    if cmd == "score":
        runpy.run_module("reflex.score", run_name="__main__")
    elif cmd == "harness":
        runpy.run_module("reflex.harness_run", run_name="__main__")
    elif cmd == "shadow":
        runpy.run_module("reflex.l2_shadow", run_name="__main__")
    elif cmd == "capture":
        runpy.run_module("reflex.capture", run_name="__main__")
    elif cmd == "hindsight":
        runpy.run_module("reflex.hindsight", run_name="__main__")
    else:
        print(USAGE)

main()