#!/usr/bin/env python3
"""GAP #1: L1->L2 end-to-end shadow batch — the L2 training-data engine.
Picks turns L1 flagged as structured-work, runs the FULL four-agent harness on them
(via the pod L2 server), executes Operator proposals only through operator_exec
(reversible-only, review mode by default), hindsight-logs outcomes. One batch per run.
Usage: python3 l2_e2e_batch.py [--apply]   (default = review mode, no file edits)"""
import os, sys, json, sqlite3, time, hashlib, urllib.request, re

sys.path.insert(0, "/home/hermes/repos/reflex")
L2_URL = "http://localhost:8001/v1/chat/completions"
L1_URL = "http://localhost:8002/v1/chat/completions"
LEDGER = os.path.expanduser("~/.hermes/reflex/ledger.db")
BATCH = 2  # e2e runs per invocation (each is 4 agent calls; keep ticks small)

def chat(base_url, system, user, max_tokens=1500):
    body = json.dumps({"model": os.environ.get("VLLM_MODEL","Qwen/Qwen3-4B-Instruct"),
                       "messages": [{"role":"system","content":system},{"role":"user","content":user}],
                       "max_tokens": max_tokens, "temperature": 0.2}).encode()
    req = urllib.request.Request(base_url, data=body, headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"]

def log(task, preview, answer, outcome=None):
    led = sqlite3.connect(LEDGER, timeout=10)
    led.execute("INSERT INTO decisions (ts, task, state_hash, state_preview, question, answer, "
                "confidence, threshold, action_taken, outcome, model_version) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), task,
                 hashlib.sha256(preview.encode()).hexdigest(), preview[:400],
                 "e2e_batch", str(answer)[:300], 0.0, 0.0, "l2_e2e", outcome, "qwen3-4b-harness-v03"))
    led.commit(); led.close()

RESEARCH_SYS = ("You are RESEARCH, a read-only research agent. Given a user request, list the "
                "concrete evidence you would gather (file paths, searches) and what you can already "
                "conclude from the request itself. Numbered findings. No actions.")
JUDGE_SYS = ("You are JUDGE. Given request + findings, output a determination: one specific, "
             "actionable instruction for OPERATOR (a concrete reversible file edit: path + old text -> new text), "
             "or ESCALATE_HUMAN if it needs judgment beyond a simple reversible edit.")
OPERATOR_SYS = ("You are OPERATOR. Output ONE proposed file edit as JSON: "
                '{"path": "...", "old_text": "...", "new_text": "..."} — a reversible find/replace. '
                "Only propose edits to files under /home/hermes/repos/ or /tmp/. No other actions.")

def run_e2e(request_text, apply=False):
    findings = chat(L2_URL, RESEARCH_SYS, f"Request:\n{request_text[:1200]}\n\nFindings:")
    determination = chat(L2_URL, JUDGE_SYS, f"Request:\n{request_text[:800]}\n\nFindings:\n{findings[:1500]}\n\nDetermination:")
    if "ESCALATE_HUMAN" in determination:
        log("l2_e2e_escalated", request_text, determination[:200], "escalated_to_human")
        return {"status": "escalated", "determination": determination[:300]}
    op_json = chat(L2_URL, OPERATOR_SYS, f"Determination:\n{determination[:1200]}\n\nProposed edit JSON:")
    try:
        m = re.search(r"\{.*\}", op_json, re.DOTALL)
        edit = json.loads(m.group(0))
    except Exception:
        log("l2_e2e_parse_fail", request_text, op_json[:200], "parse_failed")
        return {"status": "parse_failed", "raw": op_json[:300]}
    from reflex.operator_exec import execute_patch
    result = execute_patch(edit.get("path",""), edit.get("old_text",""), edit.get("new_text",""),
                           require_review=not apply)
    log("l2_e2e_exec", request_text, json.dumps(edit)[:300], result.get("reason") or result.get("outcome") or "review")
    return {"status": "ok", "edit": edit, "exec": {k: v for k, v in result.items() if k != 'diff'}}

def main():
    apply = "--apply" in sys.argv
    led = sqlite3.connect(LEDGER, timeout=10)
    led.execute("""CREATE TABLE IF NOT EXISTS l2_e2e (
        id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT, ts TEXT, request TEXT,
        status TEXT, detail TEXT)""")
    # pick turns L1 scored as first_tool in structured/edit classes, not yet e2e'd
    rows = led.execute("""
        SELECT c.message_id, c.state_preview FROM captured_turns c
        JOIN l1_scores s ON s.message_id = c.message_id
        LEFT JOIN l2_e2e e ON e.message_id = c.message_id
        WHERE e.message_id IS NULL AND s.first_tool IN ('edit_files','read_files','find_files')
            AND c.session_id NOT LIKE 'cron_%'
        ORDER BY c.ts DESC LIMIT ?""", (BATCH,)).fetchall()
    if not rows:
        print("l2_e2e: no eligible turns"); return
    for mid, preview in rows:
        body = re.sub(r"^\[Triggering message id:[^\]]*\]\s*", "", preview or "").strip()
        body = re.sub(r"^\[[a-z0-9_ ]+\]\s*", "", body).strip()
        if len(body) < 20: continue
        try:
            r = run_e2e(body, apply=apply)
            led.execute("INSERT INTO l2_e2e (message_id, ts, request, status, detail) VALUES (?,?,?,?,?)",
                        (mid, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         body[:400], r["status"], json.dumps(r)[:600]))
            led.commit()
            print(f"l2_e2e [{mid[:8]}]: {r['status']}")
        except Exception as e:
            print(f"l2_e2e [{mid[:8]}] ERROR: {str(e)[:100]}"); break
    n = led.execute("SELECT COUNT(*) FROM l2_e2e").fetchone()[0]
    print(f"l2_e2e total: {n}")

if __name__ == "__main__":
    main()