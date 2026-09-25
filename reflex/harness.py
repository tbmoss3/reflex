#!/usr/bin/env python3
"""P4 Four-Agent Harness v0.1 — Research/Judge/Report/Operator (dry-run Operator).

Usage:
  python3 harness.py --task task.json [--execute]
  task.json: {"request": "...", "evidence_paths": [...], "deliverable": "summary|diff|report"}

Each agent = prompt template + model call + schema-checked output. All calls logged to the
Reflex decision ledger. Operator PROPOSES only; --execute additionally requires L0 grant check.
"""
import argparse, json, os, sys, sqlite3, time, re, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
GRANTS = json.load(open(os.path.join(HERE, "grants.json")))
LEDGER = os.path.expanduser("~/.hermes/reflex/ledger.db")

# ---------------- Model layer ----------------
class Model:
    """One interface for local (vllm/pod) and API models."""
    def __init__(self, backend, model_id):
        self.backend, self.model_id = backend, model_id
    def chat(self, system, user, max_tokens=2000):
        if self.backend == "openrouter":
            return self._openrouter(system, user, max_tokens)
        if self.backend == "vllm":
            return self._vllm(system, user, max_tokens)
        raise ValueError(self.backend)
    def _vllm(self, system, user, max_tokens):
        import urllib.request
        # vllm OpenAI-compatible server on reflex pod; URL from env
        url = os.environ.get("VLLM_URL", "http://localhost:8001/v1/chat/completions")
        body = json.dumps({"model": os.environ.get("VLLM_MODEL", self.model_id), "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0.2}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]
    def _openrouter(self, system, user, max_tokens):
        import urllib.request
        key = os.environ["OPENROUTER_API_KEY"]
        url = "https://openrouter.ai/api/v1/chat/completions"
        body = json.dumps({"model": self.model_id, "messages": [
            {"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": 0.2}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
        return json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]["content"]

MODELS = {
    "research": Model("vllm", "qwen3-4b-instruct"),
    "judge":    Model("vllm", "qwen3-4b-instruct"),   # v0.1: local judge; hard cases -> L3 flag
    "report":   Model("vllm", "qwen3-4b-instruct"),
    "operator": Model("vllm", "qwen3-4b-instruct"),
}

# ---------------- Ledger ----------------
def log_call(role, prompt_digest, answer, confidence=1.0, outcome=None):
    led = sqlite3.connect(LEDGER, timeout=5)
    led.execute("INSERT INTO decisions (ts, task, state_hash, state_preview, question, answer, "
                "confidence, threshold, action_taken, outcome, model_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), f"p4_{role}",
                 hashlib.sha256(prompt_digest.encode()).hexdigest(), prompt_digest[:500],
                 "p4_agent_call", str(answer)[:300], confidence, 0.85, "harness_v01", outcome, "qwen3-4b-v0.1"))
    led.commit(); led.close()

# ---------------- Grant checks (L0) ----------------
def grant_check(role, tool):
    g = GRANTS[role]
    allowed = g.get("allowed_tools", [])
    if tool in allowed:
        return True
    return False

def operator_dry_run_ok(proposal_text):
    """Reject Operator proposals touching forbidden patterns — L0 static rule.
    v0.2: match on CODE/COMMAND context (code blocks, command lines), not prose."""
    import re as _re
    low = proposal_text.lower()
    # only scan fenced code blocks and lines that look like commands/diffs
    code_chunks = _re.findall(r"```.*?```", proposal_text, _re.DOTALL)
    lines_cmdish = [ln for ln in proposal_text.splitlines()
                     if _re.match(r"\s*(- )?(run |execute |sudo |git |sh |bash |python |pip |npm |cmd |terminal|\$ )", ln.lower())
                     or ln.strip().startswith(("+", "-", "$", ">", "sudo"))]
    scan = "\n".join(code_chunks + lines_cmdish).lower()
    if not scan.strip():
        return True, None
    for pat in GRANTS["operator"]["forbidden_patterns"]:
        if pat in scan:
            return False, pat
    return True, None

# ---------------- Tool layer for Research (read-only, grant-checked) ----------------
def research_read(paths, max_chars=12000):
    """Read evidence paths (read-only) for Research. Grant-checked per grants.json."""
    if not grant_check("research", "read_file"):
        return {"error": "research read_file not granted"}
    out = []
    for p in paths:
        try:
            body = open(p, errors="replace").read()[:max_chars]
            out.append({"path": p, "content": body})
        except Exception as e:
            out.append({"path": p, "error": str(e)[:200]})
    return out

def research_search(pattern, path="/home/hermes/repos/teloplex", max_hits=20):
    if not grant_check("research", "search_files"):
        return {"error": "not granted"}
    import subprocess
    r = subprocess.run(["rg", "-l", pattern, path], capture_output=True, text=True, timeout=30)
    return r.stdout.strip().splitlines()[:max_hits]

# ---------------- Agents ----------------
def research(task, model):
    sys_p = ("You are RESEARCH, a read-only research agent. You will be given the ACTUAL "
             "document contents from evidence paths. Cite them verbatim with exact quotes "
             "and the exact section headings as they appear. If something is NOT in the "
             "document, say so explicitly — never invent sections or quotes. Output: a "
             "numbered list of findings, each with an exact quote + section reference. "
             "Do not propose actions. Do not write files.")
    ev = research_read(task.get("evidence_paths", []))
    ev_s = json.dumps(ev)[:16000]
    user = f"Request: {task['request']}\n\nDOCUMENT CONTENTS (verbatim):\n{ev_s}\n\nFindings:"
    out = model.chat(sys_p, user, max_tokens=2500)
    log_call("research", task["request"], out[:200])
    return out

def judge(task, findings, model):
    sys_p = ("You are JUDGE. Given a request and research findings, make the determination: "
             "what should be done, specifically. VERIFY each finding you rely on against the "
             "quoted document text — if a finding cites a section or quote you cannot "
             "verify in the provided findings, discard it and say so. If the evidence is "
             "insufficient, say INSUFFICIENT. If the case is hard/ambiguous or touches "
             "external communications, money, or client data, say ESCALATE_HUMAN. "
             "Output: determination + rationale + confidence 0-1.")
    user = f"Request: {task['request']}\n\nFindings:\n{findings}\n\nDetermination:"
    out = model.chat(sys_p, user, max_tokens=2500)
    log_call("judge", task["request"], out[:200])
    if "ESCALATE_HUMAN" in out:
        return {"escalate": True, "text": out}
    if "INSUFFICIENT" in out:
        return {"insufficient": True, "text": out}
    return {"determination": out}

def report(task, determination, model):
    sys_p = ("You are REPORT. Render the judge's determination into the deliverable format. "
             "Do NOT re-decide, add opinions, or change any facts. Faithful rendering only.")
    user = f"Request: {task['request']}\nDeliverable type: {task.get('deliverable','report')}\n\nDetermination:\n{determination}\n\nDeliverable:"
    out = model.chat(sys_p, user)
    log_call("report", task["request"], out[:200])
    return out

def operator(task, determination, deliverable, model, execute=False):
    sys_p = ("You are OPERATOR. Given a deliverable, output a PROPOSAL of concrete file "
             "operations as markdown diff blocks or shell commands. Do not claim to execute. "
             "Format: PROPOSED ACTIONS as a list, with file paths and exact edit content.")
    user = (f"Request: {task['request']}\n\nJUDGE DETERMINATION (implement EXACTLY this, nothing else):\n{determination}\n\n"
            f"REPORT deliverable:\n{deliverable}\n\nProposed actions (must implement the determination only):")
    out = model.chat(sys_p, user, max_tokens=2500)
    ok, hit = operator_dry_run_ok(out)
    log_call("operator", task["request"], ("GRANT_BLOCK:" + hit) if not ok else out[:200],
             outcome="blocked" if not ok else "proposed")
    if not ok:
        return {"blocked": True, "pattern": hit, "proposal": out}
    if not execute:
        return {"dry_run": True, "proposal": out}
    # v0.1: execution path is intentionally unimplemented — requires L0 grant-check integration
    # with the terminal tool + human approval for irreversible classes. v0.1 ships proposals only.
    return {"dry_run": True, "proposal": out,
                "note": "execution available via reflex.operator_exec (reversible-only, L0-grant-checked, snapshotted)"}

# ---------------- Orchestration ----------------
def run_task(task, execute=False):
    print("[1/4] RESEARCH...")
    findings = research(task, MODELS["research"])
    print("[2/4] JUDGE...")
    j = judge(task, findings, MODELS["judge"])
    if j.get("escalate") or j.get("insufficient"):
        return {"status": "escalated", "detail": j}
    print("[3/4] REPORT...")
    deliverable = report(task, j["determination"], MODELS["report"])
    print("[4/4] OPERATOR (dry-run)...")
    op = operator(task, j["determination"], deliverable, MODELS["operator"], execute)
    # Final Judge alignment check: does the proposal implement the determination?
    chk = MODELS["judge"].chat(
        "You are JUDGE (final check). Given a determination and an operator proposal, answer "
        "ALIGNED or DRIFTED, then one line why. DRIFTED if the proposal implements anything "
        "other than the determination.",
        f"Determination:\n{j['determination'][:1500]}\n\nProposal:\n{op.get('proposal','')[:1500]}\n\nVerdict:")
    aligned = "ALIGNED" in chk and "DRIFTED" not in chk
    log_call("judge_final", task["request"], chk[:200], outcome="aligned" if aligned else "drifted")
    return {"status": "ok", "findings": findings, "determination": j["determination"],
            "deliverable": deliverable, "operator": op,
            "alignment": {"verdict": chk[:300], "ok": aligned}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    task = json.loads(open(args.task).read())
    result = run_task(task, args.execute)
    print(json.dumps(result, indent=1)[:4000])
