#!/usr/bin/env python3
"""L1 scorer v2 — calls the pod-served reflex-v05 (vllm) with the EXACT trained prompt,
extracts probabilities from top-logprobs at the answer position. Replaces local-CPU scoring
(this host has 1GB RAM; the 1B model serves on the RunPod 4090)."""
import json, os, sys, time, urllib.request

L1_URL = os.environ.get("REFLEX_L1_URL", "http://localhost:8002/v1/chat/completions")
MODEL = os.environ.get("REFLEX_L1_MODEL", "reflex-v05")

SKILLGATE_OPTS = ["run_commands","read_files","find_files","edit_files","load_skill",
                  "schedule","web_research","discord","memory_ops","plan","run_code",
                  "recall_history","other"]
CRITERIA = {
 'run_commands':'execute shell/build/deploy commands on a machine',
 'read_files':'read a specific file or document content',
 'find_files':'search for files or content by pattern',
 'edit_files':'write or modify files/code',
 'load_skill':'load a skill or procedural workflow before acting',
 'schedule':'create/edit/check scheduled jobs',
 'web_research':'search or fetch information from the web',
 'discord':'operate on Discord (channels, messages, roles)',
 'memory_ops':'store or recall durable memory/preferences',
 'plan':'manage a task list / plan steps',
 'run_code':'run Python computation or analysis',
 'recall_history':'search past sessions or knowledge base',
 'other':'anything else'}

def build_prompt(request_text):
    crit_s = "\n".join(f"- {k}: {CRITERIA[k]}" for k in SKILLGATE_OPTS)
    return (f"Task: choose the agent first tool.\nWhich tool class should the agent invoke first?\n"
            f"Options:\n{crit_s}\n\nRequest:\n{request_text[:1200]}\n\n"
            f"Answer with exactly one of: {', '.join(SKILLGATE_OPTS)}.\nAnswer:")

def _call(prompt, max_tokens=1):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0,
                       "logprobs": True, "top_logprobs": 20}).encode()
    req = urllib.request.Request(L1_URL, data=body, headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))

def score(request_text):
    import math
    prompt = build_prompt(request_text)
    r = _call(prompt)
    msg = r["choices"][0]
    # option scores from top_logprobs at the first generated position
    lps = msg.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
    lp_map = {}
    for lp in lps:
        raw = lp["token"]
        tok = raw.strip().lower().lstrip("▁▁ ").lstrip(" ")
        prob = math.exp(lp["logprob"])
        if not tok: continue
        # exact match first
        matched = [o for o in SKILLGATE_OPTS if tok == o.lower()]
        if not matched:
            # prefix: token is the start of an option ("web" -> web_research) or vice versa
            matched = [o for o in SKILLGATE_OPTS if len(tok) >= 2 and (o.lower().startswith(tok) or tok.startswith(o.lower()))]
        if not matched:
            # first word of multiword option
            matched = [o for o in SKILLGATE_OPTS if len(tok) >= 4 and o.lower().split("_")[0].startswith(tok[:4])]
        for m in matched:
            lp_map[m] = lp_map.get(m, 0) + prob  # sum across fragments
    Z = sum(lp_map.values()) or 1.0
    probs = {k: lp_map.get(k, 1e-9)/Z for k in SKILLGATE_OPTS}
    best = max(probs, key=lambda k: probs[k])
    # needs_full_agent head
    nprompt = prompt + "\nDoes this need the full agent loop (multi-step reasoning) rather than a single cheap handler? Answer yes or no.\nAnswer:"
    nr = _call(nprompt, max_tokens=1)
    nmsg = nr["choices"][0]
    nlps = nmsg.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
    py = pn = 1e-9
    for lp in nlps:
        t = lp["token"].strip().lower().lstrip("▁▁ ")
        if t.startswith("yes"): py = max(py, math.exp(lp["logprob"]))
        elif t.startswith("no"): pn = max(pn, math.exp(lp["logprob"]))
    p_yes = py/(py+pn)
    return {"first_tool": best, "confidence": round(probs[best], 4),
            "probabilities": {k: round(v,4) for k,v in probs.items()},
            "needs_full_agent_p": round(p_yes, 4),
            "raw_first_token": (msg["message"]["content"] or "").strip()[:30]}

if __name__ == "__main__":
    req = sys.stdin.read().strip()
    t0 = time.time()
    out = score(req)
    out["latency_s"] = round(time.time()-t0, 2)
    print(json.dumps(out, indent=1))