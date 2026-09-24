#!/usr/bin/env python3
"""Reflex v0.5 local scorer — L1 inference for the gateway hook.
Scores a request against the skill-gate heads (first_tool 13-way + needs_full_agent)
using the trained Gemma-3-1B LoRA. CPU-friendly (1B, bf16->fp32 CPU or small GPU slice).
Loads once, serves via stdin/argv or as an import. Ledger rows get REAL answers+confidence.
"""
import json, os, sys, time

ADAPTER = "/home/hermes/repos/teloplex/brain/reflex-data/adapters_backup/v05mt_local/reflex_v05mt/lora"
BASE = "unsloth/gemma-3-1b-it"

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

_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32, device_map="cpu")
    m = PeftModel.from_pretrained(base, ADAPTER)
    m.eval()
    _MODEL = (tok, m)
    return _MODEL

def score(request_text):
    """Returns {first_tool, probabilities, confidence, needs_full_agent_p}."""
    import torch, torch.nn.functional as F, math
    tok, m = get_model()
    prompt = build_prompt(request_text)
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        logits = m(**enc).logits[0, -1, :].float()
    lsm = F.log_softmax(logits, dim=-1)
    scores = []
    for opt in SKILLGATE_OPTS:
        ids = tok(" "+opt, add_special_tokens=False).input_ids
        s = sum(lsm[i].item() for i in ids) / max(1, len(ids))
        scores.append(s)
    mx = max(scores); e = [math.exp(x-mx) for x in scores]; Z = sum(e)
    probs = [x/Z for x in e]
    best = probs.index(max(probs))
    # needs_full_agent: append the binary question
    nprompt = prompt + "\nDoes this need the full agent loop (multi-step reasoning) rather than a single cheap handler? Answer yes or no.\nAnswer:"
    nenc = tok(nprompt, return_tensors="pt", truncation=True, max_length=768)
    with torch.no_grad():
        nlog = m(**nenc).logits[0, -1, :].float()
    nl = F.log_softmax(nlog, dim=-1)
    yes_id = tok(" yes", add_special_tokens=False).input_ids[0]
    no_id = tok(" no", add_special_tokens=False).input_ids[0]
    py, pn = math.exp(nl[yes_id].item()), math.exp(nl[no_id].item())
    p_yes = py/(py+pn)
    return {"first_tool": SKILLGATE_OPTS[best], "probabilities": dict(zip(SKILLGATE_OPTS, probs)),
            "confidence": probs[best], "needs_full_agent_p": p_yes}

if __name__ == "__main__":
    req = sys.stdin.read().strip()
    t0 = time.time()
    out = score(req)
    out["latency_s"] = round(time.time()-t0, 2)
    print(json.dumps(out))