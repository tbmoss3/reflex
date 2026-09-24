#!/usr/bin/env python3
"""Eval v0.4 multi-task adapter on ALL three frozen sets: compact, skillgate, stuck.
Usage: python3 eval_mt.py [adapter_dir] -> /workspace/eval_mt_v04.json"""
import json, os, sys, math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "unsloth/gemma-3-1b-it"
MAXLEN = 640
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/workspace/reflex_v04mt/lora"
OUT = "/workspace/eval_mt_v04.json"

SETS = {
    "compact":   {"file": "/workspace/eval_frozen_compact.jsonl", "options": ["keep", "summarize", "drop"]},
    "skillgate": {"file": "/workspace/eval_frozen.jsonl",         "options": ["no", "yes"]},
    "stuck":     {"file": "/workspace/eval_frozen_stuck.jsonl",   "options": ["stuck", "progressing"]},
}

def parse(it):
    for k in ("questions", "answers", "state"):
        if isinstance(it.get(k), str):
            try: it[k] = json.loads(it[k])
            except Exception: pass
    return it

def build_prompt(item, t):
    st = item["state"]
    if isinstance(st, str):
        try: st = json.loads(st)
        except Exception: st = {"message": st}
    if t == "compact":
        q = item["questions"]["keep"]
        msg = str(st.get("message", ""))[:1600]
        ctx = st.get("context", "")
        return (f"Task: context compaction decision.\n{ctx}\n"
                f"{q['instructions']}\n"
                f"Criteria:\n- keep: {q['criteria']['keep']}\n- summarize: {q['criteria']['summarize']}\n- drop: {q['criteria']['drop']}\n\n"
                f"Message:\n{msg}\n\nAnswer with exactly one word: keep, summarize, or drop.\nAnswer:")
    if t == "skillgate":
        q = item["questions"]["needs_attention"]
        out = str(st.get("output_and_error", ""))[:1600]
        job = st.get("job", "")
        return (f"Task: cron-run triage.\nJob: {job}\nLast output:\n{out}\n\n"
                f"{q['instructions']}\nAnswer with exactly one word: no or yes.\nAnswer:")
    if t == "stuck":
        q = item["questions"]["progressing"]
        acts = st.get("recent_actions", [])
        acts_s = "\n".join("- " + str(a)[:300] for a in acts[:12])
        return (f"Task: agent-loop progress check.\n{q['instructions']}\n"
                f"Recent actions:\n{acts_s}\n\n"
                f"Answer with exactly one word: stuck or progressing.\nAnswer:")
    raise ValueError(t)

def label(item, t):
    if t == "compact":
        return item["answers"]["keep"]["choice"]
    if t == "skillgate":
        return "yes" if round(float(item["answers"]["needs_attention"]["noul"])) == 1 else "no"
    if t == "stuck":
        return "progressing" if round(float(item["answers"]["progressing"]["noul"])) == 1 else "stuck"

def loglik(model, tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits[:, :-1]
    tgt = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].bool()
    tgt_logit = logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).float()
    lse = torch.empty_like(tgt_logit)
    for s in range(0, tgt_logit.size(1), 96):
        lse[:, s:s+96] = torch.logsumexp(logits[:, s:s+96].float(), dim=-1)
    ll = tgt_logit - lse
    return (ll * mask).sum(1)

def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    out = {"adapter": ADAPTER}
    for t, cfg in SETS.items():
        items = [parse(json.loads(l)) for l in open(cfg["file"]) if l.strip()]
        if t == "skillgate":
            items = [it for it in items if "needs_attention" in it["questions"]]
        opts = cfg["options"]
        lls_all, labels = [], []
        B = 8
        for i in range(0, len(items), B):
            batch = items[i:i+B]
            prompts = [build_prompt(it, t) for it in batch]
            lls = torch.stack([loglik(model, tok, [p + " " + o for p in prompts]) for o in opts], dim=1)
            lls_all.append(lls); labels += [opts.index(label(it, t)) for it in batch]
        lls = torch.cat(lls_all); labels = torch.tensor(labels).cuda()
        raw = lls.tolist()
        def metrics(T):
            logp = F.log_softmax(torch.tensor(raw) / T, dim=1).cuda()
            conf, pred = logp.exp().max(1)
            acc = (pred == labels).float().mean().item()
            ece = 0.0
            for b in range(10):
                m = (conf >= b/10) & (conf < (b+1)/10)
                if m.sum() > 0:
                    ece += (m.float().sum()/len(labels)).item() * abs(conf[m].mean().item() - (pred[m] == labels[m]).float().mean().item())
            nll = F.nll_loss(logp, labels).item()
            return acc, ece, nll
        best = None
        sweep = {}
        for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]:
            acc, ece, nll = metrics(T)
            sweep[T] = (round(acc, 4), round(ece, 4), round(nll, 4))
            if best is None or nll < best[1]: best = (T, nll, acc, ece)
        out[t] = {"n": len(items), "temp_sweep": sweep, "best_temp": best[0],
                  "best_acc": round(best[2], 4), "best_ece": round(best[3], 4)}
        print(f"[{t}] n={len(items)} best T={best[0]} acc={best[2]:.4f} ECE={best[3]:.4f}", flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("MT_EVAL_DONE", flush=True)

if __name__ == "__main__":
    main()
