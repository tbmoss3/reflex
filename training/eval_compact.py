#!/usr/bin/env python3
"""Eval compaction adapter on eval_frozen_compact.jsonl: accuracy, ECE, temperature fit.
Usage: python3 eval_compact.py [adapter_dir]"""
import json, os, sys, math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "unsloth/gemma-3-1b-it"
DATA = "/workspace/eval_frozen_compact.jsonl"
OPTIONS = ["keep", "summarize", "drop"]
MAXLEN = 640
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "/workspace/reflex_v04c/lora"
OUT = "/workspace/eval_compact_v04c.json"

def build_prompt(item):
    q = item["questions"]["keep"]
    st = item["state"]
    if isinstance(st, str):
        try: st = json.loads(st)
        except Exception: st = {"message": st}
    msg = st.get("message", "")[:1600]
    ctx = st.get("context", "")
    return (f"Task: context compaction decision.\n{ctx}\n"
            f"{q['instructions']}\n"
            f"Criteria:\n- keep: {q['criteria']['keep']}\n- summarize: {q['criteria']['summarize']}\n- drop: {q['criteria']['drop']}\n\n"
            f"Message:\n{msg}\n\nAnswer with exactly one word: keep, summarize, or drop.\nAnswer:")

def loglik(model, tok, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
    with torch.no_grad():
        logits = model(**enc).logits[:, :-1]
    tgt = enc["input_ids"][:, 1:]
    mask = enc["attention_mask"][:, 1:].bool()
    ll = F.log_softmax(logits.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (ll * mask).sum(1)

def main():
    items = [json.loads(l) for l in open(DATA) if l.strip()]
    for it in items:
        for k in ("questions", "answers", "state"):
            if isinstance(it.get(k), str): it[k] = json.loads(it[k])
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    lls_all, labels = [], []
    B = 16
    for i in range(0, len(items), B):
        batch = items[i:i+B]
        prompts = [build_prompt(it) for it in batch]
        lls = torch.stack([loglik(model, tok, [p + " " + o for p in prompts]) for o in OPTIONS], dim=1)
        lls_all.append(lls); labels += [OPTIONS.index(it["answers"]["keep"]["choice"]) for it in batch]
    lls = torch.cat(lls_all); labels = torch.tensor(labels).cuda()
    raw = lls.tolist()
    def metrics(T):
        logp = F.log_softmax(torch.tensor(raw) / T, dim=1).cuda()
        conf, pred = logp.exp().max(1)
        acc = (pred == labels).float().mean().item()
        # ECE 10 bins
        ece = 0.0
        for b in range(10):
            m = (conf >= b/10) & (conf < (b+1)/10)
            if m.sum() > 0:
                ece += (m.float().sum()/len(labels)).item() * abs(conf[m].mean().item() - (pred[m] == labels[m]).float().mean().item())
        nll = F.nll_loss(logp, labels).item()
        return acc, ece, nll
    best = None
    temps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0]
    results = {}
    for T in temps:
        acc, ece, nll = metrics(T)
        results[T] = (round(acc, 4), round(ece, 4), round(nll, 4))
        print(f"T={T}: acc={acc:.4f} ECE={ece:.4f} nll={nll:.4f}", flush=True)
        if best is None or nll < best[1]: best = (T, nll, acc, ece)
    out = {"adapter": ADAPTER, "n": len(items), "temp_sweep": results,
           "best_temp": best[0], "best_acc": round(best[2], 4), "best_ece": round(best[3], 4)}
    json.dump(out, open(OUT, "w"), indent=1)
    print("EVAL_DONE", json.dumps(out["best_temp"]), flush=True)

if __name__ == "__main__":
    main()
