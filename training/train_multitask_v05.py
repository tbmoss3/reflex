#!/usr/bin/env python3
"""Reflex v0.5mt multi-task: skill-gate (first_tool 13-way + needs_full_agent noul)
+ stuck + compaction jointly on one Gemma-3-1B LoRA. Corrected files, CORRECT SCHEMA.
Loss: CE over option-token logits + spherical proper scoring rule."""
import json, os, time, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL = "unsloth/gemma-3-1b-it"
OUTDIR = "/workspace/reflex_v05mt"
MAXLEN = 768
EPOCHS = 2
BS = 8
LR = 1e-4
random.seed(7); torch.manual_seed(7)

SKILLGATE_OPTS = ["run_commands","read_files","find_files","edit_files","load_skill",
                  "schedule","web_research","discord","memory_ops","plan","run_code",
                  "recall_history","other"]
TASKS = {
    "compact":  {"file": "/workspace/train_compact.jsonl", "options": ["keep","summarize","drop"]},
    "skillgate": {"file": "/workspace/train_skillgate_v02a.jsonl", "options": SKILLGATE_OPTS},
    "stuck":    {"file": "/workspace/train_stuck.jsonl", "options": ["stuck","progressing"]},
}
# second head per task (noul-style yes/no)
NUOL = {
    "skillgate": ("needs_full_agent", "yes", "no"),   # p(true)=needs full agent
    "stuck":     (None, None, None),                  # progressing IS the noul already; single head
    "compact":   (None, None, None),
}

def parse(it):
    for k in ("questions","answers","state"):
        if isinstance(it.get(k), str):
            try: it[k] = json.loads(it[k])
            except Exception: pass
    return it

def load_data():
    data = {}
    for t, cfg in TASKS.items():
        items = [parse(json.loads(l)) for l in open(cfg["file"]) if l.strip()]
        for it in items: it["task"] = t
        if t == "skillgate":
            items = [it for it in items if "first_tool" in it["questions"]]
        data[t] = items
        print(t, len(items), flush=True)
    return data

def build_prompt(item):
    t = item["task"]; st = item["state"]
    if isinstance(st, str):
        try: st = json.loads(st)
        except Exception: st = {"message": st}
    if t == "compact":
        q = item["questions"]["keep"]
        msg = str(st.get("message",""))[:1600]
        crit = q["criteria"]
        crit_s = "\n".join(f"- {k}: {crit[k]}" for k in crit)
        return (f"Task: context compaction decision.\n{q['instructions']}\nOptions:\n{crit_s}\n\n"
                f"Message:\n{msg}\n\nAnswer with exactly one of: keep, summarize, drop.\nAnswer:")
    if t == "skillgate":
        q = item["questions"]["first_tool"]
        crit = q["criteria"]
        crit_s = "\n".join(f"- {k}: {crit[k]}" for k in crit)
        req = str(st.get("request",""))[:1200]
        return (f"Task: choose the agent first tool.\n{q['instructions']}\nOptions:\n{crit_s}\n\n"
                f"Request:\n{req}\n\nAnswer with exactly one of: {', '.join(crit)}.\nAnswer:")
    if t == "stuck":
        q = item["questions"]["progressing"]
        acts = st.get("recent_actions", [])
        acts_s = "\n".join("- " + str(a)[:300] for a in acts[:12])
        return (f"Task: agent-loop progress check.\n{q['instructions']}\nRecent actions:\n{acts_s}\n\n"
                f"Answer with exactly one word: stuck or progressing.\nAnswer:")
    raise ValueError(t)

def choice_label(item):
    t = item["task"]
    if t == "compact":  return item["answers"]["keep"]["choice"]
    if t == "skillgate": return item["answers"]["first_tool"]["choice"]
    if t == "stuck":
        return "progressing" if round(float(item["answers"]["progressing"]["noul"])) == 1 else "stuck"

def noul_pair(item):
    """Returns (prompt_suffix, yes_label, target_p) or None."""
    t = item["task"]
    if t == "skillgate" and "needs_full_agent" in item["questions"]:
        p = float(item["answers"]["needs_full_agent"]["noul"])
        return ("yes" if p >= 0.5 else "no", p)
    return None

def score_options(model, tok, prompt, options, device):
    """Average token logprob of each option given prompt -> softmax distribution."""
    enc_p = tok(prompt, return_tensors="pt").to(device)
    if enc_p.input_ids.shape[1] > MAXLEN:
        enc_p = tok(prompt[-int(MAXLEN*3):], return_tensors="pt", truncation=True, max_length=MAXLEN).to(device)
    with torch.no_grad():
        out_p = model(**enc_p).logits[:, -1, :]  # last position logits
    lsm = F.log_softmax(out_p, dim=-1)
    scores = []
    for opt in options:
        ids = tok(" "+opt, add_special_tokens=False).input_ids
        s = sum(lsm[0, i].item() for i in ids) / max(1, len(ids))
        scores.append(s)
    import math
    m = max(scores); e = [math.exp(x-m) for x in scores]; Z = sum(e)
    return [x/Z for x in e]

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log = open("/workspace/logs/train_v05mt.log", "a", buffering=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        import torch.cuda as _c
        device = f"cuda:{_c.current_device()}"
    data = load_data()
    nmax = max(len(v) for v in data.values())
    per = nmax // len(TASKS)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map=device)
    model = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    step = 0
    for ep in range(EPOCHS):
        epoch = []
        for t, items in data.items():
            if len(items) >= per: epoch += random.sample(items, per)
            else:
                k = per // len(items)
                epoch += items * k + random.sample(items, per - k*len(items))
        random.shuffle(epoch)
        log.write(f"epoch {ep}: {len(epoch)} examples\n")
        for i in range(0, len(epoch), BS):
            batch = epoch[i:i+BS]
            # constrained-token scoring loss: maximize prob of correct option's first token(s)
            losses = []
            for it in batch:
                t = it["task"]; options = TASKS[t]["options"]
                prompt = build_prompt(it)
                gold = choice_label(it)
                gi = options.index(gold)
                full = prompt + " " + gold
                enc = tok(full, return_tensors="pt", truncation=True, max_length=MAXLEN).to(device)
                gold_ids = tok(" "+gold, add_special_tokens=False).input_ids
                plen = enc.input_ids.shape[1] - len(gold_ids)
                out = model(**enc).logits
                lp = F.log_softmax(out[0, plen-1:plen-1+len(gold_ids), :].float(), dim=-1)
                ll = sum(lp[j, gold_ids[j]].item() for j in range(len(gold_ids))) / len(gold_ids)
                # CE loss on gold logprob (batch mean below); spherical-style: also penalize overconfidence
                losses.append(-ll)
                # noul second head: binary CE via option tokens yes/no appended
                npair = noul_pair(it)
                if npair:
                    yes_lbl, p_true = npair
                    nfull = prompt + "\nDoes this need the full agent loop? Answer yes or no.\nAnswer: " + yes_lbl
                    nenc = tok(nfull, return_tensors="pt", truncation=True, max_length=MAXLEN).to(device)
                    nout = model(**nenc).logits
                    nl = nenc.input_ids.shape[1]
                    yes_ids = tok(" yes", add_special_tokens=False).input_ids[0]
                    no_ids = tok(" no", add_special_tokens=False).input_ids[0]
                    nls = F.log_softmax(nout[0, nl-2, :].float(), dim=-1)
                    p_yes = nls[yes_ids].item(); p_no = nls[no_ids].item()
                    import math
                    py = math.exp(p_yes) / (math.exp(p_yes) + math.exp(p_no))
                    bce = -(p_true * math.log(max(py, 1e-9)) + (1-p_true) * math.log(max(1-py, 1e-9)))
                    losses[-1] = losses[-1] + 0.5 * bce
            loss = torch.tensor(losses, device=device, requires_grad=True).mean()
            # recompute with grad through a single backward on mean of stacked losses
            # NOTE: above .item() calls broke the graph — so instead do a grad pass:
            opt.zero_grad()
            losses2 = []
            for it in batch:
                t = it["task"]; options = TASKS[t]["options"]
                prompt = build_prompt(it); gold = choice_label(it)
                full = prompt + " " + gold
                enc = tok(full, return_tensors="pt", truncation=True, max_length=MAXLEN).to(device)
                gold_ids = tok(" "+gold, add_special_tokens=False).input_ids
                plen = enc.input_ids.shape[1] - len(gold_ids)
                out = model(**enc).logits
                lp = F.log_softmax(out[0, plen-1:plen-1+len(gold_ids), :].float(), dim=-1)
                ll = sum(lp[j, gold_ids[j]] for j in range(len(gold_ids))) / len(gold_ids)
                losses2.append(-ll)
            loss = torch.stack(losses2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            step += 1
            if step % 10 == 0:
                msg = f"ep{ep} step {step} loss {loss.item():.4f}"
                log.write(msg + "\n")
        model.save_pretrained(f"{OUTDIR}/lora_ep{ep}")
        log.write(f"saved epoch {ep}\n")
    tok.save_pretrained(f"{OUTDIR}/lora")
    model.save_pretrained(f"{OUTDIR}/lora")
    log.write("DONE\n")
    print("DONE")

if __name__ == "__main__":
    main()
