#!/usr/bin/env python3
"""Reflex v0.4 compaction task: Gemma-3-1B-it LoRA, constrained choice scoring over keep/summarize/drop.
Loss = CE over option loglik + spherical proper scoring rule term. Per-step loss to /workspace/logs/compact.log."""
import json, os, math, time, argparse, random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

MODEL = "unsloth/gemma-3-1b-it"
DATA = "/workspace/train_compact.jsonl"
OUTDIR = "/workspace/reflex_v04c"
OPTIONS = ["keep", "summarize", "drop"]
MAXLEN = 640

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

def load_model(train=True, adapter=None):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="eager")
    if train:
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"],
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    elif adapter:
        model = PeftModel.from_pretrained(model, adapter)
    return model, tok

def option_loglik(model, tok, prompts, option):
    # batch of prompts; returns log p(option tokens | prompt) per example (memory-efficient)
    inner = 4
    outs = []
    for j in range(0, len(prompts), inner):
        ps = prompts[j:j+inner]
        texts = [p + " " + option for p in ps]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
        enc_p = tok(ps, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to("cuda")
        with torch.no_grad() if not model.training else torch.enable_grad():
            logits = model(**enc).logits[:, :-1]          # [b, L, V] bf16
            tgt = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            plen = enc_p["input_ids"].shape[1]
            opt_mask = torch.zeros_like(mask)
            opt_mask[:, plen-1:] = True
            m = (mask & opt_mask)
            rows, cols = m.nonzero(as_tuple=True)          # positions of option tokens
            sel = logits[rows, cols].float()               # [N, V] small
            ll_tok = F.log_softmax(sel, dim=-1).gather(-1, tgt[rows, cols].unsqueeze(-1)).squeeze(-1)
            ll = torch.zeros(len(ps), device=logits.device)
            ll.index_add_(0, rows, ll_tok)
        outs.append(ll)
    return torch.cat(outs)

def forward_choices(model, tok, prompts):
    lls = torch.stack([option_loglik(model, tok, prompts, o) for o in OPTIONS], dim=1)  # [B,3]
    return lls

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--sphere_w", type=float, default=0.25)
    args = ap.parse_args()
    random.seed(7); torch.manual_seed(7)
    os.makedirs(OUTDIR + "/lora", exist_ok=True)
    os.makedirs("/workspace/logs", exist_ok=True)
    log = open("/workspace/logs/compact.log", "a")
    items = [json.loads(l) for l in open(DATA) if l.strip()]
    # parse questions/answers if stringified
    for it in items:
        for k in ("questions", "answers", "state"):
            if isinstance(it.get(k), str):
                it[k] = json.loads(it[k])
    print(f"{len(items)} examples", flush=True)
    model, tok = load_model(train=True)
    model.cuda(); model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        random.shuffle(items)
        for i in range(0, len(items), args.bs):
            batch = items[i:i+args.bs]
            prompts = [build_prompt(it) for it in batch]
            lls = forward_choices(model, tok, prompts)  # [B,3]
            labels = torch.tensor([OPTIONS.index(it["answers"]["keep"]["choice"]) for it in batch]).cuda()
            logp = F.log_softmax(lls, dim=1)
            ce = F.nll_loss(logp, labels)
            # spherical proper scoring rule: S = (y . p)/||p|| ; loss = 1 - S
            p = logp.exp()
            y = F.one_hot(labels, 3).float()
            sphere = 1.0 - (y * p).sum(1) / p.norm(dim=1).clamp(min=1e-6)
            loss = ce + args.sphere_w * sphere.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0:
                acc = (logp.argmax(1) == labels).float().mean().item()
                msg = f"step {step} ep {ep} loss {loss.item():.4f} ce {ce.item():.4f} acc {acc:.3f} elapsed {time.time()-t0:.0f}s"
                print(msg, flush=True); log.write(msg + "\n"); log.flush()
            if step % 200 == 0:
                model.save_pretrained(OUTDIR + "/lora")
    model.save_pretrained(OUTDIR + "/lora")
    tok.save_pretrained(OUTDIR + "/lora")
    print("TRAIN_DONE", flush=True); log.write("TRAIN_DONE\n")

if __name__ == "__main__":
    main()
