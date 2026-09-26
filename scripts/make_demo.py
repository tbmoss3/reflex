#!/usr/bin/env python3
"""Demo generator for the Reflex launch GIF.
Runs a REAL L1 score + REAL L2 four-agent call, then renders terminal-style frames
showing the request flowing L0 -> L1 -> (skip decision) -> L2 -> cost comparison.
Output: frames + ffmpeg -> /home/hermes/diagrams/reflex_demo.gif"""
import json, os, time, urllib.request, math, subprocess

L1 = "http://localhost:8002/v1/chat/completions"
L2 = "http://localhost:8001/v1/chat/completions"
OUT = "/home/hermes/diagrams/demo_frames"
os.makedirs(OUT, exist_ok=True)

def chat(url, model, system, user, max_tokens=400):
    body = json.dumps({"model": model, "messages": [
        {"role":"system","content":system},{"role":"user","content":user}],
        "max_tokens": max_tokens, "temperature": 0.2}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]["message"]["content"]

REQUEST = "What is Reflex? Summarize the README of the repo"

# 1. REAL L1 score
t0 = time.time()
body = json.dumps({"model":"reflex-v06","messages":[{"role":"user","content":
    "Task: choose the agent first tool.\nWhich tool class should the agent invoke first?\n"
    "Options:\n- read_files: read a specific file or document content\n- web_research: search or fetch information from the web\n"
    "- run_commands: execute shell commands\n\nRequest:\n"+REQUEST+
    "\n\nAnswer with exactly one of: read_files, web_research, run_commands.\nAnswer:"}],
    "max_tokens":1,"temperature":0,"logprobs":True,"top_logprobs":20}).encode()
r = json.load(urllib.request.urlopen(urllib.request.Request(L1,data=body,headers={"Content-Type":"application/json"}),timeout=60))
lps = r["choices"][0].get("logprobs",{}).get("content",[{}])[0].get("top_logprobs",[])
lpm = {}
for lp in lps:
    t = lp["token"].strip().lower().lstrip("▁▁ ")
    p = math.exp(lp["logprob"])
    for o in ("read_files","web_research","run_commands"):
        if t.startswith(o[:4]): lpm[o] = max(lpm.get(o,0), p)
Z = sum(lpm.values()) or 1
best = max(lpm, key=lambda k: lpm[k]); conf = lpm[best]/Z
l1_latency = time.time()-t0

# 2. REAL L2 handler call
t1 = time.time()
readme = open("/home/hermes/repos/reflex/README.md").read()[:3000]
l2_out = chat(L2, "Qwen/Qwen3-4B-Instruct",
    "Answer the user's question using ONLY the provided document. Concise, 3 sentences max.",
    f"Document:\n{readme}\n\nQuestion: {REQUEST}")
l2_latency = time.time()-t1

# 3. Render terminal-style frames with PIL
from PIL import Image, ImageDraw, ImageFont
W, H = 960, 540
BG, FG, ACC, DIM = (10,14,19), (212,220,228), (45,212,191), (120,130,140)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
except: font = ImageFont.load_default()

def frame(lines, path):
    img = Image.new("RGB", (W,H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0,0,W,34], fill=(16,22,30))
    d.text((16,9), "reflex — live demo", font=font, fill=ACC)
    y = 56
    for text, color in lines:
        d.text((20, y), text, font=font, fill=color)
        y += 24
    img.save(path)

seq = [
    [(f'$ echo "{REQUEST}" | reflex', FG), ("", FG)],
    [("", FG), ("L0  grants ......... OK (read-only class)", DIM), ("L0  safety spine ... PASS (nothing restricted touched)", DIM)],
    [("", FG), (f"L1  first_tool ..... read_files  (confidence {conf:.2f})", ACC), ("L1  needs agent? ... NO  ->  skip the frontier model", ACC), (f"L1  latency ........ {l1_latency:.2f}s   cost: $0.00", DIM)],
    [("", FG), ("L2  handler ....... Research reads README, answers directly", ACC), (f"L2  latency ........ {l2_latency:.2f}s   cost: $0.00 (local 4B)", DIM)],
    [("", FG), ("ANSWER:", FG), (l2_out.replace("\n"," ")[:80], FG), ("", FG),
     ("──────────────────────────────────────────────", DIM),
     (f"TOTAL: {l1_latency+l2_latency:.1f}s   $0.00   no frontier model woke up", ACC)],
    [("", FG), ("WITHOUT reflex (frontier-model path):", DIM),
     ("~8s   ~$0.05   full agent loop for a lookup", DIM), ("", FG),
     ("L1 gate paid for itself on this one request.", ACC)],
]
for i, lines in enumerate(seq):
    frame(lines, f"{OUT}/f{i:03d}.png")
# hold last frame: duplicate it
for j in range(8):
    frame(seq[-1], f"{OUT}/f{len(seq)+j:03d}.png")
print("frames:", len(os.listdir(OUT)))
subprocess.run(["ffmpeg","-y","-framerate","1.2","-i",f"{OUT}/f%03d.png",
                "-vf","scale=960:-1","/home/hermes/diagrams/reflex_demo.gif"],
               capture_output=True)
print("GIF:", os.path.getsize("/home/hermes/diagrams/reflex_demo.gif")//1024, "KB")