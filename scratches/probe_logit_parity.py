"""Probe: Logit-Parität gemma-3-1b-it vs. argon_1b (Kalibrierung für Benchmark-Suite)."""
import json, os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

token = None
env_file = "/run/media/julian/ML4/open-mythos_p2/.env"
if os.path.exists(env_file):
    import dotenv; dotenv.load_dotenv(env_file); token = os.getenv("HF_TOKEN")

tok = AutoTokenizer.from_pretrained("./argon_1b")
gemma = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it", torch_dtype=torch.bfloat16).eval()
with open("argon_1b/config.json") as f:
    cfg = ArgonConfig(**json.load(f))
argon = ArgonForConditionalGeneration(cfg).to(torch.bfloat16)
argon.load_state_dict(load_file("argon_1b/model.safetensors"), strict=False)
argon.eval()

prompts = [
    "The capital of France is",
    "In quantum mechanics, the Schrödinger equation describes",
    "def fibonacci(n):\n    if n <= 1:\n        return",
]
cos_sum, top1 = 0.0, 0
for p in prompts:
    inp = tok(p, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        lg = gemma(inp).logits[:, -1, :].float()
        la = argon(inp).logits[:, -1, :].float()
    c = F.cosine_similarity(lg, la, dim=-1).item()
    t1g, t1a = tok.decode(lg.argmax(-1)), tok.decode(la.argmax(-1))
    cos_sum += c
    top1 += int(t1g == t1a)
    print(f"  {p!r:55s} cos={c:.4f} gemma_top1={t1g!r} argon_top1={t1a!r}")
print(f"mean cos = {cos_sum/len(prompts):.4f}, top1 agreement = {top1}/{len(prompts)}")