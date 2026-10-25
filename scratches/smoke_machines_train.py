"""Smoke für die Götterdreiheit-Extension von train_moe_router_specialization.py
(Schritt 5): resolve_corpus(machines|all), measure_domain_routing(corpus=...),
train_moe_specialization(prebuilt_model, corpus_choice="machines"),
router_diff-Save (nur geänderte gate.weight/expert_bias-Keys) und
_emit_machine_presets. Mini-Modell (bf16/CPU), dieselben Funktionen wie im
Real-Lauf; Real-Checkpoint wird NICHT angefasst."""
import json
import os
import sys

import torch

sys.path.insert(0, ".")

from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from modeling_argon import ArgonConfig, ArgonForConditionalGeneration
from train_moe_router_specialization import (
    MACHINE_DOMAINS,
    resolve_corpus,
    train_moe_specialization,
)

torch.manual_seed(0)

# 1) resolve_corpus
calib_m = resolve_corpus("machines")
assert set(calib_m) == {"goedel", "turing", "wittgenstein"}, list(calib_m)
for dom, tgt, pref in [("goedel", 1.30, [2, 7]), ("turing", 1.65, [0, 6]),
                       ("wittgenstein", 1.90, [1, 3])]:
    info = calib_m[dom]
    assert abs(info["target_entropy"] - tgt) < 1e-9, (dom, info["target_entropy"])
    assert info["preferred_experts"] == pref, (dom, info["preferred_experts"])
    assert len(info["samples"]) == 24, (dom, len(info["samples"]))
    assert all(isinstance(s, str) and s for s in info["samples"])
calib_all = resolve_corpus("all")
assert len(calib_all) == 6 and set(calib_all) >= {"code", "math_logic", "prose"}
print("[1] resolve_corpus OK — machines 3×24, all=6 Domänen")

# 2) Scratch-Base (bf16-Datei wie im Real-Lauf) + echte Mini-ID-Map
cfg_kwargs = dict(vocab_size=262144, num_hidden_layers=6, hidden_size=128,
                  intermediate_size=256, num_attention_heads=2, head_dim=48,
                  qk_nope_head_dim=32, qk_rope_head_dim=16, kv_lora_rank=48,
                  q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2,
                  tie_word_embeddings=True, num_key_value_heads=1,
                  shared_intermediate_size=256)
scratch_base = "scratches/smoke_machines_base"
out_dir = "scratches/smoke_machines_out"
for d in (scratch_base, out_dir):
    if os.path.isdir(d):
        import shutil
        shutil.rmtree(d)
os.makedirs(scratch_base)

model = ArgonForConditionalGeneration(ArgonConfig(**cfg_kwargs)).to(dtype=torch.bfloat16)
json.dump(cfg_kwargs, open(os.path.join(scratch_base, "config.json"), "w"), indent=2)
save_file({k: v.detach().clone().contiguous() for k, v in model.state_dict().items()},
          os.path.join(scratch_base, "model.safetensors"))
print("[2] Scratch-Base geschrieben:", scratch_base)

# 3) Joint Mini-Run, corpus machines, router_diff, Presets
tok = AutoTokenizer.from_pretrained("./argon_1b")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

train_moe_specialization(
    checkpoint_dir=scratch_base,
    output_dir=out_dir,
    steps=2,
    entropy_reg_weight=0.35,
    domain_sep_weight=0.30,
    train_experts=False,          # router_only im Real-Lauf
    max_length=128,
    corpus_choice="machines",
    save_mode="router_diff",
    emit_presets=True,
    prebuilt_model=model,
    tokenizer=tok,
    sanity_check=False,           # Mini: Paris-Assert auf Zufallsgewichten sinnlos
    steer=50.0,
)

# 4) router_diff-Save-Audit
diff = load_file(os.path.join(out_dir, "router_diff.safetensors"))
assert len(diff) > 0 and len(diff) <= 2 * 6, f"diff-Keys unvernünftig: {len(diff)}"
for k in diff:
    assert "moe.gate." in k, f"Diff-Key außerhalb des Routers: {k}"
    assert k.endswith("gate.weight") or k.endswith("gate.expert_bias"), k
meta = json.load(open(os.path.join(out_dir, "router_diff_meta.json")))
assert meta["corpus_choice"] == "machines" and meta["steps"] == 2
assert meta["n_keys"] == len(diff) and os.path.isabs(meta["base_checkpoint"])
cfg_saved = json.load(open(os.path.join(out_dir, "config.json")))
assert cfg_saved["num_hidden_layers"] == 6
print(f"[4] router_diff OK — {len(diff)} Keys: {sorted(diff)[:3]}...")

# 5) Presets
for dom, tgt, pref in [("goedel", 1.30, None), ("turing", 1.65, None),
                       ("wittgenstein", 1.90, None)]:
    p = json.load(open(os.path.join(out_dir, "machine_presets", f"{dom}.json")))
    assert p["domain"] == dom and abs(p["entropy_target"] - tgt) < 1e-9
    assert len(p["experts"]) == 2 and len(set(p["experts"])) == 2
    assert p["steer"] == 50.0
    bias = p["expert_bias"]
    assert bias["layers"] == 6 and len(bias["values"]) == 6 * 8
    chosen = {i for i, v in enumerate(bias["values"][:8]) if v > 0}
    assert chosen == set(p["experts"]), (chosen, p["experts"])
    assert len(p["routing_pct"]) == 8
combined = json.load(open(os.path.join(out_dir, "machine_presets", "combined.json")))
all_experts = set().union(*(json.load(open(os.path.join(out_dir, "machine_presets", f"{d}.json")))["experts"]
                            for d in MACHINE_DOMAINS))
assert combined["experts"] == sorted(all_experts), (combined["experts"], sorted(all_experts))
cb = combined["expert_bias"]["values"][:8]
assert {i for i, v in enumerate(cb) if v > 0} == set(combined["experts"])
print("[5] Presets OK —", {d: json.load(open(os.path.join(out_dir, 'machine_presets', f'{d}.json')))["experts"]
                           for d in MACHINE_DOMAINS}, "combined:", combined["experts"])

print("smoke_machines_train: OK — resolve_corpus, Joint-Run, router_diff, machine_presets")