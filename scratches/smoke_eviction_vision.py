"""Smoke-Test: O(1)-Ringpuffer-Eviction der lokalen MQA-Caches + Fix der
2D-Vision-RoPE-Rekalkulation im Decode-State (kv_modality_mask)."""
import torch
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

torch.manual_seed(0)
cfg = ArgonConfig(vocab_size=2000, num_hidden_layers=6, hidden_size=128, intermediate_size=256,
                  num_attention_heads=2, head_dim=48, qk_nope_head_dim=32, qk_rope_head_dim=16,
                  kv_lora_rank=48, q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2, tie_word_embeddings=True,
                  num_key_value_heads=1, shared_intermediate_size=256)
model = ArgonForConditionalGeneration(cfg).eval()

# ---------- Test A: Ringpuffer-Eviction (600 Token Prefill + 30 Decode-Schritte) ----------
L0 = 600
toks = torch.randint(10, 2000, (1, L0 + 30))
pre = model(toks[:, :L0], use_cache=True)
layers = [(l.keys, l.values) for l in pre.past_key_values.layers]
local_len = layers[0][0].shape[2]
abs_len = layers[5][0].shape[1]
print(f"A: local cache nach Eviction = {local_len} (sliding_window=512), global (MLA) = {abs_len}")
assert local_len == 512, f"Eviction fehlgeschlagen: {local_len}"
assert abs_len == L0, f"MLA-Cache wächst falsch: {abs_len}"

running_cache = pre.past_key_values
worst = 0.0
for step in range(1, 31):
    d = model(toks[:, L0 + step - 1: L0 + step], past_key_values=running_cache, use_cache=True)
    full = model(toks[:, : L0 + step])
    diff = (d.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()
    worst = max(worst, diff)
    assert diff < 1e-3, f"Eviction-Decode abweichend bei T={L0+step}: {diff}"
    running_cache = d.past_key_values
layers = [(l.keys, l.values) for l in running_cache.layers]
print(f"A ✅ Decode-vs-Vollforward über 30 Schritte, worst diff = {worst:.3e}")
print(f"   nach 30 Steps wieder konstant: local={layers[0][0].shape[2]}, global={layers[5][0].shape[1]}")
assert layers[0][0].shape[2] == 512 and layers[5][0].shape[1] == L0 + 30

# ---------- Test B: 2D-Vision-RoPE im Decode-State ----------
P = 40  # Prefill mit Vision-Run [5..13) (9 Patches -> 3x3-Gitter)
mm = torch.zeros((1, P), dtype=torch.long)
mm[0, 5:14] = 1
vis = torch.randn(1, P, cfg.vision_config["vision_hidden_size"], dtype=torch.float32) * 0.5
vis = vis.to(torch.float32)
# Prefill (mit Vision)
pre_v = model(toks[:, :P].clone(), modality_mask=mm, vision_patches=vis.to(model.dtype), use_cache=True)

# Decode 2 Tokens, mit kv_modality_mask (Fix)
for step in range(1, 3):
    T = P + step
    cur = toks[:, T - 1: T]
    d_fixed = model(cur, past_key_values=pre_v.past_key_values, use_cache=True,
                    kv_modality_mask=mm)
    full = model(toks[:, :T], modality_mask=torch.cat([mm, torch.zeros((1, T - P), dtype=torch.long)], dim=1),
                 vision_patches=torch.cat([vis, torch.zeros(1, T - P, vis.shape[2])], dim=1).to(model.dtype))
    diff_fixed = (d_fixed.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()

    d_broken = model(cur, past_key_values=pre_v.past_key_values, use_cache=True)
    diff_1d = (d_broken.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()
    print(f"B: T={T} diff mit kv_modality_mask = {diff_fixed:.4e} | ohne (1D-Rekalkulation, Bug) = {diff_1d:.4e}")
    assert diff_fixed < 1e-3, f"Vision-Decode-Fix fehlgeschlagen bei T={T}: {diff_fixed}"
    running_cache = d_fixed.past_key_values
    pre_v = d_fixed  # Cache weiterlaufen lassen

print("✅ Eviction + Vision-RoPE-Freeze OK")