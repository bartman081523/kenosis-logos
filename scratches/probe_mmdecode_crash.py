"""Repro: 3b-Crash der Suite (Pv=24, Vision-Run [5..14), Decode mit kv_modality_mask)."""
import torch
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

torch.manual_seed(0)
cfg = ArgonConfig(vocab_size=2000, num_hidden_layers=6, hidden_size=128, intermediate_size=256,
                  num_attention_heads=2, head_dim=48, qk_nope_head_dim=32, qk_rope_head_dim=16,
                  kv_lora_rank=48, q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2, tie_word_embeddings=True,
                  num_key_value_heads=1, shared_intermediate_size=256)
model = ArgonForConditionalGeneration(cfg).eval()

Pv = 24
mv_ids = torch.randint(100, 2000, (1, Pv))
mmv = torch.zeros((1, Pv), dtype=torch.long)
mmv[0, 5:14] = 1
visv = torch.randn(1, Pv, cfg.vision_config["vision_hidden_size"], dtype=torch.float32) * 0.5
with torch.no_grad():
    pre_v = model(mv_ids, modality_mask=mmv, vision_patches=visv, use_cache=True)
    running = pre_v.past_key_values
    mm_full = torch.cat([mmv, torch.zeros((1, 1), dtype=torch.long)], dim=1)
    full = model(torch.cat([mv_ids, mv_ids[:, -1:]], dim=1), modality_mask=mm_full)
    print("full forward OK", tuple(full.logits.shape))
    fixed = model(mv_ids[:, 23:24], past_key_values=running, use_cache=True, kv_modality_mask=mmv)
print("fixed OK", tuple(fixed.logits.shape))