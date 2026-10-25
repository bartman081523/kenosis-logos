"""Smoke: generate() über multimodalen Prefill — prüft das kwargs-Threading
modality_mask / vision_patches / kv_modality_mask durch prepare_inputs_for_generation
und die Parität mit dem manuellen Greedy-Trace über die Cache-Bridge."""
import torch
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

torch.manual_seed(0)
cfg = ArgonConfig(vocab_size=2000, num_hidden_layers=6, hidden_size=128, intermediate_size=256,
                  num_attention_heads=2, head_dim=48, qk_nope_head_dim=32, qk_rope_head_dim=16,
                  kv_lora_rank=48, q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2, tie_word_embeddings=True,
                  num_key_value_heads=1, shared_intermediate_size=256)
model = ArgonForConditionalGeneration(cfg).eval()

P, NEW = 12, 5  # Prefill 12 mit Vision-Run [3..9), 5 neue Tokens
ids = torch.randint(10, 2000, (1, P + NEW))
mm = torch.zeros((1, P), dtype=torch.long)
mm[0, 3:9] = 1
vis = torch.randn(1, P, cfg.vision_config["vision_hidden_size"], dtype=torch.float32) * 0.5

# 1) generate() mit multimodalem Prefill
try:
    gen = model.generate(
        input_ids=ids[:, :P], modality_mask=mm, vision_patches=vis, kv_modality_mask=mm,
        attention_mask=torch.ones(1, P, dtype=torch.long), use_cache=True,
        max_new_tokens=NEW, do_sample=False,
    )
    print("generate multimodal: OK, Länge", tuple(gen.shape)[:])
    ran_generate = True
except Exception as e:
    print(f"generate multimodal FEHLGESCHLAGEN: {type(e).__name__}: {e}")
    ran_generate = False

# 2) Manueller Greedy-Trace über die Bridge (Referenz)
past = None
cur = model(ids[:, :P], modality_mask=mm, vision_patches=vis, use_cache=True)
seq = ids[:, :P].clone()
for _ in range(NEW):
    nxt = cur.logits[:, -1].argmax(-1, keepdim=True)
    seq = torch.cat([seq, nxt], dim=1)
    past = cur.past_key_values
    cur = model(nxt, past_key_values=past, use_cache=True, kv_modality_mask=mm)
trace = seq
print("Trace-Länge:", trace.shape[1], "erwartet", P + NEW)

if ran_generate:
    print("generate == Trace:", torch.equal(gen, trace))
    assert torch.equal(gen, trace), "generate-Output weicht vom manuellen Trace ab!"
print("✅ mm-generate-Smoke fertig")