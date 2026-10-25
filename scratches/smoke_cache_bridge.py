"""Smoke-Test: Cache-Bridge (transformers 5.13) mit Zufallsgewichten — vor Konversion."""
import torch
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration
from transformers.cache_utils import DynamicCache

torch.manual_seed(0)
cfg = ArgonConfig(vocab_size=2000, num_hidden_layers=6, hidden_size=128, intermediate_size=256,
                  num_attention_heads=2, head_dim=48, qk_nope_head_dim=32, qk_rope_head_dim=16,
                  kv_lora_rank=48, q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2, tie_word_embeddings=True,
                  num_key_value_heads=1, shared_intermediate_size=256)
model = ArgonForConditionalGeneration(cfg).eval()
toks = torch.randint(10, 2000, (1, 9))

# 1) Direkt-Forward mit Cache-Return
out = model(toks, use_cache=True)
cache = out.past_key_values
assert isinstance(cache, DynamicCache), type(cache)
print("past returns DynamicCache:", cache.get_seq_length(), "seq_len ok")
tuples = [(l.keys, l.values) for l in cache.layers]
assert tuples[0][0].dim() == 4                                   # lokal
assert tuples[-1][0].shape[-1] == 48 and tuples[-1][0].dim() == 3  # MLA latent
print("MLA latent k:", tuple(tuples[-1][0].shape), "rope:", tuple(tuples[-1][1].shape))

# 2) Decode mit Cache-Objekt (nicht Liste!): Prefill 8 Token, nächstes Token dekodieren
pre = model(toks[:, :8], use_cache=True)
full = model(toks)
d1 = model(toks[:, 8:9], past_key_values=pre.past_key_values, use_cache=True)
diff = (d1.logits[:, -1, :] - full.logits[:, -1, :]).abs().max().item()
print("decode-via-Cache vs full logits diff:", diff)
assert diff < 1e-3

# 3) generate() über die reale 5.13-Schleife (do_sample, 8 neue Tokens)
gen = model.generate(input_ids=toks, max_new_tokens=8, do_sample=False, use_cache=True)
# 3b) Greedy-Parität: generate-Ausgang == Full-Forward-Greedy-Trace
ids = toks
for _ in range(8):
    nxt = model(ids).logits[:, -1].argmax(-1, keepdim=True)
    ids = torch.cat([ids, nxt], dim=1)
print("generate == greedy trace:", torch.equal(gen, ids))
assert torch.equal(gen, ids)
print("✅ Cache-Bridge-Smoke OK")