"""Smoke für train_moe_router_specialization.py — Mini-Modell (fp32/CPU), aber dieselben
Funktionen: configure_trainable_moe_parameters, DifferentiableRouterHook, der
Trainings-Loop-Mechanismus (forward mit labels + router_* kwargs, KL-Sep-Loss mit
reduction="sum", Backward, Clip, optimizer.step, Post-Step expert_bias-Update)
und der Sanity-Generate. Kein checkpoint_dir nötig."""
import sys, torch
sys.path.insert(0, ".")
import torch.nn.functional as F
from modeling_argon import ArgonConfig, ArgonForConditionalGeneration
from train_moe_router_specialization import (
    configure_trainable_moe_parameters, DifferentiableRouterHook,
)

torch.manual_seed(0)
cfg = ArgonConfig(vocab_size=2000, num_hidden_layers=6, hidden_size=128, intermediate_size=256,
                  num_attention_heads=2, head_dim=48, qk_nope_head_dim=32, qk_rope_head_dim=16,
                  kv_lora_rank=48, q_lora_rank=64, v_head_dim=48, moe_inter_dim=32,
                  n_routed_experts=8, num_experts_per_tok=2, tie_word_embeddings=True,
                  num_key_value_heads=1, shared_intermediate_size=256)
model = ArgonForConditionalGeneration(cfg)  # fp32, CPU
n_experts = 8
B, L = 3, 24
batches = {}
for d, (pref, tgt) in {"code": ([0, 1, 4], 1.35),
                       "math_logic": ([2, 5, 6], 1.85),
                       "prose": ([3, 7], 0.85)}.items():
    ids = torch.randint(10, 2000, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    attn[0, -4:] = 0  # Padding-Fall mitüben
    labels = ids.clone(); labels[attn == 0] = -100
    pm = torch.full((n_experts,), 0.05)
    for e in pref: pm[e] = 1.0
    batches[d] = (ids, attn, labels, pm / pm.sum(), tgt)

# 1) Freeze
router_params, expert_params = configure_trainable_moe_parameters(model, train_experts=True)
assert not model.model.layers[0].self_attn.q_proj.weight.requires_grad, "Attention nicht gefroren!"
assert model.model.layers[0].moe.shared_experts[0].gate_proj.weight.requires_grad is False, "Shared nicht gefroren!"
assert all(p.requires_grad for p in router_params), "Router nicht freigeschaltet!"
assert len(expert_params) > 0, "Experts nicht freigeschaltet!"
# Inline-Balancing-Flag
assert all(not layer.moe.gate.inline_balancing_enabled for layer in model.model.layers), "Flag nicht deaktiviert!"

# 2) Loop-Mechanik (2 Schritte)
router_hook = DifferentiableRouterHook(model)
opt = torch.optim.AdamW([{"params": router_params, "lr": 3e-3},
                         {"params": expert_params, "lr": 1e-4}])
bias_before = [layer.moe.gate.expert_bias.clone() for layer in model.model.layers]
for step in (1, 2):
    model.train(); opt.zero_grad()
    counts = [torch.zeros(n_experts) for _ in model.model.layers]
    for d in batches:
        ids, attn, labels, pref, tgt = batches[d]
        router_hook.clear()
        out = model(input_ids=ids, attention_mask=attn, labels=labels,
                    router_entropy_weight=0.35, router_target_entropy=tgt)
        assert torch.isfinite(out.loss), f"loss non-finite {d} step {step}"
        probs = torch.stack(router_hook.layer_probs, dim=0).mean(dim=0)
        sep = F.kl_div((probs + 1e-8).log(), pref, reduction="sum")
        assert torch.isfinite(sep) and sep.item() >= 0, f"sep_loss {sep.item()}"
        ((out.loss + sep) / 3).backward()
        for i, layer in enumerate(model.model.layers):
            assert hasattr(layer.moe.gate, "last_counts"), f"last_counts fehlt in Schritt {step}"
            counts[i].add_(layer.moe.gate.last_counts)
    gn = torch.nn.utils.clip_grad_norm_(router_params + expert_params, max_norm=1.0)
    assert torch.isfinite(gn), f"Grad-Norm non-finite: {gn}"
    opt.step()
    for i, layer in enumerate(model.model.layers):
        layer.moe.gate.expert_bias.add_(0.002 * torch.sign(counts[i].mean() - counts[i]))
bias_after = [layer.moe.gate.expert_bias for layer in model.model.layers]
n_bias = sum(1 if not torch.equal(b, a) else 0 for b, a in zip(bias_before, bias_after))
assert n_bias > 0, "expert_bias wurde im Post-Step-Update nicht verändert!"

# 3) Router-Gradients: gate.weight hat echten Grad aus CE + Sep, aber NICHT die Attention
assert model.model.layers[0].moe.gate.weight.grad is not None, "kein Gate-Grad!"
assert model.model.layers[0].self_attn.q_proj.weight.grad is None, "Attention-Grad erzeugt (nicht gefroren)?"

# 4) Router-Only-Variante: train_experts=False
router_params2, expert_params2 = configure_trainable_moe_parameters(model, train_experts=False)
assert expert_params2 == [], "router_only: Experten dürfen nicht freigeschaltet sein!"

# 5) Sanity-Generate (greedy) nach 2 Mini-Schritten
model.eval()
gen = model.generate(input_ids=batches["code"][0][:, :5], attention_mask=torch.ones(1, 5, dtype=torch.long),
                     max_new_tokens=3, do_sample=False, use_cache=True)
assert tuple(gen.shape)[1] == 8, f"Generate-Länge falsch: {tuple(gen.shape)}"
print("smoke_router_finetune: OK — Freeze, Hook, Sep-Kl(sum), Post-Step-Bias, Ent-Reg, Generate")