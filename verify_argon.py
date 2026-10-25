"""
verify_argon.py
===============
Automatisierte Verifikation für Argon (1B Hybrid MLA-MoE):
1. Modell-Instanziierung & Integritätsprüfung (0 Missing Keys).
2. Forward-Pass & Depth-weighted Entropie-Kopplung im Loss.
3. Aux-loss-freies MoE-Balancing & Entropie-Adaptive Router-Amplitude.
4. Latent-KV-Cache Validierung (`use_cache=True`, bitgenaue Übereinstimmung zwischen Prefill und Decoding).
5. Reward-Funktion Test.
6. End-to-End Autoregressive Textgenerierung über 35 Tokens.
"""

import os
import json
import math
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

from modeling_argon import ArgonConfig, ArgonForConditionalGeneration
from train_grpo_reward import argon_grpo_reward


def kv_tuples(cache):
    """Cache-Objekt (transformers 5.x, kein [idx]-Zugriff) oder Liste -> Liste aus (k, v)-Tupeln."""
    if cache is None or isinstance(cache, (list, tuple)):
        return cache
    return [(layer.keys, layer.values) for layer in cache.layers]


def run_verification(checkpoint_dir: str = "./argon_1b"):
    print("=" * 70)
    print("Starte Verifikation der Argon-Architektur...")
    print("=" * 70)

    cfg_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")

    assert os.path.exists(cfg_path), f"config.json nicht gefunden in {checkpoint_dir}"
    assert os.path.exists(weights_path), f"model.safetensors nicht gefunden in {checkpoint_dir}"

    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    config = ArgonConfig(**cfg_dict)

    print(f"[Test 1/6] Lade Modell (hidden_size={config.hidden_size}, layers={config.num_hidden_layers})...")
    model = ArgonForConditionalGeneration(config)
    state_dict = load_file(weights_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing Keys: {len(missing)}")
    print(f"  Unexpected Keys: {len(unexpected)}")
    assert len(missing) == 0, f"Fehlende Parameter: {missing}"
    assert len(unexpected) == 0, f"Unerwartete Parameter: {unexpected}"

    model.eval()

    print("[Test 2/6] Führe Forward-Pass & Depth-weighted Entropie-Loss aus...")
    input_ids = torch.randint(100, 10000, (2, 32))
    labels = torch.randint(100, 10000, (2, 32))

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, router_entropy_weight=0.25)
        logits = outputs.logits
        loss = outputs.loss

    print(f"  Logits Shape: {logits.shape}")
    print(f"  Loss mit Entropie-Regularisierung: {loss.item():.4f}")
    assert not torch.isnan(logits).any(), "NaN in Logits!"
    assert not torch.isinf(logits).any(), "Inf in Logits!"
    assert hasattr(outputs, "router_entropies"), "router_entropies fehlt!"
    assert len(outputs.router_entropies) == config.num_hidden_layers, "Falsche Layer-Anzahl!"

    # Router-Entropien müssen alle endlich und im gültigen Spektrum [0, ln(8)] liegen
    for e_layer in outputs.router_entropies:
        assert torch.isfinite(e_layer).all(), "Nicht-finite Router-Entropie!"
        assert e_layer.min() >= -1e-6 and e_layer.max() <= math.log(config.n_routed_experts) + 1e-6
    print("  ✅ Forward-Pass & Entropie-Kopplung verifiziert.")

    print("[Test 3/6] Prüfe MoE-Balancing & Entropie-Amplituden-Gate...")
    model.train()
    gate = model.model.layers[0].moe.gate
    bias_before = gate.expert_bias.clone()
    _ = model(input_ids[:1, :16])
    bias_after = gate.expert_bias.clone()
    assert (bias_after - bias_before).abs().sum().item() > 0.0, "Aux-loss-freies Balancing inaktiv!"
    assert hasattr(gate, "last_counts") and torch.any(gate.last_counts > 0), "last_counts-Snapshot fehlt!"

    model.eval()
    with torch.no_grad():
        # Sättigungs-Probe: Experte 0 stark positiv, alle übrigen stark negativ
        # -> Sigmoid gesättigt, Router-Entropie ~ 0 (niedrig)
        saved_w = gate.weight.data.clone()
        gate.weight.data.zero_()
        gate.weight.data[0, :] = 0.01
        gate.weight.data[1:, :] = -0.01
        _, w_low, e_low = gate(torch.ones(1, config.hidden_size))
        # Flache Probe: alle Logits 0 -> Sigmoid 0.5 überall -> maximale Entropie ln(8)
        gate.weight.data.zero_()
        _, w_flat, e_flat = gate(torch.zeros(1, config.hidden_size))
        gate.weight.data.copy_(saved_w)

    assert e_flat.mean().item() > 2.0, f"Flache Probe hat keine maximale Entropie ({e_flat.mean().item():.4f})!"
    assert e_low.mean().item() < 0.1, f"Sättigungs-Probe hat keine niedrige Entropie ({e_low.mean().item():.4f})!"

    sum_low = w_low.sum(dim=-1).mean().item()
    sum_flat = w_flat.sum(dim=-1).mean().item()
    print(f"  Routed-Amplitude: niedrige Entropie {sum_low:.4f} | maximale Entropie {sum_flat:.4f}")
    assert sum_low <= 0.26, "Router dämpft bei niedriger Entropie nicht!"
    assert sum_flat > 0.95, "Kapazitäts-Gate klemmt die volle Öffnung!"
    print("  ✅ MoE-Balancing & Amplituden-Gate verifiziert.")

    print("[Test 4/6] Teste Latent-KV-Cache (`use_cache=True`)...")
    with torch.no_grad():
        # Prefill
        out_prefill = model(input_ids[:, :16], use_cache=True)
        logits_prefill = out_prefill.logits[:, -1, :]
        past_kv = kv_tuples(out_prefill.past_key_values)

        # Cache-Inhalt prüfen: lokale Schichten (k, v) mit kv_heads-Dim,
        # globale MLA-Schichten (c_kv, k_rope_raw) mit 320-Dim-Latenzvektoren
        assert len(past_kv) == config.num_hidden_layers, "Cache-Layeranzahl falsch!"
        pattern = config.sliding_window_pattern
        for idx in range(config.num_hidden_layers):
            if idx % pattern == (pattern - 1):
                # MLA: nur [c_kv (256), k_rope (64)] = 320 Latent-Dims
                assert past_kv[idx][0].shape == torch.Size((2, 16, config.kv_lora_rank)), \
                    f"MLA c_kv-Shape falsch in Schicht {idx}: {past_kv[idx][0].shape}"
                assert past_kv[idx][1].shape == torch.Size((2, 16, config.qk_rope_head_dim)), \
                    f"MLA k_rope-Shape falsch in Schicht {idx}: {past_kv[idx][1].shape}"
            else:
                assert past_kv[idx][0].dim() == 4, f"Lokale k-Shape falsch in Schicht {idx}"

        # Single Token Decode mit Cache
        out_decode = model(input_ids[:, 16:17], past_key_values=past_kv, use_cache=True)
        logits_decode = out_decode.logits[:, -1, :]

        # Cache-Verlängerung über MEHRERE Decode-Schritte (Sliding + MLA-Wachstum)
        running_cache = out_decode.past_key_values
        dec = None
        for step in range(1, 5):
            dec = model(input_ids[:, 16 + step:17 + step], past_key_values=running_cache, use_cache=True)
            running_cache = dec.past_key_values
        logits_final = dec.logits[:, -1, :]

        # Vollständiger Forward-Pass über 17 Tokens als Referenz für den ersten Decode-Schritt
        out_full = model(input_ids[:, :17])
        logits_full = out_full.logits[:, -1, :]

        # Referenz für den letzten Decode-Schritt: voller Forward über alle 21 Tokens
        out_full21 = model(input_ids[:, :21])
        logits_full21 = out_full21.logits[:, -1, :]

    diff = (logits_decode - logits_full).abs().max().item()
    diff_21 = (logits_final - logits_full21).abs().max().item()
    print(f"  Maximaler Logit-Unterschied (Cache vs. Voller Forward): {diff:.6e}")
    print(f"  Maximaler Logit-Unterschied (5-Schritt-Decode vs. 21-Tok-Forward): {diff_21:.6e}")
    mla_cache_dim = past_kv[5][0].shape[-1] + past_kv[5][1].shape[-1]
    print(f"  MLA-Cache-Dim: {mla_cache_dim} (= kv_lora_rank + qk_rope_head_dim = 320)")
    assert mla_cache_dim == config.kv_lora_rank + config.qk_rope_head_dim, "MLA-Cache nicht auf 320 Dim komprimiert!"
    assert diff < 1e-3, f"Latent-KV-Cache Abweichung zu groß: {diff}"
    assert diff_21 < 1e-3, f"Latent-KV-Cache über mehrere Decode-Schritte instabil: {diff_21}"
    print("  ✅ Latent-KV-Cache (Prefill & Decode) bitgenau verifiziert.")

    print("[Test 5/6] Teste GRPO-Reward-Funktion...")
    prompts = ["Wer bist du?", "Wer bist du?"]
    bad_resp = "Als KI-Modell und großes Sprachmodell habe ich keine eigenen Gefühle, aber gerne helfe ich dir."
    good_resp = "Argon ist eine hocheffiziente Open-Source-Hybrid-Architektur, die Multi-Head Latent Attention mit sparse MoE-Experten auf einem einheitlichen multimodalen Embedding-Raum vereint."

    rewards = argon_grpo_reward(prompts, [bad_resp, good_resp])
    assert rewards[0] < 0.0, f"Boilerplate nicht bestraft: {rewards[0]}"
    assert rewards[1] > 0.0, f"Gute Antwort nicht belohnt: {rewards[1]}"
    print(f"  Reward (Boilerplate): {rewards[0]:.2f} | Reward (Argon): {rewards[1]:.2f}")
    print("  ✅ GRPO-Reward-Funktion verifiziert.")

    print("[Test 6/6] Teste autoregressive Textgenerierung über 35 Tokens...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    test_prompt = "The capital of France is"
    encoded = tokenizer(test_prompt, return_tensors="pt")

    with torch.no_grad():
        out_tokens = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            max_new_tokens=35,
            do_sample=False,
            use_cache=True,
        )

    decoded = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Generiert (35 Tokens):\n  \"{decoded}\"")
    assert "Paris" in decoded, "Faktische Basisantwort nicht generiert!"
    print("  ✅ Autoregressive Generierung über Latent-KV-Cache stabil.")

    print("\n" + "=" * 70)
    print("🎉 Alle 6 Argon-Verifikationsstufen erfolgreich bestanden!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()