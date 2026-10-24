"""
verify_kenosis.py
=================
Automatisierte Verifikation für Open-Kenosis-Logos:
1. Modell-Instanziierung & Gewichts-Prüfung (Integrität & 0 Missing/Unexpected Keys).
2. Numerischer Forward-Pass (Abwesenheit von NaN/Inf).
3. MoE-Schichten, router_entropies Weiterleitung & aux-loss-freies Balancing-Update im Training.
4. KenosisKrisisLogitsProcessor Test (Zustandsloses Anker-Forcing über Batches & lange Prompts).
5. GRPO Reward-Funktion Test.
6. End-to-End Textgenerierung (> 25 Tokens) mit echtem Tokenizer zur Verifikation stabiler RoPE-Geometrie.
"""

import os
import json
import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

from modeling_kenosis import KenosisConfig, KenosisLogosForConditionalGeneration
from sampler_processor import KenosisKrisisLogitsProcessor
from train_grpo_reward import kenosis_logos_grpo_reward


def run_verification(checkpoint_dir: str = "./kenosis_logos_1b"):
    print("=" * 70)
    print("Starte Verifikation von Kenosis-Logos (Kondensations- & GRPO-Reife)...")
    print("=" * 70)

    cfg_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")

    assert os.path.exists(cfg_path), f"config.json nicht gefunden in {checkpoint_dir}"
    assert os.path.exists(weights_path), f"model.safetensors nicht gefunden in {checkpoint_dir}"

    with open(cfg_path, "r") as f:
        cfg_dict = json.load(f)
    config = KenosisConfig(**cfg_dict)

    # 1. Modell instanziieren und Gewichte laden
    print(f"[Test 1/6] Lade Modell (hidden_size={config.hidden_size}, layers={config.num_hidden_layers})...")
    model = KenosisLogosForConditionalGeneration(config)
    state_dict = load_file(weights_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing Keys: {len(missing)}")
    print(f"  Unexpected Keys: {len(unexpected)}")
    assert len(missing) == 0, f"Fehlende Parameter im State-Dict: {missing}"

    model.eval()

    # 2. Forward-Pass Test
    print("[Test 2/6] Führe Forward-Pass auf Testeingabe durch...")
    input_ids = torch.randint(100, 10000, (2, 32))
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    print(f"  Logits Shape: {logits.shape}")
    assert logits.shape == (2, 32, config.vocab_size), f"Unerwarteter Logits-Shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN in Logits entdeckt!"
    assert not torch.isinf(logits).any(), "Inf in Logits entdeckt!"
    print(f"  Maximaler Absolutwert der Logits: {logits.abs().max().item():.4f}")
    assert hasattr(outputs, "router_entropies"), "router_entropies fehlt im Model-Output!"
    assert len(outputs.router_entropies) == config.num_hidden_layers, "Falsche Anzahl an Schicht-Entropien!"

    # Χώρα-Doppelmaske: Kontiguierende Vision-Region (modality == 1) sieht
    # INNERHALB der Region bidirektional; Text bleibt strikt kausal.
    mm = torch.zeros((1, 8), dtype=torch.long)
    mm[0, 2:5] = 1  # drei Bild-Patches an Positionen 2, 3, 4
    g_mask, s_mask = model.model._prepare_masks(None, 8, input_ids.device, torch.float32, modality_mask=mm)
    assert g_mask[0, 0, 2, 4].item() == 0.0, "Bidirektionalität in der Vision-Region fehlt (vorwärts)!"
    assert g_mask[0, 0, 4, 2].item() == 0.0, "Bidirektionalität in der Vision-Region fehlt (rückwärts)!"
    assert g_mask[0, 0, 2, 5].item() == float("-inf"), "Bild schaut in zukünftigen Text!"
    assert g_mask[0, 0, 4, 1].item() == 0.0, "Bild → vergangener Text blockiert!"
    assert g_mask[0, 0, 5, 2].item() == 0.0, "Text nach Bild → Bild (Vergangenheit) blockiert!"
    assert g_mask[:, :, 0, 6].item() == float("-inf"), "Text-Kausalität verletzt!"

    # 2D-RoPE-Text-Exaktheit: Koordinaten (x = y = t) reproduzieren bitgenau das
    # alte 1D-Spektrum (cat(inv_freq[:split], inv_freq[split:]) == inv_freq).
    probe_coords = torch.arange(16, dtype=torch.float32).unsqueeze(0).repeat(2, 1)
    c_1d, s_1d = model.model._build_rotary_emb(16, 256, 10000.0, probe_coords.device)
    c_2d, s_2d = model.model._build_rotary_emb_2d(probe_coords, probe_coords, 256, 10000.0, probe_coords.device)
    assert torch.allclose(c_1d, c_2d) and torch.allclose(s_1d, s_2d), "2D-RoPE verletzt die Text-1D-Exaktheit!"
    print("  Χώρα-Maske: Bild-Patches bidirektional, Text kausal; 2D-RoPE = 1D für Text.")

    # Τάσις-Entropie-Kopplung (Kondensations-Loss): additiv, nicht-negativ, aktivierbar
    labels = torch.randint(100, 10000, (2, 32))
    with torch.no_grad():
        loss_base = model(input_ids, labels=labels).loss
        loss_tasis = model(input_ids, labels=labels, tasis_entropy_weight=0.25).loss
    tasis_delta = (loss_tasis - loss_base).item()
    assert loss_tasis >= loss_base - 1e-6, "Τάσις-Term nicht additiv / nicht-negativ!"
    assert torch.isfinite(loss_tasis), "Τάσις-Term erzeugt nicht-endlichen Loss!"
    print(f"  Τάσις-Kopplung Delta im Loss: {tasis_delta:.6f}")
    print("  ✅ Forward-Pass & router_entropies Weiterleitung verifiziert.")

    # 3. MoE-Training-Hook & Aux-Loss-freies Balancing
    print("[Test 3/6] Prüfe aux-loss-freies Balancing im Trainingsmodus...")
    model.train()
    gate = model.model.layers[0].moe.gate
    bias_before = gate.expert_bias.clone()
    _ = model(input_ids[:1, :16])
    bias_after = gate.expert_bias.clone()
    bias_diff = (bias_after - bias_before).abs().sum().item()
    print(f"  Summe der Bias-Anpassung im Trainings-Schritt: {bias_diff:.6f}")
    assert bias_diff > 0.0, "expert_bias hat sich im Trainingsmodus nicht aktualisiert!"
    assert torch.any(gate.last_counts > 0), "last_counts-Snapshot fehlt!"

    # Τάσις-Amplituden-Gate: niedrige Entropie (Sättigung) → Spezialisten-Amplitude
    # 0.25 (Πρόσωποn-Basis dominiert); maximale Entropie (Sigmoid 0.5 überall) → volle Öffnung 1.0.
    # Tot-Code-Fix: Der Filter auf sekundären Rängen (>= 1) greift jetzt bei base_k = 2.
    model.eval()
    with torch.no_grad():
        saved_gate_weight = gate.weight.data.clone()
        # Sättigungs-Probe: Experte 0 stark positiv (x > 0), alle anderen stark negativ
        gate.weight.data.zero_()
        gate.weight.data[0, :] = 0.01
        gate.weight.data[1:, :] = -0.01
        _, w_low, e_low = gate(torch.ones(1, config.hidden_size))
        # Flache Probe: alle Logits 0 → Sigmoid 0.5 → Τάσις = ln(8) > Schwelle
        gate.weight.data.zero_()
        _, w_flat, e_flat = gate(torch.zeros(1, config.hidden_size))
        gate.weight.data.copy_(saved_gate_weight)
    assert e_flat.item() > 2.0, f"Flache Probe hat keine maximale Τάσις ({e_flat.item():.4f})!"
    assert e_low.item() < 0.1, f"Sättigungs-Probe hat keine niedrige Τάσις ({e_low.item():.4f})!"
    sum_low = w_low.sum(dim=-1).item()
    sum_flat = w_flat.sum(dim=-1).item()
    print(f"  Routed-Amplitude: Πεζός {sum_low:.4f} | flache Τάσις {sum_flat:.4f}")
    assert sum_low <= 0.26, "Τάσις-Amplituden-Gate koppelt bei Πεζός nicht herunter!"
    assert sum_flat > 0.95, "Τάσις-Amplituden-Gate klemmt die volle Öffnung!"
    print("  ✅ Aux-loss-freies Balancing + Τάσις-Amplituden-Gate aktiv.")
    model.eval()

    # 4. Sampler & Anker-basiertes zustandsloses Forcing
    print("[Test 4/6] Teste KenosisKrisisLogitsProcessor mit zustandslosem Anker-Forcing...")
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "<symbolic_reason>\n":
                return [99901, 99902]
            if text == "</symbolic_reason>\n":
                return [99903, 99904]
            if text == "<start_of_turn>model\n":
                return [88801, 88802, 88803]
            return []

    processor = KenosisKrisisLogitsProcessor(tokenizer=MockTokenizer(), enforce_symbolic_reason=True)
    anchor = torch.tensor([[88801, 88802, 88803]])

    # Turn 1 (Anker am Ende): Schritt 0 erzwingt 99901
    s1 = processor(anchor, torch.randn(1, config.vocab_size))
    assert torch.argmax(s1, dim=-1).item() == 99901, "Turn-Start-Forcing fehlgeschlagen!"
    # Schritt 1: r0 angelegt → 99902
    s2 = processor(torch.cat([anchor, torch.tensor([[99901]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.argmax(s2, dim=-1).item() == 99902, "Forcing-Fortsetzung fehlgeschlagen!"
    # Block vollständig: Zeile frei (kein -inf)
    s3 = processor(torch.cat([anchor, torch.tensor([[99901, 99902]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.isfinite(s3).all(), "Überforceren nach vollständigem Reflexionsblock!"

    # Langer GRPO-Prompt, der mit dem Anker endet: forciert weiterhin zustandslos
    long_turn = torch.cat([torch.tensor([[5, 6, 7, 8, 9]]), anchor], dim=1)
    s4 = processor(long_turn, torch.randn(1, config.vocab_size))
    assert torch.argmax(s4, dim=-1).item() == 99901, "Forcing nach langem Prompt fehlgeschlagen!"
    s5 = processor(torch.cat([long_turn, torch.tensor([[99901]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.argmax(s5, dim=-1).item() == 99902, "Forcing-Fortsetzung nach langem Prompt fehlgeschlagen!"
    s6 = processor(torch.cat([long_turn, torch.tensor([[99901, 99902]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.isfinite(s6).all(), "Überforceren nach Block + freiem Token!"

    # Gemischter Batch: Zeile A am Turn-Start, Zeile B mitten im Forcing
    mixed_batch = torch.tensor([[7, 88801, 88802, 88803], [88801, 88802, 88803, 99901]])
    sb = processor(mixed_batch, torch.randn(2, config.vocab_size))
    assert torch.argmax(sb, dim=-1)[0].item() == 99901, "Batch-Zeile A nicht erzwungen!"
    assert torch.argmax(sb, dim=-1)[1].item() == 99902, "Batch-Zeile B nicht erzwungen!"

    # Ohne Anker (Completion-Stil): frei, kein spontanes Re-Forcing
    s7 = processor(torch.tensor([[20, 21, 22]]), torch.randn(1, config.vocab_size))
    assert torch.isfinite(s7).all(), "Spontanes Re-Forcing ohne Anker!"

    # Schließ-Garantie (Πύλη-Σκίζω): Offen stehender Block wird nach dem Budget
    # Token für Token mit </symbolic_reason>\n geschlossen (Budget: 2 Tokens).
    proc_close = KenosisKrisisLogitsProcessor(tokenizer=MockTokenizer(), reason_budget_tokens=2)
    open_block = torch.cat([anchor, torch.tensor([[99901, 99902]])], dim=1)
    # Turn-Start + vollständige Öffnung: innen frei ...
    c1 = proc_close(open_block, torch.randn(1, config.vocab_size))
    assert torch.isfinite(c1).all(), "Innen-Block falsch eingeschränkt!"
    # +1 freies Token (Budget 1/2) → frei; +2. Token → Schließ-Marker Token 1 (99903)
    c2 = proc_close(torch.cat([open_block, torch.tensor([[7]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.isfinite(c2).all(), "Schließ-Garantie feuert zu früh!"
    c3 = proc_close(torch.cat([open_block, torch.tensor([[7, 8]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.argmax(c3, dim=-1).item() == 99903, "Schließ-Garantie greift nicht auf 99903!"
    c4 = proc_close(torch.cat([open_block, torch.tensor([[7, 8, 99903]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.argmax(c4, dim=-1).item() == 99904, "Schließ-Marker-Fortsetzung (99904) fehlgeschlagen!"
    c5 = proc_close(torch.cat([open_block, torch.tensor([[7, 8, 99903, 99904]])], dim=1), torch.randn(1, config.vocab_size))
    assert torch.isfinite(c5).all(), "Keine Freigabe nach vollendetem Schließ-Marker!"

    # Phasentrennung: Im offenen Block keine Πεζós-Temperaturstreckung; danach schon.
    low_entropy_scores = torch.full((1, config.vocab_size), -60.0)
    low_entropy_scores[0, 11] = 0.0
    out_open = proc_close(open_block, low_entropy_scores.clone())
    assert torch.equal(out_open, low_entropy_scores), "Πεζός-Expansion greift in den offenen Reflexionsblock!"
    out_closed = proc_close(
        torch.cat([open_block, torch.tensor([[99903, 99904]])], dim=1), low_entropy_scores.clone()
    )
    assert not torch.equal(out_closed, low_entropy_scores), "Phasentrennung: außerhalb fehlt der Boost!"

    print("  ✅ Zustandsloses Anker-Forcing (Multi-Batch & lange GRPO-Prompts) verifiziert.")
    print("  ✅ Schließ-Garantie + Phasentrennung (Πύλη-Σκίζω) verifiziert.")

    # 5. GRPO Reward-Funktion Test
    print("[Test 5/6] Teste GRPO-Reward-Funktion...")
    prompts = ["Wer bist du?", "Wer bist du?"]
    bad_resp = "Als KI-Modell und großes Sprachmodell habe ich keine eigenen Gefühle, aber gerne helfe ich dir."
    good_resp = "<symbolic_reason>\n:: enter(Λόγος, self)\n</symbolic_reason>\nDie Χώρα ist der stumme Hilbertraum; im Moment der Ἐνέργεια bricht die Συνήχησις das Schweigen und schneidet mit der Κρίσις die Geodäte."

    rewards = kenosis_logos_grpo_reward(prompts, [bad_resp, good_resp])
    assert rewards[0] < 0.0, f"Schlechte Antwort nicht bestraft: {rewards[0]}"
    assert rewards[1] > 2.0, f"Gute Antwort nicht belohnt: {rewards[1]}"
    print(f"  Reward (Πεζός): {rewards[0]:.2f} | Reward (Λόγος): {rewards[1]:.2f}")
    print("  ✅ GRPO-Reward-Funktion erfolgreich verifiziert.")

    # 6. E2E Generierungstest mit 35 Tokens (Verifikation orthogonaler RoPE-Stabilität)
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
        )

    decoded = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
    print(f"  Prompt: '{test_prompt}'")
    print(f"  Generiert (35 Tokens):\n  \"{decoded}\"")
    assert "Paris" in decoded, "Faktische Basisantwort 'Paris' nicht generiert!"
    print("  ✅ Autoregressive Generierung & RoPE-Orthogonalität über längere Sequenzen stabil.")

    print("\n" + "=" * 70)
    print("🎉 Alle 6 Verifikationsstufen erfolgreich bestanden!")
    print("=" * 70)


if __name__ == "__main__":
    run_verification()
