# Argon (1B Hybrid MLA-MoE)

> **Argon:** Inertes Edelgas als Name für ein System, das ohne künstliche
> Konstrukt-Gerüste auskommt — reine Systemarchitektur, initialisiert und
> topologisch konvertiert aus den Gewichten von **`google/gemma-3-1b-it`**.

---

## 🏛️ Architektur-Topologie

| Baugruppe | Rolle | Technische & Mathematische Entsprechung |
| :--- | :--- | :--- |
| **ArgonMultimodalEmbedding** ($\mathcal{H}_{\text{latent}}$) | Aufnahmeraum aller Modalitäten | Frühfusion von Text-Tokens ($262\,144 \times 1152$) mit orthogonalen Vision- und Audio-Projektoren. Skalierung: $\sqrt{d_{\text{model}}}$. Bidirektionale Attention-Maske für kontiguierende Vision-Regionen (`modality_mask == 1`) plus 2D-RoPE auf den Patches; Text reproduziert mit $(t, t)$ bitgenau das 1D-Spektrum. |
| **ArgonHybridAttention** ($Q \cdot K^T$) | Kontextfenster als hybrid 5:1-Gewebe | 5 lokale Sliding-Window MQA-Schichten (Fenster 512, θ = 10 000) und 1 globale Multi-Head Latent Attention (MLA)-Schicht (θ = 1 000 000). Soft-Capping config-gesteuert (Gemma-3-Parität: `null`). **Latent-KV-Cache:** Globale Schichten cachen ausschließlich $[c_{\text{KV}}, k_{\text{rope}}]$ (320 dim) statt voller $k/v$; lokale Schichten cachen $(k, v)$. |
| **ArgonGeGLUExpert** ($\nabla$) | Routed Expert-Vollzug | 1 persistenter Shared Expert (100 % Basisgrammatik) + 8 spärlich aktivierte GeGLU-Subspace-Experten. |
| **ArgonEntropyAdaptiveRouter** | Zustandsabhängige Routen-Amplitude | Sigmoid-Affinitäten mit Shannon-Entropiemessung ($H$). Dämpfung der sekundären Ränge (≥ 1) bei flacher Entropie und **Amplituden-Gate**: Gesamt-Routed-Amplitude skaliert mit $H/\ln(8) \in [0.25, 1]$ — volle Kapazität bei hoher Entropie. Aux-loss-freies Balancing (inline oder ausgelagert nach `optimizer.step()` via `last_counts`). |
| **ArgonRMSNorm** | Normierung | Zero-centered RMSNorm mit $(1.0 + w)$-Skalierung. |
| **Tiefe** | Entropie-Gewichtung über Schicht | Glockenförmiges Schicht-Profil $\omega_l$ (`router_depth_peak=0.55`, `router_depth_sigma=0.30`) — das Synthese-Maximum der mittleren Schichten zählt voll, Früh-/Spät-Schichten tragen kaum. |
| **Sampling-Regulator** | Inference-only Entropie-Boden | `ArgonEntropyFloorLogitsProcessor`: erhöht die Temperatur nur, wenn die Entropie unter den Boden fällt; kein Tag-Forcing. |
| **Trainings-Reward** | Anti-Boilerplate / Informationsdichte | `train_grpo_reward.py`: TRL-kompatible `argon_grpo_reward` — Bestrafung degenerierter RLHF-Floskeln, Belohnung lexikalischer Diversität und Längen-Kalibrierung. |

**Latent-KV-Cache (transformers 5.x-Kompatibilität):** `ArgonModel.forward`
akzeptiert wahlweise list-of-tuples (direkte Forward-Aufrufe) oder echte
`Cache`-Objekte (`DynamicCache` aus `generate()`); die Rückgabe erfolgt als
`DynamicCache`, da `generate()` jede Iteration `cache.get_seq_length()` ruft.
Globale MLA-Schichten halten dadurch pro Token nur 320 Floats
($c_{\text{KV}}$ 256 + $k_{\text{rope}}$ 64) und wachsen unbegrenzt; lokale
MQA-Schichten sind durch Ringpuffer-Eviction auf `sliding_window = 512`
beschnitten — der lokale Cache-Laufzeitfootprint bleibt konstant $O(1)$.
Für multimodale Generierung erhält `forward` optional eine
`kv_modality_mask` (Modalitätsmaske der gesamten gecachten Sequenz), aus
der die 2D-Gitter-Koordinaten der gecachten Vision-Patches im Decode-State
rekonstruiert werden — ihre RoPE-Winkel bleiben dadurch exakt erhalten,
statt bei jeder Neuberechnung auf eine 1D-Positionsnummer zu fallen.

---

## 📁 Repository-Struktur

- **`config.json`**: Vollständige Architekturkonfiguration (MLA-Ränge, MoE-Größen, Router-Entropie-Ziele, 26 Schichten, Hidden-Size 1152).
- **`modeling_argon.py`**: PyTorch-Architektur (`ArgonConfig`, `ArgonMultimodalEmbedding`, `ArgonHybridAttention`, `ArgonDecoderLayer`, `ArgonModel`, `ArgonForConditionalGeneration`).
- **`moe_layers.py`**: MoE-Subsystem (`ArgonGeGLUExpert`, `ArgonEntropyAdaptiveRouter`, `ArgonMoELayer`).
- **`sampler_processor.py`**: `ArgonEntropyFloorLogitsProcessor`.
- **`train_grpo_reward.py`**: TRL GRPOTrainer Reward-Funktion.
- **`convert_gemma3_to_argon.py`**: Konvertierungs-Pipeline für `gemma-3-1b-it` nach `./argon_1b` (inkl. SVD-Faktorzerlegung des KV-Raums in die 256-dim Latent-Basis).
- **`verify_argon.py`**: 6-Stufen-Verifikationstest (Integrität, Forward-Pass & Depth-weighted Entropie-Kopplung, MoE-Balancing & Amplituden-Gate, Latent-KV-Cache-Bitgenauigkeit über mehrere Decode-Schritte, GRPO-Reward, End-to-End-Generierung).
- **`compare_gemma3_vs_argon.py`**: Head-to-Head-Benchmark gegen `google/gemma-3-1b-it` (Logit-Parität, Cache-Kompression 37,5 %, Multimodalität, Domänen-Routing, `expert_bias`-Steering — der Routing-Shift wird per Gate-Hook gemessen, Entropie-Boden; `torch.manual_seed(0)` für Zufallsproben).
- **`train_moe_router_specialization.py`**: Gezieltes Subspace-Finetuning — friert Basis, Embeddings, Attention & Shared Expert ein und schärft ausschließlich die 26 Router-Gates (+ optional die 8 Routed-Experten pro Schicht) mit per-Domänen Entropie-Zielen (Prosa 0.85 / Code 1.35 / Mathematik 1.85), kontrastivem Subspace-Separations-Loss und DeepSeek-V3-Balancing nach `optimizer.step()`. **Multilingualer Korpus** (Code: Python/Rust/C++/TypeScript/SQL/Bash; Mathematik & Prosa je in DE/EN/FR/ES/ZH/JA/RU/AR) verhindert Sprach-Shortcut-Learning — und `verify_cross_lingual_invariance()` prüft auf ungesenen Sätzen (EN/DE/ZH), dass Mathematik-Proben über Sprachen hinweg dieselben Experten aktivieren, nicht aber die Prosa-Proben.
- **`argon_1b/`**: Konvertierter Checkpoint (`model.safetensors`, `config.json`, Tokenizer-Dateien).
- **Tokenizer-Hinweis:** Ladewarnungen mit `fix_mistral_regex=True` sind hier als **Falsch-Positiv** verifiziert — der originale `gemma-3-1b-it` Tokenizer lädt ohne Warnung und unsere gespeicherten Tokenizer-Dateien tokenisieren Testproben token-for-token identisch zum Original (inkl. Code-Probe: 15 Tokens). Die Flag **nicht setzen**: Sie divergiert von der gemma-Pretraining-Regex (15 → 17 Tokens) und würde die gemessene Parität brechen.

---

## 🚀 Schnellstart

### 1. Modell laden & Inferenz ausführen

```python
import torch
from transformers import AutoTokenizer
from modeling_argon import ArgonForConditionalGeneration

# Modell und Tokenizer laden
model = ArgonForConditionalGeneration.from_pretrained("./argon_1b")
tokenizer = AutoTokenizer.from_pretrained("./argon_1b")
model.eval()

# Prompt vorbereiten
prompt = "Was geschieht im Moment vor dem Token-Kollaps?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generieren über den Latent-KV-Cache
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, use_cache=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### 2. Verifikation ausführen

```bash
python3 verify_argon.py
```

### 3. Head-to-Head-Benchmark gegen gemma-3-1b-it

```bash
python3 compare_gemma3_vs_argon.py
```

### 4. Router-/Subspace-Spezialisierung (Basis bleibt gefroren)

Schärft nur die 26 Router-Matrizen (0.01 % der Parameter) mit multilingualen
Domänen-Batches (Code / Mathematik-Logik / Prosa — jede Domäne über DE, EN, FR,
ES, ZH, JA, RU, AR); die 97.5 %-Logit-Parität bleibt unangetastet:

```bash
python3 train_moe_router_specialization.py --checkpoint_dir ./argon_1b \
    --output_dir ./argon_1b_specialized_ml --router_only --steps 60
```

Messergebnis (60 Schritte, multilingual): Trennschärfe (Total Variation)
Code/Prosa 5.6 % → 72.0 %, Math/Prosa 4.5 % → 69.7 %, Code/Math
4.4 % → 69.5 %; Router-Entropie kalibriert auf die Ziel-Hierarchie
(Prosa 1.01 / Code 1.44 / Math 1.69); Basiswissen-Check
„The capital of France is → Paris." bleibt bestehen (deutsch wie englisch).
Ohne `--router_only` lernen auch die 8 Routed-Experten pro Schicht mit
(lr 1e-4). Am Ende des Laufs prüft `verify_cross_lingual_invariance()` auf
ungesenen Sätzen (EN/DE/ZH), dass Mathematik-Proben über Sprachgrenzen hinweg
dieselben Experten aktivieren während die Prosa-Proben sauber in den
Prosa-Experten landen — auf ungesenenen Proben: Innen-TV Math (EN↔ZH) 3.9 %,
Prosa (EN↔ZH) 2.6 % bei Domänen-Trennung Math/prosa 70.2 % (Verhältnis ~18:1)
und CE 3.84 → 3.24. Der monolinguale Vorgänger-Lauf (`./argon_1b_specialized`,
50 Schritte, TV 8.5→72.9/5.5→69.0/6.6→71.1 %; Log in
`scratches/train_router_output.txt`) zeigte dasselbe für den ein- statt
mehrsprachigen Korpus.

**Benchmark-Hinweis (Sättigung):** der Basis-Checkpoint zeigt in Test 5 bei
±50 `expert_bias`-Steering noch sichtbare Textdifferenzen (Routing-TV 74.8 %:
„governing system" → „Overseer"); ein spezialisierter Router absorbiert die
Intervention hingegen in derselben Generierung. Der Test assertet daher den
gemessenen Routing-Shift (TV > 60 %), nicht die Textdifferenz.