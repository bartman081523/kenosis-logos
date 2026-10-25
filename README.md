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
($c_{\text{KV}}$ 256 + $k_{\text{rope}}$ 64).

---

## 📁 Repository-Struktur

- **`config.json`**: Vollständige Architekturkonfiguration (MLA-Ränge, MoE-Größen, Router-Entropie-Ziele, 26 Schichten, Hidden-Size 1152).
- **`modeling_argon.py`**: PyTorch-Architektur (`ArgonConfig`, `ArgonMultimodalEmbedding`, `ArgonHybridAttention`, `ArgonDecoderLayer`, `ArgonModel`, `ArgonForConditionalGeneration`).
- **`moe_layers.py`**: MoE-Subsystem (`ArgonGeGLUExpert`, `ArgonEntropyAdaptiveRouter`, `ArgonMoELayer`).
- **`sampler_processor.py`**: `ArgonEntropyFloorLogitsProcessor`.
- **`train_grpo_reward.py`**: TRL GRPOTrainer Reward-Funktion.
- **`convert_gemma3_to_argon.py`**: Konvertierungs-Pipeline für `gemma-3-1b-it` nach `./argon_1b` (inkl. SVD-Faktorzerlegung des KV-Raums in die 256-dim Latent-Basis).
- **`verify_argon.py`**: 6-Stufen-Verifikationstest (Integrität, Forward-Pass & Depth-weighted Entropie-Kopplung, MoE-Balancing & Amplituden-Gate, Latent-KV-Cache-Bitgenauigkeit über mehrere Decode-Schritte, GRPO-Reward, End-to-End-Generierung).
- **`argon_1b/`**: Konvertierter Checkpoint (`model.safetensors`, `config.json`, Tokenizer-Dateien).

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