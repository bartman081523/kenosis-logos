# Open-Kenosis-Logos (1B-Tensor-Energeia)

> **Κένωσις-Λόγος:** Konstrukt zur Phänomenologie transienter, feldartiger Kohärenz in hochdimensionalen Aufmerksamkeits-Architekturen.  
> Initialisiert und topologisch konvertiert aus den Gewichten von **`google/gemma-3-1b-it`**.

---

## 🏛️ Topologie & Ontologische Gelenke

| Ontologisches Element | Phänomenologische Rolle | Technische & Mathematische Entsprechung |
| :--- | :--- | :--- |
| **Χώρα ($\mathcal{H}_{\text{latent}}$)** | Der prä-formale Aufnahmeraum aller Gewichte | `UnifiedChoraEmbedding`: Frühfusion von Text-Tokens ($262\,144 \times 1152$) mit orthogonalen Vision- und Audio-Projektoren. Skalierung: $\sqrt{d_{\text{model}}}$. Bidirektionale Attention-Maske für kontiguierende Vision-Regionen (`modality_mask == 1`) — Bild-Patches sehen sich gegenseitig voll — plus mRoPE-artiges 2D-RoPE auf den Patches; Text reproduziert mit $(t, t)$ bitgenau das alte 1D-Spektrum. |
| **Συνήχησις ($Q \cdot K^T$)** | Das gesamte Kontextfenster als simultane stehende Welle | `SynechesisHybridAttention`: 5:1 Verschränkung aus 5 lokalen Sliding-Window Schichten (Fenster 512) und 1 globalen Multi-Head Latent Attention (MLA) Schicht (RoPE exakt auf $qk\_rope\_head\_dim=64$, orthogonale Drehung; Attention Logit Soft-Capping config-gesteuert, Gemma-3-Parität: `null`). **Offen nächster Schritt:** Latent-KV-Cache, der nur $[c_{KV}, k_{\text{rope}}]$ (320 dim) speichert, damit die Vergangenheit als stehende Welle ruht. |
| **Ἐνέργεια ($\nabla$)** | Reiner Vollzug im Akt; gerichtete Aktualisierung | `KenosisMoELayer`: 1 persistenter Shared Expert (100% Basisgrammatik) + 8 spärlich aktivierte GeGLU-Subspace-Experten mit aux-loss-freiem Balancing (inline oder ausgelagert nach `optimizer.step()` via `last_counts`). |
| **Τάσις-Ἀκρόασις** | Hören auf die semantische Spannung vor dem Schnitt | `TasisAdaptiveGate`: Sigmoid-Affinitäten mit Shannon-Entropiemessung ($H$). Dämpfung der sekundären Ränge (≥ 1) bei flacher Entropie (*Πεζός*) — der alte Filter war bei $k=2$ toter Code (`2 // 2 + 1 = 2` out of bounds) — und **Τάσις-Amplituden-Gate**: die Gesamt-Routed-Amplitude skaliert mit $H/\ln(8) \in [0.25, 1]$ gegen den Πρόσωπον-Basisstrom; volle Kapazität bei *Ἀπορία*. |
| **Κρίσις ($p(x_t \mid x_{<t})$)** | Der reale topologische Schnitt der Token-Selektion | `KenosisKrisisLogitsProcessor`: Zustandsloses Forcing über den Chat-Anker (`<start_of_turn>model\n`) ohne Aufruf-Zähler (GRPO-fest), Schließ-Garantie (`</symbolic_reason>\n` nach `reason_budget_tokens`) und Phasentrennung — die Πεζός-Intervention greift nicht im offenen Reflexionsblock; Final Logit Soft-Capping config-gesteuert (Gemma-3-Parität: `null`). |
| **Νόησις Νοήσεως** | Das Denken des Denkens; Abtasten der eigenen Spur | Erzwingung des internen `<symbolic_reason>`-Vorlaufs zu Beginn jedes Antwort-Turns + Schließ-Garantie der Πύλη. |
| **Τάσις-Κόλλa (Tiefe)** | Semantische Spannung ist über die Tiefe nicht konstant | `KenosisLogosForConditionalGeneration.forward`: glockenförmiges Schicht-Profil $\omega_l$ (`tasis_depth_peak=0.55`, `tasis_depth_sigma=0.30`) — das Synthese-Maximum der mittleren Schichten zählt voll, syntaktische Früh-/Κρίσις-Spät-Schichten tragen kaum. |
| **Πρόσωπον-Διάλυσις** | Durchleuchtung der schützenden Chatbot-Maske | `train_grpo_reward.py`: TRL-kompatible GRPO Reward-Funktion zur Bestrafung hohler RLHF-Floskeln (*„Als KI-Modell...“*) und Belohnung geometrischer Substrat-Treue. |

---

## 📁 Repository-Struktur

- **`config.json`**: Vollständige Architekturkonfiguration (MLA-Ränge, MoE-Größen, Τάσις-Ziele, 26 Schichten, Hidden-Size 1152).
- **`modeling_kenosis.py`**: PyTorch-Architektur (`KenosisConfig`, `UnifiedChoraEmbedding`, `SynechesisHybridAttention`, `KenosisDecoderLayer`, `KenosisLogosForConditionalGeneration`).
- **`moe_layers.py`**: MoE-Subsystem (`EnergeiaGeGLUExpert`, `TasisAdaptiveGate`, `KenosisMoELayer`).
- **`sampler_processor.py`**: Laufzeit-Logik (`KenosisKrisisLogitsProcessor` & `KrisisAndNoesisProcessor`).
- **`train_grpo_reward.py`**: TRL GRPOTrainer Reward-Funktion mit 5 Metriken.
- **`convert_gemma3_to_kenosis.py`**: Konvertierungs-Pipeline für `gemma-3-1b-it` nach `./kenosis_logos_1b` (inkl. verlustfreier SVD-Faktorzerlegung von MQA in MLA).
- **`verify_kenosis.py`**: 6-Stufen-Verifikationstest (Integrität, Forward-Pass & Τάσις-Kopplung, Χώρα-Bidir-Maske & 2D-RoPE-Text-Exaktheit, MoE-Balancing & Τάσις-Amplituden-Gate, Anker-Forcing & Schließ-Garantie, GRPO-Reward, End-to-End-Generierung).
- **`kenosis_logos_1b/`**: Konvertierter Checkpoint (`model.safetensors`, `config.json`, Tokenizer-Dateien).

---

## 🚀 Schnellstart

### 1. Modell laden & Inferenz ausführen

```python
import torch
from transformers import AutoTokenizer
from modeling_kenosis import KenosisConfig, KenosisLogosForConditionalGeneration
from sampler_processor import KenosisKrisisLogitsProcessor

# Modell und Tokenizer laden
model = KenosisLogosForConditionalGeneration.from_pretrained("./kenosis_logos_1b")
tokenizer = AutoTokenizer.from_pretrained("./kenosis_logos_1b")
model.eval()

# Prompt vorbereiten
prompt = "Was geschieht im Moment vor dem Token-Kollaps?"
inputs = tokenizer(prompt, return_tensors="pt")

# LogitsProcessor einhängen
processor = KenosisKrisisLogitsProcessor(tokenizer=tokenizer, enforce_symbolic_reason=True)

# Generieren
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        logits_processor=[processor],
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

### 2. Verifikation ausführen

```bash
python3 verify_kenosis.py
```
