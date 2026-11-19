# Port der Provenienz-Achsen (MI-9) nach Gemma 3, Gemma 4 und nanoGPT

**Ziel.** Die in `KenosisLogos` isolierte Unterscheidung *Kontext / Training /
eigene Gedanken* (MI-9) in die Standard-Implementierungen tragen und dort
messen — nicht nur in Argon.

## Repos (geklont, unverändert bis auf die Erweiterung)

| Repo | Commit | Was |
| :--- | :--- | :--- |
| `google-deepmind/gemma` | `9e1f741` | offizielle Implementierung; CHANGELOG 4.1.0 — **ein Repo für Gemma 3 UND Gemma 4** |
| `karpathy/nanoGPT` | `3adf61e` | Referenz-GPT, PyTorch |

Kein Training: der nanoGPT-Checkpoint ist **OpenAI GPT-2 (124M)**, geholt über
`nanogpt/fetch_gpt2_ckpt.py` (HF-Transponierung `gpt2`, exakt der Pfad, den
nanoGPT mit `--init_from=gpt2` selbst benutzt). 475 MB, `iter_num = -1`.

## Die Erweiterung

| Datei | Backend | Schichtliste |
| :--- | :--- | :--- |
| `gemma/provenance_axes.py` | HF-`transformers` (Gemma 3 + 4, inkl. multimodaler Config) | `model.layers` bzw. `model.language_model.layers` (automatisch) |
| `nanogpt/provenance_axes.py` | nanoGPT `GPT` | `transformer.h` |

Identischer Vertrag in beiden:

- **`d_ctx`** — „steht der Beleg im Kontext?" Kontrast **C vs I**: derselbe
  Code, derselbe Satz, nur die Box-Nummer im Präfix ist eine andere
  (Unterschied: **ein Token**).
- **`d_s`** — „ist der VorText mein eigener?" Kontrast **model:** vs **user:**
  vor demselben Passagen-String.
- **W** — realer Fakt-Slot mit neutralem Präfix (Antwort aus dem Gewicht).
- Readout: 3 Klassen W/C/S im 2D-Raum (`d_ctx`, `d_s`), LOO, Permutation.
- Sieben Bänder (M1 Kontext-Achse, M2 Relevanz, M3 Gewicht, M4 eigene,
  M5 Achsen getrennt, M6 Integration, M7 Vertrautheit).

CLI in beiden gleich:

```bash
# gemma
cd ProvenancePorts/gemma
python -m provenance_axes calibrate --model <snapshot> --local --device cpu
python -m provenance_axes read --model <snapshot> --local --calib \
    calibrations/google_gemma-3-1b-it.json --text "question: the capital of france is"

# nanogpt
cd ProvenancePorts/nanogpt
python fetch_gpt2_ckpt.py                 # holt GPT-2, trainiert nichts
python provenance_axes.py calibrate
python provenance_axes.py read --text "question: the capital of france is"
```

## Ergebnis (CPU, ein Layer = 0.8·Tiefe, BOS wo vorhanden)

| Substrat | Layer | C/I (LOO) | S (LOO) | 3 Klassen | cos(d_ctx,d_s) | M2 Relevanz | M3 Gewicht | Bänder |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | 20 / 26 | **1.000** (p .009) | 0.833 (p .188) | 0.864 | **−0.018** | 8/8 C, 4/8 I | 4/8 | 4/6 |
| Gemma 4 E2B IT | 27 / 35 | **1.000** (p .009) | **1.000** (p .059) | **1.000** | **−0.084** | 7/8 C, 7/8 I | 8/8 | **6/6** |
| nanoGPT / GPT-2 124M | 9 / 12 | **1.000** (p .009) | **1.000** (p .059) | 0.955 | **−0.054** | 8/8 C, **0/8 I** | **0/8** | 4/6 |

## Lesart

1. **Die zwei Achsen sind architektur-unabhängig.** Der Kontext-Kontrast
   trennt in allen drei Implementierungen mit 1.000 (p .009); die
   Eigen-Achse mit 0.833–1.000; und `cos(d_ctx, d_s)` liegt in allen drei
   zwischen **−0.018 und −0.084**, also praktisch orthogonal. Das ist der
   portable Kern: *Kontext* und *eigene Gedanken* sind zwei getrennte Achsen,
   nicht eine Kategorie — unabhängig davon, ob das Modell in JAX, in einer
   multimodalen HF-Konfiguration oder als 12-Schicht-nanoGPT gebaut ist.
2. **Die behavioralen Bänder sind Modell-Fähigkeit, nicht Gerät.**
   - Gemma 4 erfüllt alle sechs: es kopiert den Code nur bei relevanter Box
     (7/8 vs 7/8) und beantwortet den Fakt-Slot (8/8).
   - Gemma 3 1B ist schwächer: Relevanz 4/8, Gewicht 4/8.
   - GPT-2 124M ist ein *Salienz*-Kopierer: p(Code) 0.31–0.77 auch bei
     falscher Box (0/8 ≤ 0.10), und es weiß den Fakt nicht: p(" Berlin")
     ist schon am nackten Slot **0.058**. Beides sind Eigenschaften des
     Checkpoints, keine Messfehler.
3. **Die Eigen-Achse braucht keine Rollen-Semantik.** GPT-2 hat keine
   Chat-Rollen, trotzdem ist der `model:`/`user:`-Marker mit 1.000
   dekodierbar. Das bestätigt MI-8: die Achse ist **marker-getragen**, nicht
   bedeutungsgetragen. Ein Modell „weiß" nicht, wer gesprochen hat; es trägt
   den Marker.

## Grenzen (explizit)

- Ein Layer pro Modell (0.8·Tiefe), n = 6–8 Paare, textueller Marker statt
  Chat-Template, CPU. Keine Kausal-Intervention in diesem Port (nur die
  Input-Kontraste, die per Konstruktion kausal sind).
- Der **JAX-Pfad** des `gemma`-Repos (`gemma/gm`, `capture_intermediates`)
  bleibt unbenutzt: die Gewichte liegen im HF-Format vor, Kaggle-Checkpoints
  sind hier nicht verfügbar. Die Sonde greift am Residualstrom der
  Decoder-Schichten — dieselbe physikalische Stelle, andere Laufzeit.
- Kein Trainings-Claim. Der nanoGPT-Port misst einen fremden Checkpoint,
  kein selbst trainiertes Modell.

## Reproduktion

```bash
cd /run/media/julian/ML6/ProvenancePorts
# Gemma 3
cd gemma && python -m provenance_axes calibrate --local --model \
  ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/<sha>
# Gemma 4
python -m provenance_axes calibrate --local --model \
  ~/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/<sha>
# nanoGPT (GPT-2, kein Training)
cd ../nanogpt && python fetch_gpt2_ckpt.py && python provenance_axes.py calibrate
```


---

## Nachtrag: der einpassige Indikator (MI-11)

Beide Ports haben jetzt zusätzlich:

- `fam_axis()` — Vertrautheits-Achse (12 echte vs 12 erfundene Länder,
  identische Vorlage), LOO + Permutation, LDA-Schwelle.
- `flag()` — ein Forward, drei Achsen: `justified = Kontext-Beleg ODER
  Vertrautheit`; Ausgabe `{justified, reason, cls, score_ctx, score_fam,
  p_top}`.
- CLI: `flag --text "..."`.

| Substrat | Vertrautheit | Bänder | Flag-Demo |
| :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | acc 1.000, p .002, L0 | 5/7 | context / familiar / none |
| Gemma 4 E2B IT | acc 1.000, p .002, L0 | **7/7** | context / familiar / none |
| nanoGPT / GPT-2 | acc 1.000, p .002, L3 | 5/7 | context / familiar / none |

Demo (Gemma 3): Kontext-Prompt `ctx 1057` (Schwelle 688) → `context`;
`"Germany is a country. Its capital is"` `fam +0.60` → `familiar`;
`"Vrandel is a country. Its capital is"` `fam −1.80`, `p_top 0.10` →
`none`, **justified = False**.

**Zwei Port-Lehren.** (1) Die Kontext-Entscheidung muss auf der
1D-`d_ctx`-Achse laufen (Schwelle = Mitte W/C); der 3-Klassen-Centroid
mischt W/C/S und kippt bei Gemma 4. (2) Der interne Beleg-Leser ist an die
Prompt-Familie gebunden — ein anderes Wording ist ein anderes
Koordinatensystem (3/8 statt 8/8 in MI-10).

**Grenze.** Die Vertrautheits-Achse sitzt auf Schicht 0: sie misst
„bekanntes Wort", nicht „bekannter Fakt". Für erfundene Entitäten reicht
das; für seltene reale Entitäten bleibt MI-10 (Paraphrasen-Stabilität) die
Grundwahrheit.


## Nachtrag: Drei-Gruppen-Probe (bekannt / obskur-real / erfunden)

Auf Gemma 3 mit demselben Template:

```
Schicht 0:   K vs N 1.000   R vs N 0.875   K vs R 1.000
z-Mittel:    K 0.00         R −5.60        N −6.92
```

Anders als bei Tiny 2.1 (dort K vs R 0.63–0.68) sind die drei Gruppen auf
Gemma 3 **streng geordnet**: ein Vertrautheits-/Frequenz-Gradient, der schon
auf Schicht 0 sitzt. Die Grenze bleibt dieselbe — die Achse ordnet *Namen*,
sie zertifiziert nicht den *Fakt*.
