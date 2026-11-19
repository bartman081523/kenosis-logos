# Project specs, tools, build & conventions

> **Zweck.** Diese Datei trägt jeden projektspezifischen Fakt, den ein
> AI-Coding-Agent (oder menschlicher Beitragender) braucht. Sie wird
> **on demand vom Metaprompt in [`AGENTS.md`](../../AGENTS.md)** geladen —
> zusammen mit [`research.md`](research.md) (Methodik) und
> [`devmind.md`](devmind.md) (Arbeits-Haltung).
>
> **Evidenzregeln.** Jede Aussage hier ist belegt: Datei im Repo (als
> `file:line`), Git-Historie, CI-Config oder ein tatsächlich ausgeführtes
> Kommando. Dokumentierte Kommandos tragen einen Marker
> `<!-- verified YYYY-MM-DD: <Beobachtung> -->` oder
> `<!-- UNVERIFIED — source: <Fundstelle> -->`.

## Build Commands

Kein Build-Schritt: reines Python. Es gibt **kein** `setup.py`,
`pyproject.toml`, `requirements*.txt` oder `environment.yml` — verifiziert
per `ls` (2026-11-19). Nötig ist ein Interpreter mit `torch` und
`transformers`; der System-`python3` (3.14.10) hat sie **nicht**
(`ModuleNotFoundError: No module named 'transformers'`, verifiziert
2026-11-19). Verifizierte Toolchain: `torch 2.12.0+cu130`,
`transformers 5.13.0`, Python 3, `sbcl 2.6.9`.

```bash
# Architektur + Gewichte verifizieren: 6 Stufen, lädt den 4,8-GB-Checkpoint
python verify_argon.py

# Maschinen-Verifikation Stufen 1-3 (CPU, kein Modell, wenige Sekunden)
python machines/verify_machines.py --stages 1,2,3

# Stufe 4: Live-SBCL-Parität der Lisp-Brücke
python machines/verify_machines.py --stages 4

# Maschinen-Korpus deterministisch neu erzeugen (Seed 42)
PYTHONPATH=. python machines/build_machine_corpus.py

# Router-/Subspace-Spezialisierung (Basis bleibt eingefroren)
python train_moe_router_specialization.py --checkpoint_dir ./argon_1b \
    --output_dir ./argon_1b_specialized_ml --router_only --steps 60
```
<!-- verified 2026-11-19: verify_argon.py exit 0, 6/6 Stufen GRÜN; --stages 1,2,3 exit 0; --stages 4 exit 0 (SBCL); build_machine_corpus exit 0, sha256 unverändert -->
<!-- verified 2026-11-19: train_moe_router_specialization.py --help exit 0 -->
<!-- UNVERIFIED — source: README.md:99-103 (60-Schritte-Lauf), nicht in dieser Session wiederholt; Messwerte stammen aus dem dokumentierten Lauf -->

Der Trainingsbefehl braucht denselben Interpreter (`import torch`,
`from transformers import ...`). `PYTHONPATH=.` ist nur für Skripte nötig,
die `machines.*` absolut importieren (`machines/build_machine_corpus.py:42`).

## Specifications & references

- **`google/gemma-3-1b-it`** — Paritäts-Referenz. Argon entsteht per
  topologischer Konvertierung aus diesen Gewichten (`README.md:3-5`,
  `convert_gemma3_to_argon.py`). Gemessen: 97.5 % Logit-Parität
  (`README.md:95`).
- **`config.json`** — Architektur-Kontrakt: 26 Layer, `hidden_size` 1152,
  4 Attention-Heads, `head_dim` 256, `kv_lora_rank` 256,
  `sliding_window` 512 mit Muster 6 (5 lokal + 1 global),
  `n_shared_experts` 1, `n_routed_experts` 8, `num_experts_per_tok` 2.
- **`Topologische Initialisierung des Kenosis-Logos.md`** — Design-Dokument
  (720 Zeilen) zur Konvertierung und Architektur-Begründung.
- **`machines/lisp/vendor/`** — Lisp-Canon (u. a. `uni_241`,
  `geometric-tokenizer.lisp`), MIT, Autorenzeile Julian 2026-02
  (`machines/lisp/README.md:8-15`). Die Python-Seite zitiert den Canon in
  Docstrings mit `datei:Zeile` (`machines/machine_base.py:9-16`).
- **`machines/parity/python_fixtures.json`** — eingefrorene Goldwerte
  („Erstersperrung"), gegen die `verify_machines.py:63` prüft.
- **`machine_registry.json`** — gemessene Top-2-Experten je Maschinen-Domäne
  plus Korpus-`sha256` (`f6c0b5de…`).

## Linting / Formatting

Nichts ist erzwungen: kein `.editorconfig`, keine `ruff`/`black`/`flake8`-
Config, kein Lint-Job (es gibt kein CI). Verifiziert per `ls` und
`find . -maxdepth 2 -name '.github'` (2026-11-19). Stil wird per
Konvention gehalten (siehe unten), nicht per Tool.

## Coding Conventions

Beobachtet an Stichproben, nicht verordnet:

- **4 Leerzeichen, keine Tabs** — `grep -rlP "^\t" --include=*.py .` liefert
  nichts (2026-11-19).
- **Deutscher Modul-Docstring** mit Dateinamen und `=====`-Unterstrich:
  `moe_layers.py:1-6`, `verify_machines.py:1-30`.
- **Type Hints in Signaturen** — `modeling_argon.py` 11, `machines/machine_base.py`
  39, `train_moe_router_specialization.py` 24 Treffer für `-> `.
- **Canon-Zitate im Docstring** statt Prosa-Behauptung: `datei:Zeile` +
  wörtliche Formel (`machines/machine_base.py:9-16`).
- **Determinismus ist Pflicht**: `RANDOM_SEED = 42`
  (`machines/build_machine_corpus.py:50`), `MachineConfig(seed=42, ...)`
  (`machines/verify_machines.py:80`); dieselbe Erzeugung muss byte-identisch
  sein (`machines/build_machine_corpus.py:24-26`).
- **Exit-Codes im Modul-Docstring dokumentiert** und im Code eingehalten:
  `0` grün, `1` Fehlschlag, `3` SKIP ohne stillen False-Pass
  (`machines/verify_machines.py:5-8`).
- **Keine Lizenz-Header in `.py`-Dateien**; die MIT-Angabe lebt bei den
  Vendor-Lisp-Dateien (`machines/lisp/README.md:15`). Im Repo-Root liegt
  keine `LICENSE` (verifiziert 2026-11-19).

## Commit Message Style

Aus `git log` abgeleitet, nichts erzwungen: `Bereich: Gegenstand`
(deutsch), Bereich ist eine Runden-/Meilenstein-Kennung (`MI-8`, `SP17`)
oder ein Modulname (`repo2txt`). Roh-Läufe und Berichte werden getrennt
committet, Fehlläufe bleiben in der Historie.

```
MI-8: provenance-geometry device + embedding proof + raw runs (Erstlauf broken kept)
SP17: diverse-examples device + raw runs (Erstlauf broken kept, extern gestoppt Lauf 4 kept)
MI-10: hallucination device + raw runs (Rev-1 widerlegt kept, Rev-2..4)
repo2txt: Runden-Paket-Regime — geaendert + ausgefuehrt + Ergebnisse
MI-6-P4 Berichte+Konkatenat: die Bänder zünden — exit 0 GRÜN
```

## Common Workflows

1. **Vor jeder Änderung an Modell oder Maschinen verifizieren**
   `python verify_argon.py` und `python machines/verify_machines.py
   --stages 1,2,3` — beide exit 0 als Baseline.
2. **Eine Maschine prüfen („ein Test")**
   `python machines/verify_machines.py --stages 3` (Kanon-Parität) bzw.
   `--stages 1` (Mini-Smoke).
3. **Experiment-Runde (Hausordnung)**
   Gerät als `scratches/smoke_<kennung>_*.py`, Roh-Ergebnis als `.log`/`.json`
   daneben, Auswertung in `scratches/*_bericht.md` /
   `laufprotokolle_befund_report.md`; anschließend `repo2txt`-Konkatenat.
   Als getrennte Commits: erst Gerät+Roh, dann Berichte.
4. **Router schärfen**
   `train_moe_router_specialization.py --router_only --steps 60`, danach
   `verify_cross_lingual_invariance()` im Lauf beobachten (README.md:96-99).
5. **Maschinen-Presets erzeugen**
   `--corpus all --router_only --emit_presets` schreibt
   `machine_presets/{goedel,turing,wittgenstein,combined}.json`
   (`train_moe_router_specialization.py:641`).
6. **Korpus regenerieren und Determinismus prüfen**
   `PYTHONPATH=. python machines/build_machine_corpus.py` → sha256 muss
   `f6c0b5de7c96…` bleiben (verifiziert 2026-11-19).

## Architecture Overview

1. **Embedding** — `ArgonMultimodalEmbedding`: Text-Tokens (262 144 × 1152),
   Vision-/Audio-Projektoren orthogonal, Skalierung `sqrt(d_model)`,
   `modality_mask` für Vision-Attention (`modeling_argon.py`).
2. **26 Hybrid-Schichten** — Muster 6: fünf lokale MQA-Schichten
   (Fenster 512, θ=10 000) plus eine globale MLA-Schicht (θ=1 000 000);
   globaler Cache nur `[c_KV (256), k_rope (64)]` = 320 Floats/Token.
3. **MoE** — 1 Shared Expert + Top-2 aus 8 GeGLU-Experten; Router mit
   Sigmoid-Affinitäten, Shannon-Entropie und Amplituden-Gate
   `H/ln(8) ∈ [0.25, 1]` (`moe_layers.py`).
4. **Norm/Kopf** — zero-centered RMSNorm `(1+w)`, `tie_word_embeddings`,
   `lm_head` an die Embedding-Matrix gebunden (`config.json`).
5. **Inferenz-Regler** — `ArgonEntropyFloorLogitsProcessor` hebt nur die
   Temperatur, wenn die Entropie unter den Boden fällt
   (`sampler_processor.py`).
6. **Maschinen-Ebene** — Gödel/Turing/Wittgenstein als Ports über derselben
   Canon-Gap-Mechanik, mit `ArgonOracle` für Router-Steering
   (`machines/machine_base.py`).

| Datei | Rolle |
| :--- | :--- |
| `modeling_argon.py` | Architektur (`ArgonConfig`, Attention, Decoder, `ArgonForConditionalGeneration`) |
| `moe_layers.py` | Experten, Entropie-Router, MoE-Layer |
| `sampler_processor.py` | Entropie-Boden im Sampling |
| `train_grpo_reward.py` | TRL-kompatible Reward-Funktion |
| `convert_gemma3_to_argon.py` | Konvertierung `gemma-3-1b-it` → `./argon_1b` |
| `train_moe_router_specialization.py` | Router-/Subspace-Finetuning, Preset-Emission |
| `verify_argon.py` | 6-Stufen-Verifikation des Modells |
| `compare_gemma3_vs_argon.py` | Head-to-Head-Benchmark gegen Gemma-3 |
| `machines/` | Götterdreiheit-Ports, Oracle, Korpus, Lisp-Brücke |
| `scratches/` | Geräte, Roh-Läufe, Berichte, Konkatenate |

## Testing

Es gibt **kein** `pytest`/`unittest`-Suite; die Verifikationsskripte sind
das Test-Regime (verifiziert: `grep -rln "import pytest"` leer, 2026-11-19).

- **Modell:** `python verify_argon.py` — 6 nummerierte Stufen mit
  PASS/FAIL, Exit 0 nur bei vollständigem Durchlauf (`verify_argon.py:1-15`).
- **Maschinen:** `python machines/verify_machines.py --stages <liste>` —
  Stufen 1-6; Exit `0` grün, `1` Fehlschlag, `3` nur bei SBCL-SKIP
  (`machines/verify_machines.py:5-8`). Einzelfall = eine Stufe, z. B.
  `--stages 3`.
- **Golden Files:** `machines/parity/python_fixtures.json` wird nicht
  automatisch fortgeschrieben. Ändern nur als bewusste Neu-Erstersperrung,
  mit dem geänderten Kanon als Begründung im Commit.
- **Determinismus:** Korpus-Erzeugung ist byte-identisch
  (`build_machine_corpus.py` meldet `sha256`; verifiziert 2026-11-19).

## CI

Keine. Es existiert kein `.github/`, keine `.gitlab-ci.yml`, kein
`tox.ini`/`noxfile.py` (verifiziert 2026-11-19). CI-Ersatz ist der
Verifikationslauf aus „Build Commands" plus die Commit-Hausordnung
(Gerät+Roh getrennt von Berichten, Fehlläufe bleiben sichtbar).

## Runtime Configuration / Usage

- **Checkpoint-Verzeichnisse** (relativ zum Repo-Root):
  `./argon_1b` (Basis, 4,8 GB `model.safetensors`, gitignored),
  `./argon_1b_specialized{,_ml}` (Router-Diff),
  `./argon_1b_machines` (Maschinen-Router + `machine_presets/`).
- **CLI-Flags** (`--help` verifiziert 2026-11-19):
  `train_moe_router_specialization.py`: `--checkpoint_dir`, `--output_dir`,
  `--steps` (60), `--max_length` (128), `--lr_router` (3e-3),
  `--lr_experts` (1e-4), `--router_only`, `--corpus {builtin,machines,all}`,
  `--save_mode {router_diff,full}`, `--emit_presets`;
  `machines/verify_machines.py`: `--checkpoint_base`, `--checkpoint_machines`,
  `--stages`, `--skip_gpu`;
  `convert_gemma3_to_argon.py`: `--model_id`, `--output_dir`, `--env_file`,
  `--routed_scale`.
- **Umgebung:** `HF_TOKEN` für gated Downloads
  (`convert_gemma3_to_argon.py:201`, `compare_gemma3_vs_argon.py:36`);
  `--env_file` lädt eine `.env` für den Token.
- **Tokenizer-Falle:** `fix_mistral_regex=True` **nicht** setzen — die
  gespeicherten Tokenizer-Dateien sind token-for-token identisch zum
  Original, die Flag bricht die Parität (15 → 17 Tokens, `README.md:50`).
- **SBCL** wird nur für Stufe 4/5 der Maschinen-Verifikation und die
  Lisp-Brücke gebraucht (`sbcl --script machines/lisp/run-bridge.lisp`,
  kein Quicklisp, kein Netz — `README.md:180-186`).

## Dependencies

- **Kein Manifest** (verifiziert 2026-11-19); Abhängigkeiten ergeben sich
  aus den Imports: `torch`, `transformers`, `safetensors`, `trl`
  (Reward-Funktion, nur für GRPO-Training).
- **Verifizierte Versionen:** `torch 2.12.0+cu130`, `transformers 5.13.0`.
  Die Transformer-5.x-`Cache`-API wird vorausgesetzt
  (`modeling_argon.py`, `verify_argon.py:22-26`).
- **System:** `sbcl` für die Lisp-Stufen (verifiziert 2.6.9).
- **Gewichte:** `google/gemma-3-1b-it` (HF), lokal als `./argon_1b`
  (4,8 GB, nicht im Repo).

## Source Layout

```
./                          Architektur, Training, Verifikation (Top-Level-Skripte)
machines/                   Götterdreiheit-Ports, Oracle, Korpus, Parität
machines/corpus/            machine_corpus.jsonl (generiert, 72 Zeilen)
machines/parity/            python_fixtures.json (Goldwerte)
machines/lisp/              Lisp-Brücke + vendor/ (MIT-Canon)
argon_1b/                   konvertierter Basis-Checkpoint (gitignored Gewichte)
argon_1b_specialized*/      Router-Diff-Checkpoints (gitignored)
argon_1b_machines/          Maschinen-Router + machine_presets/ (gitignored)
scratches/                  Geräte, Roh-Logs, Berichte, Konkatenate (145 Einträge)
```

Build-Output gibt es nicht — Skripte schreiben Checkpoints in die
`argon_1b*`-Verzeichnisse und Ergebnisse nach `scratches/`.

## Generated artefacts (never hand-edit)

| Datei | Quelle | Kommando |
| :--- | :--- | :--- |
| `argon_1b/` (Gewichte, Tokenizer) | `google/gemma-3-1b-it` | `python convert_gemma3_to_argon.py --model_id google/gemma-3-1b-it --output_dir ./argon_1b` <!-- UNVERIFIED — source: convert_gemma3_to_argon.py:194-210, 4,8 GB Konvertierung nicht erneut gefahren --> |
| `argon_1b_specialized{,_ml}/` | `argon_1b` + Router-Training | `train_moe_router_specialization.py --router_only --steps 60` <!-- UNVERIFIED — source: README.md:99-103 --> |
| `argon_1b_machines/machine_presets/*.json` | Router-Analyse | `... --corpus all --router_only --emit_presets` (`train_moe_router_specialization.py:619-641`) |
| `machines/corpus/machine_corpus.jsonl` | Maschinen-Ports, Seed 42 | `PYTHONPATH=. python machines/build_machine_corpus.py` <!-- verified 2026-11-19: sha256 f6c0b5de… unverändert --> |
| `machines/parity/python_fixtures.json` | Kanon-Werte (Erstersperrung) | kein Generator im Repo — nur bewusst per Review ändern (`verify_machines.py:63`) |

## Writing pull requests / issues

Keine `CONTRIBUTING.md`, kein `.github/ISSUE_TEMPLATE/`, kein
PR-Template (verifiziert 2026-11-19). Nichts ist erzwungen; die gelebte
Norm ist die Commit-Hausordnung oben. Wenn Contribution-Infrastruktur
gewünscht ist, wird sie aus diesem Repo abgeleitet (nicht als
Copy-Paste-Template).

## Verification log

| Command | Status | Evidence | Date |
| :--- | :--- | :--- | :--- |
| `python verify_argon.py` | verified | exit 0, 6/6 Stufen, „Paris." generiert | 2026-11-19 |
| `python machines/verify_machines.py --stages 1,2,3` | verified | exit 0, GRÜN (Mini/Korpus/Kanon) | 2026-11-19 |
| `python machines/verify_machines.py --stages 4` | verified | exit 0, `philosopher_p2 PASS` (SBCL 2.6.9) | 2026-11-19 |
| `PYTHONPATH=. python machines/build_machine_corpus.py` | verified | exit 0, sha256 unverändert | 2026-11-19 |
| `python train_moe_router_specialization.py --help` | verified | exit 0, usage | 2026-11-19 |
| `python convert_gemma3_to_argon.py --help` | verified | exit 0, usage | 2026-11-19 |
| `python machines/verify_machines.py --help` | verified | exit 0, usage | 2026-11-19 |
| `python compare_gemma3_vs_argon.py` | UNVERIFIED | README.md:73, lädt zwei 4,8-GB-Modelle | 2026-11-19 |
| `python3 verify_argon.py` (System-Python) | FAILED | `No module named 'transformers'` (3.14.10) | 2026-11-19 |
