# Befund-Report: Lauf-Protokolle der Götterdreiheit-Erweiterung

Quellen (alle in `scratches/`): `argon_1b_machines_train.log` (Joint-6-Domänen-Lauf),
`build_corpus_re_sync.log` (Korpus-Re-Auth + Phase-A), `verify_machines_full.log`
(6-Stufen-Verifikation), plus `machine_registry.json` (Golden-Digests, version 2).

---

## 1. Trainingslauf — 60 Schritte, Joint über 6 Domänen

| Größe | Wert |
| :--- | :--- |
| Trainierbare Parameter (Freeze) | 239.616 von 2.245.605.632 (**0,01 %**) — nur Router-Gates; Basis/Embeddings/Attention/Shared Expert eingefroren |
| CE+Ent-Loss | 4.8446 → 4.3638 (**−9,9 %**) |
| Subspace-Sep-Loss | 0.2389 → 0.0370 (**−84,5 %**; nach ~30 Schritten weit konvergiert: 0.0500 → Rest 0.037) |
| Zeit | 974.8 s (≈ 16,2 s/Schritt, CUDA) |
| Korpus | `--corpus all` (6 Domänen × 24 = 72 Zeilen), sha `f6c0b5de…` |
| Save-Modus | `router_diff`: 52 geänderte Keys (Gate.weight + expert_bias über 26 Layer) statt 5-GB-Vollkopie |

**Korpus-Re-Auth (Re-Sync-Lauf):** die Signum-Korrektur (`err = actual − prediction`,
uni_241:220) veränderte die Zahlenschwänze der Gödel-Trajektorien-Samples → kanonischer
Rebuild `f6c0b5de…`, Phase-A-Baseline unverändert kohärent (TV zwischen Maschinen
2,87–4,12 %), Expertenwahl je Domäne „plan-defaults (baseline collision)" (Baseline-Top-2
kollidierten auf [1,4]/[1,0]; disjunkte Defaults {2,7}/{0,6}/{1,3} gegriffen).

## 2. Entropie-Kalibrierung — Solls vs. Messwerte

| Domäne | Ziel H | gemessen H | Δ | Top-2 (Anteile) |
| :--- | :--- | :--- | :--- | :--- |
| prose | 0.85 | 1.01 | +0.16 | E3 38.8 % / E7 38.2 % |
| code | 1.35 | 1.42 | +0.07 | E4 33.1 % / E1 26.5 % |
| math_logic | 1.85 | 1.67 | −0.18 | E5 35.4 % / E2 21.0 % |
| **goedel** | **1.30** | **1.41** | **+0.11** | **E2 33.1 % / E7 30.7 %** |
| **turing** | **1.65** | **1.57** | **−0.08** | **E6 32.6 % / E0 28.4 %** |
| **wittgenstein** | **1.90** | **1.68** | **−0.22** | **E3 30.6 % / E1 28.2 %** |

Alle sechs Domänen liegen in einem **±0.22-Band** um die Solls — die Hierarchie
(Prosa eng → Sprachspiel breit) ist erhalten, keine Domäne hat das Soll übersprungen.
Die Maschinen-H-Werte sind inzwischen als `trained_entropy` in der Registry
persistiert (Herkunftsfeld `trained_entropy_source`; Presets persistieren `mean_h`
selbst ab `cd6c009`).

**Baselines:** vor dem Training alle 6 Domänen H≈2.06 (≈ ln 8 = 2.079, diffus) —
das Training hat die Entropie-Hierarchie aus der diffusen Basis geformt.

## 3. Routing-Trennung und Subspace-Kopplung

- TV-Paare **vor**: 2.6–5.6 % (diffuse Basis) → **nach**: 40.3–71.1 %
  (min code/wittgenstein 40.3 %, max math/prose 71.1 %).
- Maschinen-Maschinen-Paare: goedel/turing 56.3 %, goedel/wittgenstein 55.8 %,
  turing/wittgenstein 52,1 % — überdemäßig scharf getrennt bei nur 24 Samples/Domäne.
- **Ko-Subspace-Kopplung (Befund, kein Fehler):** die Maschinen-Cluster überlappen
  mit den Alt-Domänen-Stärken — E2/E6 (math), E3/E7 (prose), E0/E1 (code); E4/E5
  bleiben Top-Anchor von Code/Math (Brücken-Subspace). Die Maschinen-Korpora sind
  aus diesen Domänen-Stilen gebaut (Trajektorien-, AST-, Transitions-Formate), der
  geteilte Subspace ist die *Anbindung* an das antrainierte Verständnis — genau das
  Plan-Ziel „auf dem antrainierten Verständnis aufbauen“.

## 4. Sprachinvarianz & Wissens-Erhalt

| Messgröße | vor | nach | Befund |
| :--- | :--- | :--- | :--- |
| Math EN↔ZH (ungesehene Probes) | 5.5 % | 8.5 % | **Kosten-Note** (eine gegenläufige Bewegung; unter dem 20-%-Schwell) |
| Math DE↔ZH | 4.5 % | 9.1 % | ebenso |
| Prose EN↔ZH | 7.2 % | **3.1 %** | verbessert |
| Trennung Math(EN)/Prose(EN) | 7.8 % | **70.6 %** | Domänen-Schärfe ≈ 9:1-Verhältnis |
| Paris EN/DE | ✓ | ✓ | Basiswissen geschützt |

## 5. Verifikation — alle 6 Stufen grün (Exit 0)

| Stufe | Befund |
| :--- | :--- |
| 1 Mini-Smoke | Digests `1594f06485ef` (goedel) / `459575a8eb35` (turing) / `094df3c57a2f` (wittgenstein) / `660ffead6c40` (composite) — bitidentisch in 2 Läufen, NaN-frei |
| 2 Korpus | 72 Zeilen, 24 je Domäne, kinds ≥ 3; Raw-Bytes == Registry-sha (roh-gelesen, universal-newline-Falle vermieden) |
| 3 Canon-Parität | alle Kanon-Werte == Goldwerte (`parity/python_fixtures.json`) |
| 4 SBCL-Parität | P1 bitgenau + P2 strukturell (Philosopher: len 52, thoughts 6) |
| 5 Real-Lauf | steer Δ-TVs 0.362/0.390/0.411 (Δ-Semantik auf dem trainierten Router, Schwell 0.3), Anteil 1.000, `copy_`-Restaurierung True; Konvergenz 0.7437 → 0.3777; Trennung min 0.52 (Maschinen) / 0.40 (vs. Alt-Domänen); Invarianz 8.5/3.1 % + Paris |
| 6 Regression | Paradigm ratio 2.60 RISING; Axiom-Expansion LOOP + „ALPHA“; Registry version 2 incl. Golden-Digests; compare-Suite unangetastet |

**Kohärenz-Quercheck:** die Stufe-5-Re-Messung der Invarianz-Probes gibt bitidentisch
die Trainings-Werte zurück (H 1.63/1.68/1.64/1.07/1.03/1.01, TVs 8.5/3.1/70.6) —
der Verifikator misst dieselben Zahlen wie der Trainingslauf, keine Drift zwischen
dem Save-Zeitpunkt und der Nach-Betrachtung.

## 6. Notizen zu Log-Ordnung und Fehlalarmen

- **Druck-Ordnung im Verifikations-Log:** die Invarianz-Tabelle steht *vor* STUFE 1,
  weil Stufen-Findings gesammelt und am Ende gedruckt werden, während der
  Invarianz-Druck von Stufe 5 während der Ausführung bereits erscheint — kein
  Ausführung-Anomalie.
- **Tokenizer-Warnung (`fix_mistral_regex`)** erscheint in allen drei Logs —
  dokumentierter **Falsch-Positiv** (README, Tokenizer-Hinweis): der Test-Tokenizer
  tokenisiert Proben token-für-token identisch zum Original, die Flag darf nicht
  gesetzt werden.
- **Idempotenz der Stufe 6:** `trained_entropy` wird aus `preset.mean_h` übernommen
  bzw. ein vorhandener Messwert nicht auf null zurückgesetzt — mehrfache
  Verifikationsläufe überschreiben keine Messwerte mehr.

---

## 7. Experiment: CompositeMachine — Oracle vs. In-Loop unter Preset-Steering

(`scratches/smoke_composite_oracle.py` → `composite_oracle_experiment.log` / `.json`;
6 Lanes à 16 Schritte, Seed 42, Checkpoint `argon_1b_machines`)

| Lane | Modus | Steering | Digest | H-Probe | Gap-Verlauf (16 Schritte) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L1/L2 | in_loop | — | `d63f6d6cec55` (≈) | 1.7432 | 0.68 → **5.47/6.39/8.02-Ausreißer** → 1.86 (volatil) |
| L3/L4 | oracle | — | `2565609bc3dc` (≈) | 1.7432 | 0.61…0.81 (ruhiges Band), Δ 0.74→0.73 |
| L5 | oracle | combined {0,1,2,3,6,7} | `0d1479f2f804` | 1.7441 | 0.48…0.67 — **systematisch unter ungesteuert** |
| L6 | oracle | goedel {2,7} | `b4cd8edcf5da` | 1.7420 | 0.48…0.67 — nahezu identisch zu L5 |

**Befunde (alle PASS):** Determinismus in beiden Kopplungsmodis (in_loop und oracle
bitidentisch über zwei Läufe — und über zwei Prozesse hinweg), Gaps endlich,
Bias-Restaurierung in jeder Steering-Lane.

**Zwei semantische Befunde:**

1. **Das H-Meter ist bias-frei (Design, experimentell belegt):** |ΔH| unter ±50-Steering
   ≈ 0.001, weil `moe_layers.py:60-66` die Router-Entropie aus den **bias-freien**
   Sigmoid-Affinitäten berechnet, während `expert_bias` ausschließlich die Top-K-Auswahl
   verschiebt (Zeile 65-66) — die Gewichte selbst kommen ebenfalls aus den bias-freien
   Scores (Zeile 79). Das Steering-Signal liegt also in der *Auswahl* (Stufe 5:
   Anteil 1.000, TV 0.36–0.41) und in der Trajektorie (L5/L6 ≠ L3): gleiche Angriffseigentümer,
   andere Messgrößen. Erste Experiment-Kriterium-Version hatte ΔH als Failure-Wert —
   der Fehler lag im Kriterium, nicht im Steering; korrigiert und dokumentiert.

2. **In-Loop-Volatilität vs. Oracle-Band:** der In-Loop-Composite-Gap steigt in
   Ausreißer bis 8.02 (mean-of-3-Gaps; die submaschinelle Rückkopplung schlägt durch),
   während der Oracle-Gap in einem ruhigen Band von 0.49–0.81 bleibt und unter
   Steering um ~0.1–0.2 **absinkt** — das externe Selbstmodell stabilisiert die
   Komposition und die steered-Experten (Maschinen-Cluster) reduzieren den Gap.
   Der LOOP→`expand_basis`-Pfad bleibt in allen Lanes leer: composite-Programme
   halten per Konstruktion (Präfix-Walk bis `halt`); die Expansion wird auf der
   nackten Turing-Maschine in Stufe 6 („looper“ + „ALPHA“) getrieben.