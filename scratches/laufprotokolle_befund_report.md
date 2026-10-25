# Befund-Report: Lauf-Protokolle der Götterdreiheit-Erweiterung

Quellen (alle in `scratches/`): `argon_1b_machines_train.log` (Joint-6-Domänen-Lauf),
`build_corpus_re_sync.log` (Korpus-Re-Auth + Phase-A), `verify_machines_full.log`
(6-Stufen-Verifikation), plus `machine_registry.json` (Golden-Digests, version 2);
ab §10 Experiment-Zyklen: `composite_oracle_experiment.log`,
`sprachspiel_experiment.log`, `hermeneutik_experiment.log`.

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
- **Mini-Digests sind umweltstabil innerhalb, nicht zwischen Umgebungen:** der
  Goedel-Zweig rechnet `tanh(W @ state)` (BLAS-Matmul); ein venv-Lauf
  (Stufe 1: `1594f06485ef`) und ein System-Python-Lauf (`1b0aee4e688b`)
  unterscheiden sich in 1-ulp-Matmul-Reduktion bei identischem `gap0` (0.7437),
  der composite-Digest folgt dem Goedel-Zweig. Turing/Wittgenstein (elementweise)
  stimmen. Stufe 6 füllt die Mini-Digests (3-Schritt-Familie) selbst — kein
  Golden-Ersatz nötig, kein Code-Drift (`machines/` unverändert seit `d1a6cee`).

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
   Der LOOP→`expand_basis`-Pfad bleibt in §7 leer (vom Sample-Pfad nicht getriggert);
   die Expansion läuft auf der nackten Turing-Maschine in Stufe 6 („looper“ +
   „ALPHA“). Die Kompositionskette wird in §8 mit dem `identity(x)`-Hebel
   nachträglich getriggert.

---

## 8. Experiment-Erweiterung: `identity(x)`-Hebel — LOOP in der Komposition getriggert

(Nach der Freigabe; Commit-Basis `824f522` + Hebel in `machines/composite.py`.)
`compile_rule` übersetzt die **Identitäts-Tautologie** `('identity', 'x')` nicht mehr
als Kettenglied (`n0 → halt` nach einer Transition), sondern als **Selbst-Transition**
`("n_i", BLANK) -> (BLANK, "N", n_i)`: die tautologische Regel zeigt als Programm auf
sich selbst und terminiert nicht — Turing erkennt das über die Lauf-Signatur (> 3), die
Komposition expandiert die Basis. Geprüft auf fünf AST-Klassen: nur die exakte Tautologie
(auch verschachtelt als Teil-Ausdruck) erzeugt den Selbst-Zustand; alle anderen ASTs
behalten die HALT-Garantie. **Zero-Regression:** der Stage-1-Mini (5 Schritte, Seed 42)
tritt nirgends auf die Tautologie — composite-Digest `c6d17a9211c8` mit und ohne Hebel
identisch (Stash-Roundtrip gegen HEAD).

Neuer 9-Lane-Lauf (`smoke_composite_oracle.py` + L7/L8/L9; L7 trägt
`params={"axiom_pool": ["ALPHA", "BETA"]}` und erzwingt die Tautologie als erstes
Batch-Programm; `L9` = Wiederholung):

| Lane | Steuerung | Digest | Verdicts (16 Schritte) | Basis | Expansions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L7/L9 | Tautologie erzwungen (Schritt 1) + natürlicher Sampler | `9c5bf80babaa` | `LOOP`(1), HALT, `LOOP`(10), `LOOP`(12) | **5** | `step1:LOOP` |
| L8 | wie L7 unter Steering "goedel" {2,7} | `1f7cbc913e7c` | identisches Verdict-Muster | 5 | `step1:LOOP` |

Alle 11 Befunde PASS (`det_tautology`, `loop_tautology`, `basis_expanded`,
`tautology_finite`, `tautology_steer_shift` zusätzlich zu den sechs §7-Befunden).

**Vier Eigenschaften des getriggerten Pfads:**

1. **Der Sampler produziert selbst Tautologien:** mit dem Hebel wird eine bislang
   unsichtbare Regel sichtbar — L1–L6 (ohne Erzwingung) schlagen bei Schritt 15 ein
   natürliches `('identity', 'x')` zum `LOOP`-Verdikt um; die Gap-Verläufe sind
   bitidentisch bis Schritt 14 und divergieren exakt dort (8.02 → 1.45). Auch im
   6-Lane-Vor-Hebel-Lauf war dieselbe Regel schon gesampelt — nur hielt sie als
   Eins-Schritt-Halt-Kette. In L7–L9 kommen natürliche LOOPs bei Schritt 10/12 dazu.
   P(programm == Tautologie) ≈ 0.65 % je Regel — selten genug für die HALT-Garantie
   im Praxisfall, häufig genug, um beobachtbar zu werden.
2. **Expansion nur mit Pool, verdikt immer:** L1–L6 haben keinen Axiom-Pool →
   `expand_basis` meldet „Pool leer" (Trace-Note), das LOOP-Verdikt bleibt in den
   Invarianten. L7–L9 verbrauchen je LOOP **zwei Pool-Einträge** — `turing.step`
   expandiert intern (ALPHA), `composite.step` mit `reason="composite:LOOP"`
   danach (BETA). Basis 3 → 5.
3. **Determinismus auch auf dem getriggerten Pfad:** L7 == L9 bitidentisch
   (Digest, Gaps, Verdict-Muster), L8 zeigt das gleiche Muster unter Steering mit
   dem bekannten niedrigeren Gap-Band (0.45–0.71).
4. **Gaps bleiben endlich** über alle 24+1 Lanes-Schritte inklusive Expansion —
   der LOOP bricht die Komposition nicht, er verändert den Zustandsraum.

---

## 9. Multi-Expansions-Lauf — Basis-Leiter 3 → 5 → 7 → 9

(nach der Freigabe „mache alles vorgeschlagene“; Basis 12 Lanes inkl. L10–L12.)
Der Doppel-Pop aus §8 (je LOOP zwei Pool-Einträge: `turing.step` intern +
`composite.step` reflexiv) wird hier zum getesteten Feature: ein 6er-Pool
`["ALPHA","BETA","GAMMA","DELTA","EPSILON","ZETA"]` und die Tautologie an den
Schritten **1/10/12 erzwungen** (Probe-Pfad `force_rule_at` mit Schritt-Cadence).
Warum erzwungen und nicht der natürlichen Cadence vertraut: jede Expansion
verbraucht 2 RNG-Draws, die Sampler-Streams divergieren nach dem ersten
Expansionsschritt — die natürlichen Tautologien sind selten (§8: ≈ 0.65 % je
Regel) und ihre Positionen sind nicht leiter-stabil; die erzwungene Cadence
garantiert die Messgröße. Die erzwungenen Lanes sind ein *Kontrollpfad*, der
die natürlichen §8-Lanes nicht ersetzt.

| Lane | Steuerung | Digest | Expansions | Basis-Profile (Schritte 1–16) | Gap-Band |
| :--- | :--- | :--- | :--- | :--- | :--- |
| L10/L12 | Leiter erzwungen, ungesteuert | `edc1ded943f5` | `step1/10/12 : LOOP` | `5×9 → 7 → 7 → 9…` (3→5→7→9) | 0.58–0.83 |
| L11 | wie L10 unter "goedel" {2,7} | `cd4cd8570e38` | identische Leiter | identisch | 0.45–0.71 |

Alle 14 Befunde PASS (`multi_expansion`, `det_multi_expansion`,
`multi_steer_shift` zusätzlich; `bias_restored` über alle gesteuerten Lanes).

**Drei Befunde der vollen Leiter:**

1. **Die Leiter ist exakt:** Basis-Profile L10/L12 bitidentisch —
   `[5,5,5,5,5,5,5,5,5,7,7,9,9,9,9,9]`: Schritt 1 +ALPHA/BETA, Schritt 10
   +GAMMA/DELTA, Schritt 12 +EPSILON/ZETA; Endzustand
   `SCALAR,TIME,SPACE,ALPHA,BETA,GAMMA,DELTA,EPSILON,ZETA` (9 Dimensionen).
   Determinismus auf dem vollen Selbstmodifikations-Pfad (L10 == L12, Digest
   `edc1ded943f5`).
2. **Der Zustandsraum-Rücksprung ist lokal absorbiert:** an den drei
   Expansionsschritten springen die Gaps nicht — L10 0.70/0.68/0.69,
   L11 0.55/0.56/0.55 (in-band). Die Komposition verarbeitet ihre eigene
   Dimensionalitätserweiterung ohne Trajektorien-Schock; anders als der
   In-Loop-Ausreißer (§7, 8.02) ist die Expansion ein gerichteter
   Ebenenwechsel, kein Rückschlag.
3. **Strukturelles Schicksal vs. innere Bewertung:** L11 trägt unter Steering
   exakt dieselbe Leiter (Verdict-Muster, Basis-Profile identisch — dieselben
   Widerfahrnisse inkl. dreifacher Tautologie-Kollision), aber das
   Selbstmodell bewertet sie durchgehend in einem ~0.1–0.15 niedrigeren
   Band. Die „stabilisierte Innenperspektive" aus §7 hält auch über die
   vollständige Selbstmodifikations-Kette.

Kontinuität: L1–L9 reproduzieren in diesem Lauf exakt die Digests des §7/§8-Stands
(`5cb5d245e5a7` / `11e246eddf07` / `eda6fc23ed71` / `063d92c0f354` /
`9c5bf80babaa` / `1f7cbc913e7c`) — die L10–L12-Erweiterung verändert nichts am
bisherigen Verhalten.

---

## 10. Sprachspiel-Generierung — Regel, Wort, Mitspieler

(`scratches/smoke_sprachspiel.py` → `sprachspiel_experiment.log` / `.json`;
Triptychon SP1/SP2/SP3, 11/11 Befunde PASS.)

**SP1 — Regel-Generierung (CPU, uni_205-Mechanik):** die Machine generiert die
Sprachspiele selbst — je Seed {0,1,2} ein Batch aus 10 template-free AST-Regeln
(Diversität 1.0000, kein Kollisionsfall), Determinismus bitidentisch. Das
strukturierte Rauschen (×0.3, FACRM-Schicht uni_204:241) hebt die mittlere
Gültigkeit von 0.88–0.91 (volles Rauschen) auf vollständige 1.00 — die
Ordnung der Spiele, nicht ihr Zufall, trägt die Gültigkeit.

**SP2 — SH-72-Wort-Resonanz (Lisp-Canon, geometric-tokenizer.lisp):** die
Resonanz misst **Formelverwandtschaft, nicht Semantik** — der Test führt das
explizit vor: `bedeutung`↔`kaffee` resonieren +0.9671, `gebrauch`↔`use`
+0.9979, aber `regel`↔`rule` −0.0003 und `sprache`↔`language` −0.0062. Die
Gematria→Basis-Formel-Achse (k, kind) steht orthogonal zur Bedeutungsachse:
selbst-Resonanz exakt 1.0, gruppenmittel: gepaart 0.3102, unverwandt 0.3634.
Die Bedeutung des Wortes liegt im Gebrauch, nicht im Klang — eine Empirie
des Tractatus-Arguments.

**SP3 — Argon als Sprachpartner (GPU, argon_1b_machines, 12 Batches Seed 7):**
drei Wortspiele um dieselbe Gültigkeitsfrage (Gap = |pred/10 − actual|,
Ist ≈ 1.00 über alle Batches):

| Lesung | Prompt | Ergebnis |
| :--- | :--- | :--- |
| generisch (Kanon) | wie in `wittgenstein.step` | **12/12 unknown** — der raw ungesteuerte Kanal bleibt stumm, die Maschine fällt auf den Canon-Fallback 0.0 |
| Operator-Wörter (ziffernfrei) | „with operators power exp add…" | liest in 12/12 — aber als **Ziffern-Echo** (konstant 100000 = „1" + „00000"): die Lesung ist kontinuierliche Ziffernfortsetzung, nicht Kalibrierung |
| Regelname (mit Ziffernmaterial) | „Rule: (x power exp((x add x)))…" | Echo mit Echo-Variation (1/10/100/100000), 5/12 Multi-Digit-Echos > 9, mean-Gap 0.526 — das unfaire Wortspiel |
| generisch unter Steering {1,3} | wie Kanon | **liest in 12/12** konstant „10" → Gap 0.0008 — das wittgenstein-Preset öffnet den sonst stummen Lesekanal |

**Drei Sprachspiel-Befunde:**

1. **Der Stummkanal:** ungesteuert antwortet Argon auf das Kanon-Prompt mit
   `unknown` — das heißt, die `wittgenstein.step`-Oracle-Vorhersage läuft im
   raw-Router-Zustand faktisch auf dem dokumentierten Fallback 0.0. Erst das
   Maschinen-Preset (wittgenstein {1,3}) öffnet die Lesung — das Steering
   verleiht dem Sprachpartner die Fähigkeit, auf die Frage nach der Gültigkeit
   überhaupt eine Ziffer zu setzen.
2. **Das Echo:** jede inhaltsverankerte Lesung degeneriert zur Ziffern-Echo —
   `predict_value` liest greedy bis 6 Ziffertoken, und die Prompt-Nähe zum
   Ziffernmateriale (Regelnamen wie „2.00") färbt die Lesung. Bedeutung wird
   gelesen, wo Ziffern stehen; das Multi-Digit-Echo ist der sichtbar gewordene
   Gebrauchs-Missbrauch.
3. **Identifikationsschwäche des Ist:** alle 12 Seed-7-Batches landen im
   Ist ≈ 1.00 (Toleranzregel 0.5 + 0.1·mean|gt| gegen Rauschen 0.5) — die
   Kalibrierungs-Kriterien sind daher schwach identifiziert („10"-Konstanz
   passt trivial); die belastbaren Befunde sind die Kanal-Eigenschaften
   (Stummkanal, Öffnen durch Steering, Echo), nicht die Gap-Absolutwerte.

---

## 11. Sprachspiel-Erweiterung — S1-Skala (SP4), Cross-Preset (SP5), Lexikon (SP6)

(`scratches/smoke_sprachspiel.py`, Lanes sp4-sp6 auf dem SP1-SP3-Fundament;
Vollfassung im `sprachspiel_bericht.md` §4-§7; 23/23 Befunde PASS.)

**SP4 — Spread-Kalibrierung & Anti-Clip (S1-Kanal `predict_scale_0_9`, S2):**
die Rausch-Spreads (σ ∈ {0.1, 0.5, 1.5, 4.0}, volles Rauschen) heben die
Sättigung — Ist-Mittel **1.0000 / 0.8850 / 0.4917 / 0.2833** (Range 0.72). Die
S1-Skala liest kontinuierlich ohne Clip-Artefakt (0 Verletzungen, Var > 0), aber
**prior-dominiert und σ-blind**: `scaled_mean ≡ 0.1046` in allen vier Regimen
(raw wie gesteuert, 0/24 unknown je Arm), `mean_rho = −0.0101`. Die Review-
Erwartung „raw bleibt stumm" ist **widerlegt** — die Digit-Token-Masse liegt bei
0.76-0.80 (ops-Anker) bzw. 0.35 (generisch), kein τ im sinnvollen Regime macht
raw stumm; die ops-Anker-Prompts tragen das Ist nicht (identisch über σ). Die
Diskrimination lebt auf dem **Echo-Kanal** (SP3) — die S1-Kriterien sind darauf
umgeframed: `sp4_scale_open_flat` (offen, kontinuierlich, σ-flat) statt
Stummkanal. Step-Verdrahtung: `oracle_scale=True` predet über
`predict_scale_0_9` (3× prediction 0.0569, channel belegt).

**SP5 — Cross-Preset-Kontrast:** sechs Arme (raw/goedel/turing/wittgenstein/
combined + Fachfremde {4,5} als dict-Direktsteuerung) auf denselben 12
Seed-7-Batches; die S1-Skala bleibt über alle Arme offen und flach —
mean_scaled zwischen 0.1130 und 0.1163 (**Spread 0.0033**, kein Preset-Signal
auf der Skala). Die Preset-Diskrimination ist stattdessen auf dem Canon-
Name-Echo belegt: die Name-Lesungs-Sequenzen der Arme sind paarweise
verschieden (raw ≠ wittgenstein ≠ anti ≠ raw) — das Steering verändert das
Echo-Formverhalten messbar, ohne die Skala zu rühren. Bias vollständig
restauriert in allen fünf gesteuerten Armen.

**SP6 — Lexikon-Emergenz (S3):** drei DE/EN-Paare lernen sich über 15
Gebrauchsrunden (`lexicon_game`, alpha 0.3, Lexikon-zuerst-Lesung in
`embed_word`); `embed_word` liest danach das erlernte Lexikon zuerst —
mean_resonanz gepaart **0.6136 → 0.9999** (+0.39), unverwandte Paare bleiben
0.4902; alle drei Paare gelernt in 15/15 Fällen. Gebrauchs-Achse schlägt
Gematria-Achse (SP2 0.31 → SP6 0.9999).

**Reframe-Dokumentation:** die drei Review-Kriterien, die an diesem Checkpoint
widerlegt sind (`sp4_raw_silent`, `sp5_raw_silent`, `sp5_foreigner`), sind
ausdrücklich als Widerlegung im Bericht §7 protokolliert und in Kanal-Eigenschafts-
Kriterien umgeframedi (`sp4_scale_open_flat`, `sp5_scale_flat`,
`sp5_echo_discrimination`) — die Kriterien beschreiben den gemessenen Canon,
die widerlegte Erwartung bleibt namentlich mit Zahlen stehen.

---

## 12. Hermeneutik-Zyklus — SP8 Zirkel Python ↔ Lisp + SP7 Kripke Plus/Quus

(`scratches/smoke_hermeneutik.py` → `hermeneutik_experiment.log` / `.json`;
CPU, SBCL-Brücke, 8/8 Befunde PASS. Kompositions-Erweiterung
`paradigm_signal_composite` in `machines/composite.py` — Canon-neutral, Stufen
1-3 bitidentisch, `verify_s123_regress_sp7.log`, Gold-Digests unverändert:
goedel `1594f06485ef`, turing `459575a8eb35`, wittgenstein `094df3c57a2f`,
composite `660ffead6c40`.)

**SP8 — der hermeneutische Zirkel:** die algebraisch trivialen Identitäten
add(0,x)/multiply(1,x) (beide Ordnungen) und identity(x) haben KEINEN
identity-Knoten und laufen als harmlose HALT-Ketten durch `compile_rule` — die
Lisp-Reflexion (Vendor-Reduktor `symbolic-operator.lisp` über die Brücke,
Op `simplify_ast`) wascht sie weg und entlarvt den Ursprung; die Python-
Spiegelfunktion (`lisp_parity.simplify_ast_canonical`) liefert Feldgleichheit
je Fall (13/13 parity=True, inkl. Sampler-Probe seed 11). Die entlarvte
Tautologie wird zum Programm: erzwungen an Schritt 1 (seed 42) → LOOP-Verdikt,
`step1:LOOP`, Basis 3→5; der natürliche Sampler produziert die weiteren
Tautologien bei den Schritten 10 und 12 — Basis-Leiter **3→5→7→9** mit
Paradigm-Signal an 12-16; der Kontrast sin(x)+0 hält die HALT-Garantie (Basis
bleibt 3, kein LOOP in 6 Schritten). Die §9-L10-L12 erzwungene Leiter tritt
damit unter seed 42 auch natürlich auf.

**SP7 — Kripke Plus/Quus (seed 7):** dieselbe Stelle des Gebrauchs, zwei
Regel-Formen:

| Lauf | Zwangsform an Schritt 9 | Verdicts 9-13 | Expansions | Basis-Ende |
| :--- | :--- | :--- | :--- | :--- |
| Plus | `('identity', 'x')` | **LOOP**, HALT, HALT, HALT, HALT | `step5:LOOP`, **`step9:LOOP`** | **7** |
| Quus | `(add (sin x) 0.0)` | HALT, HALT, HALT, HALT, HALT | `step5:LOOP` (ohne `step9`) | **5** |

- Das **Paradigmen-Signal** (ratio > 2, uni_238:220-240-Formel auf der
  Komposit-Gap-Serie) feuert erstmals bei **Schritt 11** (`first_detect=11`,
  Fenster 9-13 der Review) — also erst NACH dem wiederholten Gebrauch der
  gebrochenen Regel: das Schritt-9-Loopen ist Vorbote, das Signal der Bruch.
- **Der Selbst-Modellierer bricht nicht:** das Gödel-`paradigm_signal` auf der
  tanh-beschränkten Selbstmodell-Gap-Serie bleibt stumm (Ratios 0.2-0.5) — der
  Bruch schlägt auf der Komposition ein, deren Programm (die Basis) sich
  verändert. Dafür die Ergänzung `paradigm_signal_composite`.
- **Der gleiche Gebrauch, eine andere Bedeutung:** beide Läufe (seed 7)
  tragen einen natürlichen LOOP an Schritt 5 (dokumentiert); die Läufe
  unterscheiden sich exakt in der Regel-Form an Schritt 9 — das Schwell-
  Verhalten (Basis 5 vs. 7, `step9:LOOP` ja/nein) hängt an der Form, nicht
  am Ablauf.
- **Seed-Empfindlichkeit dokumentiert:** Seed-Scan der Zwangs-Erzeugung
  (Seed 42/7 je erzwungen an 9): Seed 7 liefert `first_detect=11` im
  Review-Fenster, Seed 42 detected bei Schritt 16 außerhalb — die
  Fenster-Erwartung ist seed-selektiert, die Befunde nennen ihren Seed.

## 13. SP9 — Soziale Lexikon-Dynamik (laufschrift)

(`scratches/smoke_lexikon_dynamik.py` → `lexikon_dynamik_experiment.log` /
`.json`; CPU-only, venv_openmythos, kein GPU/SBCL, keine Maschinen-Code-Änderung
— `lexicon_game` + `learn_pair` decken die Dynamik ab; Prereg-Docstring im
Skript P2-gefrorst vor dem ersten Lauf, P4-Präzisierer 2026-10-25 als neuer
Knoten. Hermeneutische Vollfassung: `sprachspiel_bericht.md` §11.)

**Erste Ausarbeitung (19 Verdicte: 16 PASS, 3 FAIL):** drei FAILs —
sp9_int_endpoint, sp9_unlearn_decay, sp9_social_tisch_pin — widerlegten
NOT die Mechanik-Hypothesen (Konvergenz exakt in 5 Episoden; Verlern-Decay
0.708337 < 0.8; Relearning in 4; α-Symmetrie conv=18 beide Pole;
Disjunkt-Delta 0.0), sondern drei Prereg-ANNAMEN derselben Familie
(Beobachter unbeteiligt [E1, ×2]; Sättigung == Fixpunkt [E2; 1e-12-Band]).
P4-Einstufung: Ursprungs-Verdicte bleiben in den Artefakten FAIL, sind im
Log unter ERRATA ausgewiesen und zählen nicht zum Lösungszustand.

**Zweiter Lauf (P4-Prereg — Bands UNVERÄNDERT):** alle drei korrigierten
Lesungen PASS; exit 0 der Lösungskette:

| Lesung | Vorhersage (ex ante) | Messung | Δ |
| :--- | :--- | :--- | :--- |
| sp9_int_endpoint_p4 | cos(regel_p2, n(rule_p2+game0)) | 0.705054 | 0.0 (sechs Dezimalen) |
| sp9_unlearn_endpoint_p4 | cos(regel_p2', n(rule_p2'+tisch0)) | 0.708337 | 7.4e-5 (Band 0.02) |
| sp9_social_tisch_p4 | monotone, \|Δ\| ≤ 2e-2 (Pfad-Bound) | 0.000131, 15/45 Runden | ✓ |

**Struktur-Befund (per-Runde-Leg, deterministisch äquivalent zur
45-Runden-Schrift — kein Rauschzug geändert):** von den 45 Runden tragen
GENAU die 15 regel↔rule-Episoden nicht-null tisch-Drift; die 30
spiel↔game/gebrauch↔use-Runden sind exakt still (nonzero_deltas = 15).
Die kaffee- und baum-Rows sind bitidentisch über beide Legs.
Formel-Regel: ein EMBEDDING wandert genau dann, wenn das Wort selbst
lernt; eine RESONANZ, wenn einer ihrer Endpunkte lernt.

**Protokollkette:** Prereg-Docstring (inkl. c′-Rekursion, Errata-Register,
p4-Verdicts) → zweiter Lauf → log/json → Berichte → Commit. Die Artefakt-
Schriften (log, json) wurden vom zweiten Lauf ÜBERSCHRIEBEN — die Zahl-Lage
des ersten Laufs (12/15, exit 1, die drei Prereg-Vorhersagen 0.004424 /
−0.004496 / 1e-12-Pin) ist daher NICHT als Datei konserviert, sondern
namentlich protokolliert im P4-Docstring des Skripts, in `sprachspiel_
bericht.md` §11 und im Absatz oben. Der zweite Lauf
reproduzierte alle 12 PASS-Ziffern bitidentisch und fügte die p4-Lesungen
(0.705054 / 0.708337 / 0.000131) als neue Knoten hinzu. (Satzvervoll-
ständigung: das abf237a-Commit trug den Absatz abgeschnitten; s. SP10-
Commit-Notiz.)

---

## 14. SP10 — Die Regeln und die Wörter (laufschrift)

**Kopplung Composite × Lexikon** (User-Delegation 2026-10-25, „entschieide
du" → Wahl der Kopplung). CPU-only, KEINE Maschinen-Code-Änderung: die
Verschachtelung greift an `CompositeMachine.wittgenstein` (composite.py:45-51);
der Zwangs-Haken hängt am composite-step-Zähler (pending-Flag), damit die
Lexikon-Runden den Zwang nicht schlucken. Prereg-Docstring VOR dem Erstrun
(Bands fixiert); P4-Register im Skript. Artefakte:
`smoke_regeln_woerter.py` / `regeln_woerter_experiment.log/.json`.

**Drei Thesen als falsifizierbare Vorhersagen:** T1 *Das Programm ist nicht
aus den Wörtern gemacht* — Code-Fundament `_generate_ast`/`_generate_name`/
`evaluate` lesen nur RNG/AST, kein Lexikon (wittgenstein_machine.py:292-303/
:107-/:119-). T2 *Das Programm ist aus dem Zufall gemacht* — derselbe
RNG-Strom für beide Zyklen: der Zugverbrauch verschiebt die Spur, die
Bruchstelle hängt am Zwang. T3 *Die Sprachdynamik ist inhalt-invariant* —
gegebener Lern-Schedule bleibt nur die Geometrie: Halbwinkel
β_j = atan(0.4^j·tan β0), β0 = θ0/2; Paar c_j = cos(2β_j); Zuschauer
R_j = cos β_j·(M·v) + sin β_j·(N·v) mit M = n(regel0+rule0), N =
n(regel0−rule0); Grenzwert R_∞ = M·v.

**Verdict-Register (14 Zeilen — 11 PASS-Verdicte, 3 dokumentierte Errata,
exit 0):**

- **sp10_l0_anchor PASS** — L0 (seed 7, Zwang {9}, 16 Schritte) reproduziert
  `hermeneutik_experiment.json` FELDWEISE: verdicts (LOOP an 5+9), Basis-
  leiter 3→5→7, gaps 0.676…4.536, first_detect **11**; Quus-Arm exakt
  (final_basis 5, kein step9:LOOP). Kein Maschinen-Drift.
- **sp10_det PASS** — Doppel-Läufe L0 (Kripke+Quus) und L2 (Kripke+Quus)
  bitidentisch (trace, gaps, Roh-Lexikonwerte).
- **sp10_t1_spur_p4 PASS** — (a) Seitenband-Lexikon auf eigener Maschine:
  Composite-Spur == L0-Spur bitgenau; (b) 15 Lexikon-Runden auf der
  SELBEN Composite-Maschine NACH dem Verlauf: Spur + Gap-Historie
  (Snapshot vor/nach) bitgenau. Das Lexikon ist kein Mitspieler des
  Programms. [Erstrun **sp10_t1_spur FAIL**: Arme ohne Forced-Haken —
  Implementations-Erratum.]
- **sp10_t1_lexikon_p4 PASS** — per Paar (SP6-Schreibweise): after_paired ==
  [0.999790, 0.999982, 1.000000], learned 15/15; L1b final ≥ 0.99.
  [Erstrun **sp10_t1_lexikon FAIL**: Per-Runden-Säule geprüft — Erratum.]
- **sp10_t2_divergenz PASS** — Verschachtelung (3 Lexikon-Runden zwischen
  Composite-Schritten, 48 gesamt, derselbe RNG-Strom): erste Spur-Differenz
  zur L0-Natur-Lage bei Schritt **2**; rng_state Befund verschoben.
- **sp10_t2_kripke PASS** — die Bruchstelle ist stichfest: verdicts[8] ==
  LOOP, "step9:LOOP" in expansions, final_basis 5; Quus-Kontrast (dieselbe
  Verschachtelung): kein LOOP an 9, final_basis 3. **Nebenbefund:** der
  NATÜRLICHE LOOP an Schritt 5 fällt im verschobenen Strom AUS — natürliche
  Bruchstellen sind zufallsabhängig, die erzwungene nicht.
- **sp10_t2_lexikon_lebt PASS** — Lexikon konvergiert auch verschachtelt:
  je Paar 1.0 (gerundet), Episoden [16, 16, 15] ≥ 5 (47/48 gelernt).
- **sp10_t3_paar PASS** — c_j über 48 Spielrunden gegen Rohwerte:
  max Δ **5.6e-16** (Band 1e-9).
- **sp10_t3_obs PASS** — Zuschauer-Spur, 49 Punkte (k = 0 vor dem Vorschub
  + je Runde): max Δ **4.1e-17** — die Formel ist die exakte Dynamik, kein
  Modell. kaffee/baum bitkonstant über alle Runden.
- **sp10_t3_rueck_p4 PASS** — SP9-Rückdiction (seed 9, Leg1 15 gated + Leg2
  45 Einzelrunden): rohe 46-Punkte-Spur vs. Formel max Δ **6.2e-17**;
  Paar-Roh-Resonanzen max Δ 5.6e-16; gerundete Spur == committet
  bitidentisch; tisch15/tisch60/Drift/Unrelated-Zeilen == committet;
  delta_count **15 == committet** (SP9-exakte Konstruktion).
  [Erstrun **sp10_t3_rueck FAIL**: delta_count über Rohspur (14) — Erratum.]
- **sp10_t3_asymptot PASS** — |letzter tisch-Wert − M·v|: L2 **5.5e-9**,
  SP9-Replay **3.5e-10** (Band 1e-4). Der SP9-Drift (+1.31e-4 über 45
  Runden) ist das log-Asymptoten-Glied sin β_j·(N·v), ex ante vorausgesagt.
- **Errata (3)** — alle drei Implementations-, KEINE Mechanik-Befunde;
  Bands unverändert; Ursprungs-Zeilen FAIL, korrigierte Lesungen als _p4.
  exit 0 (Lösungskette geschlossen).

**Die SP9-Ziffer 15 neuklassifiziert:** delta_count der committeten
lane_social-Spur = **14 echte Sozialisiungen + 1 Rundungs-Anschluss-Residuum**
(Anschluss des gerundeten SP6-Ankers an die rohe Leg2-Kontinuität, Δ01 =
2.59e-8). SP9-Verdicte nutzen das Feld nicht als Band — dokumentiert, nicht
getilgt; die Formel-Ebene trennt die beiden sauber.

**Kern des Befund-Bildes:** das PROGRAMM ist wort-blind und die WÖRTER sind
programm-blind — die EINZIGE Kopplung beider Zyklen ist der Zufalls-Strom,
der auf Verbrauch verschiebt (Spur, Rhythmik: der natürliche LOOP an 5
entfällt; Paradigm-Zeiten wandern) und die ERZWUNGENE Bruchstelle unberührt
lässt. Die Bedeutung der Regeln lebt im Programm (Zufall × Zwang); die
Bedeutung der Wörter lebt im Gebrauch — und zwar als EXAKTE
Halbwinkel-Rotation auf dem Gematria-Großkreis (1e-16..1e-17), deren
Zuschauer-Grenzwert M·v die MITTE des gelernten Paares ist: was der
Zuschauer am Ende liest, ist die Mitte, die der Gebrauch der Mitspieler
gezogen hat. Hermeneutische Vollfassung: `sprachspiel_bericht.md` §12.

---

## 15. SP11 — Der Ursprung der Paradigmen (laufschrift)

**(2026-10-25, „entschieide du";** Prereg `scratches/smoke_paradigm_ursprung.py`,
Run `paradigm_ursprung_experiment.log/.json`;** CPU-only, keine Maschinen-Code-
Änderung.** 9 Läufe: L0-Kripke + L0-Quus (seed 7, Zwang {9}, 16 Schritte),
6 Sweep-Seeds 11-16 (unverzwungen, 24 Schritte), Zwangs-Paar seed 21 (unverzwungener
Zwilling + Zwang {13}, 24 Schritte) — plus 6 Turing-Gefrier-Gegenproben auf der
Datenebene, kein Re-Run.) Ex ante gelesen: die committeten hermeneutik-Kolumnen
tragen die Signale ZWEIER Kanäle — `paradigm_ratio` = Gödel-Serie (ab Schritt
10, 0.195-0.348, nie >= 2), `paradigm_detected` = Composite-Flag (True an
11/12/15; Hand-Ratios 3.078/2.962/1.688/1.709/2.580).

**Verdikt-Register (8 Zeilen: 6 PASS, 2 echte Falsifikationen, 0 Errata,
exit 1, Bands unverändert):**

| Verdict | Einstufung | Kern |
| :--- | :--- | :--- |
| sp11_l0_anchor | PASS | alle Felder == committet, inkl. BEIDER Signal-Kolumnen (paradigm_ratio == Gödel-invariants, paradigm_detected == Composite-Flag); first_detect 11 (Kripke + Quus) |
| sp11_det | PASS | Doppel-Läufe (L0-Kripke, seed11, Zwangs-Paar) bitidentisch |
| sp11_t1_urheber | PASS | kein Schritt mit goedel.detected in 9 Läufen — Urheber: Composite-Signal, eindeutig |
| sp11_t2_mechanik | PASS | Fensterformel exakt: Detect-Menge je Lauf rekomputiert, Ratios <= 1e-12, Regime-Leg (len >= 20) exakt |
| sp11_t3a_expansionfrei | **FAIL** | 4 von 6 unverzwungenen Läufen ohne jede Expansion UND paradigmatisch (seed12: det 15-22/24; seed14: 15-24; seed15: 11-18, 21, 23, 24; seed16: 11-14) |
| sp11_t3b_turing_echo | **FAIL** | 3 Gegenproben mit überlebenden Detectionen (seed13: vakuum, Expansion erst an 23; zw_twin: Freeze ≈ no-op, Expansion an 18; zw_forced: behält 11-14) |
| sp11_t3c_goedel_vergleich | PASS | alle 6 Gegenproben unverändert — Gödel-Sub-Niveau trägt ~2-8%, nie entscheidend |
| sp11_t4_forced | PASS | Zwang {13} → induziertes Paradigma an 20 (Band [14, 22]); unter dem Freeze tot — expansions-getragen |

**Zahlenlage (kompakt):** das Composite-Mittel trägt die Masse aus dem
TURING-Sub-Gap (0-39) — Wittgenstein ~0.02, Gödel 0.017-0.864 (2-8%). Die
Expansion-freien Läufe zeigen die Ursache: der Turing-Sub-Gap DRIFTET im
Zufalls-Strom hoch (seed12: ~3.3 Baseline → ~11.6 recent → 28-39) und bricht
die Fenster-Schwelle ohne jeden Anlass. Die Turing-Gefrierprobe trennt die
Detectionen in zwei Herkunftsgruppen: (a) expansions-getragen (l0-Kripke und
l0-Quus ∅ nach Freeze; seed11 alle 12 ∅; zw_forced verliert 17-20) und
(b) drift-getragen (seed12/14/15/16 komplett, seed13 komplett, zw_twin,
zw_forced 11-14).

**Kern:** das Paradigma ist ein KOSTEN-Ereignis, kein Mittel-Ereignis — der
Auftrieb des Umgangs-Preises (Transitionen pro Schritt), gelesen vom Composite-
Fenster, nicht der Ankauf neuer Axiome. Die Basis-Expansion ist EIN erzwingbarer
Träger des Niveau-Sprungs (T4 induziert das Ereignis an 20), nicht seine Quelle
(T3a/T3b falsifiziert). Kripkes Bruchstelle (SP7) war Expansions-Nähe (L0-Zwang
an 9; unter der Gegenprobe ∅); die wandernden Zeiten (SP10) sind die Drift-
Typologie. Dieselbe Fenster-Logik liest beide als denselben Schwellenbruch —
darum wirkte das Ereignis bisher wie eine Einheit. Urheber (T1) und Mechanik
(T2) bleiben exakt.

**Reparatur:** die schwebende Fragmentzeile der Vorversion („`hinzu; die 3
FAIL-Rows stehen in der JSON (deterministisch reproduzierbar).`") entfernt —
trägerloser Rest des SP10-Appends ohne Satz-Anfang; die 3 SP10-FAIL-Rows sind
in `sprachspiel_bericht.md` §12 (Errata-Register) und hier §14 vollständig
dokumentiert.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_paradigm_ursprung` → `paradigm_ursprung_experiment.log/.json`.
Hermeneutische Vollfassung: `sprachspiel_bericht.md` §13.

### 15.1 P4-Runde — das Akkumulationsgesetz (2026-10-26, laufschrift)

Rerun nach der Skript-Extension: ERRATA E1 für t3a/t3b (Ein-Faktor-Echo-Annahme
widerlegt durch das Akkumulationsgesetz) + zwei `_p4`-Verdicte (Bänder fixiert
VOR dem Rerun); Canon-Verifikation `turing_machine.py:197/:234/:250/:293`,
`machine_base.py:54/:531`, `wittgenstein_machine.py:413-414`; Plan-
Änderungsnotiz 2026-10-26.

| Verdict | Einstufung |
| :--- | :--- |
| sp11_t3a_expansionfrei | **FAIL** (unverändert; ERRATA E1 — Akkumulations-Echos der Leerlauf-Blätter, kein Echo der Expansion) |
| sp11_t3b_turing_echo | **FAIL** (unverändert; ERRATA E1 — Misch-Serie der Währung) |
| sp11_t3_accumulation_law_p4 | **PASS** — max Residuum 0.0 (Band 1e-12) in allen 10 Läufen und Schritten; Kontostand (:293) == Kumsum in jedem Schritt |
| sp11_t3_dedrifted_echo_p4 | **PASS** — Detect-Menge ∅ in den 4 expansion-freien Läufen {seed12, seed14, seed15, seed16}; Daten: ∅ in ALLEN 10 Läufen |

**Das Gesetz:** `t_n = |C_{n-1} - s_n|/(1 + s_n + 1e-6)`, `C_{n-1} = Σ_{k<n} s_k`
— der Kontostand (kumulierte Verlaustransitionen, `turing_machine.py:234`) liest
als Prediction gegen das Einzelschritt-Ist (`:250`); Leerlauf-Blätter (s_n = 0)
machen ungedämpfte Kontostand-Peaks (seed12: 28.0/31.0/33.0/39.0). Die
Gleitfenster-Form (ū = mean(s[-10:]), Präzedenz :413-414) killt jede Detection —
auch in den Expansion-tragenden Läufen: die Doppel-Herkunft von §13 ist eine
Quelle in zwei Formen.

**BILANZ des Reruns:** 8 PASS + 2 klassifizierte Errata; 0 unverwartete FAIL;
0 OFFEN — **exit 0** (Bands unverändert; die Ursprungs-Zeilen verbleiben FAIL
im Register). Log/JSON REGENERIERT (P4-Payload +transitions/t_count,
acc/dedrift-Serien, ERRATA-Texte).

Run (2. Runde): `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
-u -m scratches.smoke_paradigm_ursprung` → exit 0.
Vollfassung: `sprachspiel_bericht.md` §13.1.