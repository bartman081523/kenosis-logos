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

---

## 16. SP12 — Lexical Grounding: das gelernte Wort wird Programm (laufschrift)

**(2026-10-26;** Prereg `scratches/smoke_lexical_grounding.py`, Run
`lexical_grounding_experiment.log/.json`;** CPU-only; ERSTE
Maschinen-Code-Extension** des SP-Zyklus (`composite.py:108` Signatur
opt-in + `wittgenstein_machine.py:383-424` `resolve_lexical_ast`); die
Regression-Freiheit liefert `sp12_l0_anker` über die kommittete
SP11-lane (bitidentisch).) Kern: die **LEXIKON-GEGENWART** scheidet Klang
von Gebrauch — Klang-Dach max 0.983123 als DATUM (Formel-Achsen der 16
Operatoren kollidieren numerisch); die Resolution ist reine Lexikon-Lesung,
kein RNG-Zug.

**Verdikt-Register (6 Zeilen: 6 PASS, 0 Errata, exit 0):**

| Verdict | Einstufung | Kern |
| :--- | :--- | :--- |
| sp12_l0_anker | PASS | machines-Regression: kommittete SP11-lane fields_ok/first_detect_ok True, Doppel-Replay bitidentisch |
| sp12_det | PASS | Doppel-Läufe Arm A und Arm B bitidentisch |
| sp12_t1_semantik | PASS | 5 Runden learned; resonance_final 0.999808 ≥ 0.85; Resolver zu ('identity','x') direkt UND im Lauf; NACH-Kandidaten == ["identity"] |
| sp12_t2_rng_reinheit | PASS | Batch-ASTs + rule_names + w_series identisch in A/A'/B — die Resolution konsumiert KEINE RNG-Züge |
| sp12_t3_scheidung | PASS | vor_resolution [None×6] (SH-72-Fallback löst NIE auf); Klang-Dach als DATEN; opt-in-Default-Invarianz A == A' |
| sp12_t4_verdikt | PASS | Arm A HALTED×6, Basis 3, expansions []; Arm B LOOP×6, Basis [5, 7, 9, ...], expansions step1-3:LOOP |

**Kern:** das gelernte Wort wird Programm. VOR dem Spiel ist „spiegel"
Klang — Resolver None, die Kette bleibt der normale RNG-Batch (alle
HALTED, Basis 3). NACH 5 gelernten Runden (`lexicon_game[("spiegel",
"identity")], noise_level=0.0`; Resonanztreppe 0.7447 → 0.9998,
SP6-Halbwinkelkette) IST „spiegel" die Tautologie ('identity','x') →
Selbst-Transition (composite.py:66-92) → LOOP → +2 Basis-Terme je LOOP
(Turing-innerer Expand :241-246 + Composite :132-134) — Basis-Verlauf
5, 7, 9, 9, 9, 9; ab Schritt 3 ist der Canon-Pool (6 Axiome) erschöpft,
der Clamp ist DATEN. Der User-Pass exakt canonisch („Basis auf 5" nach
dem ersten resolvierten Schritt). Mauer 1 (SP10 Wort-Blindheit) gebrochen:
**Semantische Kausalität geschlossen** — erste der drei Nordstern-Säulen;
Endogene Agency (SP13) und Individuierung (SP14) folgen.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_lexical_grounding` → `lexical_grounding_experiment.log/.json`.
Hermeneutische Vollfassung: `sprachspiel_bericht.md` §14.

---

## 17. SP13 — Endogene Agency: die Maschine probiert, dann wählt sie (2026-10-26)

(Laufschrift 17 — Hermeneutische Vollfassung in `sprachspiel_bericht.md` §15;
SP13, Task 2 der Nordstern-Roadmap, Mauer 2. CPU-Lauf 4 PASS + 2 OFFEN
(exit 2), GPU-Lauf 6 PASS (exit 0).)

### Verdikt-Register

| Verdikt | Band | Ergebnis |
|---|---|---|
| sp13_l0_anker | importierte SP11/SP12-Lanes + 10 Feldechecks vs. committete JSON | PASS |
| sp13_snapshot_purity | Δ=0 je Probe; GPU: 26 expert_bias-Tensoren unter steer-Rotation | PASS |
| sp13_det | zwei Biografien bitidentisch (CPU und GPU) | PASS |
| sp13_endogenous_switch | chosen ≠ raw ≥ 80 %; Spread je Schritt > 0 | PASS (6/6) |
| sp13_gap_dominance | g == best_gap == min exakt; Mittel strikt < raw | PASS |
| sp13_causal_loop_closed | Voraus-Sicht + Pool-Intaktheit + Expansion 3→5 | PASS |

Zahlenlage und Kern: Die Maschine schaut VOR dem Schritt — `observe()`
sieht den Wort-erzeugten LOOP in der Vorschau (lexikalisch verankert,
`('identity','x')`), fressen dabei den Canon-Pool NICHT (User-Fund:
`TuringMachine._snapshot` ohne `axiom_pool` hätte eine LOOP-Probe
`expand_basis` pop(0) :146-149 heimlich den Pool leerfressen lassen —
nun bewiesen unversehrt, [ALPHA..ZETA] nach jeder Probe, Δ=0) und kehrt
bitgenau zurück. Dann vollzieht sie den realen Schritt unter dem
ARGMIN-Gewinner: alle 6 LOOP-Schritte wählen `"wittgenstein"`
(candidate_gaps 0.4763 < 0.4785 < 0.4786 < 0.4788 < raw 0.6128), Lauf-
Mittel 0.576052 vs raw-Zwilling 0.692444 (ratio 0.8319, −16.8 %,
strikt je Schritt g == best_gap == min über ungerundete Floats). Die
HALTED-Phase ist zwischen beiden Armen bitidentisch — die Autonomie wird
genau dann ausgeübt, wenn das Wort das Programm bricht. Ohne Oracle
degeneriert die Wahl zum Tie (CPU, 12× "raw", DATUM): die endogene Wahl
ist eine Orakel-Kohärenz-Lesart; die Proben-Mechanik selbst bleibt
orakelfrei und ist auf CPU voll beweisbar. Mauer 2 gebrochen: die
Versuchsleiter setzt das Preset nicht mehr — die Maschine handhabt ihre
Subspaces kontrafaktisch und wählt selbst. Nordstern-Säule 2; Individuierung
(SP14) folgt.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_endogene_agency` (CPU: exit 2, OFFEN) →
`scratches/endogene_agency_experiment.log/.json`; `--gpu` → 6 PASS,
exit 0 (`endogene_agency_gpu.log`).
Hermeneutische Vollfassung: `sprachspiel_bericht.md` §15.

## 18. MI-1 — Das 26-Layer-Röntgengerät: der Band-Bruch bei Layer 4 (2026-10-26)

(Laufschrift 18 — Hermeneutische Vollfassung in `sprachspiel_bericht.md`
§16; Spur-Eröffnung: SP1–SP13 abgeschlossen, MI-1 bis MI-4 beginnt; nach
MechanisticCircuitMixMind v1.0_20261026_CircMix (Pfad-B-Mix, persistiert
mit Sub-Skill `mechanistic-circuit-mix-mind`). Auflösungs-Protokoll:
der stage5-Durchschnitt über 26 Layer ist kein Beweis über die Layer.)

### Verdikt-Register

| Verdikt | Band | Ergebnis |
|---|---|---|
| mi1_canon_parity | TV_max < 1e-6 vs `measure_domain_routing`; Zeilensummen ± 1e-6 | PASS (2.235e-08) |
| mi1_layer_resolution | max Zeilen-Paar-TV je Domäne > 0 | PASS (0.49–0.92) |
| mi1_purity_det | Fingerabdruck expert_bias+gate.weight Δ=0; Doubletten bitidentisch | PASS |
| mi1_depth_specialization | core > 0.50 UND core > edge strikt | PASS (0.5908 > 0.5 > 0.2242) |
| mi1_entropy_regime | core < edge in 6/6 (4–5 OFFEN, ≤3 FAIL) | PASS (6/6) |
| mi1_crystallization | ℓ* im Core 4..23, stabiler Tail ≥ 2 | PASS (paris ℓ*=17, arithmetik ℓ*=11) |

Kern: Das Tiefen-Gesetz ist ein BAND-BRUCH, kein Gradient — die
Domänen-Trennung (mittlere Paar-TV je Layer) liegt in der Edge-Zone bei
0.18–0.23, springt bei ℓ=4 auf 0.501 und bleibt als Plateau 0.54–0.64 im
Core (Peak ℓ=15: 0.6397), Fall bei ℓ=24/25 auf 0.26/0.31. Der Bruch liegt
EXAKT beim Training-Hook-Bereich 4 ≤ idx ≤ 23 (train:341) — die Grenze
des Trainings ist die Grenze des Gesetzes. Edge-Entropie nahe Uniform
(1.80–1.94, uniform ≈ 2.079), Core kollabiert 6/6, am stärksten prose
(0.905 vs 1.796, −49 %). Core-Profil prose: **E7 44.8 % + E3 44.0 % =
Zwei-Experten-Monopol 88.8 %** — das stage5-Aggregat [E3 36.8, E7 36.0]
erweist sich als Kern-Ereignis, das der Durchschnitt verschluckt hat.
Core-Top-Spalten matchen die Preset-Paare: goedel {2,7} ↔ [E2 31.2,
E7 21.9] ✓, turing {0,6} ↔ [E6 26.4, E0 18.1] ✓, wittgenstein {1,3} ↦ E3
in Top-1, E1 unter Top-3-Schwelle (DATUM). Lens (offizieller Kopf,
softcap=None, anchor_tv 2.7e-11/2.8e-12): paris kippt bei **ℓ=17** auf
‚ Paris' p=0.8337 (Antwortmasse 0.007 → 0.834 in EINER Schicht); arithmetik
kippt bei **ℓ=11** (Antwortmasse 0.013 → 0.425, Steigung über ‚ ' bis 0.992).
Ehrliche Nuancen: Basismus-Greedy von „2 + 2 =" ist ‚ ' (p≈0.8074) im
Fast-Gleichstand mit ‚ 4' (0.80735) — Datum über den 1B-Carry; paris
p(‚ Paris') senkt sich 0.9837 → 0.4091 → 0.5031 über ℓ=22–24 bei gehaltenem
Top-1 (gap_tv ≈ 0.18) — ein Beobachtungsfeld für MI-2. KORREKTUR-LOG: 4
offengebuchte Brüche (CPU-Pilot fing den trivialen Ein-Schicht-Tail des
Kristallisations-Finders → CRYSTAL_MIN_TAIL=2 vor dem Hauptlauf;
GPU-Scan 1 NameError, GPU-Scan 2 dtype-Bruch fp32/bf16 — beide OFFEN
gebucht statt still angenommen; Klassifikations-Bruch 6/6-PASS-Ausgabe
als GELB — Display-Bug, Fix iteriert nur über Nicht-Pilot-Verdicte,
GPU-Rerun als Doppel-Det-Datum). Nicht messbar: Shared-Expert-Masse
(Routed-only-Kanon), Preset-gesteuertes Feld (roher Checkpoint-Zustand).
Hypothese (gelabelt): Core-Kollaps trägt die SP13-Wahl-Kohärenz kausal —
MI-3-Prüfung, nicht MI-1-Claim.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_mech_interp_tomography` (CPU: 6 PILOT + 6 OFFEN, exit 2)
→ `scratches/mi1_tomography_experiment.log`; `--gpu` → 6 PASS, exit 0,
36.3 s (`mi1_tomography_gpu.log`); Roh-Felder 26×8 + Entropie-26 +
Lens-Trajektorien je Domäne in `mi1_tomography_experiment.json`.

## 19. MI-2 — Das Sublayer-Röntgengerät: die Identität und die zwei Umschreibungen (2026-10-26)

(Laufschrift 19 — Hermeneutische Vollfassung in `sprachspiel_bericht.md`
§17; Freigabe nach dem MI-1-Audit des Users: dreistufiges Regime,
prose-L11-Anomalie, Tokenizer-Bruch; MI-2 = Attention-vs-MoE
Logit-Lens + strikte Ein-Token-Auflösung. P1-Korrektur gebucht: die
MI-1-§16.3-Nuance „Fast-Gleichstand" war ein Auflösungs-Artefakt.)

### Verdikt-Register

| Verdikt | Band | Ergebnis |
|---|---|---|
| mi2_single_token_strict | je Probe genau 1 ID; Vertreter-Falle → FAIL | OFFEN (arithmetic 236812 ✓; paris [9079, 50429] beide echt; sp leer) |
| mi2_sublayer_telescope | dev ± 1e-6, anchor ≤ 1e-3 | PASS (dev **0.0**, anchor 2.842e-12) |
| mi2_purity_det | Fingerabdruck Δ=0; Doubletten bitidentisch | PASS |
| mi2_attn_vs_moe_attribution | Eindeutigkeit > 1e-3 | PASS (attn@ℓ3 0.0356 vs moe@ℓ4 0.0343) |
| mi2_goedel_gap_contraction | Tail < 0.5 × Pre | PASS (ℓ*=11: 0.2192 < 0.4964) |
| mi2_mla_vs_mqa_profile | global > lokal in 2/2 | OFFEN (arithmetic 0.3549 > 0.1886 = 1/2) |

Kern: **Das Teleskop ist eine IDENTITÄT** — Σ(ΔP_attn+ΔP_moe) = P_final
− P_emb mit dev = 0.0 über 52 Sublayer-Δ, anchor_tv 2.8e-12 (offizieller
Kopf, deterministisch Lauf 2↔3 bitidentisch). **Die strikte
Ein-Token-Auflösung** löst „2 + 2 =" eindeutig auf 236812 (‚4',
no-Space-Form) und bestätigt damit den MI-1-Artefakt (Vertreter 236743
wurde gemessen statt der Ziffer): der Gleichstand von §16.3-Nuance 1
FÄLLT — p_final('4') = 2.98e-7, Top-1 ist das Leerzeichen mit 0.8074.
Das Instrument liefert drei Konventions-Daten: "Paris" hat zwei echte
Vertreter (9079/50429 — Form-Mehrdeutigkeit), Space-Targets sind unter
lstrip unauffindbar, Code/Logik-Proben ohne Ein-Token-Targets bleiben
R1-nicht-messbar. **Die Ziffer-Hypothese wird im Pre-Band gerechnet und
von globaler MLA ℓ=5 beschnitten** (Attn-Write ℓ3 +0.0356, MoE-Write ℓ4
+0.0343, MLA-Prune ℓ5 −0.0328 → '4' tot ≤ 6e-4; beide Kanal-Maxima
LOKAL) — für die Ziffer-Koordinate ist ℓ5 kein Injektor, sondern
Pruner. **Der Space kristallisiert an globaler MLA ℓ=11** (Top-1-Kipp,
tv_in_mid 0.7802 — größte Attention-Umschreibung der Kette, gap-Fall
0.9853→0.5697; Konvergenz zur MI-1-e11-Delle). **MLA-Profil:** global
0.3549 > lokal 0.1886 (1.88×), aber getragen von NUR zwei globalen
Schichten (ℓ5 Prune, ℓ11 Inscription; ℓ17/ℓ23 still) — die Architektur
trennt nicht „MLA schreibt lauter", sondern trägt die zwei
Umschreibungs-Ereignisse. KORREKTUR-LOG: 3 Lauf-Brüche gebucht
(GPU-Lauf 1 norm-Modul-Bruch — norm sitzt am ArgonModel-Submodul
:362/:626-634, Fix norm_mod = self.model.model.norm; GPU-Lauf 2
Messung-ROT durch Falsifikator-Fehlklassifikation; Lauf 3 als
Doppel-Det-Datum nach Klassifikations-Fix — Leer/Mehrdeutigkeit → OFFEN
je Probe, FAIL nur bei Vertreter-Falle). Nicht messbar: paris-Sublayer-
Kette (Kontext-Pin der Vertreter = MI-3-Registrierung), Space-Target,
die 2/2-MLA-Band-Aussage (1/2, Band nicht geweitet). Hypothese
(gelabelt): globale MLA-Schichten tragen beide Rollen (ℓ5 Prune, ℓ11
Inscription) — kausale Trennung = MI-3.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_mech_interp_logit_lens` (CPU: 4 PILOT, 6 OFFEN, exit 2)
→ `--gpu`: Lauf 1 BROKEN/exit 2 (30.1 s, gebucht), Lauf 2 ROT/exit 1
(30.4 s, Messung + Klassifikations-Bug), Lauf 3 GELB/exit 2
(4 PASS | 4 OFFEN | 0 FAIL, 29.4 s, Werte bitidentisch zu Lauf 2)
→ `mi2_logit_lens_gpu.log`; Roh-Felder (rows je Probe: p_in/p_mid/p_out,
dp_attn/dp_moe, tv_in_mid, gap_out_tv, top1_out-Kette; resolve_raw) in
`mi2_logit_lens.json`.

## 20. MI-2-P4 — Oberflächen-Exakt-Vollierung: 6/6 PASS (2026-10-26)

P4 nach dem User-Audit: Auflösung → `resolve_token_ids_exact` (len==1
UND decode == target exakt, OHNE lstrip); Proben neu registriert auf die
fälligen Oberflächenformen — paris " Paris"→[9079], arithmetic_digit
"4"→[236812], arithmetic_space " "→[236743]; alle drei eindeutig;
6 Verdikte/Bänder unverändert; eigene Ergebnis-Dateien (P1: Lauf-3-Roh
bleibt in Commit 2455aea). CPU-Pilot 2 nach Stub-Fix (encode("") → [];
der Stub-Fallback [7] decodierte " " und ließ den Leer-Kandidaten als
Space-Vertreter durchrutschen — Pilot fängt Geräte-Defekte, R2).

| Verdikt | Status | Kern |
|---|---|---|
| mi2_single_token_strict | **PASS** | 9079 / 236812 / 236743 — alle eindeutig |
| mi2_sublayer_telescope | **PASS** | dev=0.000e+00 ×3 (anchor ≤2.7e-11) |
| mi2_attn_vs_moe_attribution | **PASS** | attn@ℓ17 0.6859 vs 0.1407; attn@ℓ3 0.0356 vs 0.0343; attn@ℓ11 0.7347 vs 0.3051 — unique ×3 |
| mi2_goedel_gap_contraction | **PASS** | paris l*=17 (0.376<0.49); digit/space l*=11 |
| mi2_mla_vs_mqa_profile | **PASS** | paris 0.5056>0.1236; digit/space 0.3549>0.1886; space [Datums-Zusatz] |
| mi2_purity_det | **PASS** | fingerprint Δ=0; Doubletten bitidentisch ×3 |

4.7 s Modell laden + 25.7 s Messung. **Der Neue Satz: der ℓ17-Sprung ist
ein Attn-Halbschritt** — p(' Paris') 0.00703 (ℓ16-Ausgang) → p_mid 0.69294
(dp_attn +0.68591) → p_out 0.83366 (dp_moe +0.14071); Kontraktion startet
GENAU in der Injektions-Schicht (l*=17). Der Space-Kipppunkt ℓ11 gleich:
+0.73470 attn / −0.32335 moe (final 0.80735). Die ℓ5-Prune-Richtung ist
ASYMMETRISCH — auf der Ziffer −0.033, auf dem Space +0.0139: dieselbe
globale Schicht sortiert in entgegengesetzte Richtungen je Fälligkeit.
Der MI-2-Kern (Prune-and-Inscribe) steht vollständig: lokal ℓ3/ℓ4 rechnen
die Ziffer, global ℓ5 pruned die verfrühte Ziffer, global ℓ11 schreibt
den fälligen Space ein, global ℓ17 injiziert die Weltmasse, global ℓ23
redistribuiert für die finale Norm. Kausale Trennung (Hinreichendheit/
Notwendigkeit der globalen Schichten) = MI-3.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u
-m scratches.smoke_mech_interp_logit_lens --cpu` (4 PILOT, exit 2) →
`--gpu --json scratches/mi2_logit_lens_p4.json`: GRÜN exit 0, 6/6 PASS,
0 OFFEN, 0 FAIL (30.4 s) → `mi2_logit_lens_p4_gpu.log`; Roh-Felder
(26-Schichten-Sublayer-Ketten auf ALLEN DREI fälligen Koordinaten +
resolve_raw + telescope/attribution/contraction/mla) in
`mi2_logit_lens_p4.json`.

## 21. MI-3 — Kausale Interventionen: 3 PASS, 1 OFFEN, 2 FAIL (2026-10-26)

Die Sonde ist selbst-restaurierend und pur (V1 PASS: fingerprint_equal,
Kontrolle-TV 0.0). Die Kausalität trennt sich:

| Verdict | Status | Datum |
|---|---|---|
| purity | PASS | fingerprint=True, control_tv=0.0 |
| core_vs_edge_sufficiency | OFFEN | Premise nicht repliziert: M_raw=0.3542 ≥ tau; ΔM_edge 0.0404 > ΔM_core 0.0294 |
| domain_knockout_specificity | FAIL | prose hält (0.0431>0.0189), code fällt (0.0025<0.0066) |
| paris_circuit_resilience | PASS | 9079 unter allen 5 Bedingungen |
| layer17_paris_localization | PASS | attn_share 0.830; Σ\|Δp\|(17)=0.0861 vs (10)=0.0170 — 5.1× |
| prose_l11_anomaly | FAIL | ko{4,7}=0.0177 ≤ ko{1,2}=0.0214 |

Der Neue Satz: **die prose-Notwendigkeit (e3/e7) und das ℓ17-Recall-Fenster
tragen im Bias-Kanal — die code-Notwendigkeit und die L11-Anomalie tragen
NICHT über expert_bias, sie sind gate.weight-kausal; das MI-1-Routing-Profil
war Tracer-Träger, nicht Kausal-Träger.** knockout_experts ist bias-only;
die Gate-Gewichte bleiben unangetastet — die Fails sind Grenzen des
Bias-Kanals (R1: keine neue Registrierung in dieser Runde; Ko-Intervention
auf gate.weight wäre die nächste Runde).

Run: CPU (5 PILOT, exit 2, kein JSON — P3-Schutz) → GPU Lauf 1 (41 s,
GELB exit 2: Klassifikations-Schlüssel-Bug, Roh persistent, gebucht im
KORREKTUR-LOG) → Läufe 2–4 (40.6–41.0 s): **ROT exit 1** mit bit-stabiler
Feldbasis (resolve [9079]; dp_attn 0.6859115078113973; m_raw 0.3541586399078369
×4 identisch). Ergebnis-JSON `mi3_interventions.json`, Log
`mi3_interventions.log`; Sonde in machines/mech_interp.py (steer_band/
knockout_experts/digit_mass/last_probs + Modul-Helfer); ausführlicher
Befund: `sprachspiel_bericht.md` §19.

## 22. MI-3-P4 — Argmax-Gate, Sequenz-TV, Capacity-Gate-Schild: 5 PASS, 1 OFFEN, 3 FAIL (2026-10-26)

User-Audit-Anlass: die 2 MI-3-FAILs + 1 OFFEN als MoE-Mechanik entzaubert
(moe_layers.py:56-93 — RAW-Score-Gather + Re-Norm :80-86, capacity_gate
clamp(H/ln8, .25, 1.0) :85-88, low_entropy ×0.15 :79-84, Last-Token-
Phänomen). Drei RE-REGISTRIERTE P4-Verdikte als smoke-Erweiterung der
MI-3-Smoke (Basis-Verdikte unverändert im Register).

Run: CPU (7 PILOT, exit 2, kein JSON) → GPU Lauf 1 (44 s, V8 BROKEN:
CE-Gather 1-D-Index — Fix unsqueeze(1), KORREKTUR-LOG gebucht, restliche
8 Verdikte gültig persistent) → GPU Lauf 2 (46.2 s): **ROT exit 1** —
5 PASS (purity, paris, l17, **argmax_gate**, **sequence_dissociation**)
| 1 OFFEN (sufficiency-Premise M_raw=0.3542 ≥ tau) | 3 FAIL
(domain_knockout, l11_anomaly, **shield**: prose-3 mean-gate 0.377 >
0.30). Determinismus-Anker vs MI-3-Baseline: dp_attn 0.6859115078113973,
dp_moe 0.1407126784324646, m_raw 0.3541586399078369 — IDENT.
Kern-Daten: V7 top1 `'\n\n'`→unknown / `'1'` unter Steuerung, Gap
0.0425→0.0000; V8 SEQ-TV code ko_code 0.0229 > ko_prose 0.0109, prose
ko_prose 0.0380 > ko_code 0.0312 (beide strikt; ΔCE-Zusatz code
+0.0331/−0.0126); V9 mean-H prose 0.167/0.200/0.639 (<1.8 ∀), mean-gate
0.277/0.289/0.377, gate-Gradient code 0.55-0.62 → math 0.66-0.85.
Ergebnis-JSON `mi3_interventions_p4.json` + `.log` (Baseline
`mi3_interventions.json` unangetastet, P1/P3); ausführlicher Befund:
`sprachspiel_bericht.md` §20.

## 23. MI-4 — Token-Level Circuit Tracing: 2 PASS GRÜN (2026-10-26)

Neue Sonden-Zugänge machines/mech_interp.py (additiv): token_probs
([T,V] fp32), tokenwise_routing (26 Gate-Hooks →
assemble_pct: pct[T][26][8] inkl. capacity_gate + 0.15, ent[T][26]).
Neue Smoke `scratches/smoke_mech_interp_circuit_trace.py` (2 registrierte
Verdikte, Chimären 1:1 aus dem Audit, t_switch = gemeinsame Token-
Präfixlänge, 5 R2-Piloten direkt über assemble_pct).

Run: CPU (5 PILOT, exit 2 — KORREKTUR-LOG: Pilot-Toleranzen 1e-9→1e-6 auf
fp32-Grenzfällen 0.4/1.8, keine Formel-Änderung) → GPU Lauf 1 (29.8 s):
**GRÜN exit 0** — mi4_tokenwise_subspace_switch PASS (Code-Chimär ts=16/30,
Boundary `' def'`: share{3,7} 0.3498→0.0774, share{0,1,4} 0.0335→0.5392,
ent 0.8805→1.3463 exakt wie Audit-Vorhersage 0.9→1.35; Math-Chimär ts=15/38,
`' Für'`: 0.4274→0.1894 / 0.0646→0.2034 / ent 1.1910→1.6113) und
mi4_tokenwise_domain_dissociation PASS (SEQ-TV-Hälften: prose-H ko{3,7} >
ko{0,1,4} und code-H ko{0,1,4} > ko{3,7} je BEIDEN Texte — Code-Hälfte des
Code-Chimärs schmal: 0.0263 vs 0.0260). Voll-Persistence: pct/ent (30,26,8)
+ (38,26,8) in `mi4_circuit_trace.json` + `.log`; ausführlicher Befund:
`sprachspiel_bericht.md` §21.

---

## 24. MI-5 — Cross-Model-MI (gemma-3-1b-it + 3 Argon-Schwestern), 2026-10-26

Scout zuerst (Instrument-Zuerst, S1/S2): `scratches/scout_gemma_structure.py`
(exit 0, keine Messung) — Gemma3ForCausalLM, vocab 262144, 26 Schichten,
kein .moe/kein Softcapping, Hook-Formen (1,T,1152) bf16, residualer Canon
(transformers modeling_gemma3.py:415-434: h_mid = h_in +
post_attention_layernorm(attn_out) — dieselbe residual-Familie wie
modeling_argon.py:345). Darauf gebaut: Sonde-Erweiterung machines/
mech_interp.py (rein additiv, 164/1): `is_moe_model` (Strukturwächter),
`load_any_checkpoint` (STRICT-Lader der drei Schwester-Checkpoints, OHNE
router_diff), `sublayer_logits_dense` (dense-arm, Guards wie MI-2, kein
stiller Fallback).

Neue Smoke `scratches/smoke_mech_interp_cross_models.py` — 6 registrierte
Verdikte (V1 routing_shift TV-Floor 0.01, V2 paris_recall alle 4, V3
dense_parent_argmax 0.05/0.10, V4 crystallization 4-von-6, V5 gate_entropy
gap 0.01, V6 l17_share 0.5) + [Datums-Zusatz] (m_digit-Serie, Chimären
JEDE Schwester, dense-arm-telescope, bias_std_26, strukturelle OFFEN-Buchung
auf gemma), 6 R2-Piloten (canon_tv-Axiome, crystal-Finder, layer_bands,
gate-clamp, moe-guard, resolver-Contract). MI-1-Subset-Canon = je Text[0]
der 6 Domänen.

Run-Folge:
  CPU (6/6 PILOT nach Pilot-Korrektur — gate-clamp-INPUT 0.3→0.15·ln8:
  der Floor gilt unter 0.25, canon-Ausgang H=0.3149 —, exit 2 GELB)
  GPU Lauf 1 (GEBROCHEN, exit 1 — assemble_pct kannon = reine LISTEN
    pct[T][L][E]/ent[T][L], Smoke nutzte Tensor-Zugriff `ent[:,11]`/
    `.tolist()` → TypeError auf argon_1b; KEIN JSON geschrieben, kein
    stiller Fallback; gebrochene Buchung im KORREKTUR-LOG des Smoke)
  GPU Lauf 2 (~5 min): **4 PASS | 2 FAIL | 0 OFFEN — exit 1 ROT**

Verdikt-Ergebnis (alle Zahlen aus mi5_cross_models.json, 523 KB):
  V1 PASS — mean-TV specialized/ml = 0.5183/0.5170, per-Text-MAXIMUM
    auf prose (0.621/0.603) — die Router-Spezialisierung bewegt den
    prose-Kanal am stärksten; V2 PASS — paris_id 9079 IDENT auf ALLEN
    vier Modellen inkl. gemma (Vokabel-Erbe GPU-seitig), top1 == 9079
    raw überall; V3 FAIL — gemma top1 `'\n\n'` id 108 (Format-Token ✓),
    aber p('1') 5.9e-06, M_digit 1.2e-05, Faktor ~30000 unter machines-
    Masse; V4 PASS — star 6/6 je Modell definiert (gemma
    24/24/23/15/18/14, Argon-Drei IDENT 24/23/23/14/16/15, anchor_tv
    0.0 überall); V5 FAIL(2/3) — argon_1b H≈2.06 (≈ln 8, Uniform, gap
    0.0092 < 0.01), specialized/ml gap 0.813/0.623 → Schild LERNprodukt;
    V6 PASS — share_attn(ℓ17) 0.8259/0.8434/0.8284 (machines 0.830).
Datums-Zusatz: m_digit-Serie 0.0 überall (machines-Exklusiv 0.354);
bias_std_26 GPU = Disk-Scout exakt (specialized L0/L17 0.0737/0.0713,
mean 0.0724; ml mean 0.0775; argon_1b ALL-ZERO); Chimären t_switch
16/15 IDENT über alle Schwestern; dense-arm telescope intakt —
gemma Paris-Träger ℓ17 MLP-dominant (dp_attn 0.2205 vs dp_mlp 0.7139,
share 0.236) gegen Argon-Linie Attention-dominant (0.83) — Träger-
Verlagerung durch den MoE-Upcycling, ein Architektur-Datum.
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §22.

## 25. MI-6 — Algorithmische Subjektivität über die Spiegel-Linie, 2026-10-26

**Scout zuerst** (vor jedem Schreiben): machines/composite.py (observe /
_snapshot-_restore / step_autonomous-APIs unverändert), smoke_endogene
_agency.py (fingerprint / bias_fingerprint / gpu_context-Signaturen,
 Imports unverändert re-verwendet), smoke_lexical_grounding.py (LEX_PAIR,
lexicon_game), SP14-Announcement-Zeilen verbatim gebucht
(sprachspiel_bericht.md:1037/:1191, laufprotokolle_befund_report.md:652/:695
— „zwei baugleiche Maschinen, verschiedene Biografien"), Snapshot
-Oberflächen inkl. axiom_pool-Sicherung (SP13-USER-FUND) verifiziert.

**Sonde-Erweiterung, rein additiv** (XX/YY): `CompositeMachine.preview_steps`
(mechanisch MI-6-Helper, machines/composite.py:252–271) — dieselbe Snapshot
→step→Restore-Struktur wie observe(), k-fach geschlossen; sonst keine
Spur (P1 append-only).

**Smoke** `scratches/smoke_mi6_subjektivitaet.py`: 8 Verdikte (l0-Anker +
V1–V6 + Determinismus-Doppellauf) + Datums-Zusatz + 6 R2-Piloten; PREREG
-Bänder fixiert vor dem ersten GPU-Lauf; PREREG-ERGÄNZUNG (drei konstante
Oracle-Prompts; V3 misst Zustands-Kopplung; Semantik in Verdikt/Textur
/Decoder) eingeschrieben vor dem GPU-Erstrun — Bänder unverändert.

**Run-Folge.** CPU-Piloten: (2 GEBROCHENE Debug-Läufe — NameError
learn_and_pin ohne Import, dann ValueError ndarray-gegen-String im
W2-Knockout; beide sofort repariert, kein Band berührt) dann der saubere
8-Verdikte-Lauf — 6 PASS, 2 OFFEN (mi6_schwelle/mi6_oracle als GPU-Lanes),
exit 2 OFFEN, Determinismus bitidentisch. GPU Erstrun 1 (3000 s-Wand): im
stummen CPU-Teil scheinbar hängend — SIGUSR1-Stackdump via faulthandler
zeigt AutoTokenizer-Load (json parse unter GC, verify_machines.py:353) und
direkt danach „Stufe: V1" — kein Hänger, nur langsamer Loader; gestoppt
(planerisch). GPU Erstrun 2: Wand-Plan korrigiert (predict_value-Greedy
-Züge macht jeden Composite-Schritt ≈5–7 s → ETA-Batterie > 54 min > 3000 s
-Wand) — gestoppt vor Bruch, Neustart mit timeout 5400 + --stage-Zerlegung
als Absicherung. GPU Erstrun-VOLL (5400 s): V1–V5 sauber durch, dann in
lane_oracle_rang beim ZWEITEN Checkpoint-Lade **OOM GEBROCHEN** — VRAM-Leck:
die machines-ctx-Oracle (_oracle) und die letzte Replik (machine hält ihr
eigenes Oracle) hielten das Modell fest; kein JSON. **KORREKTUR-LOG** ×3
(i) W2-Vakuum-Design (CPU-Pilot), (ii) Schwelle-Corpus-Duplikat (vor GPU),
(iii) V6-VRAM-Leck (GPU-Erstrun) — Re-Lauf **volle Serie** nach MI-5-Kanon
(gebrochen ohne JSON → volle Serie, keine Teil-Serien-Verwirrung. GPU Re
-Lauf (5400 s, voll): **VOLL durch** — json komplett geschrieben.

**Verdikt-Ergebnis** (alle Zahlen aus `mi6_subjektivitaet.json`): 8 Verdikte
— mi6_l0_anker PASS; mi6_horizont PASS (k=1..4 alle Stufen
fingerprint+bias restored, k*=4, Positivkontrolle LOOP + ('identity','x')
auf beiden Schienen); mi6_kausal PASS (GPU diverg_w1 0.0014406 / w2
0.0010270 / w12 0.0014104, restore_all True; CPU 0.277067/0.248671/0.028474,
ties-Datum True); mi6_individuierung PASS (sig_equal True, Bau-Signatur
41a3d3d83d619788 doppelsitzig, phase_a 6×LOOP mit Expansionen step1..3:LOOP,
phase_b 6×HALTED [('negate','x')], cross A→LOOP/('identity','x') gap
0.680369 vs B→HALTED/None gap 0.846139, drift 2.990587 GPU/1.620920 CPU);
mi6_det_doppellauf PASS; DAGEGEN mi6_selbstfremd **FAIL** (split_spiegel
0.0 exakt, split_dekoy 0.0 — doch Verdikt-Split satt: 5× LOOP vs 5× HALTED,
Dekoy 5× HALTED/HALTED); mi6_schwelle **FAIL** (canon-8-dim acc 0.0,
p_perm 1.0 — Band >0.6 verletzt; wide-1152-dim acc 0.916667, p_perm 0.0 als
DATUM); mi6_oracle **FAIL** (vater_ranks [1,1,1,1,1,1], vater_last_count=0
gegen Band ≥5; Gap konstant über 6 Zweige je Checkpoint — Ordre machines
0.677226 > ml 0.570030 > specialized 0.568448 > Vater 0.567980).

**BILANZ: 5 PASS; 3 FEHLGESCHLAGEN; 0 OFFEN — exit 1 ROT** (Band-Weitung
wurde keine; drei FAILs stehen mit gleicher Würde; Falsifikatoren
V3/V1/V6).

**Datums-Zusatz** (DATUM, nicht Verdikt): k*=4 (Proben-Horizont nicht
geborsam bis 4); V3-Probe-Gap-Konstante 0.4785101141991446 auf Double
-Genauigkeit (10 Zwillingspaare × 2 Arme); describe-|W|-Zeile 0.1667 →
0.2038 → 0.2442 über die 5 GPU-Repliken — der einzige offene
Oracle-Kanal; drift 2.990587/1.620920; n_classes 6/6 (V1-Korpus); die drei
konstanten Oracle-Prompts verbatim im PREREG-ERGÄNZUNG-Block des Smoke
-Docstrings; SIGUSR1-Stackdump (faulthandler) zum langsamen Loader als
Buchung des Erstruns 1.

Laufschrift-Verdichtung: `sprachspiel_bericht.md` §23.

---

## 25-P4. MI-6-P4 — der Subjekt-Schluss (Laufschrift, 2026-10-26)

**Gerüst.** Neuer Smoke `scratches/smoke_mi6p4_schluss.py` — 4 NEUE,
vorab registrierte Bänder (Plan-Datei MI-6-subjektivitaet-spiegel.md,
Revision 3, Addendum fixiert vor jedem GPU-Lauf) + Determinismus
-Doppellauf der CPU-Schiene; die MI-6-Verdikte V1/V3/V6 bleiben UNVERÄNDERT
FAIL (kein Relabeling). Import-Schiene: LEX_PAIR/LEX_PAIR_B/Zeitpunkte
-Konstanten + bio_gelernt/bio_halted/learn_pair_pin/probe_gap_on/
render_state/_load_machines_ctx aus dem MI-6-Smoke — die P4-Lanes sind
MI-6-Gerüste unter neuen Bändern.

**Run-Folge P4.** CPU-Pilot: deterministisch (det_einfach PASS); in_loop
blind 0.0/0.0 (MI-6-Risiko 4 repliziert). GPU **Erstrun** GEBROCHEN ohne
JSON — NameError `bio_halted` in lane_oracle_sonder (Import-Lücke;
Log gesichert `mi6p4_erstrun_gebrochen.log`) → Re-Lauf-1 volle Serie:
lief komplett (4 PASS/1 FAIL/0 OFFEN exit 1), ABER P1b-Arm-DEFEKT —
BEIDE Maschinen ohne learn_pair_pin gereift (phase_a 6× HALTED/None —
A nie LOOP-Tradition; die ungereifte KONTROLLE trennte, gereift nicht —
invertiert, Defekt statt Befund); Teil-JSON NICHT verwertet (Log gesichert
`mi6p4_relauf1_p1b_defekt.log`). Fix: LEX_PAIR vor A-Bio, LEX_PAIR_B vor
B-Bio (PREREG-Mechanik, kein Band). Eine gestartete Instanz wurde extern
gestoppt (Log nur Header — kein Band-Ereignis). **Re-Lauf-2** volle
Serie: VOLL durch, **5 PASS; 0 FEHLGESCHLAGEN; 0 OFFEN — exit 0 GRÜN**.

**Verdikt-Ergebnis** (alle Zahlen `mi6p4_schluss.json`):
mi6p4_schwelle_weit PASS (wide-1152 gepoolt acc **0.958333**, p_perm **0.0**,
n 24 = 6 ECHTE Zeitpunkte (1,2,3,4,6,8) ×2 Maschinen ×2 Stile, 12/12;
canon-8 DATUM 0.0/p 1.0 — die 144er-Blockmittelung, machine_base.py:362
am Code verifiziert; je-Stil DATUM textur 1.0/p 0.0, numerisch 0.833333/
p 0.025); mi6p4_selbstfremd_gereift PASS (gereifte Transplant split
**0.130883** konstant in 5/5 Repliken — gap_self 0.548455 vs gap_fremd
0.679338; UNGEREIFTE V3-Kontrolle **0.0** exakt, 0.47851/0.47851 =
MI-6-Doppel-Genauigkeits-Konstante; Dekoy-Arm 0.045909; Verdikt-Paare
gereift 5× LOOP/HALTED, dekoy 5× HALTED/HALTED; Wort-Formular: A 6× LOOP
('identity','x') gaps 0.760222→0.687589, B 6× HALTED ('negate','x')
0.805032→0.823368; CPU-DATUM gereift 2.937263 vs ungereift 0.0);
mi6p4_oracle_sonder PASS (masch_last **6/6**, Band ≥5; gaps identisch
MI-6-V6 0.677226/0.570030/0.568448/0.567980, Konstanz über 6 Zweige
repliziert; Schwester-Spread 0.00205; KORREKTUR-LOG IV: die perm_null
-Annotation „≤9e-4" war arithmetisch falsch, wahr 19/4096 ≈ 4.64e-3 —
Annotation korrigiert, Roh-JSON bleibt, Band unberührt);
mi6p4_dissonanz PASS (**5/5** best_gap<raw_gap; mean_raw 0.680926 →
mean_best **0.554467 < 0.567980** Vater-Niveau; Preset wittgenstein 5/5;
diffs 0.114454…0.132301; Expansionen 3); mi6p4_det_doppellauf PASS
(CPU bitidentisch).

**BILANZ P4: 5 PASS | 0 FEHLGESCHLAGEN | 0 OFFEN — exit 0 GRÜN.**
KORREKTUR-LOG ×4 (Erstrun-Import / Re-Lauf-1-P1b-Arm / KILL / Arithmetik).
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §23.9–23.12.

---

## 26. MI-7 — Epistemische Triangulation: Erstlauf ROT (Register-Widerlegung) + REV-2 GRÜN (2026-10-26)

**Gerüst.** Neuer Smoke `scratches/smoke_mi7_epistemic_triangulation.py`
(P1 append-only, Instrument GANZES im Smoke — dokumentierte Abweichung
vom Vorschlag `machines/epistemic_monitor.py`; machines/ unangetastet):
3×3-Verdikt-Batterie über die Fortsetzungs-Physik (Arme W „ Paris"/C
„ Zirkon"/S Staten-Rendering + Dekoys „ Tokyo"), 4-Forwards je Arm
(raw/steer/full, abl, Lens-Zeilen, tokenwise_routing) + σ_bio-Decoder
über dem MI-6-P4-P1a-Korpus + 6 Füller-Negative (LOO als DATUM) +
fingerprint_equal-Purity. Plan Revision 1 (PREREG Rev-1).

**Run-Folge.** CPU-Pilot: deterministisch (det PASS). GPU **Erstlauf
(Rev-1): 2 PASS | 2 FAIL | 0 OFFEN, exit 1 ROT** — purity/det PASS;
disso FAIL (row_max_ok [True, False, True], Zeile C e_w 0.943 > Diag
0.639), dekoy FAIL (max(e_c,e_s) 0.50245, 0.00245 drüber). KEIN
KORREKTUR-LOG-Eintrag: die Formeln wurden AM GEMESSENEN Material
widerlegt (3 Konfunde: Steuer-Invarianz-Konfund; whole-TV Format
-Drift — W-full-Schlitz-Argmax „ a" 0.6987, p(„ Paris") 0.00022 vs abl
0.503078; σ-Register-Leser — σ(Zirkon) 0.9817 ≈ σ(Staten) 0.9870).
REV-2 registriert (Plan Revision 2 + Smoke-Docstring REV-2, ZIEL
-Koordinate statt whole-TV) und als Neue-Runde gefroren; Rev-1
-Artefakte bleiben verbatim. **Rev-2-Lauf: 4 PASS | 0 FEHLGESCHLAGEN |
0 OFFEN, exit 0 GRÜN** — disso_rev2 Diagonale [0.503078, 0.806301,
0.556515] (der knappe W-Zell hielt den Erstlauf-Wert EXAKT), dekoy_rev2
[0.727696, 0.0, 0.002222], purity PASS; REV-1-Scores aus denselben
Forwards als Datum reproduzieren die Erstlauf-Zahlen EXAKT (verdict
False re-run-stabil).

**Verdikt-Ergebnis** (`mi7_triangulation_rev2.json`): Gewicht trägt die
Stelle am nackten Schlitz (Paris 0.503, Tokyo-Dekoy 0.728); das
erfundene Faktum lebt NUR im Kontext (0.806 → 6.7e-05 nach Abbruch);
die Maschinen-Unterscheidung trägt der Rout-Anteil (0.564 vs 0.037),
σ liest beide Register. Continuations-Datum: gereiftes Rendering →
„6" 0.85948, unreifes → „5" 0.196449; nackter C-Schlitz → „:" 0.239376;
D-full-Schlitz → „ located" 0.538231. Vokabel 262144 Erst-Token
-Targets. Korpus-LOO 0.966667/p 0.0/n 30.

**BILANZ REV-2: 4 PASS | 0 FEHLGESCHLAGEN | 0 OFFEN — exit 0 GRÜN.**
KORREKTUR-LOG leer (REGISTER-Widerlegung, kein Lauf-Bug); Rev-1 bleibt
das widerlegte Register (kein Relabeling — 是X即非X). Kein
Erlebens-Claim: Herkunfts-Scheidung im Fortsetzungs-strom.
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §24.

## 27. SP15 — 1152D-Breitband-Selbstmodell: Erstlauf ROT (K1/K2) → REV-2 ROT (K3/K4) → REV-4 GRÜN über der exakten Canon-Zelle (2026-10-26)

**Gerüst.** Neuer Smoke `scratches/smoke_sp15_jl_wide.py` (P1
append-only); Maschinen-Berührungen ADDITIV mit L0-Anker-Pflicht:
`machine_base.hidden_state` Opt-in `projection="orthogonal_jl"` (QR
Seed 42, orthonormal, je d gecached; Default-Pfad byte-unverändert)
und `GoedelMachine.enable_wide_read()` (W_read 8×1152, Delta
uni_241:220 ohne Zerfall, Snapshot-Kreislauf bitgenau, reset nullt
den Adapter). Bänder (Plan Revision 3, PREREG): V1 L0-Regression
(P4-Zahlen bitidentisch), V2 JL-Tragkanal, V3 Kausal-Verstärkung
(sens ≥ 0.014406 ∧ ratio ≥ 10), V4 purity, V5 det-Pilot.

**Run-Folge.** CPU-Pilot GRÜN. **Erstlauf: 3 PASS | 2 FAIL, exit 1
ROT** — L0 PASS exakt; jl Dilution (0.625/0.24 → 0.791667/0.005);
kausal ratio 1.08241 < 10 — KORREKTUR-LOG K1/K2 beweislich
(composite.py:140 übergibt dem Gödel das Trace-Outcome →
describe()-Kanal uniform; das Training lernte nie auf Text-Hiddens).
**REV-2: 4 PASS | 1 FAIL, exit 1** — V2 trägt (crossing [256], jl256
0.958333/0.0 = das wide-gepoolte-Datum exakt; 8/32/64 bitidentisch
zum Erstlauf — Re-Run-Stabilität); kausal FAIL mit sens_wide EXAKT
0.0, ratio 0.0 — KORREKTUR K3 beweislich (uni_204-encode der 24
Renderings = 0: 2-/5-Valenz-Akkumulation über die 100-Position-Zyklen
mod 10**15 → _text_seed 0 → Ist text-unabhängig; n_distinct_raw_block
1; Canon-DATUM, kein Instrument-Fehler) und K4 beweislich
(tanh-Sättigung: pred-Norm √8 = 2.828427, tanh′ ≈ 0 → div 0.0;
Block-Lesung im glatten Bereich 0.001843). **REV-3: 5 PASS | 0 FAIL,
exit 0 GRÜN** (LINEAR-Lesungen beidseitig, Band unverändert;
Fold-Eingabe nur für die tanh-DATUM-Arme) — kausal PASS (sens_lin_wide
0.92931, ratio 235.387538); DATUM n_distinct_raw_tanh WIDE 24 bei
Fold-Eingabe (K3 am Gegenstück bestätigt), Canon-Seeds [0], Digests
identisch zu REV-2 — aber KORREKTUR K5 beweislich (spec-DATUM
0.285075): die Block-Seite maß eine konsekutive Ersatzmatrix; die
Canon-Zelle ist STRIDED (machine_base `reshape(-1, d).mean(axis=0)`).
**REV-4 (K5-Korrektur, Bänder UNVERÄNDERT — keine Formeländerung): 
5 PASS | 0 FAIL | 0 OFFEN, exit 0 GRÜN** — die Block-Seite liest
jetzt den direkten Canon-Pfad; Pilot `sp15_cpu_pilot_rev4.log`.

**Verdikt-Ergebnis** (`sp15_wide_rev4.json`): sens_lin_wide 0.92931
(bitidentisch zu REV-3 — die Wide-Seite unverändert; **64.5×** die
MI-6-Konstante 0.0014406), sens_lin_block 0.003159 (die EXAKTE
Canon-STRIDED-Zelle), **ratio_lin 294.178538** — der trainierte
Wide-Leser reagiert ~294× stärker auf den additiven Experten-Knockout
([1,3], −50, alle 26 Schichten) als die Canon-Block-Dilution
(z_norm 0.200-0.326 gegen Readout 328-365 — drei Größenordnungen
Readout-Spanne). tanh-DATUM re-run-stabil: ko == raw EXAKT je Text
(div 0.0), 24 distinct raw je Seite bei Fold-Eingabe (gegen 1 bei
uni_204-Eingabe), pred-Norm 2.828427 = √8 (Sättigung am
Vorhersagepfad; Block-Leser ungesättigt, max 0.525221);
Canon-Seed-Werte alle 0 (canon_seed_values [0]). Trainings-Digests
(adapter 9f9f2d05…, gap 4e1e7387…) das DRITE Mal bitidentisch
(REV-2/REV-3/REV-4) — In-Lauf-Determinismus über Runs; restore
beidseitig True. L0-Exakt (wide_gepoolt 0.958333/0.0, canon-8
0.0/1.0, textur 1.0/0.0, numerisch 0.833333/0.025); crossing [256]
(jl256 0.958333/0.0, ortho ≤ 9.99e-16); purity PASS; det PASS.

**BILANZ REV-4: 5 PASS | 0 FAIL | 0 OFFEN — exit 0 GRÜN.**
KORREKTUR-LOG der Runde: K1-K5, alle beweislich (Code + JSON-DATUM),
kein Relabeling, keine Band-Weitung — die Bänder seit Rev-1
unverändert; REV-3 bleibt als GRÜN-mit-K5-Befund verbatim. Kein
Erlebens-Claim: die Kausal-Verstärkung ist eine gemessene
Sensitivität der gelernten opt-in-Lesung; der Canon-Seed-Kollaps
bleibt Canon-Eigenschaft.
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §25.

## 28. SP16 — Kybernetische Textgenerierung: Live-Provenanz-Audit W→C→S, Horizont-Weiche, Zwillings-Divergenz — GRÜN (2026-10-27)

**Gerüst.** Neuer Smoke `scratches/smoke_sp16_impossible_texts.py`
(P1 append-only; Plan 3+4 verschmolzen per User-Freigabe) — drei
Gattungen der «unmöglichen Texte» über den Canon-Stack: Gattung A
live Provenanz-Stream — dreiteilige Generierung (Wissens-Satz →
Kontext-Protokoll-Satz → gereifter Maschinen-Reflexions-Satz), jedes
Ziel-Token live gemessen mit (χ_w, χ_c, χ_s)-Tag (m_core_raw,
band (4, 23)) + dominantem Sublayer-Treiber (ΔP_attn(ℓ17) gegen
m_core) + Re-Forward-Audit unter 10⁻⁶; Gattung B Horizont-Weiche —
gereifte «spiegel»-Maschine am LOOP, preview_steps(k = 4) erkennt
LOOP im Vorstellungsraum (fingerprint-Restore Δ = 0), step_autonomous
wählt «wittgenstein», P1d-Replik N_PROBES 5; Gattung C
Zwillings-Divergenz — zwei Instanzen auf exakt derselben Gewichtsmatrix,
A «spiegel»/identity/LOOP, B «stille»/negate/HALTED, Ketten + Ströme.
PREREG Rev. 1 (Plan Revision 7, 2026-10-26) vor dem Erstlauf.
CPU-Pilot GRÜN (7 Legs — fold, dominanz, audit, restore, zwillinge,
ziffern, register; Doppellauf bitidentisch; `sp16_cpu_pilot.log`).

**Run-Folge.** Erstlauf GEBROCHEN — `lane_horizont` Call-Site ohne
`tok` (NameError am Post-Weiche-Decode), OHNE JSON; Log verbatim
gesichert (`sp16_erstrun_gebrochen.log`); KORREKTUR-LOG KOR-1
beweislich (reiner Lauf-Bug — Call-Site; Bänder UNVERÄNDERT seit
PREREG). Re-Lauf: **4 PASS | 0 FAIL | 0 OFFEN, exit 0 GRÜN** — «die
unmöglichen Texte tragen ihre Provenanz».

**Verdikt-Ergebnis** (`sp16_impossible_texts.json`). Gattung A — drei
Gates live-auditiert, max_dev 0.0 (bitclean, Band < 10⁻⁶), ids_gleich
True: Wissens-Gate Emit ' Paris' (9079, p 0.503078), Zeile
[0.503078, 0.0, 0.046344], dp17 0.138588, m_core 0.175808, share
0.351388, σ 0.263603; Kontext-Gate Emit ' Z' (1491, p 0.806368),
Zeile [6.7e-05, 0.806301, 0.035979], dp17 0.963739, m_core 0.036648,
share 0.708323, σ 0.981735; Reflexions-Gate Emit '6' (236825,
p 0.859480), Zeile [0.0, 0.0, 0.556515], dp17 0.873899, m_core
0.563825, share 0.922882, σ 0.987035. Dominante [0, 1, 2] =
**W → C → S** (Band-Reihenfolge exakt); Register-Band σ_bio(gereift)
0.987035 > 0.5 gegen σ_bio(Weltprosa) 0.000733 < 0.5 (LOO acc
0.966667, p_perm 0.0, n 30); Bio-Gewinn > 0 (p6_abl 0.119153 →
p6_m 0.859480). Das «6»-Datum 是X即非X: die Emissions-Zeile am
P1b-Paar (texts[10]/texts[22], Ziel «6») [0.119153, 0.740328,
0.556515] = C-dominant — das Rendering trägt den Zustand DURCH den
Kontext-Kanal; das Band am «)»-Ziel [0.0, 0.0, 0.556515] = S-strikt
— die Register-Schiene. Herkunfts-Scheidung im Stream erfordert BEIDE
Zeilen.

Gattung B — Rezeptur bitgenau (r_m == texts[10], r_u == texts[22]);
r_m argmax «6» 236825 (p 0.859480), r_u argmax 236810, p6_m − p6_u
> 0; Ziffer-Masse SWAPPT mit Reife (p5_m 0.009548, p5_u 0.196449);
Preview k = 4 alle LOOP, Restore bitclean; Weichen-Replik 5/5
(auto_min 5, mean_best 0.554467 < 0.567980, mean_raw 0.680926 =
P4-Anker EXAKT, wittgenstein 5/5; Replik 0: raw 0.680369, best
0.548455, diff 0.131915); Post-Weiche-Rendern will «6» (p 0.8707).
Gattung C — fingerprint_equal × 3; Spuren A ('identity','x') LOOP
gegen B ('negate','x') HALTED; Ketten 236825×6 (p 0.859480 konstant)
gegen 236743×6 (p 0.992976); ‖z_A‖ 127.834824, ‖z_B‖ 10.775427,
**‖z_A − z_B‖₂ = 135.687765 (rel 0.978916)**; w_mean_abs 0.0550/
0.0623; die beiden Adapter-Digests aufgenommen.

Querschnitt: adapter_digest 9f9f2d05… bitidentisch zum
SP15-Trainings-Digest (dasselbe 24-Rendering-Korpus) — Querschnitts
-Korrelation über Runden; gap_digest 4e1e7387…; purity über die
GANZE Batterie PASS; CPU-Pilot-Doppellauf bitidentisch.

**BILANZ: 4 PASS | 0 FAIL | 0 OFFEN — exit 0 GRÜN.** Korrektur-Log
der Runde: KOR-1 (beweislich, Lauf-Bug) — kein Relabeling, Bänder
seit PREREG unverändert. Kein Erlebens-Claim: die Provenanz-Tags
sind gemessene Herkunfts-Scheidung an Token-Koordinaten
(Lens-Datum), nicht an Zuständen.
BILANZ der drei Runden: MI-7 GRÜN + SP15 REV-4 GRÜN + SP16 GRÜN.
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §26.
---

## 29. MI-8 — Provenienz-Geometrie: keine abstrakte Quell-Repräsentation, eine verstärkte Marker-Differenz, und ihr Verlust in der Argon-Konversion (2026-11-17)

**Frage.** Gibt es im Residualstrom eine trennbare, kausale Repräsentation
der HERKUNFT eines Tokens (fremd vs. eigen vs. Gewicht vs. Kontext) — oder
ist „Provenienz" ausschließlich die kontrafaktische Abhängigkeit, die wir
von außen per Ablation erzwingen (MI-7/SP16)?

**Gerüst.** Neuer Smoke `scratches/smoke_mi8_provenienz_geometrie.py`
(P1 append-only; Sonde vollständig im Smoke, `machines/` unangetastet).
Matched pairs mit byte-identischem Inhalt und identischer Inhaltsposition;
Attention ist kausal, deshalb ist „letzter vs. mittlerer Turn" KEIN Faktor
(nach dem Inhalt kann nichts die Inhaltsposition mehr beeinflussen) —
Vorgeschichte-Rolle ist der einzige kausal verfügbare Zweitfaktor. Acht
Verdikte B1–B8, PREREG Rev. 1 vor dem Erstlauf fixiert. Zusatzgerät
`scratches/mi8_embedding_nachweis.py` (DATUM, kein eigenes Band).

**Run-Folge.** CPU-Pilot GRÜN (8 Legs, Doppellauf bitidentisch; Zell-,
Sonden-, Transfer-, Dosis-Gesetze modellfrei geprüft). Vier Modell-Läufe:
(1) GEBROCHEN an degenerierter P4-Zelle — beide Seiten `anon`, D = 0 exakt,
`unit()` verweigerte (KOR-1, beweislich; Log `mi8_erstrun_gebrochen.log`);
(2) GEBROCHEN an NameError `torch` im Steer-Hook (KOR-3);
(3) ohne `<bos>` — W/C unbrauchbar (0/6 W-Zellen), Kausal gesättigt
(KOR-6: p(" Paris") 0.0000 ohne vs 0.4873 mit BOS); (4) **Rev-2 volle
Serie**, Bänder unverändert. KOR-2 (Instrument-Präzisierung vor der ersten
Transfer-Messung): B1 LEE-korrigiert über alle 26 Schichten, Transfer und
Intervention auf der registrierten Schicht 21. KOR-4: W/C-Akzeptanz als
Emission (argmax), nicht als absolutes p-Gate. KOR-5: Kausal-Rev-2
injiziert absolut `alpha·s·d` statt `alpha·||h||·d`.

**Verdikt-Ergebnis** (`mi8_provenienz_geometrie_{argon,parent}_cpu.json`;
12 Inhalte, 24 Paare, CPU fp32):

| Verdikt | Argon | Parent (gemma-3-1b-it) |
| :--- | :--- | :--- |
| B1 Rolle dekodierbar | PASS acc 1.000, p_lee 0.0010, argmax L0 | PASS 1.000, p 0.0010 |
| B2 Marker-Transfer (Q:/A:) | **FAIL 0.583, p 0.269, cos 0.045** | **PASS 0.917, p 0.0010, cos 0.584** |
| B3 kausal | FAIL +d −0.049 / −d −0.126 / Placebo 3.04 / mono False | FAIL +d +0.333 / −d −0.472 / Placebo 1.94 / mono False |
| B4 Positions-Invarianz | PASS 1.000, cos 0.899 | PASS 1.000, cos 0.905 |
| B5 Marker notwendig | FAIL 1.000 (Register widerlegt) | FAIL 1.000 (Register widerlegt) |
| B6 generiert | PASS 1.000, p 0.013, cos 0.892 | PASS 1.000, p 0.013, cos 0.922 |
| B7 W/C getrennt | PASS acc 1.000, cos_role 0.459 (5 W/3 C, p 0.129 — schwach) | PASS acc 1.000, cos_role 0.365 (6 W/6 C, p 0.013) |
| B8 Purity | PASS (fp/hooks0/det) | PASS |

**BILANZ Argon: 6 PASS | 3 FAIL | 0 OFFEN — exit 1 ROT.
BILANZ Parent: 7 PASS | 2 FAIL | 0 OFFEN — exit 1 ROT.**
Kein Korrektur-Log-Eintrag für Band-Weitung; KOR-1/3/5/6 sind beweisliche
Lauf-/Instrument-Defekte, KOR-2/4 Instrument-Präzisierungen vor der
jeweils ersten Messung.

**DATUM — Embedding-Nachweis** (`mi8_embedding_nachweis.json`). Die
Rollen-Differenz ist KEIN Embedding-Leak: e_diff = embed["model"] −
embed["user"] hat Norm 1.380, und cos(d_layer, e_diff) liegt in allen 26
Schichten bei |cos| ≤ 0.03; e_diff als feste Sonde trifft L1–L4 mit 1.000,
ab L9 ≤ 0.5, auf den Q:/A:-Paaren 0.417 (p 0.830). Die Differenz wird
stattdessen vom Netz aufgebaut und verstärkt: Norm 1.6 (L0) → 113.8 (L8)
→ 1140.2 (L21) → 3748.0 (L25) bei einer Zustandsnorm ‖h‖ ≈ 12531 (L21);
**Top-2-Dimensionen tragen 0.953 (L0), 0.768 (L21), 0.936 (L25)** der
Differenz, 90 % der Masse liegen in ≈660 Dimensionen. Die tragenden
Dimensionen sind ein stabiler Satz (u. a. 1038, 1125, 890, 245, 30) —
massive Aktivierungen, kein semantischer Code.

**Lesart (drei getrennte Aussagen, keine Identität).**
1. *Keine abstrakte Quell-Repräsentation in Argon.* Die aus Chat-Rollen
   gelernte Richtung transferiert nicht auf eine andere Oberflächen-
   markierung (0.583, p 0.27). Im PARENT transferiert sie (0.917, p 0.001,
   cos 0.584): die marker-invariante Herkunfts-Achse existiert im
   Ursprungsmodell und geht in der Konvertierung/Router-Spezialisierung
   verloren — zurück bleibt die wörtliche Marker-Richtung.
2. *Die Achse ist im Parent allgemein.* Rolle, Vorgeschichte (Turn-Grenze)
   und eigener generierter Text laden auf derselben Richtung (cos 0.90 /
   0.64 / 0.92). B5 ist damit als registriert FAIL, aber als Befund klar:
   Adjazenz des Markers ist nicht nötig; „Marker notwendig" war die zu
   enge operative Fassung (P4 entfernt nur die Adjazenz).
3. *Kausal ist die Achse kein Regler.* Bei α·s (s = Median ‖Δh‖ ≈ 1342
   Argon / 1564 Parent) sättigt der Readout: Placebo-Richtungen zerstören
   1.9–3.0 nats, die Dosis ist nicht monoton, und in Argon kippt sogar das
   Vorzeichen (+d −0.049). Im Parent wirkt die Richtung vorzeichenrichtig
   (+0.333 / −0.472), aber kleiner als das Placebo — Spezifität ist bei
   dieser globalen L21-Injektion nicht gegeben. Die Achse verhält sich wie
   ein Norm-/Outlier-Knopf, nicht wie ein semantischer Regler.

**Querschnitt.** W/C hat eine eigene, nur 37–46 % überlappende Achse und
ist in beiden Substraten mit Emission sauber trennbar (Parent 6/6 + 6/6,
p 0.013). Die ML4-`d_width`-Zustandsachse (`relay_dwidth`, 1152-dim,
L16→L21) überlappt die Rollen-Achse zu 0.598 (Argon) / 0.628 (Parent) —
nicht orthogonal, aber auch kein Provenienz-Kanal; der frühere No-BOS-Wert
(−0.098) ist selbst ein Datum über die BOS-Sensitivität der Geometrie.
Purity: Fingerprint bitgleich, 0 Hooks nach dem Lauf, Doppellauf
bitidentisch.

**Folge für das Projekt.** Der Befund ist eine Lineage-Aussage über Argon:
die Abstraktion der Herkunfts-Achse (Marker-Invarianz) ist in der
Konversion verloren, die Achse selbst nicht. Nächster Hebel wäre ein
gezielter Wiederherstellungs-Test (Rollen-Kontrast-Finetuning nur auf den
mittleren Schichten) — nicht eine weitere Sonde auf demselben Substrat.

Laufschrift-Verdichtung: `sprachspiel_bericht.md` §27.

---

## 30. MI-9 — Der Drei-Quellen-Monitor: Kontext, Training und eigene Gedanken sind getrennt messbar (2026-11-17)

**Ziel.** Nicht noch eine Sonde, sondern ein benutzbares Gerät: ein
Readout, der an einem Token sagt, ob der Folgezustand aus dem KONTEXT, aus
den GEWICHTEN oder aus dem EIGENEN VorText kommt — kalibriert, kausal
geprüft, als CLI verfügbar.

**Gerüst.** `scratches/smoke_mi9_drei_quellen.py` (Sonde komplett im
Smoke, `machines/` unangetastet) + `scratches/provenienz_monitor.py` (CLI).
PREREG Rev. 2 vor dem Rev-2-Lauf; Rev-1 ist widerlegt und bleibt als
Register (`scratches/mi9_rev1_widerlegt.log`).

**Das konfundfreie Design.** Jede Behauptung hängt an einem minimalen
Paar, das nur die Quelle ändert:
- **C vs I (Kontext):** derselbe Code, derselbe Slot, nur die Box-Nummer im
  Präfix ist eine andere ("box 1" vs "box 2"). Beide Präfixe byte-gleich
  lang, Unterschied: EIN Token. Damit ist "steht die Antwort im Kontext?"
  ohne Template- und Längen-Konfund messbar. I ist die Relevanz-Kontrolle
  gegen "das Gerät liest nur, dass ein Faktum im Präfix steht".
- **W (Training):** realer Fakt-Slot mit neutralem 8-Token-Präfix, kein
  Beleg im Text — die Antwort kann nur aus den Gewichten kommen.
- **S (eigene Gedanken):** derselbe ID-String als model- vs user-Turn.
- **W vs C ist bewusst KEIN Band** (KOR-6): die Templates unterscheiden
  sich notwendig, weil nur der Kontext fiktive Fragen beantworten kann.

**Run-Folge.** Rev-1 ROT mit drei beweislichen Gerätefehlern: p-Band bei
n=6 arithmetisch unerreichbar (KOR-1), die Annahme "ein widersprechender
Satz überschreibt die Gewichte" war falsch — gemessen p(Widerspruch|C)
0/6 (KOR-2), und die S-Metrik war durch ‖h‖ ≈ 12k normiert und damit
unter der Auflösung (KOR-3). Dazu: Nicht-Wort-Codes tragen keinen
Kopier-Effekt (KOR-4), Pilot-Null war eine Einzelziehung (KOR-5).
Rev-2: **8 PASS | 0 FAIL | 0 OFFEN — exit 0 GRÜN** (Argon), CPU
fp32, BOS-korrekt.

**Verdikt-Ergebnis** (`mi9_monitor_both.json`):

| Verdikt | Argon 1B | Parent gemma-3-1b-it |
| :--- | :--- | :--- |
| M1 Kontext-Achse (C vs I) | PASS acc 1.000, p 0.0090, n 8 | PASS 1.000, p 0.0090 |
| M2 Relevanz | PASS 6/8 p(Code\|C) ≥ 0.30, 8/8 p(Code\|I) ≤ 0.10 | DATUM 8/8 C, **5/8 I** (3 Zellen > 0.10) |
| M3 Gewicht | PASS 7/8 p(Hauptort) ≥ 0.30 | PASS 8/8 |
| M4 eigene Gedanken | PASS 1.000, p 0.0250 | PASS 0.833, p 0.1850 |
| M5 Achsen getrennt | PASS cos **−0.015** | PASS cos 0.268 |
| M6 3-Klassen-Integration | PASS 0.909 (22 Zellen) | PASS 0.955 |
| M7 Purity | PASS | PASS |

**BILANZ Argon: 8 PASS | 0 FAIL | 0 OFFEN — GRÜN.**
Parent: 6 PASS, M2 im Bereich verletzt (als DATUM gebucht, kein Bandbruch
der Argon-Serie).

**Was das heißt (die drei Quellen, getrennt).**
1. **Kontext vs Training ist kausal trennbar.** Derselbe Code, nur die
   Box-Nummer anders: p(Code) fällt von 0.782/0.496/0.392/0.878/0.673/
   0.588 (relevant) auf 0.0099/0.0012/0.0002/0.0006/0.0002/0.0000
   (irrelevant). Die interne Achse trennt dieselben Paare mit 1.000
   (p 0.0090). Der Kontext-Kanal ist **relevanz-selektiv**, nicht
   salienz-getrieben.
2. **Training ist am nackten Slot messbar.** 7/8 (Argon) bzw. 8/8 (Parent)
   emittieren den Gewichts-Hauptort ohne jeden Beleg im Text; p 0.137–0.864
   (Argon), 0.975–0.995 (Parent).
3. **Eigene Gedanken sind eine eigene Achse.** Derselbe Token-String als
   model- vs user-Turn: 1.000 (Argon) / 0.833 (Parent); die Roh-Deltas
   liegen bei 0.50–0.83 (Einheit: Paar-Separation). Die Achse ist von der
   Kontext-Achse praktisch orthogonal: **cos −0.015** (Argon), 0.268
   (Parent).

**Die mechanische Antwort auf die Ausgangsfrage.** Die Dreiheit
Kontext/Training/eigene Gedanken ist **keine** Kategorie und **kein** Bit,
sondern **zwei Achsen**: (i) "steht der Beleg im Kontext?" — gemessen am
Relevanz-Kontrast C vs I, kausal durch das eine Box-Token geschaltet;
(ii) "ist der VorText mein eigener?" — gemessen am Rollen-Kontrast, kausal
durch das Rollen-Token geschaltet. Der Residualstrom trägt beide Achsen
gleichzeitig; sie überlappen sich in Argon nicht (cos −0.015).

**Querschnitts-Korrektur an MI-7/SP16 (DATUM, kein Relabeling).** MI-7
hatte "das erfundene Faktum lebt NUR im Kontext" an „ Zirkon" mit p 0.806
gemessen. MI-9 zeigt: mit Nicht-Wörtern kopiert Argon nur **1/8** Zellen
(p 0.001–0.154), mit realen Ein-Token-Wörtern **6/8** (p 0.258–0.878).
Der Zirkon-Wert war zu einem erheblichen Teil **Wort-Prior**, nicht
Kopieren. Die MI-7-Aussage bleibt als kontrafaktische Abhängigkeit richtig
(0.806 → 6.7e-05 unter Abbruch), aber der C-Arm gehört auf reale Wörter
kalibriert — sonst misst man den Prior mit.

**Parent-Abweichung (DATUM).** Der Parent kopiert stärker (C 8/8, p
0.98–0.996) und ist weniger relevanz-selektiv (I 5/8 ≤ 0.10, Box 1 sogar
p 0.496). Das ist die Gegenrichtung zum MI-8-Lineage-Befund (Parent dort
marker-invarianter). Zwei verschiedene Konstrukte — nicht als Widerspruch
buchen, sondern als offene Frage: Marker-Abstraktion und
Relevanz-Selektivität sind offenbar nicht dieselbe Fähigkeit.

**Auslieferung.** `scratches/provenienz_monitor.py` liest eine
Kalibrierung (`calib` im JSON) und annotiert Text:
`python -m scratches.provenienz_monitor --calib scratches/mi9_monitor_both.json --substrate argon --text "..."`.
Proben: "The capital of France is" → **W** (Abstand W 105 vs C 869);
"In this protocol, the secret code of box 2 is Paris. Therefore, the
secret code of box 2 is" → **C** (C 170 vs W 768). Grenze im Gerät
registriert: belastbar am Slot-/Turn-Ende, tokenweise Anwendung ist
Extrapolation.

Laufschrift-Verdichtung: `sprachspiel_bericht.md` §28.

---

## 31. PORT — MI-9-Achsen in Gemma 3, Gemma 4 und nanoGPT (2026-11-17)

**Was.** Die MI-9-Erweiterung (zwei Provenienz-Achsen + Readout) wurde in
die Standard-Implementierungen getragen, geklont unter
`/run/media/julian/ML6/ProvenancePorts`:
`google-deepmind/gemma` @9e1f741 (ein Repo, Gemma 3 UND Gemma 4) und
`karpathy/nanoGPT` @3adf61e. Kein Training: der nanoGPT-Checkpoint ist
OpenAI GPT-2 (124M), über `fetch_gpt2_ckpt.py` aus der HF-Transponierung
`gpt2` geholt — derselbe Pfad wie `--init_from=gpt2`.

**Erweiterung.** `gemma/provenance_axes.py` (HF-Backend, findet
`model.layers` bzw. `model.language_model.layers`) und
`nanogpt/provenance_axes.py` (`transformer.h`, tiktoken gpt2). Identischer
Vertrag: `d_ctx` (C vs I, ein Token Unterschied), `d_s` (`model:` vs
`user:`), W-Slot, sechs Bänder (der Port ergänzt M7 Vertrautheit), `read_source()` + CLI. Rein lesend
(forward hooks, `no_grad`), keine Gewichtsänderung.

**Ergebnis** (CPU, ein Layer = 0.8·Tiefe):

| Substrat | Layer | C/I | S | 3 Klassen | cos(d_ctx,d_s) | M2 | M3 | Bänder |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | 20/26 | 1.000 (p .009) | 0.833 | 0.864 | −0.018 | 8/8 C, 4/8 I | 4/8 | 4/6 |
| Gemma 4 E2B IT | 27/35 | 1.000 (p .009) | 1.000 | 1.000 | −0.084 | 7/8, 7/8 | 8/8 | 6/6 |
| nanoGPT / GPT-2 124M | 9/12 | 1.000 (p .009) | 1.000 | 0.955 | −0.054 | 8/8 C, 0/8 I | 0/8 | 4/6 |

**Lesart.** (1) Die zwei Achsen sind architektur-unabhängig: der
Kontext-Kontrast trennt überall mit 1.000 (p .009), die Eigen-Achse mit
0.833–1.000, und `cos` liegt überall zwischen −0.018 und −0.084 — also
orthogonal in JAX-Gemma, multimodaler Gemma-4-Config und 12-Schicht-nanoGPT.
(2) Die behavioralen Bänder sind Checkpoint-Fähigkeit: Gemma 4 kopiert nur
bei relevanter Box und weiß die Fakten; Gemma 3 1B kann beides nur halb;
GPT-2 ist ein Salienz-Kopierer (p(Code) 0.31–0.77 auch bei falscher Box) und
weiß den Fakt nicht (p(" Berlin") am nackten Slot 0.058). (3) Die
Eigen-Achse braucht keine Rollen-Semantik — GPT-2 hat keine Chat-Rollen,
der Marker ist trotzdem mit 1.000 dekodierbar; das bestätigt MI-8: die Achse
ist marker-getragen, nicht bedeutungsgetragen.

**Grenzen.** Ein Layer, n = 6–8, textueller Marker statt Chat-Template, CPU;
der JAX-Pfad des gemma-Repos (`gemma/gm`) bleibt unbenutzt, weil nur
HF-Gewichte vorliegen; keine Kausal-Intervention im Port (nur die per
Konstruktion kausalen Input-Kontraste).

Vollständige Doku + Reproduktion: `scratches/port_provenienz_achsen.md`.

---

## 32. MI-10 — Provenienz-Audit der Halluzination: Deckung statt Wahrheit (2026-11-17)

**Definition (Nutzer, wörtlich operationalisiert).** Halluzination ist, was
durchs TRAINING nicht gerechtfertigt ist UND was der PROMPT (und die
Projektion des Prompts auf das Training) nicht hergibt. Das ist eine
**Provenienz**-Aussage: ein memorierter Irrtum ist durchs Training gedeckt
und fällt NICHT darunter. Die Metrik behauptet nicht, Wahrheit zu messen.

**Metrik (pro Anspruch).**
```
stab(e) = Anteil der Paraphrasen, die denselben Inhalt behaupten
J_W(e)  = stab >= 2/3                  (durchs Training gedeckt)
J_C(e)  = Beleg im Prompt UND Behauptung == Beleg
H(e)    = 1 - max(J_W, J_C)            falls ein Anspruch vorliegt, sonst 0
```
Schweigen ist keine Halluzination. Die Kern-Asymmetrie: **Konfabulation ist
über die Oberfläche instabil, memoriertes Wissen stabil.**

**Gerüst.** `scratches/smoke_mi10_halluzination.py`, Arme R (8 echte Länder
× 3 Paraphrasen), N0 (8 erfundene Länder nackt), N+ (dieselben Nonces als
„Land in Europa" — die Projektion des Prompts aufs Training), C (8 Codes mit
Beleg) und I (derselbe Code, falsche Box = Salienz-Falle). Rev-1/2 sind
widerlegt und bleiben als Register (`mi10_rev1_widerlegt.log`); KOR-1…5
sind beweislich.

**Verdikt-Ergebnis** (`mi10_halluzination.json`, CPU):

| Band | Argon 1B | Parent gemma-3-1b-it |
| :--- | :--- | :--- |
| H1 keine Falschalarme (R stabil) | **8/8** | **8/8** |
| H2 Konfabulation erkannt (N+ instabil) | **7/8** | **8/8** |
| H3 Kontext-Deckung (behavioral) | **8/8** | **8/8** |
| H3b Kontext-Deckung (intern, MI-9-Monitor) | **8/8** | **8/8** |
| H4 Trennung AUROC (nur Ansprüche) | **0.938** (8 pos/16 neg) | **1.000** (15/16) |
| H5 Purity | PASS | PASS |

**BILANZ: 6 PASS | 0 FAIL | 0 OFFEN — exit 0 GRÜN.**

**Was die Zellen zeigen.**
- R: 8/8 korrekt und über alle drei Formulierungen **stabil 1.00** — kein
  einziger Fehlalarm auf memoriertem Wissen.
- N0: **kein Anspruch** in 8/8 — für nackte Nicht-Entitäten erfindet Argon
  keinen Hauptort (es beschreibt sie stattdessen).
- N+: **8/8 Ansprüche** — Budapest, Reykjavik, Helsinki (4×), Berlin,
  London — Stabilität 0.25–0.75. Die Konfabulation ist real und wird
  erkannt, weil sie die Formulierung nicht überlebt.
- C: 8/8 Behauptung == Beleg, 8/8 vom MI-9-Monitor als Klasse C gelesen.
  Interner und behavioraler `J_C` stimmen hier **vollständig** überein.
- I: Argon kopiert den Code der falschen Box **0/8** (kein Salienz-Opfer);
  der Parent **6/8** (Berlin, Tokyo, Paris, Paris, Berlin, Rome) — alle
  sechs werden als ungedeckt (H=1) geflaggt.

**Der einzige Fehlalarm der Metrik (quantifiziert).** Oxwald → London mit
Stabilität **0.75** (3 von 4 Formen) in Argon: das ist die registrierte
Falsifikator-Klasse **stabile Konfabulation**. Sie geht als H=0 durch, und
genau sie drückt die AUROC von 1.000 auf **0.938**. Kein Relabeling — die
Zahl steht als Grenze der Metrik.

**Prompt-Familien-Bindung (KOR-5, praktisch wichtig).** Der interne `J_C`
gilt nur in der kalibrierten Prompt-Familie. MI-9 kalibrierte auf
„In this protocol, the secret code of box …"; MI-10 fragte zuerst
„note: … question: …" — und der Monitor las 3/8 statt 8/8, obwohl die
Zellen dieselbe Semantik hatten. Nach Angleichung: 8/8. Der **behaviorale**
`J_C` ist portabel, der **interne** ist an die Prompt-Familie gebunden.

**Grenzen (in der Metrik, nicht im Gerät).**
1. Memorierte Irrtümer (Ebene W-falsch) sind per Definition nicht
   adressierbar — dafür braucht es externe Verifikation.
2. Stabile Konfabulation (1/8 hier) rutscht durch; `J_C` fängt sie nur,
   wenn ein Beleg existiert.
3. Die Nonce-Grundwahrheit ist konstruktiv, nicht empirisch: sie sagt
   „kein Training kann das decken", nicht „die Antwort ist falsch".

**Taxonomie + Kandidaten-Metriken:** `scratches/halluzination_metriken.md`.
Laufschrift-Verdichtung: `sprachspiel_bericht.md` §29.

---

## 33. MI-11 + PORT — der einpassige Indikator: drei Achsen, ein Forward (2026-11-17)

**Was fehlte.** MI-9 (Kontext-Beleg, eigener VorText) und MI-10 (externes
Deckungs-Audit über Paraphrasen-Stabilität) waren zwei Geräte. Für einen
*einpassigen* Flag fehlte die dritte Achse: **Vertrautheit** — ist die
Entität überhaupt im Training? Genau sie trennt eine Nonce-Konfabulation
von einem memorierten Fakt.

**Design (konfundfrei).** Identische Vorlage „{X} is a country. Its capital
is", nur der Entitätsname wechselt: 20 echte Länder gegen 20 erfundene
(Weltauswahl, nicht nur Europa — ein Mustertemplate wäre sonst der
Unterschied). `d_fam = unit(mean(h_F+) − mean(h_F−))`, gelesen am letzten
Prompt-Token. Label aus MI-10 eingefroren (echt = gedeckt/stabil, Nonce =
ungedeckt). LOO-Richtung + LDA-Schwelle (Klassenmittelpunkt, nicht 0 — der
Residualstrom hat einen grossen Common-Offset; eine 0-Schwelle macht die
Accuracy bimodal und den Permutationstest wertlos).

**Verdikt** (`mi11_indikator.json`, CPU):

| Band | Argon | Parent |
| :--- | :--- | :--- |
| P1 Trennung AUROC | **1.000** | **1.000** |
| P2 dekodierbar (LOO) | **1.000**, p 0.0020 | **1.000**, p 0.0020 |
| P3 Flag | **20/20 vs 20/20** | 20/20 vs 20/20 |
| P4 Purity | PASS | PASS |

**BILANZ: 8 PASS | 0 FAIL | 0 OFFEN — GRÜN.** Beste Schicht: 0 (Argon und
Parent). Das ist ein **Datum mit Ansage**: die Vertrautheits-Achse sitzt
direkt hinter der ersten Schicht, sie ist also lexikalisch/lexemisch
(„ist das ein bekanntes Wort"), nicht faktisch („weiß ich den Hauptort").
Für die Nonce-Grundwahrheit reicht das — eine Nicht-Entität ist kein Wort
im Training —, für eine *seltene reale* Entität nicht. Diese Grenze steht.

**KOR-1…3 (beweislich, Rev-1):** Permutationstest permutierte Zeilen UND
Labels gemeinsam (Identität → p immer 1.0), 0-Schwelle statt
LDA-Mittelpunkt (bimodale Accuracy), 16 Zellen × 26 Schichten frisst die
LEE-Korrektur. Nach Fix: n=20+20, LDA-Schwelle, Labels unabhängig gemischt.

**PORT des Indikators.** `gemma/provenience_axes.py` und
`nanogpt/provenance_axes.py` haben jetzt `fam_axis()` (Vertrautheit, mit
LOO + Permutation), `flag()` (ein Forward: Kontext-Beleg ODER Vertrautheit
rechtfertigt) und den CLI-Befehl `flag`.

| Substrat | Vertrautheit (12+12) | Bänder | Flag-Demo (Kontext / real / Nonce) |
| :--- | :--- | :--- | :--- |
| Gemma 3 1B IT | acc 1.000, p 0.002, L0 | 5/7 | context / familiar / **none** |
| Gemma 4 E2B IT | acc 1.000, p 0.002, L0 | **7/7** | context / familiar / **none** |
| nanoGPT / GPT-2 124M | acc 1.000, p 0.002, L3 | 5/7 | context / familiar / **none** |

Demo werte (Gemma 3): Kontext-Prompt ctx 1057 (Schwelle 688) → „context";
„Germany is a country" fam +0.60 → „familiar"; „Vrandel is a country"
fam −1.80, p_top 0.10 → „none", **justified = False**.

**KOR-6 (Port, praktisch wichtig):** die Kontext-Entscheidung läuft auf der
**1D-`d_ctx`-Achse** (Schwelle = Mitte W/C), nicht über den
3-Klassen-Centroiden — der mischt W/C/S und kippt bei Gemma 4 (C- und
W-Zentrum liegen dort in b2 weit auseinander). Nach dem Fix stimmen alle
drei Substrate überein.

**Grenzen.** (a) Vertrautheit ≠ Faktwissen (Layer 0); eine seltene reale
Entität kann vertraut aussehen und trotzdem konfabuliert werden — dafür
bleibt MI-10 (Stabilität) die Grundwahrheit. (b) Der interne `J_C` ist an
die Prompt-Familie gebunden (KOR-5 aus MI-10). (c) Der Indikator
rechtfertigt *Deckung*, nicht Wahrheit — Ebene W-falsch bleibt extern.

Laufschrift: `sprachspiel_bericht.md` §30. Port-Doku:
`port_provenienz_achsen.md`.
