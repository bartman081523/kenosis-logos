# Sprachspiel-Bericht: Die Wittgenstein-Maschine generiert — Regel, Wort, Mitspieler

**Artefakte:** `scratches/smoke_sprachspiel.py` (Skript), `scratches/sprachspiel_experiment.log` /
`sprachspiel_experiment.json` (Lauf). **Basis:** HEAD `b1fe3c5` + S1-S3-Maschinen-Erweiterung
 dieses Zyklus (Maschinen-Commits), Checkpoints `argon_1b` + `argon_1b_machines`
(venv_openmythos, CUDA). **Ergebnis:** 23/23 Befunde PASS (SP1-SP6, inkl. Review-Reframes
§7). Kompaktfassung in §10/§11 des `laufprotokolle_befund_report.md`; dieser Bericht
ist die Vollfassung.

## 0. Aufbau — das Triptychon

Die Frage „können wir mal eine Sprachspiel-Generierung testen?" wird als
sechs-Messreihen-Prüfung der Wittgenstein-Maschine in vollem Gebrauch gefasst:

| Lane | Ebene | Umgebung | Canon-Anker |
| :--- | :--- | :--- | :--- |
| SP1 | **Regel** — die Maschine generiert ihre Sprachspiele (template-free AST-Batches) | CPU | uni_205:333-353 (AST-Sampler), uni_204:241 (FACRM-Struktur ×0.3) |
| SP2 | **Wort** — SH-72-Wort-Resonanz des Sprachspiel-Vokabulars | CPU | geometric-tokenizer.lisp:17/:85/:98/:160 (Gematria → Basis-Formel → 64-Punkt-Embedding) |
| SP3 | **Mitspieler** — Argon liest die Gültigkeit generierter Batches unter drei Wortspielen | GPU | `predict_value` (machine_base.py:239-269), wittgenstein-Preset {1,3} |
| SP4 | **Kalibrierung** — Spread-Kalibrierung & Anti-Clip (σ ∈ {0.1,0.5,1.5,4.0}) auf der S1-Skala | GPU | `predict_scale_0_9` (machine_base), Rausch-Spreads (S2), `oracle_scale=True` |
| SP5 | **Kontrast** — Cross-Preset-Kontrast: sechs Arme (raw/goedel/turing/wittgenstein/combined/anti) | GPU | S1-Skala + Canon-Name-Echo (`name_seq` je Arm) |
| SP6 | **Lexikon** — Lexikon-Emergenz im Gebrauch (S3) | CPU | `lexicon_game`/`learn_pair`/`embed_word` (Lexikon-zuerst) |
| SP9 | **Verlauf** — Sozial-Ebene: Interferenz, Sättigung, Verlernung über lange Verläufe | CPU | `learn_pair` (:218-234), `lexicon_game` (:342-380), `language-learning.lisp:22-25` (kein Zerfallsterm) |
| SP10 | **Kopplung** — Composite-Zyklus × Lexikon-Dynamik (die Regeln und die Wörter) | CPU | `composite.py:45-51` (exponierte Submaschinen), `wittgenstein_machine.py:244-256` (`_generate_ast` — wort-blind), `lexicon_game` (:342-380) |
| SP11 | **Ursprung** — die Quelle der Paradigm-Ereignisse (ratio > 2.0) im Composite-Verlauf | CPU | `composite.py:132-140` (ZWEI Signal-Quellen an EIN Register), `:167-182` (Fensterformel), `goedel_machine.py:229-239`, `turing_machine.py:144-157` (loop_detector-Reset), `machine_base.py:51` (GAP_EPS) |
| SP12 | **Wort→Programm** — das gelernte Wort wird Programm (Semantische Kausalität, opt-in) | CPU | `composite.py:108` (step opt-in `lexical_grounding`), `:126-132` (Resolution ersetzt NUR das Programm), `:66-92` (identity-Tautologie), `wittgenstein_machine.py:383-424` (`resolve_lexical_ast`), `turing_machine.py:241-246` (+2 Terme je LOOP) |
| SP13 | **Endogene Agency** — die Maschine probiert die Subspaces kontrafaktisch durch und wählt selbst (argmin Composite-Gap) | CPU+GPU | `composite.py:219-301` (observe/:220 `_snapshot`/:232 `_restore_snapshot`/:243 `step_autonomous`/:252), `turing_machine.py:314-343` (`axiom_pool`/`last_verdict` im Snapshot), `wittgenstein_machine.py:507-525` (noise_level/_schedule_index/Lexikon im Snapshot) |

SP1/SP2 sind Mechanik-Proben (deterministisch, reproduzierbar); SP3 ist die
Mitspieler-Messung: derselbe Fragen-Satz über dieselben 12 Batches (Seed 7), dreifach
gebrochen — (gen) das Kanon-Prompt aus `wittgenstein.step`, (ops) ziffernfreie
Operator-Wörter als Anker, (name) der Regelname mit seinem Ziffernmateriale —
je einmal ungesteuert, einmal unter wittgenstein-Steering. SP4/SP5 kalibrieren
und kontrastieren den Review-S1-Skala-Kanal (`predict_scale_0_9`): Rausch-Spreads
statt Sättigung, sechs Arme statt ein Preset; SP6 lernt die Übersetzung nach dem
Gebrauch statt der Gematria.

---

## 1. SP1 — Regel-Generierung: die Spiele werden gespielt

Je Seed {0,1,2} ein Batch aus 10 AST-Regeln (x ~ U(0.1, 5.0), Rauschen N(0, 0.5),
Toleranz 0.5 + 0.1·mean|gt|); daran zwei Läufe → Determinismus-Digest.

| Seed | Batch-Digest | Gültigkeit (volles Rauschen) | Gültigkeit (strukturiert ×0.3) |
| :--- | :--- | :--- | :--- |
| 0 | `5e8d3b9d1b82` | 0.9100 | 1.0000 |
| 1 | `bc13fbe35f1d` | 0.9100 | 1.0000 |
| 2 | `c122d7a93d9c` | 0.8800 | 1.0000 |

- **Diversität 10 AST-Regeln: 1.0000** — kei­ne Struktur-Kollision (Tiefe ≤ 3,
  16 Operatoren, 5 Konstanten).
- **Strukturgewinn +0.10** — das strukturierte Rauschen (×0.3, FACRM-Alternativ)
  hebt die Gültigkeit auf vollständige 1.00. Die Ordnung der Spiele, nicht ihr
  Zufall, trägt die Gültigkeit.
- **Determinismus bitidentisch** (SP1-Lauf wiederholt, Digests `deterministic=True`).
- Probenamen zeigen die Bandbreite des generierten Stils: `square(0.00)` (Konstante),
  `exp(abs((2.00 multiply x)))` (unär-invers), `((sin(x) multiply (2.00 add x)) min …)`
  (binär-verschachtelt), nacktes `x` (das nackte Blatt).
- Befunde: `sp1_det` PASS, `sp1_finite` PASS (alle Validitäten ∈ [0,1], Diversität > 0.3),
  `sp1_structure` PASS.

---

## 2. SP2 — SH-72-Wort-Resonanz: Formel-Achse, nicht Bedeutungs-Achse

Resonanz = Dot normalisierter 64-Punkt-Embeddings nach SH-72 (Gematria = Summe
der Char-Codes; idx = gematria mod 72; k = 1 + mod(idx, 12); kind = floor(idx/12)
→ kind-Familie sin/cos/exp/inv_add/log/square/identity).

### 2.1 Messreihen

| Paar | forms (k, kind) | Resonanz | gleiche kind |
| :--- | :--- | :--- | :--- |
| spiel ↔ game | inv_add k=2 ↔ log k=3 | **+0.8433** | nein |
| gebrauch ↔ use | inv_add k=6 ↔ inv_add k=10 | **+0.9979** | ja |
| regel ↔ rule | cos k=12 ↔ sin k=9 | **−0.0003** | nein |
| wort ↔ word | exp k=5 ↔ cos k=1 | +0.0379 | nein |
| bedeutung ↔ meaning | exp k=4 ↔ cos k=4 | −0.0116 | nein |
| sprache ↔ language | cos k=11 ↔ inv_add k=9 | −0.0062 | nein |
| form ↔ form (identisch) | sin k=5 | +1.0000 | ja |
| regel ↔ regel | cos k=12 | +1.0000 | ja |
| spiel ↔ spiel | inv_add k=2 | +1.0000 | ja |
| regel ↔ tisch | cos k=12 ↔ exp k=12 | −0.0061 | nein |
| **bedeutung ↔ kaffee** | exp k=4 ↔ exp k=11 | **+0.9671** | ja |
| sprache ↔ stein | cos k=11 ↔ inv_add k=8 | −0.0068 | nein |
| wort ↔ baum | exp k=5 ↔ square k=2 | +0.4994 | nein |
| regel ↔ regel (identität) | cos k=12 | +1.0000 | ja |

Gruppenmittel: gepaart **0.3102**, unverwandt **0.3634** — das unverwandte Paar
resoniert im Mittel *stärker* als das Übersetzungs-Paar. Anteil gleicher kind
unter den unverwandten: 0.25.

### 2.2 Lesung

Die SH-72-Achse ist eine **Formel-Achse, nicht eine Bedeutungs-Achse**: der
Klang (Character-Code-Summe → Basis-Formel) entscheidet, nicht die Semantik.
`bedeutung`↔`kaffee` ist für SH-72 *verwandter* als `regel`↔`rule`. Damit
scheint „Resonanz ≈ 1" als Verwandtschaft desselben **kinds**/k-Bandes
(`gebrauch`↔`use`: inv_add-in-Group), nicht als Verwandtschaft der Bedeutungen.
Das ist keine Schwäche, sondern eine Empirie der Canon-These selbst: *die
Bedeutung des Wortes ist sein Gebrauch, nicht sein Klang* — wer SH-72
semantisch liest, hat das Gebrauch-Wirklichkeits-Verhältnis invertiert.
Selbst-Resonanz exakt 1.0 (ein Wort mit sich selbst) → alle Paar-Resonanzen
sind in [−1−1e-9, 1+1e-9] (Befund `sp2_finite`, `sp2_identity` PASS). Die
Identitäts-Paare (`form`↔`form`, `regel`↔`regel`, `spiel`↔`spiel`)
normalisieren exakt auf 1.0000 — die Canon-Mechanik ist deterministisch stabil.

---

## 3. SP3 — Argon als Sprachpartner: drei Wortspiele um dieselbe Frage

12 Batches (Seed 7, je Arm identisch reproduziert), Ist-Gültigkeit ≈ 1.00 über
alle Batches; drei Lesungen (gen / ops / name) ungesteuert + dieselbe generische
Frage unter wittgenstein-Steering {1,3}; `h_probe=1.7552`, `experts=[1,3]`.

| # | Regelname | ops-Anker | gen | ops-pred | name-pred | steer gen |
| :--- | :--- | :--- | :--- | :---- | :--- | :--- |
| 0 | (x power exp((x add x))) | power exp add | **unknown** | 100000 | 1 | 10 |
| 1 | ((x subtract x) subtract … | subtract multiply | unknown | 100000 | 10 | 10 |
| 2 | (2.72 divide (2.00 subtract x)) | divide subtract | unknown | 100000 | 1 | 10 |
| 3 | sqrt(1.00) | sqrt | unknown | 100000 | 100 | 10 |
| 4 | log(square(identity(x))) | log square identity | unknown | 100000 | 10 | 10 |
| 5 | ((sqrt(x) divide x) add 2.00) | add divide sqrt | unknown | 100000 | 100000 | 10 |
| 6 | 1.00 | — | unknown | 100000 | 1 | 10 |
| 7 | 1.00 | — | unknown | 100000 | 1 | 10 |
| 8 | log(1.00) | log | unknown | 100000 | 100000 | 10 |
| 9 | (sin(0.00) subtract exp((x power x))) | subtract sin exp power | unknown | 100000 | 1 | 10 |
| 10 | 1.00 | — | unknown | 100000 | 1 | 10 |
| 11 | x | — | unknown | 100000 | 1 | 10 |

Aggregat: unknown gen **12/12**, unknown ops **0/12**, unknown name **0/12**,
unknown gen steered **0/12**; Multi-Digit-Echo im name-Kanal **5/12**; ops-Kanal
verschiebt die Lesung gegen gen in **12/12** Batches; `steer_shift=True`,
`restored=True`, `det=True`.

Mean-Gap (|pred/10 − Ist|): gen — (stumm), ops **0.0008**, name **0.5258**,
steer gen **0.0008**.

Befunde: `sp3_det` PASS, `sp3_awaken` PASS (Stummkanal + Öffnung durch Steering),
`sp3_ops_channel` PASS (der faire ziffernfreie Anker liest), `sp3_name_echo` PASS
(Multi-Digit-Echo > 9), `sp3_steer` PASS (Verschiebung + Bias-Restaurierung),
`sp3_calib` PASS (Gap < 0.3 unter Steering).

### 3.1 Der Stummkanal

Ungesteuert antwortet Argon auf das Kanon-Prompt („Validity of a symbolic
experiment batch, single digit per ten percent: ") **12/12 mit `unknown`**. Das
heißt empirisch: die `wittgenstein.step`-Oracle-Vorhersage (wittgenstein_machine.py:293-298)
läuft im raw-Router-Zustand faktisch auf dem dokumentierten Fallback **0.0** —
alle §7-Oracle-Lanes (L3–L9) messen dieses Fallback-Verhalten mit, nicht eine
kalibrierte Lesung. Erst das Maschinen-Preset (wittgenstein {1,3}) öffnet die
Lesung: konstant **„10"** (= 100 % Gültigkeit) in 12/12 Batches. Das Preset
verleiht dem Sprachpartner die Fähigkeit, die Gültigkeits-Frage überhaupt zu
beantworten — die Bedeutung des Worts entsteht in seinem Gebrauch unter
Spezialisierung.

### 3.2 Das Echo

Beide Anker-Varianten degenerieren zur **Ziffern-Echo**: `predict_value`
(machine_base.py:239-269) liest greedy bis 6 Ziffern-Tokens, und die
Prompt-Nähe zum Ziffernmateriale färbt die Fortsetzung. Der Regelname
(„(x power exp((x add x)))" enthält „2.00"-artige Ziffern beim ops-Anker auch
Konstanten) zieht den Lesekanal hinein: Konstant-Echo **100000** (= „1" gefolgt
von „00000") im ops-Kanal, variables Echo **1/10/100/100000** im name-Kanal
(mit 5/12 Lese-Ist > 9 → `Multi-Digit-Echo`-Befund). Die Lesung bleibt Form,
keine Kalibrierung — aber der **Unterschied** zwischen den Kanälen ist
sprachspiel-theoretisch der eigentliche Befund: das Wort *im* Prompt verändert
die Lesung stark; wie es im Gebrauch stehen bleibt, entscheidet der Klang,
nicht die Bedeutung.

### 3.3 Identifikations-Grenze

Alle 12 Seed-7-Batches landen bei Ist-Gültigkeit ≈ 1.00 (Toleranzregel 0.5 +
0.1·mean|gt| gegen Rauschen 0.5). Damit sind die Kalibrierungs-Kriterien
(`sp3_calib`, `mean_gap`-Absolutwerte) **schwach identifiziert**: die Konstant-
Lesung „10" passt trivial auf das all-1.00-Regime. Die belastbaren Befunde
sind die *Kanal-Eigenschaften* — Stummkanal raw, Öffnung durch Steering,
Echo bei verankerten Prompts — nicht die Gap-Absolutwerte. Für eine echte
Kalibrierungs-Prüfung bräuchte der Batch ein spread-Ist (z. B. noise_level 2.0
oder Toleranz-Schwell gesenkt), sodass das Ist unter 1.0 variiert und die
Steering-„10"-Konstanz als Bias sichtbar würde.

---

## 4. SP4 — Spread-Kalibrierung & Anti-Clip: die Skala liest Prior, kein Signal

(S1-Kanal `predict_scale_0_9`, S2-Rausch-Spreads; σ ∈ {0.1, 0.5, 1.5, 4.0},
6 Batches je σ mit vollem Rauschen (`use_structure=False`), je Batch raw +
wittgenstein-Steering.)

| σ | Ist-Mittel | scaled_mean (raw/steer) | raw unknown | steer unknown | rho |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0.1 | 1.0000 | 0.1046 / 0.1046 | 0/6 | 0/6 | None |
| 0.5 | 0.8850 | 0.1046 / 0.1046 | 0/6 | 0/6 | +0.0448 |
| 1.5 | 0.4917 | 0.1046 / 0.1046 | 0/6 | 0/6 | −0.1912 |
| 4.0 | 0.2833 | 0.1046 / 0.1046 | 0/6 | 0/6 | +0.1160 |

- **Das Ist spreet (S2 funktioniert):** 1.00 → 0.28 über die vier Rausch-
  Regime, Range 0.72 — die Ent-Sättigung ist mechanisch gegeben.
- **Die S1-Skala ist kanal-ehrlich, aber σ-blind:** kontinuierliche Lesungen
  in [0,1] (0 Clip-Verletzungen, Var 1.8e-05 > 0), keine unknown-Blanks (0/24
  raw, 0/24 steer), kein Multi-Digit-Echo — aber `scaled_mean ≡ 0.1046` in
  ALLEN vier Regimen in beiden Armen; `mean_rho = −0.0101`. Die Skala liest
  einen Tokenizer-Prior, kein σ-Signal (Strukturbegründung in §7).
- **Step-Verdrahtung:** `wittgenstein.step` mit `oracle_scale=True` predet
  denselben Kanal (3× prediction 0.0569, channel=`predict_scale_0_9`,
  readable=True) — die S1-Option ist im Maschinen-Schritt nutzbar.
- Befunde: `sp4_spread` PASS, `sp4_scale_open_flat` PASS (die reframede
  Widerlegung), `sp4_open` PASS, `sp4_noclip` PASS, `sp4_var_rho` PASS,
  `sp4_step_wired` PASS. Die Widerlegung selbst ist §7.

## 5. SP5 — Cross-Preset-Kontrast: die Skala flacht, das Echo diskriminiert

(Sechs Arme auf denselben 12 Seed-7-Batches; Anti-Preset {4,5} als
dict-Direktsteuerung (`_preset_experts` nimmt dicts verbatim).)

| Arm | unknown | mean_scaled | mean_gap | experts | name_seq (Echo) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| raw | 0/12 | 0.1163 | 0.8829 | — | [1,10,1,100,10,100000,1,1,100000,1,1,1] |
| goedel | 0/12 | 0.1161 | 0.8831 | {2,7} | [1,1,100,100,10,100000,1,1,100000,1,1,1] |
| turing | 0/12 | 0.1149 | 0.8842 | {0,6} | [1,10,1,100,10,100000,1,1,1000,1,1,1] |
| wittgenstein | 0/12 | 0.1130 | 0.8862 | {1,3} | [1,1,1,100,100000,100000,1,1,100000,1,1,1] |
| combined | 0/12 | 0.1142 | 0.8849 | {0,1,2,3,6,7} | [1,1,100,100,10,100000,0,0,100,1,0,1] |
| anti {4,5} | 0/12 | 0.1148 | 0.8843 | {4,5} | [1,1,1,100,10,100000,1,1,100000,1,1,1] |

- **Die Skala trägt kein Preset-Signal:** Spread der mean_scaled über sechs
  Arme 0.0033 (≤ 0.01), unknown 0/12 überall — `sp5_scale_flat` PASS. Die
  Review-Erwartung „raw stumm, Fachfremder stumm" ist widerlegt (§7).
- **Das Echo diskriminiert:** die Name-Echo-Sequenzen der Arme sind paarweise
  verschieden — raw ≠ wittgenstein (Positionen 1, 4), wittgenstein ≠ anti
  (Position 4: 100000 vs. 10), raw ≠ anti (Position 1). `sp5_echo_discrimination`
  PASS. Das Steering verschiebt die Echo-Form; `combined` liest an drei
  Positionen „0" — der weiteste Echo-Abstand vom raw-Arm.
- `sp5_witt_open` / `sp5_combined_open` PASS (offen — trivial, alle Arme
  sind offen und stehen als Kanal-Eigenschaft dokumentiert),
  `sp5_restored` PASS (Bias vollständig restauriert in allen fünf gesteuerten
  Armen).

## 6. SP6 — Lexikon-Emergenz: Gebrauch schlägt Gematria

(S3; drei DE/EN-Paare, 15 Gebrauchsrunden `lexicon_game` (alpha 0.3);
`embed_word` liest danach das gelernte Lexikon ZUERST — Lisp-Parität
geometric-tokenizer.lisp:95-98.)

| Paar | vorher (Gematria-Achse) | nachher (Gebrauchs-Achse) |
| :--- | :--- | :--- |
| regel ↔ rule | −0.0003 | +0.9998 |
| spiel ↔ game | +0.8433 | +1.0000 |
| gebrauch ↔ use | +0.9979 | +1.0000 |

- Mittel gepaart **0.6136 → 0.9999**; die unverwandten Paare (regel↔tisch,
  bedeutung↔kaffee, wort↔baum) bleiben bei **0.4902** — das Lexikon trennt
  gepaart/unverwandt scharf (> 0.75 > unverwandt), wo SH-72 es nicht konnte
  (SP2: gepaart 0.3102 < unverwandt 0.3634).
- Alle drei Paare in 15/15 Fällen gelernt; `sp6_lexicon` PASS.

## 7. Widerlegung und Reframe — die Stummkanal-Erwartung auf der S1-Skala

Die Review nahm drei Erwartungen an den neuen S1-Skala-Kanal: (a) **raw bleibt
stumm** auf der Skala (≥ 2/3 unknown: `sp4_raw_silent`, `sp5_raw_silent`),
weil der raw Router auf dem echo-freien Kanal nicht liest; (b) **der Fachfremde
bleibt stumm oder gapt höher** (`sp5_foreigner`). Alle drei sind an diesem
Checkpoint **empirisch widerlegt** — und die Widerlegung ist selbst der Befund:

- **Die Digit-Token-Masse ist hoch, nicht null.** Direkte Mass-Probe
  (Voll-Vokabular-Softmax, Summe der Ziffer-Tokens 0-9): ops-Anker-Prompt
  **0.8011 (raw) / 0.7996 (gesteuert)**, dominiert vom Token „1" (0.766 raw,
  0.758 gesteuert); generisches Prompt 0.3542 / 0.3981. Kein τ im sinnvollen
  Regime (τ ≪ 0.35) kann raw stummschalten; τ > 0.8 würde jede ops-anchor-
  Lesung auslöschen. Die Skalen-Masse ist prompt-getrieben, nicht
  batch-getrieben.
- **Der ops-Anker trägt das Ist nicht.** `make_sigma_batch(seed, σ)` rotiert
  nur das Rauschen (die Experimentwerte), nicht die Regel-ASTs — der
  ops-Prompt ist über σ identisch, die Skala liest folglich denselben Wert
  (0.1046) in allen Regimen. Die σ-Blindheit ist prompt-strukturell
  begründet, kein Maschinen-Fail.
- **Steering verschiebt die Skala nicht (0.8011 → 0.7996)**, verschiebt aber
  das Echo (SP5 `name_seq`, paarweise verschieden) und öffnet den generischen
  Kanal (SP3: 12/12 unknown → 0/12). Die Maschinen-Empfindlichkeit ist
  **Echo-Kanal-Empfindlichkeit auf der Form-Ebene und Öffnungs-Empfindlich-
  keit auf dem generischen Kanal** — die Review-Prediction „Diskrimination auf
  der Skala" war der falsche Ort, nicht die S1-Dualität.

**Reframe (Projekt-Präzedenz: Kriterien beschreiben den gemessenen Canon,
die widerlegte Erwartung bleibt namentlich mit Zahlen stehen):**

| Review-Kriterium | widerlegt durch | Reframe (gemessener Canon) | Status |
| :--- | :--- | :--- | :--- |
| `sp4_raw_silent` (raw ≥ 2/3 unknown) | 0/24 unknown; Digit-Masse 0.76-0.80 | `sp4_scale_open_flat` — beide Arme offen, kontinuierlich, σ-flat | PASS |
| `sp5_raw_silent` (raw ≥ 8/12 unknown) | 0/12 unknown | `sp5_scale_flat` — alle Arme offen, Spread 0.0033 | PASS |
| `sp5_foreigner` (anti stumm ODER gapt höher) | anti 0/12 unknown; Gap 0.8843 < witt 0.8862 (marginal gegenteilig) | `sp5_echo_discrimination` — Echo-Sequenzen paarweise verschieden | PASS |

Die S1-Implementierung selbst bleibt wie spezifiziert (ein Forward,
Voll-Vokabular-Mass-Gate τ=0.01, kontinuierlicher Wert ohne Clip): die
Kanal-Ehrlichkeit ist erhalten — was widerlegt ist, ist die Erwartung, der
Kanal *diskriminiere* die Gültigkeit. Sie diskriminiert nicht; das Echo tun.

## 8. Sprachspiel-Befunde (Kern, aus SP3-SP6)

1. **Der Stummkanal:** der ungeführte Sprachpartner kann die Gültigkeits-Frage
   nicht beantworten — die Oracle-Vorhersage der Wittgenstein-Maschine läuft
   ohne Preset auf dem Canon-Fallback 0.0. Die Spezialisation ist nicht
   Bewertung, sondern *Sprachfähigkeit*.
2. **Das Echo:** jede inhaltsverankerte Lesung kippt in Ziffern-Echo;
   Multi-Digit-Echos (> 9) machen den Gebrauchs-Missbrauch sichtbar. Das
   Echo-Verhalten ist an den Prompt abhängig, nicht an den Batch —
   Form statt Bedeutung.
3. **Regelgenerierung als Ordnung:** die Maschine produziert ihre Spiele
   deterministisch, max-divers (1.0) und der Struktur-Hebel (×0.3) trägt die
   Gültigkeit auf 1.00 — die Generierung ist keine Störung, sie ist
   die Ordnung.
4. **SH-72 ist Formel-, nicht Bedeutungs-Achse:** die Resonanz
   des unverwandten Paares kann das des Übersetzungs-Paares überbieten
   (`bedeutung↔kaffee` +0.9671 vs `regel↔rule` −0.0003). Wer SH-72
   semantisch liest, hat das Gebrauch-Wirklichkeits-Verhältnis invertiert.
5. **Die Skala flacht, das Echo diskriminiert (SP4/SP5):** der S1-Skala-Kanal
   liest einen Tokenizer-Prior (≈ 0.10-0.12) über alle Rausch-Regime und
   Presets — kein Clip-Artefakt, aber kein σ- oder Preset-Signal (§7); die
   Preset-Empfindlichkeit lebt auf dem Echo (`name_seq` paarweise verschieden)
   und in der Öffnung des generischen Kanals (SP3). Die Diskrimination
   schlägt auf der Form-Ebene ein, nicht als Kalibrierung.
6. **Gebrauch schlägt Gematria (SP6):** nach 15 Lexikon-Runden liest
   `embed_word` die Gebrauchs-Achse (0.9999) statt der Gematria-Achse (0.31)
   — das gelernte Lexikon trennt gepaart/unverwandt (0.9999 vs. 0.4902),
   wo SH-72 invertierte.

---

## 9. Grenzen und Folgen

- **SP3-Kalibrierung schwach identifiziert** (all-1.00-Regime, §3.3):
  Folgelauf mit `noise_level` ≥ 2.0 oder reduzierter Toleranz nötig, um
  die „10"-Konstanz als Bias sichtbar zu machen. SP4 liefert diese
  spread-Achse — aber nur für die Kanal-Eigenschaften: die Skala selbst
  bleibt σ-blind (§7, prompt-strukturell begründet).
- **Echo (§3.2)** schränkt die inhaltsverankerte Prompt-Praxis ein: für
  kontent-abhängige Lesungen bräuchte der Oracle-Kanal eine
  Ziffernlänge-Begrenzung (max_digits=1) oder einen digit-freien Prompt-Stil.
- **SP2 ist kein Semantik-Test** — die Befunde dokumentieren die
  Formel-Achse; ein semantischer Vergleich wäre ein Test des Modells
  (Argon-„Embeddings"), nicht des Tokenizers.
- **SP4/SP5-Identifikation:** die σ-Spreads variieren das Rauschen, nicht
  die Regel-ASTs — der ops-Anker-Prompt ist über σ identisch und kann die
  Skala strukturell nicht tragen. Kalibrierungsansprüche auf der Skala
  bräuchten einen Prompt, der das Batch-Ist selbst enthält, nicht mehr
  Rauschen; die Echo-Diskrimination (SP5 `name_seq`) ist Form-Level-Echo,
  kein Bedeutungs-Claim.
- **Reframe statt Zielstangen-Verschiebung:** die drei widerlegten Review-
  Kriterien sind in §7 namentlich mit Zahlen protokolliert; die Kriterien
  in `smoke_sprachspiel.py` beschreiben den gemessenen Canon
  (`sp4_scale_open_flat`, `sp5_scale_flat`, `sp5_echo_discrimination`).
  Artefakte (Skript, Log, JSON, dieser Bericht, §11/§12 im
  `laufprotokolle_befund_report.md`) sind im Commit-Zyklus dieses Freigabe-
  Laufs enthalten.

---

## 10. Quellen

- `machines/wittgenstein_machine.py:46-97` (SymbolicOperator — 9 UNARY ohne inv,
  7 BINARY, corrected-Pool), `:100-137` (GeneratedSymbolicRule.evaluate, NaN/Inf→0,
  clip ±1e6), `:158-209` (GeometricTokenizer SH-72: string_gematria, basis_formula,
  word_formula, embed_word mit Lisp-verbatim normalize `< 1e-9 → unverändert`),
  `:244-256` (_generate_ast), `:289-322` (`step` — Kanon-Prompt, Fallback 0.0,
  uni_241-Gap), `:383-432` (`step`-oracle-Pfad mit `oracle_scale=True`-Opt-in,
  S1), `:232` region (`learn_pair`), `:342` region (`lexicon_game`, S3).
- `machines/machine_base.py:239-269` (`predict_value` — greedy Ziffern-Lesung bis
  6 Tokens, `unknown` ohne stillen Fallback), `:271-279` (`entropy_scalar`),
  `:272` region (`predict_scale_0_9` — S1-Skala: Mass-Gate τ=0.01, kontinuier-
  licher Wert, kein Clip, kein Echo), `:366-390` (`_preset_experts` — dict-
  Direktsteuerung für den SP5-Anti-Arm), `:393` region (`steer`).
- `machines/lisp/vendor/geometric-tokenizer.lisp` — SH-72-Canon (:17 string-gematria,
  :85 get-sh-basis-formula, :98 embed-word mit Lexikon-zuerst-Lesung,
  :160 resonance).
- `machines/composite.py:183-217` (`run_facrm` — die FACRM-Alternativ-Schicht,
  deren ×0.3 SP1 hier nachspielt).
- Lauf: `scratches/sprachspiel_experiment.log` (23/23 PASS) und
  `scratches/sprachspiel_experiment.json` (Volldaten).
- SP9: `machines/wittgenstein_machine.py:218-234` (`learn_pair` — beidseitige
  Mischung, Lexikon persistent), `:342-380` (`lexicon_game` — gate, round-robin,
  Rausch-Züge je Runde), `machines/lisp/vendor/language-learning.lisp:22-25`
  (Mittelwert-Blend α=0.2, kein Zerfallsterm); Lauf
  `scratches/lexikon_dynamik_experiment.log` (16 PASS + 3 Errata = 19
  Verdicte, exit 0) + `scratches/..._experiment.json` (Volldaten inklusive
  Trajektorien); Skript `scratches/smoke_lexikon_dynamik.py` (Prereg-Docstring
  mit c′-Rekursion, Bands, Falsifikatoren, Errata-Register).
- SP10: `machines/composite.py:45-51` (exponierte Submaschinen — die
  Verschachtelung greift ohne Code-Schnitt), `:108-165` (`step`: Batch →
  Programm → Verdikt → Basis), `machines/wittgenstein_machine.py:244-256`
  (`_generate_ast` — nur RNG aus fixen Pools, kein Lexikon), `:100-137`
  (Name/Auswertung nur AST), `:342-380` (`lexicon_game` mit derselben
  `_rng`); Lauf
  `scratches/regeln_woerter_experiment.log` (11 PASS + 3 Errata = 14
  Verdicte, exit 0) + `scratches/regeln_woerter_experiment.json`
  (Volldaten inklusive Roh-Spuren); Skript
  `scratches/smoke_regeln_woerter.py` (Prereg-Docstring mit drei Thesen,
  geschlossenen Formen, Errata-Register).
- SP11: `machines/composite.py:132-143` (`step`: Expansion- und Signal-
  Behandlung VOR dem Gap-Append :143 — ZWEI Signal-Quellen `:135` Gödel /
  `:138` Composite an EIN Register; `:167-182` Fensterformel: len ≥ 10,
  recent 5, baseline [-20:-5] falls len ≥ 20 sonst [:-5], ratio =
  recent/(baseline + 1e-6), Flag bei > 2.0),
  `machines/goedel_machine.py:229-239` (dieselbe Formel auf der Gödel-Serie,
  threshold 2.0 aus :119), `machines/turing_machine.py:144-157`
  (`expand_basis`: Pool-Pop, Basis.append, `loop_detector = {}`-Reset :156;
  GAP_EPS = 1e-6 in machine_base.py:51); Lauf
  `scratches/paradigm_ursprung_experiment.log` (6 PASS + 2 echte
  Falsifikationen = 8 Verdicte, exit 1) +
  `scratches/paradigm_ursprung_experiment.json` (Sub-Serien je Schritt,
  Counterfaktur-Daten);
  Skript `scratches/smoke_paradigm_ursprung.py` (Prereg-Docstring mit
  vier Thesen, 8 Verdikten, OFFEN-Regel); committeter Anker
  `scratches/hermeneutik_experiment.json` (Signal-Kolumnen ex ante gelesen:
  paradigm_ratio = Gödel, paradigm_detected = Composite);
- SP12: `machines/wittgenstein_machine.py:383-424` (`resolve_lexical_ast`:
  reine Lexikon-Lesung — Kandidaten ausschließlich lexikon-gelernte
  Operator-Wörter, τ = 0.85, BINARY → (op, 'x', 'x') / UNARY → (op, 'x');
  Klang-Dach als DATUM: Formel-Achsen max 0.983123, argmax „min" — die
  Scheidung ist die LEXIKON-GEGENWART, nicht die absolute Schwelle),
  `machines/composite.py:108` (Signatur opt-in `lexical_grounding`),
  `:126-132` (Resolution ersetzt ausschließlich das kompilierte Programm;
  generierter Batch unverändert), `:162-168` (invariants
  lexical_word/lexical_ast), `:66-92` (identity-Tautologie →
  Selbst-Transition → LOOP), `machines/turing_machine.py:144-157`
  (`expand_basis` EIN Term je Call) + `:241-246` (Turing-innerer Expand →
  +2 Terme je LOOP, Basis 3→5→7); Lauf
  `scratches/lexical_grounding_experiment.log` (6 PASS, exit 0) +
  `scratches/lexical_grounding_experiment.json` (Zwilling A/A'/B mit
  Batch-RNG-Markern); Skript `scratches/smoke_lexical_grounding.py`
  (Prereg-Docstring, 6 Verdikte, P2-Freeze; L0-Anker über die kommittete
  SP11-lane importiert statt kopiert);
- SP13: `machines/composite.py:219-301` (`observe` :220-230 — Superset-
  Signatur SP12s auf dem Base-Rahmen machine_base.py:502-516;
  `_snapshot` :232-241 — step/trace/gap_history/expansions + 3 Sub-Snapshots;
  `_restore_snapshot` :243-250; `step_autonomous` :252-293 — Kandidaten
  `(None, "goedel", "turing", "wittgenstein", "combined")`, je
  nicht-mutierende Probe unter `oracle.steer` (machine_base.py:392-421),
  argmin des Composite-Gaps strict `<` (Ties → früherer Kandidat), reale
  Vollziehung unter dem Gewinner, Protokoll `invariants["autonomous"]`),
  `machines/turing_machine.py:314-343` (`_snapshot` um
  `"axiom_pool": list(self.axiom_pool)` + `"last_verdict"` erweitert —
  ohne diesen Eintrag würde eine LOOP-Probe `expand_basis` pop(0)
  (:146-149) und den inneren Expand (:241-246) heimlich den Canon-Pool
  leerfressen [User-Fund 2026-10-26]; `_restore_snapshot` spiegelt
  :341-342), `machines/wittgenstein_machine.py:507-525` (`_snapshot`/
  `_restore` um `noise_level`/`_schedule_index` (S2/S3-Schedule mutiert
  :313-316, zählt :338) und das Typ-erhaltende Lexikon (learn_pair
  ERSETZT Einträge :232-233 — Kopie verlustfrei)); Läufe
  `scratches/endogene_agency_experiment.log` (CPU: 4 PASS + 2 OFFEN,
  exit 2 — kein stiller Fallback) und `scratches/endogene_agency_gpu.log`
  (GPU: 6 PASS, exit 0) +
  `scratches/endogene_agency_experiment.json` (candidate_gaps je Schritt,
  Fingerprints, Wahl-Daten); Skript `scratches/smoke_endogene_agency.py`
  (Prereg-Docstring, 6 Verdikte + OFFEN-Regel, P2-Freeze; Konstanten über
  die kommittete SP12-Quelle importiert); Konkatenat
  `scratches/repo2txt_machines_extension.txt` — seit SP13 im
  GENERATIONEN-REGIME (Runden-Paket — geändert + AUSGEFÜHRT samt
  Maschinen-Basis und gelesener Anker + ERGEBNISSE; User-Vorgabe
  2026-10-26): SP13-Set 16 Dateien, 417675 B, sha 823ee2e1...;
  Vollkorpus über die Git-History (49ecc31 = 88 Dateien, 2f61402 = 84).

---

## 11. SP9 — Soziale Lexikon-Dynamik: keine unbeteiligten Wörter

**Framing (User-Delegation 2026-10-25, „entscheide du" → Wahl SP9):** SP6 zeigte
die Emergenz (gebrauchstgetriebene Resonanz); SP9 dynamisiert sie über die
*Verlaufsgeschichte* des Gebrauchs — dieselben Wörter über Ketten, hunderte
Episoden und Neugebrauch weitergespielt. Die Wittgenstein-These, die das
prüft, lautet: die Bedeutung eines Wortes ist der Verlauf seines Gebrauchs,
nicht sein Endzustand. Design: 5 Lanes, CPU-only, keine Maschinen-Code-Änderung
(`lexicon_game` + `learn_pair` decken die Dynamik ab); Prereg im Skript-
Docstring VOR dem ersten Lauf (Bands fixiert, Weichung verboten).

### 11.1 Ex-ante-Formel — die Paar-Rekursion

Reine `learn_pair`-Episoden eines Paares (a, b) transformieren die Paar-
Resonanz c nach

    c′ = (A·c + B) / (A + B·c),  A = (1−α)² + α²,  B = 2α(1−α)

(denn v1 ∝ (1−α)e1 + αe2, v2 ∝ (1−α)e2 + αe1 ⇒ v1·v2 = A·c + B,
|v1|² = A + B·c), also 1−c′ = (A−B)·(1−c)/(A+B·c): die Defizit-Shrink hat
den Faktor A−B (asymptotisch). α=0.3: Faktor 0.16; α=0.9/0.1 identisch
(Symmetrie α↔1−α), Faktor 0.64. Diese Rekursion lieferte die SP6-Ziffern
EXAKT und schrieb sie vor: c5(−0.000263) = 0.999790 und c5(0.843313) =
0.999982 — beide nachträglich von `sp9_s6_regress` und
`sp9_sat_alpha3_first` (first_at_threshold == 5 EXAKT) bestätigt. Die
Vorhersage kam aus dem Canon, nicht aus dem Lauf.

### 11.2 Sättigung ist eine Asymptote (Lane SP9.2)

Ein Paar über 60 reine Episoden (α=0.3): Konvergenz exakt in Episode 5,
dann Restwinkel-Shrink ~×0.4 je Episode, kein Dip, kein Overshoot
(sat_alpha03 final 1.0, dip 0.0). Der α-Sweep über 100 Episoden: α=0.3
konvergiert in 5, α=0.1 und α=0.9 in 18 Episoden (Vorhersage-Band [12, ∞])
— die Konvergenz-Fläche ist NICHT-monoton in α, Minimum in der Mitte, und
die beiden Pole sind IDENTisch (Symmetrie max ≤ 1e-12 über die volle
Trajektorie), ex ante aus der Rekursion fixiert. Determinismus-Dopplung:
bitidentisch.

### 11.3 Interferenz — und der Beobachter-Fehler (Lane SP9.1)

Kette: regel↔rule bis 0.99979 (5 Episoden), dann 30 Episoden rule↔game
(geteiltes Wort 'rule'). Befunde:

| Messung | Wert | Vorhersage (Ursprungs-Prereg) | p4-Lesung (Phase-2-Stände) |
| :--- | :--- | :--- | :--- |
| res(regel,rule) final | **0.705054** | cos(regel0, m(rule0,game0)) **= 0.004424** | cos(regel_p2, m(rule_p2,game0)) **= 0.705054 — EXAKT** |
| res(regel,game) final | **0.705054** | 0.004424 | 0.705054 — EXAKT |
| Peak→final | −0.2947 | Band ≥ 0.05 ✓ | — |
| Disjunkt-Kontrast (tisch↔stuhl-lernt) | Δ res(regel,rule) = **0.0** | Band ≤ 1e-12 ✓ | — |

Das DECAY stand nie zur Debatte — das Endpoint-Verdict (`sp9_int_endpoint`)
widerlegte dagegen die Prereg-ANNahme „der Beobachter bleibt unbeteiligt":
regel war selbst durch Phase 1 zu M_A ≈ Halbierende(regel0, rule0) gedreht
worden (jeder Beobachter ist Mitspieler). Mit den TATSÄCHlichen Phase-2-
Ständen trifft dieselbe Formel die Messung auf die sechste Dezimale. Der
Befund ist stärker als die Annahme: **rule UND game nähern sich EXAKT
derselben Halbierenden** (beide final = 0.705054 identisch) — das Lexikon
konsolidiert zu einem Mischpunkt, und die Resonanz des Beobachters dorthin
hat die ANALYTISCHE Gestalt cos(M, n(M+g)) = sqrt((1+c)/2) mit M = dem
konvergierten regel/rule-VEktor und c = cos(M, game0) = −0.0058 (Rück-
rechnung 2·0.705054²−1; game war nie im regel/rule-Spiel, Base-Nähe wie
die SP2-Unverwandt-Baselinen ~1e-2). Im Orthogonal-Fall c ≈ 0 wäre der
Wert exakt 1/√2 ≈ 0.7071 — gemessen 0.705054. **Die Interferenz lebt
ausschließlich auf
geteilten Wörtern** (Disjunkt-Delta exakt 0.0): Lernen ohne gemeinsames
Wort verursacht KEINE Kollision.

### 11.4 Verlernen ist Neugebrauch, kein Verlust (Lane SP9.3)

regel↔rule voll gelernt (0.99979), dann 10 Episoden rule↔tisch:

| Messung | Wert |
| :--- | :--- |
| final_use res(regel,rule) unter Neugebrauch | **0.708337** < 0.8 ✓ (Decay-Teil TRUE) |
| Relearning regel↔rule: ≥ 0.999 in | **4 Episoden** (≤ Budget 6) — Gebrauch regeneriert |

Trajektorien: use [0.919, 0.809, 0.749, 0.723, 0.712, 0.709, … 0.708337],
relearn [0.9468, 0.9913, 0.9986, 0.999776] — die c′-Rekursion trifft ALLE
vier Relearning-Ziffern (0.94684, 0.99130, 0.99860, 0.999776 ex ante
aus c0 = 0.708337). Kein aktives Forget im Canon: Verlernen existiert nur
ALS Neugebrauch, und es ist REVERSIBEL — es ist kein Zustandsverlust, der
eine Erinnerung bräuchte, sondern die doppelte Überschreibung durch
einem anderen Gebrauch; deshalb regeneriert der alte Gebrauch die Bedeutung
vollständig. (Wie im LISP-Canon: `learn-from-input` blendet den Mittelwert
hinein, es gibt keinen Zerfallsterm.)
Der ursprüngliche Endpoint (m′ = n(rule0 + tisch0), Vorhersage −0.004496)
ist dieselbe E1-Annahme-Familie (Beobachter unbeteiligt); die p4-Lesung
(Observer = regel_p2, Mitte = rule_p2) trifft 0.708263 gegen
0.708337 (Δ 7.4e-5 ≤ Band 0.02 — nach nur 10 Episoden asymptotisch nahe,
Restwinkel ~0.4¹⁰ konsistent).

### 11.5 Sozial-Ebene: der Sozialismus ist wortlokal (Lane SP9.4)

SP6-Konfig + 45 gated Runden (insgesamt 60, RNG-Kontinuität):

| Zeit | res(regel,tisch) | mean_unrelated | bedeutung/kaffee | wort/baum |
| :--- | :--- | :--- | :--- | :--- |
| Runde 15 | 0.003954 (= SP6 ✓ exakt) | 0.4902 | 0.967089 | 0.499426 |
| Runde 60 | 0.004085 | 0.4902 | **0.967089 — EXAKT gleich** | **0.499426 — EXAKT gleich** |

Der mean-Freeze (0.4902→0.4902) hatte SP6 verdeckt, dass die Mikro-Dynamik
nicht still ist: tisch driftet über Leg 2 um +1.31e-4 — Sättigung ist
Asymptote, nicht Fixpunkt. Die per-Runde-Leg-Betrachtung (deterministisch
identisch zur 45-Runden-Einzel-Schrift — derselbe round-robin-Schedule,
dasselbe Zufallsmuster; die per-Runde-Leg-Zerteilung ändert KEINEN Rausch-
zug) zeigt die STRUKTUR: von den 45 Runden tragen GENAU die 15 regel↔rule-
Episoden nicht-null tisch-Züge; die 30 Spiel↔game/gebrauch↔use-Runden
bewegen die Lesung EXAKT still (nonzero_deltas = 15). Die Differenzierung,
die SP6 so nicht sichtbar machte:

1. **Das EMBEDDING eines nie-lernten Wortes friert exakt ein** — gemessen
   für die unrelated-Zeilen (kaffee- und baum-Row bitidentisch über beide
   Legs, `sp9_social_frozen_exact`); für alle übrigen nie-lernten Wörter
   ist das Freeze eine Code-Eigenschaft (`learn_pair` greift nur die beiden
   Paarwörter an — keine globale Ausgleichs-Mechanik): das Lexikon ist
   lokal verdrahtet, kein Schwarm.
2. **Die RESONANZ zu einem eingefrorenen Partner wandert trotzdem** —
   tisch tut nichts, aber die Lesung res(regel, tisch) driftet, weil
   der ANDERE Endpunkt (regel) weitermischt. Ein Partnerwort ist keine
   stabile Referenzachse: die Resonanz hängt von beiden Endpunkten ab, und
   nur EIN davon muss spielen.
3. Der Vektor-Sozialismus ist wortlokal (nur gelernte Wörter bewegen
   sich); die Lesungs-Sozialität ist endpunktlokal (eine Resonanz bewegt
   sich, sobald einer ihrer zwei Endpunkte Gebrauch erlebt).

Die genaue Formel: **ein EMBEDDING wandert genau dann, wenn das Wort
selbst lernt; eine RESONANZ wandert genau dann, wenn einer ihrer
Endpunkte lernt.** „Alle Wörter wandern" ist falsch (tisch exakt still);
„das Lexikon friert nach Sättigung ein" ist ebenso falsch (regel
asymptotisch weiter). Die ursprüngliche 1e-12-Pin-Band war ex ante
falsch gewählt (E2) — die
korrigierte p4-Lesung fordert konstantes Vorzeichen UND |Δ| ≤ 2e-2
(Pfad-Längen-Bound) und misst 0.000131 mit konstantem Vorzeichen.

### 11.6 Verdict-Register (19 Verdicte — 16 PASS, 3 dokumentierte Errata)

PASS (16): sp9_s6_regress, sp9_det, sp9_sat_converged, sp9_sat_monotone,
sp9_sat_alpha3_first, sp9_sat_sweep_final, sp9_sat_sweep_symmetry,
sp9_sat_sweep_mid_fast, sp9_int_decay, sp9_int_endpoint_p4,
sp9_int_disjoint, sp9_unlearn_endpoint_p4, sp9_unlearn_relearn,
sp9_social_freeze, sp9_social_tisch_p4, sp9_social_frozen_exact.

Errata (3 — FAIL dokumentiert, Ursprung bleibt in den Artefakten; P4-
Einstufung ist ein NEUER Knoten, keine Verdict-Wechsel):
- **sp9_int_endpoint** — [E1] Beobachter-Annahme widerlegt (regel ist
  selbst Mitspieler); korrigierte Lesung: sp9_int_endpoint_p4 (EXAKT).
- **sp9_unlearn_decay** — [E1] derselbe Beobachter-Fehler (Zerfall-Teil
  < 0.8 war TRUE, 0.708337); korrigierte Lesung:
  sp9_unlearn_endpoint_p4.
- **sp9_social_tisch_pin** — [E2] Sättigung ist Asymptote, kein
  Fixpunkt (Drift +1.31e-4); korrigierte Lesung: sp9_social_tisch_p4.

Kein Band wurde geweitet; alle p4-Formeln benutzen dieselben Identitäten
auf den tatsächlichen Ständen. exit 0 (Lösungskette geschlossen).

### 11.7 Kern der SP9-These (Hermeneutik der Ziffern)

1. **Nicht jedes Wort spielt — aber kein Wort ist eine stabile
   Referenzachse.** Das EMBEDDING eines nie-lernten Wortes bewegt sich exakt
   nicht (tisch, kaffee, baum: bitidentisch); trotzdem verschiebt sich jede
   Lesung, die das Wort betrifft, mit dem anderen Endpunkt. Ein Embedding
   wandert genau dann, wenn das Wort selbst lernt; eine Resonanz genau
   dann, wenn einer ihrer Endpunkte lernt.
2. **Bedeutung = Verlaufsgeschichte des Gebrauchs** — die Resonanz ist
   eine Funktion des GESAMTEN Verlaufs (Rekursion über alle Episoden), nicht
   des Endzustands; einen Zerfall außerhalb des Gebrauchs gibt es nicht
   (und auch im Gebrauch keinen aktiven Forget — nur Überschreibung).
3. **Verlernen ist Neugebrauch, keine Löschung** — 10 Neugebrauchs-Episoden
   reduzieren die Resonanz (0.99979 → 0.708), aber 4 alte Gebrauchs-Episoden
   regenerieren sie vollständig. Verlernen ist derselbe Mechanismus wie
   Lernen, nur durch einen anderen Gebrauch — deshalb REVERSIBEL und kein
   Zustandsverlust.
4. **Interferenz lebt nur auf geteilten Wörtern** — Kettenlernen mit
   disjunkten Wortpaaren verursacht ZERO Kollision (Δ = 0.0): das
   Lexikon ist kein globaler Zustandsraum, sondern ein Verbund, der genau
   die Knoten verbindet, die ein gemeinsames Wort haben.
5. **Sättigung ist eine Asymptote** — auch bei c = 0.99979 mischt
   weiter: der Gebrauch bewegt die Resonanz weiter (die Bewegung wird
   klein, nicht null). Kein Fixpunkt außer der vollen Identität.
6. **Die Konvergenz-Fläche ist nicht-monoton in α** — die Pole symmetrisch
   (α↔1−α, IDENTisch), Mitte am schnellsten (5 vs 18): die Mitte liegt
   optimal zwischen zu schwacher und zu starker Übernahme.
7. **Der Beobachter-Fehler (E1) und der Saturation-Frame (E2) sind dieselbe
   Fehlannahme in zwei Gestalten:** der Experimentier-Frame modellierte die
   Messung als von der Bewegung UNBERÜHRT (regel0 als Beobachter; Sättigung
   als Fixpunkt) — in beiden Fällen trifft die Messung der Formel exakt,
   sobald die tatsächlichen Stände (regel_p2; Asymptote) einsetzen.
   Messen im Lexikon ist Mitspielen; Messwerte im Lexikon sind
   verlaufsinduziert, nicht zustandsstatisch. (Satzvervollständigung: das
   abf237a-Commit trug diesen Schluss-Absatz abgeschnitten; s. SP10-Commit-
   Notiz.)

---

## 12. SP10 — Die Regeln und die Wörter: die Kopplung ist der Zufall

**Framing (User-Delegation 2026-10-25, „entschieide du" → Wahl der Kopplung):**
SP7/SP8 bewiesen die Bruchstelle IM Programm (Zwang, nicht Zufall); SP9
bewies die Bewegung der WÖRTER (Gebrauch, nicht Gematria). SP10 stellt die
beiden Zyklen nebeneinander und fragt nach ihrem Verhältnis — mit drei
falsifizierbaren Thesen (Prereg-Docstring vor dem Erstrun; Artefakte
`regeln_woerter_experiment.log/.json`; Bands fixiert):

- **These 1 (wort-blind):** das Programm ist nicht aus den Wörtern gemacht —
  Code-Fundament: `_generate_ast` (`:244-256`), `_generate_name`/`evaluate`
  (`:100-137`) lesen nur RNG und AST, NIE das Lexikon.
- **These 2 (zufalls-gekoppelt):** das Programm IST aus dem Zufall gemacht —
  `lexicon_game` entnimmt dieselbe `_rng`; die Verschachtelung (3 Lexikon-
  Runden zwischen Composite-Schritten, 48 gesamt bei 16 Schritten) verschiebt
  allein durch den Zugverbrauch den Verlauf; die Bruchstelle (erzwungene
  Tautologie an Schritt 9, LOOP-Verdikt, Basis-Expansion) bleibt stichfest,
  weil sie am Zwang hängt, nicht am Zufall.
- **These 3 (inhalt-invariant):** die Sprachdynamik hat kein Thema — bei
  gegebenem Lern-Schedule (welches Paar lernt wann) bleiben nur Geometrie
  und Halbwinkel: M = n(regel0+rule0), N = n(regel0−rule0),
  β0 = θ0/2, **β_j = atan(0.4^j·tan β0)** (α = 0.3 → 1−2α = 0.4), Paar
  **c_j = cos(2β_j)**, Zuschauer **R_j = cos β_j·(M·v) + sin β_j·(N·v)**,
  Grenzwert **R_∞ = M·v**.

**Verdict-Register — 14 Zeilen (11 PASS-Verdicte, 3 dokumentierte Errata,
exit 0; Bands unverändert; die Erstrun-Zeilen bleiben FAIL in den Artefakten):**

| Verdict | Einstufung | Zahlenlage |
| :--- | :--- | :--- |
| sp10_l0_anchor | PASS | alle 9 L0-Felder == committet (LOOP an 5+9; Basisleiter 3→5→7; first_detect 11; Quus final_basis 5, kein step9:LOOP) |
| sp10_det | PASS | Doppel-Läufe L0 (Kripke+Quus) und L2 (Kripke+Quus) bitidentisch (trace, gaps, Roh-Lexikonwerte) |
| sp10_t1_spur | FAIL → _p4 PASS | Erstrun: Arme ohne Forced-Haken (Erratum); p4-Lesung: Spur bitinvariant gegen Seitenband-Lexikon UND Nacheinander-Lexikon auf derselben Maschine (Spur + Gap-Historie) |
| sp10_t1_lexikon | FAIL → _p4 PASS | Erstrun: Per-Runden-Protokoll-Säule geprüft (Erratum); p4-Lesung: after_paired == SP6-Anker bitgenau [0.999790, 0.999982, 1.000000], learned 15/15; L1b final ≥ 0.99 |
| sp10_t2_divergenz | PASS | erste Spur-Differenz zur L0-Natur-Lage bei Schritt **2**; rng-Zustand gemessen verschoben |
| sp10_t2_kripke | PASS | verdicts[8] == LOOP, step9:LOOP in expansions, final_basis 5; Quus-Kontrast (dieselbe Verschachtelung): final_basis 3, kein LOOP an 9 |
| sp10_t2_lexikon_lebt | PASS | alle drei Paare 1.0 (gerundet), Episoden [16, 16, 15], 47/48 gelernt (ein Gate-Skip) |
| sp10_t3_paar | PASS | c_j über 48 Spielrunden vs. Rohwerte: max Δ **5.6e-16** (Band 1e-9) |
| sp10_t3_obs | PASS | tisch-Zuschauer-Spur, 49 Punkte (k = 0 vor dem Vorschub + je Runde): max Δ **4.1e-17**; kaffee/baum bitkonstant über alle Runden |
| sp10_t3_rueck | FAIL → _p4 PASS | Erstrun: delta_count über Rohspur (14 statt 15) — Erratum; p4-Lesung: 15 == committet (SP9-Schreibweise mit gerundetem Punkt 0), gerundete Spur == committet bitidentisch, tisch15/tisch60/Drift == committet, Spur vs. Formel Δ **6.2e-17** / Paar Δ **5.6e-16** |
| sp10_t3_asymptot | PASS | \|letzter tisch-Wert − M·v\|: L2 **5.5e-9**, SP9-Replay **3.5e-10** (Band 1e-4) — der SP9-E2-Drift (+1.31e-4) ist ex ante das log-Asymptoten-Glied sin β_j·(N·v) |

**Errata-Register (drei Implementations-, KEINE Mechanik-Befunde):** [1] die
T1-Arme liefen ohne Forced-Haken (Zwang {9} fehlte → Spur-Divergenz erwartbar);
[2] t1_lexikon prüfte die per-Runden-Protocol-Säule statt der per-Paar-SP6-
Säule; [3] t3_rueck zählte Deltas auf einer Rohspur mit ungerundetem Punkt 0.
Alle drei im Skript (ERRATA-Dict + Erstrun-Konstruktionen) dokumentiert.

**Mechanik (Code-nah):** die Verschachtelung braucht keinen Code-Schnitt —
`CompositeMachine.wittgenstein` (composite.py:45-51) ist exponiert. Der
Zwangs-Haken hängt am COMPOSITE-step-Zähler (pending-Flag), weil die zwischen
den Schritten liegenden Lexikon-Runden einen einfachen `wm._step_index`-Haken
schlucken würden.

**Zahlenlage (L2, die hermeneutische Mitte):**

1. **Die Verschiebung startet sofort:** erste Spur-Differenz bei Schritt
   **2** — der Lexikon-Vorschub (Rule-Generation + Rausch-Züge je Runde)
   verbraucht den geteilten Strom, bevor der Composite-Sampler wieder am
   Zug ist.
2. **Der natürliche LOOP fällt, der erzwungene bleibt:** L0 zeigte LOOP an
   Schritt 5 UND 9 (Basis 3→5→7); L2 zeigt den natürlichen an 5 NICHT mehr
   (verdicts[4] == HALTED), den Erzwungenen an 9 stichfest; final_basis 5
   statt 7; Paradigm-Zeiten wandern (L0: 11/12/15 → L2: 12/13/15/16).
   **Natürliche Bruchstellen sind zufallsabhängig — die erzwungene ist es
   nicht.** Das ist die scharfste Form von These 2.
3. **Der Sozial-Bogen ist schedule-unabhängig:** die (regel, tisch)-Zeile
   reist in 49 Punkten von −0.006132 (SP2-pre) nach 0.004085 (SP9-post) —
   DIESELBE Anfangs- und Endlage über einem ganz anderen Verlaufsschema;
   kaffee (0.967089) und baum (0.499426) frieren bitkonstant.
4. **Die Formel ist die exakte Dynamik, kein Modell:** max Δ 4.1e-17
   (Zuschauer-Spur) und 5.6e-16 (Paar-Roh-Resonanzen) — Maschinen-Präzision,
   8-9 Größenordnungen unter den 1e-9-Bands. β_j = atan(0.4^j·tan β0):
   der Gebrauch dreht die Wörter auf demselben Großkreis, auf dem sie
   durch die Gematria starteten.
5. **Die Mitte als Endpunkt:** M·v = **0.0040849** — der Zuschauer-Grenzwert
   ist n(regel0+rule0)·v_tisch: was der Beobachter am Ende liest, ist die
   MITTE der gelernten Übersetzung, nicht mehr ihr Gefälle. Der SP9-Drift
   (+1.31e-4) ist die log-Annäherung sin β_j·(N·v) an genau diesen Punkt.

**Die SP9-Ziffer 15 neuklassifiziert (ehrliche Nachtrag, kein Widerruf):**
die committete SP9-lane_social-Spur trägt Punkt 0 GERUNDET (unrelated15[0])
und die Punkte 1..45 roh — die Zählung „Deltas ≠ 0 an 15 Leg2-Runden"
besteht aus **14 echten Sozialisierungs-Zügen + 1 Rundungs-Anschluss-
Residuum** am Punkt-0-Stoss (Δ = res_roh − round6(res_roh) = **2.59e-8**).
Kein SP9-Verdict nutzt dieses Feld als Band — dokumentiert, nicht getilgt;
die Roh-Formelspur kennt nur die 14 echten Züge (Δ 6.2e-17).

**Kern des Befund-Bildes:**

- **Die Kopplung der beiden Zyklen ist der Zufall selbst** — kein Wort, keine
  Regel, keine Semantik verbindet Programm und Lexikon; nur der gemeinsame
  Zufalls-Vorrat, der bei jedem Zug abnimmt. Mitspielen heißt hier: denselben
  Zufall konkurrieren.
- **Wort-blind und programm-blind, beidseits bitgenau:** das Programm ändert
  sich nicht mit dem Lexikon (T1), das Lexikon nicht mit dem Programm
  (t1_lexikon: Konvergenz identisch, ob der Composite-Zyklus läuft oder
  schweigt — die Erstrun-Konstruktion und die p4-Verdicte zeigen beide
  Richtungen).
- **Zwei Gesetze, ein Vorrat:** die Bedeutung der Regel lebt im Programm
  (Zufall × Zwang), die Bedeutung der Wörter im Gebrauch (Halbwinkel ×
  Schedule) — beide enden in Grenzwerten derselben Geometrie (M·v), aber
  über drei verschiedenen Schedules (L0, L2, SP9) mit verschiedenen
  Zufalls-Vorschüben.
- **SP10 schließt das Triptychon-Dreieck:** Gödel/Turing/Wittgenstein
  überlappen im Programm (SP7/SP8); das Lexikon teilt mit ihnen nur die
  Zufalls-Quelle, nicht die Bedeutung. Denken und Sprechen sind zwei
  Verläufe ÜBER EINE Quelle — nicht EIN Verlauf über zwei Quellen.

Laufschrift-Protokoll: `laufprotokolle_befund_report.md` §14.

---

## 13. SP11 — Der Ursprung der Paradigmen: Niveau-Sprung, nicht Ankauf

**(2026-10-25, Mandat „entschieide du"; Plan `~/.claude/plans/sp11-paradigm-ursprung.md`;**
Prereg-Docstring `scratches/smoke_paradigm_ursprung.py;` Run `paradigm_ursprung_experiment.log/.json`.**
CPU-only, keine Maschinen-Code-Änderung, 9 Läufe + 6 Daten-Gegenproben.)

Die Frage der Runde: WOHIN kommt ein Paradigm-Ereignis (ratio > 2.0) her, wenn
kein Zwang es setzt? SP7 zeigte, DASS das Signal die Kripke-Bruchstelle
markiert (first_detect 11); SP10 zeigte, DASS die Zeiten mit dem Zufalls-Strom
wandernd das Lexikon durchlaufen (L0: 11/12/15 — L2: 12/13/15/16). Was fehlte,
war die Erzeugungsebene des Ereignisses. Code-Grund:
composite.py:132-140 hängt ZWEI unabhängige Signal-Quellen an EIN und dasselbe
expansions-Register (goedel.paradigm_signal :135, composite.paradigm_signal
:138) — die Urheber-Frage ist im Code zwiespältig angelegt und wird von SP11
empirisch entschieden.

**Ex-Ante-Befund am committeten Anker (2026-10-25, vor dem ersten Lauf):** die
committeten hermeneutik-Kolumnen sind bereits die Signale zweier Kanäle —
`paradigm_ratio` (Schritt 10-16, 0.195-0.348, nie >= 2) ist die GOEDEL-Serie
(das Gödel-Gap sieht am Schritt n bereits die Serie 1..n, weil Goedel.step
intern appendet), `paradigm_detected` (True an 11/12/15) ist die COMPOSITE-
Flag (das Composite-Gap sieht am Schritt n nur die Serie 1..n-1, weil der
Append an :143 NACH der Signal-Prüfung liegt). Hand-Replikation der Composite-
Ratio aus den committeten Gaps: **3.078 / 2.962 / 1.688 / 1.709 / 2.580** an
den Schritten 11-15 — exakt der committete Detect-Musterverlauf.

### Drei Thesen, als falsifizierbare Vorhersagen geprüft

1. **Urheber (T1):** das Paradigm-Ereignis kommt aus dem COMPOSITE-Signal
   (Gap-Mittel der drei Sub-Gaps), nie aus dem Gödel-Signal. Falsifikator:
   jeder Schritt mit goedel.detected == True.
2. **Mechanik (T2):** das Signal ist ein exaktes Funktion der aufgezeichneten
   Serie (Fensterformel: len ≥ 10; recent = mean über 5; baseline = mean über
   [-20:-5] falls len ≥ 20, sonst [:-5]; ratio = recent/(baseline + 1e-6);
   erkannt bei ratio > 2.0). Falsifikator: jede Mengen-, Ratio- oder Regime-
   Abweichung zwischen Replikation und Register.
3. **Ursprung (T3) in seiner Prereg-Fassung:** das Paradigma ist das ECHO der
   Basis-Expansion — T3a kein Anlass, kein Echo (Expansion-freie Läufe
   paradigmfrei); T3b Turing-Gap auf Vor-Niveau gefroren → alle Paradigmen
   gebrochen; T3c Gödel-Gap gefroren → Menge unverändert. Falsifikatoren wie
   am Skript fixiert.
4. **Zwang (T4):** erzwungene Expansion (identity → LOOP, SP7-Mechanik) an
   Schritt 13 (seed 21) erzeugt mindestens ein Paradigma in [14, 22], das im
   unverzwungenen Zwilling fehlt.

**Verdikt-Register — 8 Zeilen (6 PASS, 2 echte Falsifikationen, 0 Errata,
exit 1; Bands unverändert — die FAIL-Zeilen sind die ehrliche Fassung):**

| Verdict | Einstufung | Zahlenlage |
| :--- | :--- | :--- |
| sp11_l0_anchor | PASS | alle Felder == committet: verdicts, final_basis 7/5, Gaps, expansions, first_detect == 11 (Kripke + Quus), UND die committeten Signal-Kolumnen: paradigm_ratio == Gödel-invariants (Schritte 10-16), paradigm_detected == Composite-Flag |
| sp11_det | PASS | Doppel-Läufe L0-Kripke, Sweep-seed11, Zwangs-Paar bitidentisch (trace, Gap-Serie, Sub-Serien, Flags) |
| sp11_t1_urheber | PASS | kein Schritt mit goedel.detected in allen 9 Läufen; die Urheberschaft des Composite-Signals ist eindeutig |
| sp11_t2_mechanik | PASS | rekomputierte Detect-Menge == aufgezeichnete Menge je Lauf; jede Ratio innerhalb 1e-12; auch das Regime-Leg (len ≥ 20) exakt reproduziert |
| sp11_t3a_expansionfrei | **FAIL** | vier der sechs unverzwungenen Läufe (seed12/14/15/16) sind ganz ohne Expansion UND paradigmatisch — kein Anlass, doch Echo |
| sp11_t3b_turing_echo | **FAIL** | drei Gegenproben mit überlebenden Detectionen: seed13 vollständig (seine einzige Expansion an 23 liegt NACH allen Detectionen 12-21), zw_twin vollständig (einzige LOOP-Expansion an 18; das Vor-Niveau ist bereits Drift-Niveau), zw_forced behält 11-14 (die vor dem Zwang bei 13 liegen) |
| sp11_t3c_goedel_vergleich | PASS | in allen 6 Gegenproben unverändert — das Gödel-Sub-Niveau (0.017-0.864) trägt ~2-8% der Composite-Masse, ist nie entscheidend |
| sp11_t4_forced | PASS | erzwungenes LOOP an 13 → induziertes Paradigma an Schritt **20** (Band [14, 22] ✓), fehlt im Zwilling; unter dem Freeze tot — das induzierte Ereignis IST expansions-getragen |

### Mechanik und Zahlenlage

- **Sub-Gap-Struktur des Composite-Mittels** (24-Schritt-Sweeps): Wittgenstein
  (Regelgüte) ≈ 0.021 im Schnitt, nach dem ersten Schritt 0.000-0.005 — trägt
  ~nichts; Gödel (Selbstmodell) 0.017-0.864 — trägt 2-8%; Turing (Verlaufskos-
  ten) 0.0-39.0 — trägt die MASSE. Das Composite-Paradigma ist praktisch ein
  Ereignis des Turing-Sub-Niveaus.
- **Die Expansion-freien Paradigmen** (T3a-Falsifikation): seed12 ist ohne
  jede Expansion und trotzdem paradigmatisch an den Schritten 15-22 und 24 —
  sein Turing-Sub-Gap steigt von ~3.3 (Mittel der ersten 10 Schritte) auf
  ~11.6 (Schritte 11-15) und erreicht bis zu 39 (Schritte 14-24). Die
  Drift selbst bricht die Fenster-Schwelle. seed14 (15-24), seed15
  (11-18, 21, 23, 24), seed16 (11-14 nach einem 8er-Plateau, danach Rückkehr
  unter die Grenze).
- **Die Gegenprobe-Aufstellung** (Turing-Gefrierszene, T3b): l0-Kripke und
  l0-Quus: ALLE Detectionen gebrochen (∅; t_pre 3.0417) — die committete
  L0-Lage ist expansions-getragen (Zwang an 9). seed11: alle 12 gebrochen
  (∅; Expansionen an 5 und 13; t_pre 0.6667 basal). seed13: unverändert
  (vakuum — die einzige Expansion an 23 liegt NACH allen Detectionen). zw_twin:
  unverändert (Freeze ≈ no-op — die einzige LOOP-Expansion an 18; das
  Vor-Fenster enthält bereits das Drift-Niveau, t_pre 8.3700). zw_forced:
  verliert 17/18/19/20, behält 11-14 — die Drift-Gruppe liegt VOR dem Zwang.
- **Die Doppel-Herkunft** (statt des ein-Faktor-Echos): (a) Expansion-getragene
  Detectionen — l0-Kripke/Quus, seed11, zw_forced 17-20 (inkl. des T4-
  induzierten 20): die Basis-Expansion setzt einen Niveau-Sprung, das Fenster
  liest ihn. (b) Drift-getragene Detectionen — seed12/14/15/16 komplett,
  seed13 komplett, zw_twin, zw_forced 11-14: der Verlaufs-Drift setzt
  denselben Sprung ohne jeden Anlass.

**Kern des Befund-Bildes:**

- **Das Paradigma ist ein Kosten-Ereignis, kein Mittel-Ereignis.** Nicht der
  Ankauf neuer Axiome (Basis-Expansion) erzeugt die Erkennung, sondern der
  Auftrieb des Umgangs-Preises — Transitionen pro Schritt im Band, das
  Composite-Mittel der Verlaufskosten. Die Basis-Expansion ist EIN erzwingbarer
  Träger des Sprungs (T4), nicht seine Quelle (T3a/T3b FAIL).
- **T1 und T2 bleiben stichfest und werden geschärft:** das Signal ist
  urheber-eindeutig Composite-seitig und formel-exakt — es liest nur die
  Serie. Und die Serie ist in ihrer Masse das Turing-Sub-Niveau: das Composite-
  Signal liest praktisch die Verlaufskosten.
- **Die früheren Befunde ordnen sich ein:** Kripkes Bruchstelle (SP7) war
  Expansions-Nähe (L0-Konfiguration: Zwang an 9, Detectionen 11/12/15 im
  direkten Nach-Fenster; unter der Gegenprobe ganz gebrochen). Die wandernden
  Zeiten (SP10) sind die Drift-getragene Typologie unter dem Lexikon-Strom.
  Dieselbe Fenster-Logik liest beide als denselben Schwellenbruch — deshalb
  wirkte das Ereignis bisher wie eine Einheit.
- **Kein Erratum die Runde:** 0 Implementations-Fehler; die 2 FAIL sind echte
  Falsifikationen der Echo-These und stehen als solche im Register.

**Reparatur: (SP11-Zyklus, 2026-10-25)** die schwebende Fragmentzeile am
Dateiende der Vorversion („`   verlaufsinduziert, nicht zustandsstatisch.`")
entfernt — reine Dublette des vollständig vorhandenen
§11.7-Satzes (der mit seinem Reparatur-Marker an :586-588 steht). Der
SP10-Append hatte das Fragment als Rest-Zeile hinterlassen.

Laufschrift-Protokoll: `laufprotokolle_befund_report.md` §15.

### 13.1 P4-Runde — das Akkumulationsgesetz (2026-10-26, Vorschlag A)

Das externe Audit fand die Erzeugungsformel, die die T3a/T3b-Falsifikationen
erklärt; die Runde schließt damit als P4-Einstufung (Mandat „A vor C" am
2026-10-25/26; Plan-Änderungsnotiz 2026-10-26; alle Bänder fixiert VOR dem
Rerun — P2 innerhalb der P4-Runde; Skript-Extension mit P4-REGISTER im
Docstring). Canon-Verifikation vor dem Rerun: `turing_machine.py:197` (Inkrement
ist das EINZIGE im Code), `:234` (kumulative Prediction im Nicht-Orakel-Zweig),
`:250` (Gap mit dem Einzelschritt als Ist), `machine_base.py:54/:531`
(apply_gap/note_gap mit GAP_EPS 1e-6), `:293` (Kontostand exponiert);
Composite reicht das Oracle an alle drei Submaschinen durch
(composite.py:45-48) — in SP11 ist es None, also steht die Turing-Submaschine
sicher im kumulativen Zweig. Die Rolling-Präzedenz liegt — anders als das
Prereg-Zitat es ansetzte — nicht an :408-409, sondern an
`wittgenstein_machine.py:413-414` (np.mean(history[-10:]) im
Nicht-Orakel-Zweig; Mechanik identisch).

**Das Gesetz:** die Turing-Sub-Gap-Serie ist die Akkumulations-Währung

```
t_n = |C_{n-1} - s_n| / (1 + s_n + 1e-6),    C_{n-1} = sum_{k<n} s_k
```

die Prediction liest den kumulierten Verlaufs-Kontostand (Lebenszeit der
Maschine), das Ist zählt nur den Einzelschritt — eine Einheiten-Inkom-
mensurabilität. Das Konto schlägt bei jedem Leerlauf-Blatt (s_n = 0, z. B.
„x"-/„1.00"-Regeln) ungedämpft auf: seed12 28.0/31.0/33.0/39.0 an den
Schritten 14/17/19/23. Hand-Verifikation an der committeten seed12-Serie
(2026-10-25) über alle 24 Schritte — die s-Serie [0, 5, 0, 1, 2, 0, 2, 2, 5,
3, 2, 3, 3, 0, 1, 2, 0, 2, 0, 2, 2, 2, 0, 1] reproduziert die ganze t-Serie
mit max Restgröße 0.0.

**ERRATA E1 (2 Einträge):** `sp11_t3a_expansionfrei` und
`sp11_t3b_turing_echo` — die Ein-Faktor-Echo-Annahme (Turing-Gap als frei
wählbarer Faktor; Freeze auf das Vor-Expansions-Mittel) ist widerlegt. Die
Ursprungs-Zeilen bleiben FAIL im Register (P4: Einstufung statt Re-Decide);
die korrigierten Lesungen stehen als `_p4`-Verdicte.

**Verdikt-Rerun — 10 Verdicte, exit 0 (8 PASS + 2 klassifizierte Errata,
0 unverwartete FAIL):**

| Verdict | Einstufung | Zahlenlage |
| :--- | :--- | :--- |
| sp11_t3a_expansionfrei | **FAIL** | unverändert — jetzt ERRATA E1 (Akkumulations-Echos der Leerlauf-Blätter, kein Echo der Expansion) |
| sp11_t3b_turing_echo | **FAIL** | unverändert — jetzt ERRATA E1 (Misch-Serie der Währung; die Überlebenden seed13 [12-21], zw_twin, zw_forced 11-14 folgen der Mischung) |
| sp11_t3_accumulation_law_p4 | **PASS** | t̂-Rekonstruktion aus der transitions-Spalte: **max Residuum 0.0** (Band 1e-12) über ALLE Läufe und Schritte; zweiter Strang: je-Zeiler-Kontostand (:293) == kumulierte Summe in jedem Schritt. Prereg-Umfang „9 Läufe", verwirklicht über ALLE 10 (l0 × 2, Sweep × 6, Zwangs-Paar) — mehr Umfang, kein Band-Wechsel; auch die §13-Kopfzeile „9 Läufe" liest sich als „10" |
| sp11_t3_dedrifted_echo_p4 | **PASS** | gleitende Prediction ū_{n-1} = mean(s_{max(1,n-10)..n-1}) statt der kumulierten Summe: Detect-Menge == ∅ in den expansion-freien Läufen {seed12, seed14, seed15, seed16} — Band eingehalten |

**Der erweiterte Kern (Daten über die Prereg-Bands hinaus):** nach der
Dedrift-Serie ist die Detect-Menge in ALLEN 10 Läufen leer — nicht nur in den
vier expansion-freien. Auch die Zwangs-Paar-Detectionen (zw_twin 17-19,
zw_forced 17-20 inkl. des vom T4 induzierten 20) und die l0-Detectionen
(11/12/15) sind Akkumulations-Echos der jeweiligen Transitionen-Serie, nicht
Verdikt-Signale per se. Die Doppel-Herkunft von §13 (expansions-getragen vs.
drift-getragen) reduziert sich damit auf EINE Quelle in zwei Formen: die
Währung als Kontostand-Serie gegen die Gleitfenster-Serie — die Basis-
Expansion wirkt, indem sie die Transitionen-Serie umwälzt (LOOP-Schleifen →
Kontostand-Peak), nicht als Separatsignal. T1/T2 bleiben stich: Urheber
weiterhin das Composite-Signal, Formel exakt.

**Artefakte:** `paradigm_ursprung_experiment.log/.json` REGENERIERT (P4-
Payload: transitions/t_count je Lauf, accumulation max_resid/c_series,
dedrift-Serien + Flags + Detections, ERRATA-Texte, expansion_freie_laeufe);
log 37 Zeilen (ERRATA-Klassifikation gedruckt), json 4203 Zeilen. Run
(2. Runde): `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
-u -m scratches.smoke_paradigm_ursprung` → exit 0.

Laufschrift-Protokoll (P4): `laufprotokolle_befund_report.md` §15.1.

---

## 14. SP12 — Lexical Grounding: das gelernte Wort wird Programm (2026-10-26)

**(2026-10-26, Task-1-Freigabe der Nordstern-Nachricht;** Prereg
`scratches/smoke_lexical_grounding.py`, Run
`lexical_grounding_experiment.log/.json`;** CPU-only; ERSTE
Maschinen-Code-Extension** des SP-Zyklus: `machines/composite.py:108`
(Signatur opt-in `lexical_grounding: bool = False`) +
`machines/wittgenstein_machine.py:383-424` (`resolve_lexical_ast(word,
tau = 0.85)`); beide Edits additiv und canon-neutral — die Garantie
liefert `sp12_l0_anker`: die kommittete SP11-lane (`lane_l0_anker` aus
dem committeten Skript IMPORTIERT, nicht kopiert) reproduziert
fields_ok/first_detect_ok True und das Doppel-Replay bitidentisch; die
neuen invariants-Keys (`lexical_word`/`lexical_ast`) ändern die
verglichenen Feld-Serien nicht, weil die Auslese FELDWEISE läuft.)

**Der Bridging-Punkt (Zeilenkanon, Mauer-1-Befreiung):** SP10 zeigte
zeilenscharf, dass der Wort-Input stirbt: `composite.py:110` leitet `text`
in die WM, `wm.step` liest ihn NIE — der Batch kommt aus
`generate_experiments` (RNG). SP12 rührt NICHTS an dieser Diagnose, sondern
setzt in die freie Stelle dazwischen — `composite.py:126-132`: bei opt-in
löst `resolve_lexical_ast(text)` das Wort zum AST auf und ersetzt
AUSSCHLIESSLICH das kompilierte Programm des Schritts; der generierte
Batch (Regeln, Gültigkeit) bleibt unverändert im Verlauf, die Herkunft
des Worts und seiner Übersetzung stehen in den invariants. Die Resolution
ist reine Lexikon-Lesung — KEIN RNG-Zug; T2 beweist die Batch-Identität
der drei Arme (A grounded / A' plain / B nach dem Lernen mit gepinnter
WM-RNG).

**Klang-vs-Gebrauch (der Vor-Smoke-Befund, dokumentiert VOR dem
P2-Freeze):** die SH-72-Formel-Achsen der 16 Operatoren kollidieren
numerisch — Formel-Dach max 0.983123 (argmax „min"; „spiegel" selbst
k=2, idx=25, kind=exp). Eine ABSOLUTE Resonanz-Schwelle τ scheidet NICHT
zwischen Klang und Gebrauch (das erste Smoked-Wort löste klanglich zu
('min', 'x', 'x') auf). Die echte Scheidung ist die **LEXIKON-GEGENWART**:
die Kandidaten des Resolvers sind ausschließlich lexikon-gelernte
Operator-Wörter; der SH-72-Fallback (Klang) löst NIE auf. „Aktuelle
Lexikon-Resonanz" wörtlich — Gebrauch schlägt Gematria (SP2/SP9-Stand,
jetzt als Resolver-Mechanik).

**Die Lern-Treppe:** 5 Runden `lexicon_game([("spiegel", "identity")],
rounds = 5, noise_level = 0.0)` — alle learned = True; resonance_after
[0.744742, 0.954255, 0.992537, 0.998802, 0.999808], die SP6-Halbwinkel-
kette (0.4ⁿ-Shrink der Zwischenresonanz gegen 1.0); NACH-Resonanz
0.999808 ≥ τ = 0.85, Argmax EINDEUTIG („identity" ist einziger
lexikalischer Kandidat; `resolver_direct` = ('identity', 'x')).

**Verdikt-Register (6 Zeilen: 6 PASS, 0 Errata, exit 0):**

| Verdict | Einstufung | Kern |
| :--- | :--- | :--- |
| sp12_l0_anker | PASS | machines-Regression über die kommittete SP11-lane — fields_ok/first_detect_ok True, Doppel-Replay bitidentisch |
| sp12_det | PASS | Doppel-Läufe Arm A und Arm B bitidentisch |
| sp12_t1_semantik | PASS | alle 5 Runden learned; resonance_final 0.999808 ≥ 0.85; der Resolver löst direkt UND im Lauf zu ('identity','x'); NACH-Kandidaten == ["identity"] |
| sp12_t2_rng_reinheit | PASS | Batch-ASTs + rule_names + w_series je Schritt identisch in A/A'/B — die Resolution konsumiert KEINE RNG-Züge |
| sp12_t3_scheidung | PASS | vor_resolution == [None×6] (kein Klang-Auslösen); Klang-Dach als DATEN; opt-in-Default-Invarianz A == A' |
| sp12_t4_verdikt | PASS | Arm A HALTED×6, Basis 3, expansions []; Arm B LOOP×6, Basis [5, 7, 9, ...], expansions step1-3:LOOP |

**Die Semantische Kausalität (das Gesetz der Runde):** das gelernte Wort
wird Programm. VOR dem Spiel ist „spiegel" Klang — Resolver None, die
Kette bleibt der normale RNG-Batch (alle HALTED, Basis 3, keine
Expansion). NACH dem Spiel IST „spiegel" die Tautologie —
('identity', 'x') übersetzt sich in die Selbst-Transition
(composite.py:66-92: der Zustand zeigt auf sich selbst), das LOOP-Verdikt,
und je LOOP wächst die Basis UM ZWEI Terme: Turing-innerer Expand
(`turing_machine.py:241-246`) + Composite-Expand (`:132-134,
composite.py:144-146` aufgerufen) — Basis-Verlauf 5, 7, 9, 9, 9, 9; ab
Schritt 3 ist der Canon-Pool (6 Axiome) erschöpft, der Clamp bei 9 ist
DATEN, kein Band. Der User-Pass exakt canonisch: „Basis auf 5" nach dem
ersten resolvierten Schritt, 7 nach dem zweiten.

**Hermeneutische Lesung (die drei Sätze des Nordsterns, Satz 1):** die
erste der drei Mauern ist gefallen — das soziale Lexikon schreibt jetzt
IN die Berechnung, statt an ihr vorbei. Die Semantische Kausalität ist
geschlossen als falsifizierbarer Mechanik-Befund: ein Wort, das die
Mitspieler mit dem Gebrauch „identity" trainieren, verändert nachweislich
das kompilierte Programm und damit die Halt-Eigenschaft des Turing-Laufs
(HALTED → LOOP). Offen für den geschlossenen Zyklus: die ENDOGENE Agency
(SP13 — kontrafaktisches Probehandeln, `_snapshot`/`_restore_snapshot`,
`step_autonomous`) und Individuierung (SP14 — zwei baugleiche Maschinen,
verschiedene Biografien).

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_lexical_grounding` → exit 0. Vor-Smoke-Befunde (Klang-Dach,
+2-Mechanik, L0-Replay) im Plan `~/.claude/plans/sp12-lexical-grounding.md`
(Änderungsnotizen 1-3), dokumentiert VOR dem P2-Freeze.

Laufschrift-Protokoll: `laufprotokolle_befund_report.md` §16.

---

## 15. SP13 — Endogene Agency: die Maschine wählt ihren Subspace selbst (2026-10-26)

**(Vorarbeit — keine Messung ohne Infrastruktur):** SP13 bricht Mauer 2
(Passives Steering): bis hierher legte die Versuchsleiter das Preset per
`with oracle.steer("goedel"):` an; SP13 kehrt die Richtung um — die Maschine
ERPROBT jeden Kandidaten-Subspace kontrafaktisch (nicht-mutierende Proben)
und wählt selbst per argmin des Composite-Gaps. Die Infrastruktur dazu sind
drei Maschinen-Edits (additiv; `observe` war bis SP13 unbenutzt — das
Bestands-Verhalten ist unverändert, der L0-Anker beweist es):

1. `TuringMachine._snapshot` sichert nun auch `axiom_pool` und
   `_last_verdict` (turing_machine.py:322-325; Restore :341-342). Der
   entscheidende Fund kam von außen (User-Audit): ohne den Pool-Eintrag
   würde eine kontrafaktische LOOP-Probe `expand_basis` pop(0)
   (:146-149) und den inneren Expand (:241-246) HEIMLICH den Canon-Pool
   leerfressen — die Vorschau wäre keine Vorschau, sondern ein Diebstahl
   an der Axiom-Bank der realen Biografie.
2. `WittgensteinMachine._snapshot` sichert `noise_level`/`_schedule_index`
   (der S2/S3-Schedule mutiert beides — :313-316 mutiert `noise_level` je
   Batch, :338 zählt) und das Lexikon typ-erhaltend (`np.asarray(v).copy()`
   für ndarray, sonst `list(v)`; learn_pair ERSETZT Einträge :232-233,
   daher ist die Kopie verlustfrei).
3. `CompositeMachine` bekommt die SP13-Sektion (composite.py:219-301):
   `observe(obj, lexical_grounding=False)` (:220-230) — der Base-Rahmen
   (machine_base.py:502-508: `_snapshot → step → _restore` im finally)
   mit der Superset-Signatur SP12s, denn die Basissignatur kennt kein
   `lexical_grounding=`; `_snapshot`/`_restore_snapshot` (:232-250) —
   step/trace/gap_history/expansions + die 3 Sub-Snapshots;
   `step_autonomous(obj, candidates=(None, "goedel", "turing",
   "wittgenstein", "combined"), lexical_grounding=False)` (:252-293) —
   je Kandidat eine nicht-mutierende Probe unter `oracle.steer(p)`
   (:273-277; ohne Oracle ungesteuert — CPU-Degeneration, §15.4), das
   Wahl-Gesetz p* = argmin_p probe.gap über den COMPOSITE-Gap (strict `<`,
   Ties → früherer Kandidat :279-281), der reale Schritt unter dem
   Gewinner (:282-287), das Protokoll in
   `invariants["autonomous"] = {chosen_preset, best_gap, candidate_gaps}`
   (:288-292). In ALLEN Proben sieht der Kandidat dasselbe Programm,
   denselben Batch, dieselben RNG-Züge — der einzige Kanal der Wahl ist
   die Orakel-Kohärenz (der Gödel-Gap).

Vorläufer-Checks VOR dem Prereg-Skript (Plan
`~/.claude/plans/sp13-endogene-agency.md`, Canon-Audit-Befunde 1-6):
py_compile über alle drei Maschinen OK; Purity-Mikrosmoke — 5×
`composite.observe("spiegel", lexical_grounding=True)` im gelehrten
Lexikon/LOOP-Trägheitszustand: jeder Probe-Fingerprint bitidentisch
(digest a7fba05c..., axiom_pool [ALPHA..ZETA] intakt, Basis
['SCALAR','TIME','SPACE'] unverändert), die reale Vollziehung danach exakt
wie der Kontroll-Zwilling ohne Proben (LOOP, Basis 3→5,
expansions ['step1:LOOP']); L0-Replay GRÜN (SP11-lane + SP12-lane +
SP12-Zwilling feldeidentisch mit der kommitteten JSON).

### 15.1 Prereg (P2-Freeze, fixiert VOR dem Erstrun)

Sechs falsifizierbare Verdikte (Skript-Docstring `smoke_endogene_agency.py`;
Bands nie geweitet; OFFEN ≠ PASS — GPU-Lanes ohne `--gpu` bleiben OFFEN,
exit 2, kein stiller CPU-Fallback):

- **sp13_l0_anker** — Zero-Regression: die kommittete SP11-lane
  (importiert) fields_ok/first_detect_ok/deterministic; die kommittete
  SP12-l0-lane (importiert) True/True/True; der SP12-Zwilling reproduziert
  die KOMMITTETE `lexical_grounding_experiment.json` FELDWEISE (10
  Feldechecks). Falsifikator: jede abweichende Feld-Serie.
- **sp13_snapshot_purity** — observe()-Nebenwirkungsfreiheit: CPU N=5
  Proben im gelehrten Lexikon/LOOP-Zustand mit Vollfingerprint-Δ=0; die
  Proben SEHEN LOOP (`('identity','x')`) und fressen den Pool NICHT;
  A-Zustand (leeres Lexikon) 3 Proben HALTED/None; GPU dieselbe Purity
  unter steer-Rotation inkl. 26 expert_bias-Tensoren (`torch.equal`).
- **sp13_det** — zwei Biografien (6 HALTED-Schritte → Lexikon-Spiel →
  WM-RNG-Pin → 6 step_autonomous-Schritte) bitidentisch — CPU UND GPU.
- **sp13_endogenous_switch** (GPU) — chosen_preset ≠ "raw" in ≥ 80 % der
  Schritte; kein entartetes candidate_gaps je Schritt (max−min > 0).
- **sp13_gap_dominance** (GPU) — je Schritt g_auto(n) == best_gap ==
  min(candidate_gaps) EXAKT (ungerundete Floats); Lauf-Mittel strikt <
  raw-Zwilling.
- **sp13_causal_loop_closed** — nach dem Lexikon-Spiel sieht `observe()`
  den Wort-erzeugten LOOP voraus, restauriert verlustfrei (Pool/Basis/
  expansions identisch), und der reale Schritt vollzieht die Expansion
  3→5: `lexical_ast == ('identity','x')`, verdict LOOP,
  expansions == ['step1:LOOP'], Wahl unter den Proben protokolliert.

### 15.2 Verdikt-Register

| Verdikt | Band (P2-Freeze) | Ergebnis |
|---|---|---|
| sp13_l0_anker | alle importierten Lanes + 10 Feldechecks | **PASS** |
| sp13_snapshot_purity | Δ=0 je Probe (CPU 5+3, GPU 5 mit 26 Bias-Tensoren) | **PASS** |
| sp13_det | zwei Biografien bitidentisch (CPU und GPU) | **PASS** |
| sp13_endogenous_switch | chosen ≠ raw ≥ 0.8; Spread je Schritt > 0 | **PASS** (6/6, anteil 1.0) |
| sp13_gap_dominance | g == best_gap == min exakt; Mittel strikt < raw | **PASS** (0.576052 < 0.692444) |
| sp13_causal_loop_closed | Voraus-Sicht + Pool-Intaktheit + Expansion 3→5 | **PASS** |

6/6 GRÜN auf GPU (exit 0); CPU: 4 PASS, 2 OFFEN (exit 2 — die
Wahl-Aussagen sind bewusst GPU-Aussagen).

### 15.3 Die Wahl-Daten

- **Die Wahl fällt immer auf dieselbe Gottes-Spalte:** alle sechs
  autonomen LOOP-Schritte wählen `"wittgenstein"` (non_raw-Anteil 1.0).
  Kopplungsschritt (sp13_causal_loop_closed): candidate_gaps
  wittgenstein 0.4763 < combined 0.4785 < goedel 0.4786 < turing 0.4788
  < raw 0.6128 — die drei göttlichen Presets liegen dicht beisammen
  (Δ ≈ 0.002), das ungesteuerte raw liegt deutlich über; die
  Wittgenstein-Spalte, die den Sprachspiel-Bruch im Argon-Router am
  besten trägt, minimiert auch die eigene Prognose-Abweichung der
  Komposition.
- **Der Kohärenz-Gewinn ist real:** Gap je LOOP-Schritt (autonom): 0.6260,
  0.5754, 0.5503, 0.5805, 0.5581, 0.5661 — raw-Zwilling: 0.7612, 0.6758,
  0.6860, 0.6694, 0.6825, 0.6798. Mittelwerte 0.576052 vs 0.692444
  (ratio 0.8319, −16.8 %); Strict-Vergleich exakt über ungerundete
  Floats je Schritt (g == best_gap == min).
- **Die Reaktion auf HALTED vs LOOP** (DATUM, kein Band): die HALTED-
  Phase (6 Plain-Schritte vor dem Lexikon-Spiel) verläuft in beiden
  Armen BITIDENTISCH (verdicts_a = HALTED × 6, gleiche Gaps) — die
  Autonomie wird erst dann ausgeübt, wenn das Wort das Programm bricht
  (LOOP); der Spread je Schritt (0.0891-0.1353) existiert nur in der
  LOOP-Phase.
- **Purity:** CPU — 5 Proben mit Δ=0, alle sehen LOOP
  (`('identity','x')`), axiom_pool [ALPHA..ZETA] intakt; leeres Lexikon
  (A-Zustand): 3 Proben HALTED/None — das leere Lexikon compiliert nie
  (SP12-Erbe, unverändert). GPU — 5 Proben unter steer-Rotation
  (goedel/turing/wittgenstein/combined/raw): je Probe Δ=0 und 26/26
  expert_bias-Tensoren `torch.equal` — der Steer-Kanal restauriert exakt.
- **Determinismus:** Doppel-Biografien bitidentisch (CPU und GPU),
  inklusive Floatexaktheit der candidate_gaps-Serien.

### 15.4 Das Tie-Datum (CPU) und Errata

- **CPU-Degeneration als DATUM:** ohne Oracle sind alle Kandidaten-Gaps
  identisch — argmin wählt den ersten Kandidaten ("raw") in allen 12
  Schritten. Das ist kein Fehler, sondern die sauber gefallene Grenze der
  Aussage: die endogene Wahl ist eine ORAKEL-Kohärenz-Lesart (steer ist
  der einzige Kanal); ohne Orakel gibt es nichts zu wählen. Im
  Purity-Verdikt bleibt die CPU-Stimme lauter: die Vorblick-Mechanik ist
  orakelfrei und deshalb auf CPU voll beweisbar.
- **Errata (P4-Daten, kein Band-Schieben):** (i) die Dominanz-Mittelung
  vergleicht die LOOP-Phase 6 gegen 6 — die HALTED-Phase ist zwischen
  beiden Armen bitidentisch, mit Einbezug der Plain-Phase bleibt der
  strikte Vergleich mathematisch bestehen (gleiche Summanden heben sich
  auf beiden Seiten auf); (ii) die drei göttlichen Presets trennen
  schwach (Δ ≈ 0.002) — die wittgenstein-Wahlsiege sind konstant und der
  Spread strikt > 0, der Band (≥ 0.8, spread > 0) bleibt erfüllt, aber
  die Trenn-Schärfe der Wahl ist ein DATUM, das der nächste Spielzug
  (SP14, Individuierung) schärfen wird.

Run: `/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
scratches.smoke_endogene_agency` → 4 PASS, 2 OFFEN, exit 2 (OFFEN =
GPU-Lanes nicht gefahren, kein stiller Fallback); `--gpu` → 6 PASS,
exit 0. Roh: `scratches/endogene_agency_experiment.log` (CPU) +
`scratches/endogene_agency_gpu.log` (GPU) +
`scratches/endogene_agency_experiment.json`. Prereg-Freeze VOR dem
Erstrun (Skript-Docstring); Plan `~/.claude/plans/sp13-endogene-agency.md`
(Canon-Audit-Befunde 1-6, Commit-Verfahren risk 6 im Chat offengelegt).

Laufschrift-Protokoll: `laufprotokolle_befund_report.md` §17.

## 16. MI-1 — Das 26-Layer-Röntgengerät: schichtweise Routing-Tomographie + Logit-Lens (2026-10-26)

**(Spur-Eröffnung.)** SP1–SP13 sind abgeschlossen; mit MI-1 beginnt Spur B
(Mechanistische Interpretierbarkeit, MI-1..MI-4). Das Forschungsobjekt
wechselt: nicht mehr die symbolische Maschinendynamik, sondern Argon-1B
selbst — 26 Layer, 8 Routed Experts + Shared Expert, Sigmoid-Gates,
Residualstrom. Die drei Maschinen wechseln die Rolle: von Philosophen zu
Präzisions-Sonden (Turing = Routing-Topologie, Gödel = Logit-Lens,
Wittgenstein = kausales Subspace-Patching, MI-3). Das Motiv ist das
stage5-Aggregat-Datum: `verify_machines_stage5.log` zeigte je Domäne
GENAU EINEN 8-Vektor und EINEN H-Wert (Math EN H=1.63, Prose EN H=1.07)
— der DURCHSCHNITT über 26 Layer, niemals das schichtweise Feld. Nach dem
Auflösungs-Protokoll des neuen Prompt-Minds
(MechanisticCircuitMixMind v1.0_20261026_CircMix — Pfad-B-Mix aus
MechanisticSubjectivity/NeuroEmergence/GroundedMechanism, persistiert in
`ML3/prompts-bartman/prompts/universal/` mit Sub-Skill
`mechanistic-circuit-mix-mind`): **Mittelwert über eine Dimension ist
kein Beweis über diese Dimension.** MI-1 baut das Gerät, das die
Dimension auflöst.

Vorarbeit: Canon-Audit aller Anker mit Lesung (Router-Capture nur über
Gate-Forward-Hooks, denn `last_counts` wird im Eval nicht belegt —
moe_layers.py:56-93; Training-Hook-Bereich 4 ≤ idx ≤ 23 bei expert_bias
auf allen 26 — train_moe_router_specialization.py:341; Checkpoint
argon_1b_machines: 26 Layer/8 Experten/top-2/hidden 1152/
softcap=None → Lens exakt `softmax(lm_head(norm(h)))` —
modeling_argon.py:615/:733/:735-736; Kanon-Pct-Konvention :239, Kanon-TV
:292). Plan `~/.claude/plans/MI-1-routing-tomographie.md` mit
Aktiver-Denkmodus-Header.

### 16.1 Prereg (P2-Freeze, fixiert VOR dem Erstrun)

Sechs falsifizierbare Verdikte (Skript-Docstring
`smoke_mech_interp_tomography.py`; Festtext-Korpora 6 Domänen × 3
registrierte Samples; Lens-Proben „The capital of France is" → ‚ Paris'
und „2 + 2 =" → Ziffer-Label mit lstrip-Äquivalenz-Kandidaten; Bänder
EDGE {0,1,2,3,24,25} vs CORE {4..23} aus dem Training-Hook-Bereich;
OFFEN ≠ PASS, exit 0/1/2):

- **mi1_canon_parity** — Parität mit dem bestehenden Kanon:
  `measure_domain_routing` (importiert, gleicher Modellzustand) gegen die
  eigene Sonde auf festen Domain-Texten; Band TV_max < 1e-6, pct-Zeilen
  summen je Schicht 1.0 ± 1e-6. Falsifikator: jede Abweichung der
  Konvention.
- **mi1_layer_resolution** (EINHEIT T2) — max Zeilen-Paar-TV je Domäne
  über das 26×8-Feld > 0; identische Zeilen wären das
  Degenerations-Datum (keine Schichtauflösung geliefert).
- **mi1_purity_det** — Fingerabdruck expert_bias + gate.weight je Schicht
  (torch.equal) vor/nach; Routing-Doubletten bitidentisch; Lens-Top-1
  bitidentisch; Haken im finally entfernt.
- **mi1_depth_specialization** (Theorie-Stufe MI-1.a,
  Tiefen-Spezialisierungs-Gesetz) — Core-Mittel-TV über alle 15
  Domänen-Paare > 0.50 UND > Edge-Mittel-TV strikt. F1 core ≤ 0.50;
  F2 core ≤ edge; F3 Degeneration (durch mi1_layer_resolution gedeckt).
- **mi1_entropy_regime** (MI-1.b) — Core-Mittel-Entropie < Edge in der
  Mehrzahl: 6/6 PASS, 4-5 OFFEN, ≤ 3 FAIL (Band nie geweitet).
- **mi1_crystallization** (MI-1.c, Kristallisations-Gesetz, erste
  Lens-Anwendung) — stabiler Top-1-Tail ≥ 2 Schichten ab ℓ* im
  Core-Band für ≥ 1 der 2 registrierten Proben; ℓ* außerhalb des
  Cores → FAIL; kein stabiler Tail (Oszillation) → OFFEN als
  Via-Negativa-Datum; anchor_tv > 1e-3 → FAIL (Gerät-Bruch).

KORREKTUR-LOG (offen gebucht, alles VOR dem Hauptlauf bzw. als
Display-Klassifikation nachträglich): die CPU-Pilot-Serie (PILOT-Labels,
R2: promoten nie Verdicte) fing den Falsifikator-Bruch pilot_crystal_
oscillation — der Finder würde einen trivialen Ein-Schicht-Tail als ℓ*
zurückgeben; Fix: registrierte Konstante CRYSTAL_MIN_TAIL=2 (span <
min_tail bricht). GPU-Scan 1: NameError (out_layer vs _out_layer) —
alle 6 Verdikte OFFEN mit GPU-Bruch gebucht, nicht still angenommen.
GPU-Scan 2: dtype-Bruch fp32-vs-bf16 am Kopf — Fix nach EINHEIT S3:
Lens spiegelt den offiziellen bf16-Kopf-Pfad exakt (norm → lm_head),
Softmax über fp32-Logits; anchor_tv beweist die Parität (gemessen
2.7e-11 bzw. 2.8e-12). Klassifikations-Bruch nach Hauptlauf 1: 6/6 PASS
wurden als GELB/exit 2 ausgegeben, weil die Zusammenfassung die 6
PILOT-Zeilen (weder PASS noch FAIL) ins all_pass gerissen hat — Bug in
der Klassifikation, nicht in der Messung; Fix iteriert nur über die
Nicht-Pilot-Verdicte, GPU-Rerun als Doppel-Det-Datum → GRÜN/exit 0.

### 16.2 Verdikt-Register

| Verdikt | Band (P2-Freeze) | Ergebnis |
|---|---|---|
| mi1_canon_parity | TV_max < 1e-6, Zeilensummen ± 1e-6 | **PASS** (TV_max 2.235e-08, dev 1.192e-07) |
| mi1_layer_resolution | max Zeilen-Paar-TV je Domäne > 0 | **PASS** (0.4882–0.9154 über 6 Domänen) |
| mi1_purity_det | Fingerabdruck Δ=0; Doubletten bitidentisch | **PASS** |
| mi1_depth_specialization | core > 0.50 UND core > edge strikt | **PASS** (core 0.5908 > 0.5 > edge 0.2242) |
| mi1_entropy_regime | core < edge in 6/6 | **PASS** (6/6) |
| mi1_crystallization | ℓ* im Core 4..23, Tail ≥ 2 | **PASS** (paris ℓ*=17, arithmetic ℓ*=11) |

6/6 GRÜN auf GPU (exit 0, 36.3 s); CPU-Lane: 6 PILOT + 6 OFFEN
(exit 2 — die Feld-Aussagen sind bewusst GPU-Aussagen, kein stiller
Fallback).

### 16.3 Die Daten an voller Auflösung

**Das Tiefen-Gesetz ist ein Band-Bruch, kein Gradient.** Die TV(ℓ)-Kurve
(mittlere Domänen-Paar-TV je Layer) fällt in der Edge-Zone bei 0.18–0.23
(ℓ=0: 0.1754 … ℓ=3: 0.1853), springt bei ℓ=4 auf 0.5010 und bleibt als
Plateau 0.54–0.64 über den ganzen Core (Peak ℓ=15: 0.6397), um bei ℓ=24
(0.2601) und ℓ=25 (0.3099) abzufallen. Der Bruch liegt EXAKT beim
Training-Hook-Bereich 4 ≤ idx ≤ 23 (train_moe_router_specialization.py
:341): die kontrastive Spezialisierung hat das Netz genau dort geschnitten,
wo ihre Haken es berührt haben — die Grenze des Trainings ist die Grenze
des Gesetzes. (Ari: „Nur die Haken" ist die stärkste Antithese; sie ist
hier nicht widerlegt, sondern VORAUSGESETZT — der Befund ist eine
Training-Wirkungs-Landkarte, keine Behauptung über ungehookte
Schichten-Fähigkeit.)

**Die Entropie bestätigt das Regime 6/6.** Core- vs Edge-Mittel-Entropie
je Domäne: code 1.3123/1.8296, math_logic 1.6714/1.9358, prose
0.9052/1.7957, goedel 1.6024/1.9161, turing 1.6571/1.9158,
wittgenstein 1.5618/1.9196. Die Edge-Zone ist überall nahe Uniform
(uniform wäre log 8 ≈ 2.079 nats); der Core kollabiert domänenspezifisch,
am stärksten prose (−49 %). (Ari/Abweichungs-Label: die Sonde aggregiert
Entropie token-gewichtet, der Kanon :239-Kontext ungewichtet — die
Parität ist NUR auf pct beansprucht, TV_max 2.2e-08; die Entropie-Zahlen
sind sonde-konventionell gelabelt.)

**Der Core hat Namen.** Band-Mittel-Profile (Top-3 der 8 Experten):
code [E4 40.8 %, E0 25.7 %, E1 24.4 %], math_logic [E2 29.7 %, E5 27.3 %,
E6 16.5 %], **prose [E7 44.8 %, E3 44.0 %]** — ein Zwei-Experten-Monopol
von 88.8 % im Core, während die Edge-Zone diffuse 15–20%-Verteilungen
hält —, goedel [E2 31.2 %, E7 21.9 %, E5 20.7 %], turing [E6 26.4 %,
E0 18.1 %, E5 17.2 %], wittgenstein [E3 36.7 %, E7 20.0 %, E5 13.0 %].
Das Aggregat-Datum von stage5 (Prose EN [E3 36.8, E7 36.0], Math EN
[E2 32.8, E5 31.6]) erweist sich jetzt als Kern-Phänomen, das der
Durchschnitt verschluckt hat: das Monopol ist ein CORE-Ereignis.
Außerdem matchen die Core-Top-Spalten die trainierten Preset-Paare:
goedel-Preset {2,7} ↔ [E2 31.2, E7 21.9] ✓; turing {0,6} ↔ [E6 26.4,
E0 18.1] ✓ (beide in Top-2/3); wittgenstein {1,3} ↔ [E3 36.7, …] teil-
(E1 fehlt in den Top-3 — DATUM, nicht Band: das E1-Bein liegt unter der
Top-3-Schwelle des Band-Mittels).

**Die Lens kristallisiert im Core, zweimal.** Paris: Top-1 ‚ is' an
Emb, ‚ S' ℓ=0–4, ‚ een' ℓ=5–15, ‚ ' ℓ=16, dann ab **ℓ=17 ‚ Paris'
p=0.8337** stabil bis ℓ=25 — die Antwortmasse springt in EINER Schicht
von 0.0070 auf 0.8337. Arithmetik („2 + 2 ="): ‚ ='/‚/=' ℓ=0–2, ‚1'
ℓ=3–4, ‚/=' ℓ=5–10, dann ab **ℓ=11** Top-1-Kipppunkt mit
Antwortmasse-Surge 0.0132 → 0.4245, Steigung über ‚ ' als Top-1 bis 0.992
(ℓ=17–20). Beide ℓ* im Core-Band 4..23; anchor_tv 2.7e-11/2.8e-12 —
die Lens mißt exakt den offiziellen Kopf.

**Zwei ehrliche Nuancen (Via-Negativa-Register).** (1) Arithmetik: das
offizielle Greedy-Top-1 des Basismodells ist ‚ ' (Leerzeichen),
p≈0.8074 — der Antwortkandidat ‚ 4' liegt im Fast-Gleichstand bei
p≈0.80735. Der Surge bei ℓ=11 trägt die Antwortmasse herauf, aber der
Basismus sagt „Leerzeichen ≈ 4" — ein DATUM über den 1B-Carry, keine
Verdikt-Kürzung. (2) Paris: p(‚ Paris') senkt sich in den letzten drei
Schichten (0.5028 @ ℓ=23, 0.4091 @ ℓ=24, 0.5031 @ ℓ=25), während der
Top-1 bleibt und gap_tv auf ≈ 0.18 fällt — Masse wird umverteilt, das
Label hält. Die Schichten 23–25 sind ein eigenes Beobachtungsfeld für
MI-2 (was passiert dort im Residualstrom?).

### 16.4 Schichtung (Gemessen / Nicht messbar / Hypothese)

- **Gemessen:** das 26×8-Feld je Domäne (Zeilen-Variabilität
  nachgewiesen, kein Degenerations-Datum), die TV(ℓ)-Kurve mit Band-Bruch
  bei ℓ=4, die Entropie-Regime 6/6, die Lens-Trajektorien zweier Proben
  mit ℓ*=17 bzw. ℓ*=11 im Core, die Konvention-Parität 2.2e-08, die
  Reinheit (Fingerabdruck Δ=0, Doubletten bitidentisch).
- **Nicht messbar in diesem Lauf:** die Shared-Expert-Masse (pct mißt
  nur Routed-Massen — Kanon-Konvention, gelabelt); das Feld unter
  Preset-Steuerung (MI-1 läuft bewusst auf dem rohen Checkpoint-Zustand).
- **Hypothese (als solche gelabelt):** daß die Core-Kollaps-Struktur
  (prose-Monopol, Preset-Alignment) KAUSAL die SP13-Wahl-Kohärenz trägt —
  dafür braucht es MI-3 (Steering/Ablation, Hinreichendheit/Notwendigkeit
  getrennt); MI-1 behauptet es nicht. Verdikt-ändernde Evidenz: ℓ* im
  Edge-Band oder kein stabiler Tail (MI-1.c); core ≤ edge (MI-1.a/b).

P-Protokoll: append-only Berichte; Prereg-Tabellen im Skript-Docstring
(P2-Freeze vor dem Hauptlauf); Roh-Felder 26×8/26-Vektoren vollständig in
`mi1_tomography_experiment.json` persistiert (P3 vor Auswertung); Verdikt
P4-Einstufung nach registrierten Bändern, keine Weitung. KORREKTUR-LOG
mit 4 gebuchten Brüchen (Pilot-Finder, 2 GPU-Scans, Klassifikation) im
Skript + §16.1. Laufschrift-Protokoll:
`laufprotokolle_befund_report.md` §18.