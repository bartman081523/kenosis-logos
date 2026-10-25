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