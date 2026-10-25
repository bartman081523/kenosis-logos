# machines/lisp — SBCL-Brücke + Lisp-Referenz (Götterdreiheit)

Referenz-Parität: die Python-Maschinen (`machines/`) werden gegen die
original Lisp-Implementierung verifiziert, die in `vendor/` unverändert
liegt. **Kein Quicklisp, kein cl-json, kein Netz** — der Vendor nutzt die
.asd-Deps gar nicht (`package.lisp: (:use #:cl)`).

## Provenienz

- Quelle: `/run/media/julian/ML2/Python/faizal-rebuttal/experiments-new-grouped-13778/lisp-cognito-philosopher/`
- Kopie: `vendor/` — 19 root `.lisp` + `.asd` + `tests/` + README +
  `TEST_RESULTS.md` + `determinism_log.txt` + `stability_signature.txt`,
  byte-identisch; SHA-256 je Datei steht in `parity/lisp_fixtures.json`
  (`vendor_files_sha256`, erzeugt via `machines.lisp_bridge.vendor_digest`).
- Lizenz der Vendor-Dateien: MIT, Autorenzeile Julian 2026-02.

## Architektur

```
Python (machines/lisp_bridge.py)
  └─ subprocess: timeout --kill-after=15 45s sbcl --script machines/lisp/run-bridge.lisp
       └─ run-bridge.lisp      — lädt 10 asd-Komponenten + bridge-protocol
            └─ bridge-protocol.lisp — Mini-JSON + Framing + Op-Dispatch
```

Protokoll: **eine** S-Expression-Auftragzeile auf stdin
(:op "NAME" :key value ...), Antwort = Mini-JSON zwischen `;;; RESULT-BEGIN`
und `;;; RESULT-END`; alles andere (Banner, Philosopher-Text) verwirft
Python. Fehler → `{"error": ...}` im Frame, Exit 0 — kein stiller Pass.

- `sbcl --non-interactive --script` funktioniert **nicht** — `--non-interactive`
  setzt `--quit` und terminiert vor der Skriptausführung (SBCL 2.6.9). Korrekt:
  `sbcl --script <pfad>` allein.
- ASCII-rein (Plan-Risiko 2: Bignum-Exponenten in `expt prime char-code`,
  Locale-Freiheit); Unicode bleibt Python-seitig.
- SBCL schreibt Double-Exponenten als `d+0` — `jnumber-string` subsituitert
  zu `e`, damit `json.loads` parst.

## Paritätsstufen

| Stufe | Anspruch | Ops |
|-------|----------|-----|
| **P1** | bitgenau (pure functions) | `ping`, `goedel`, `gap` (Vektor/Skalar), `sh72`, `ast_eval` |
| **P2** | strukturell (Längen, Zähler, Kategorien — nie Bitgleichheit) | `philosopher` (sxhash-/Text-Kanäle) |

Fixtures: `parity/lisp_fixtures.json` (Goldwerte aus SBCL 2.6.9 +
Vendor-Digests); Python-Spiegel des Lisp-Kanons: `machines/lisp_parity.py`
(nicht der uni_205-Maschinenkanon — Divergenzen sind im Modul-Docstring
zeilenweise belegt: Lisp `inv`/`φ`-Konstanten, unary-Prob 0.6, Gap-Nenner
`|actual|` mit 1e-9-Guard statt uni_241s `‖state‖+1e-6`).

## Ops

- `ping` — SBCL-Version + Primzahl-/Operator-/Konstanteninventar (Selbsttest).
- `goedel` — `:expr STRING|S-EXPR` → prin1-Kanonstring + Gödelzahl (mod 10¹²,
  max 15 Zeichen, 20 Primzahlen; Strings ≥ 3 Zeichen kollabieren fast immer
  auf 0 — `2^code·5^code ≡ 0`).
- `gap` — `:pred/:actual NUMBER|LIST` → `compute-goedel-gap`
  (`goedel-encoder.lisp:87-150`); String-Zweig ist Totcode (Strings sind
  `vectorp` → NUMBER-Fehler) — bewusst gespiegelt, gefixt nicht abgefangen.
- `sh72` — `:words LIST` → Gematria + 64-D-Embeds + Resonanzen (i<j).
- `ast_eval` — `:seed/:count/:xs` → LCG-gepunktete ASTs + `sexpr`/`rendered`/
  `size`/`depth`/`values`; Fehler je Wert = `null`.
- `philosopher` — `:seed/:problem/:n-cycles/:queries/:self-modify-threshold`
  → `gap_history`/`thought_count`/`state`/`self_model` (P2).

## Nutzung

```python
from machines.lisp_bridge import LISPBridge
bridge = LISPBridge()
bridge.ping()
bridge.gap([1.0, 2.0], [1.2, 1.8])      # → gap 0.13074409009212268, "good-self-knowledge"
bridge.ast_eval(seed=42, count=3, xs=[1.0, 1.5, 2.5])
bridge.philosopher(problem="gap", cycles=2, seed=123457)
```

CLI: `python -m machines.lisp_bridge ping`.

Verifikation (Stufe 4): `machines/verify_machines.py` nutzt diese Brücke;
SBCL fehlt → explizit SKIP (Exit 3), kein stiller False-Pass.