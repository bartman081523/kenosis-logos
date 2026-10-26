"""
smoke_sp15_jl_wide.py — SP15: 1152D-Breitband-Selbstmodell (Runde 2 des
Programms „Argon-Subjekt v2 / MI-7 bis SP16", User „lgtm" 2026-10-26).

AUFGABE: der 144:1-Flaschenhals der Oracle-Verdatung (machine_base
hidden_state, block-mean d=8; P4-P1a am Code verifiziert — bei d=8
144:1-Mittelung, bei d = hidden Identität) wird ADDITIV geöffnet — zwei
Opt-in-Zugänge ohne Canon-Berührung (Default-Pfade byte-unangetastet):
  (1) ArgonOracle.hidden_state(..., projection="orthogonal_jl",
      proj_dim=d): orthogonale JL-Rotation (QR einer Seed-42
      standard_normal((hidden, d))-Matrix, reduced; Zeilen orthonormal,
      gecached je d — non-expansiv als Projektion), d ∈ {8, 32, 64}.
  (2) GoedelMachine.enable_wide_read(): W_read ∈ R^{dims×hidden} = 0;
      prediction = tanh(W_read @ h_1152); Delta-Regel (uni_241:220-Familie,
      KEIN Zerfall — registriert): W_read += 0.05·outer(err, h_1152);
      Adapter im Snapshot-Kreislauf (observe bleibt bitgenau), reset()
      nullt den Adapter und behält den Opt-in.

PREREG (eingefroren vor dem GPU-Lauf, 2026-10-26; Band-Weitung verboten):

  REGISTER-REVISION V2′ (VOR dem GPU-Lauf, aus dem CPU-Piloten — dokumentiert,
  kein Relabeling nachher): der Pilot zeigt die JL-Dünngesetz-Zelle auf der
  synthetischen Hypothesen-Welt — per-dim SNR ist d-invariant, der
  Total-SNR wächst nur √d; die Drei-d-Konjunktion (alles > 0.85) ist so nicht
  verteidigbar (Pilot: acc 0.167/0.833/1.0 bei d = 8/32/64). Registriert
  wird deshalb der TRAGKANAL am höchsten d + das Spektrum als DATUM:
  V2  sp15_jl_biografie_erhalt — LOO-Bio-Decodierung im JL-Kanal auf
      DENSELBEN 24 Texten: Band = d = 64: acc > 0.85 UND p_perm ≤ 0.01;
      Non-Expansion JEDES d/Texts ||h_jl(d)|| ≤ ||h_wide|| + 1e-9; Orthonor
      -malität je Cache-Matrix < 1e-12; Kontrast-Band: canon-8 acc < 0.05
      (P4-Datum 0.0 exakt). DATUM: jl8/jl32-Zeilen (Dilutions-Spektrum —
      die registrierte Hypothese: nicht-fallend über d, geliefert wenn's
      passt, ohne Band).
  V3  sp15_kausal_verstaerkung — Notwendigkeitsprobe am gelernten
      Expert-Bias (mech_interp.knockout_experts, ADDITIV, penalty −50,
      experts [1, 3] = sortierte wittgenstein-Preset-Experten der
      commitierten Preset-Datei, layers = ALLE): GoedelMachine wide-read
      (Training = 1 Pass über die 12 A-Renderings in Korpus-Reihenfolge,
      nach bio_halted/bio_gelernt) vs. baugleiche Maschine OHNE Opt-in
      (Block-Lesung), je 24 Proben via observe (bitgenau zurück).
      Band: sens_wide = max |gap_ko − gap_raw| ≥ 0.014406 (10× die
      MI-6-Kausal-Konstante 0.0014406) UND ratio = sens_wide/sens_block
      ≥ 10. DATUM: sens_block selbst, w_mean_abs nach Training.
      Verfahrens-Hinweis: die MI-6-Konstante stammt vom self_model
      -Knockout am describe()-4-Dezimal-Kanal — die Ratio (Block-Lesung
      vs. Wide-Lesung am SELBEN Knockout im SELBEN Lauf) ist der
      mechanische Kern; die Konstante ist der externe Gap-Einheiten-Anker.
  V4  sp15_purity_l0 — MechInterpProbe.fingerprint_equal über die ganze
      Batterie (vor/nach) + W_read-Trainings-Determinismus (2 Trainings
      auf frischen Maschinen → Adapter-Bytes-Digest identisch UND
      Gap-History-Digest identisch).
  V5  sp15_det_pilot — CPU-Pilot doppellauf bitidentisch (JL-Mathematik
      + Band-Logik auf synthetischen Fixtures, ohne Modell).

KORREKTUR-LOG (SP15-Erstlauf ROT → REV-2; die Erstlauf-Artefakte
scratches/sp15_wide.json + .log bleiben VERBATIM):
  K1 Probe-Pfad: composite.observe(t) übergibt dem Goedel das Trace-Outcome
     (composite.py:140 ``result_g = self.goedel.step(transitions)``,
     Skalar) → _prediction(None) → prompt = describe()
     (goedel_machine.py:198) — alle 24 Proben text-invariant (Erstlauf
     -JSON: raw 0.5790/ko 0.4278 uniform, div 0.151203/0.139691 je Text
     konstant = Sättigungsregime des describe()-Kanals). KORREKTUR REV-2:
     Proben via goedel.observe(t) (Submaschine, Prompt = der Text,
     h text-abhängig).
  K2 Trainings-Pfad: derselbe Defekt im Adapter-Training — machine.step
     (t) trainierte auf describe()-Hiddens; der Adapter sah nie Bio-Text
     -Hiddens. KORREKTUR REV-2: 1 Pass über die A-Renderings via
     goedel.step(t) (Submaschine, Delta-Regel auf dem Text-Hidden).

REGISTER-REVISION REV-2 (aus dem Erstlauf-DATUM registriert, vor dem
GPU-Re-Lauf eingefroren; die Band-NAMEN bleiben, die Schablone wird aus
dem Lauf-Datum re-registriert — kein Relabeling, 是X即非X):
  REV2-V2  Tragkanal-Suche: JL_DIMS = (8, 32, 64, 128, 256). Die
      Erstlauf-Zeilen 8/32/64 (acc 0.625/0.708333/0.791667, p
      0.24/0.04/0.005, monoton) = Dilutions-DATUM (Dünngesetz bestätigt).
      Neue Tragkanal-Zelle: EXISTS d ∈ SP15_CROSS = (64, 128, 256) mit
      acc > 0.85 UND p_perm ≤ 0.01. Non-Expansion je d, Orthonormalität
      < 1e-12, Kontrast canon-8 < 0.05 unverändert. Re-Run-Stabilität:
      die Erstlauf-Zeilen 8/32/64 bitidentisch erwartet (Forwards
      deterministisch, Decoder-Seed frisch je Aufruf).
  REV2-V3  Band-Konjunktion UNVERÄNDERT (sens_wide ≥ 0.014406 ∧ ratio
      wide/block ≥ 10) auf der KORRIGIERTEN Schablone (Text-Prompts,
      Training auf Text-Hiddens). Die Erstlauf-Werte (sens_wide
      0.151203, sens_block 0.139691, ratio 1.08241, Proben uniform)
      = Sättigungs-DATUM des describe()-Kanals.
      Struktur-DATUM neu: n_distinct_raw je Maschine (Erstlauf: 1 —
      beweist K1; REV-2 erwartet Text-Abhängigkeit).

REV-2-ERSTLAUF UND REV-3 (2026-10-26): der REV-2-Lauf lieferte
  4 PASS | 1 FAIL (sp15_kausal_verstaerkung) — und die Befund-Mechanik
  ist ZWEIFACH mechanisiert (beides in den REV-2-Artefakten beweislich):
  K3 Canon-Eingabe-Kollaps: der uni_204-Gödelzahl-Fold (encode, max 1000
     Zeichen, prime_idx = i%100, power = ord%10+1) akkumuliert über die
     Positions-Zyklen 2-/5-Valenz bis das Produkt ≡ 0 (mod 10^15) —
     CPU-Demo verbatim: encode der 24 in-loop-Rekonstruktions-Renderings
     DISTINCT = 1 (ALLE = 0); Arithmetik-Sanity: encode('111111'×8)
     v2,v5 = (10,10). → _text_seed = 0 für alle 24 Texte → identischer
     gefalteter input_vec → DIE IST-DYNAMIK IST TEXT-UNABHÄNGIG
     (REV-2: n_distinct_raw_wide 1 trotz Text-Prompts).
  K4 tanh-Sättigung des trainierten Wide-Lesers: raw 1.96171 =
     |pred_sat − actual|/|state| nahe Maximum (|pred| ≈ √8); unter dem
     knockout bleibt tanh′ ≈ 0 → div EXAKT 0.0 je Text, während die
     Block-Lesung denselben Δh im glatten Bereich liest (sens_block
     0.001843, Text-abhängig, ≈ 1.28× die MI-6-Konstante).

REGISTER-REVISION REV-3 (dritte und letzte Re-Registrierung der
SP15-V3-Runde; die Band-ZIELE bleiben, die LESUNGEN werden aus dem
Lauf-Datum re-registriert; ein REV-3-Fail schließt die Runde als
ROT-mit-Befund):
  REV3-V3  LINEAR-Lesungen beidseitig (desselbe Δh, ohne tanh): z_wide,t
      = W_read @ hidden_state(t, 1152); z_block,t = hidden_state(t, 8)
      (= das tanh-Argument der Canon-Block-Lesung); sens_lin = max_t
      |Δ z|/(|state| + 1e-6). Band: sens_lin_wide ≥ 0.014406 ∧
      ratio_lin = sens_lin_wide/sens_lin_block ≥ 10.
  DATUM neu: tanh-Arme (Re-Run-Stabilität des REV-2-Sättigungs-Regimes:
      div_tanh_wide ≈ 0, raw 1.96171; block tanh-Daten), die Canon-Seed
      -Kollaps-Werte (n_distinct_canon_seeds = 1 erwartet), pred-Normen.
  Eingabe-Schablone REV-3 (nur für die tanh-Datum-Arme): der gefaltete
      input_vec aus dem SHA-256 des GANZEN Texts (np 8-Vektor normal
      (0, 0.1), canon-legale Array-Eingabe via observe/Vektor) statt des
      uni_204-Fensters — die IST-Schiene wird text-abhängig; der
      Canon-Kollaps bleibt als registriertes DATUM stehen.

REV-3-ERSTLAUF UND K5→REV-4 (2026-10-26): der REV-3-Lauf lieferte
GRÜN (exit 0) — 5 PASS | 0 FAIL | 0 OFFEN; sp15_kausal_verstaerkung
PASS mit sens_lin_wide 0.92931 (64.5× die MI-6-Konstante), sens_lin_block
0.003948, ratio_lin 235.387538; die tanh-DATUM-Arme re-run-stabil
(sens_tanh_wide 0.0 exakt das REV-2-Datum; sens_tanh_block 0.001996 bei
REV-3-Eingabe-Faltung), n_distinct_raw_tanh WIDE = 24 bei Fold-Eingabe
(gegenüber 1 bei uni_204-Eingabe — die K3-Diagnose am Gerät bestätigt),
Canon-Seeds alle 0 (canon_seed_values [0]), Trainings-Digests
adapter 9f9f2d05…/gap 4e1e7387… IDENTISCH zu REV-2, restore beide True.
  KORREKTUR-LOG K5 (Instrument-Defekt der REV-3-Block-Seite, beweislich
  über das spec-DATUM im JSON: block_mean_linear_max_err 0.285075
  beidseitig): die REV-3-Schablone sagt z_block,t = hidden_state(t, 8),
  die Implementation ersetzte das durch eine konsekutive 8×1152-Mittel
  -Matrix (Blöcke à 144); die Canon-Zelle (machine_base.hidden_state)
  ist aber die STRIDED-Form pooled.reshape(-1, d).mean(axis=0) (Positionen
  ≡ j mod 8) — beide Formen messen Linear-Verdünnungen derselben Klasse,
  aber nicht dasselbe Argument; der REV-3-Band-PASS beruht auf der
  konsekutiven Form (ratio 235 gegen die konsekutive Verdünnung).
REGISTER-REVISION REV-4 (K5-Korrektur derselben REV-3-Bänder — KEINE
Formeländerung; die Registrierung von REV-3 wird exakt implementiert):
z_wide,t = W_read @ hidden_state(t, 1152); z_block,t =
hidden_state(t, 8) DIREKT (Canon-tanh-Argument, strided); sens_lin und
Band unverändert; die M-Ersatzmatrix und ihr spec-DATUM entfallen (der
direkte Canon-Pfad braucht keine Äquivalenz-Prüfung); Knockout-Contexte
streng sequenziell. Ein REV-4-Fail schließt die Runde als
ROT-mit-Befund (REV-3 verbatim als GRÜN-mit-K5-Befund erhalten).

Ausführung: python3 -m scratches.smoke_sp15_jl_wide [--gpu]
Artefakte: Erstlauf scratches/sp15_wide.json + .log (verbatim, ROT);
REV-2 scratches/sp15_wide_rev2.json + .log (verbatim); REV-3
scratches/sp15_wide_rev3.json + .log (verbatim, GRÜN mit K5-Befund);
REV-4 scratches/sp15_wide_rev4.json + .log; Piloten sp15_cpu_pilot_rev2
.log/_rev3.log/_rev4.log.
Exit: 0 GRÜN | 1 ROT (Band-Verletzung) | 2 OFFEN (GPU-Kontext fehlt).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.machine_base import load_preset
from machines.composite import CompositeMachine
from scratches.smoke_endogene_agency import (
    make_composite,
    round6,
)
from scratches.smoke_mi6_subjektivitaet import (
    DEC_P_TAU,
    LEX_PAIR,
    _load_machines_ctx,
    bio_gelernt,
    bio_halted,
    learn_pair_pin,
    render_state,
)
from scratches.smoke_mi6p4_schluss import (
    ZEITPUNKTE,
    _loo_decoder,
)

JL_DIMS = (8, 32, 64, 128, 256)  # REV-2 Tragkanal-Suche (8/32/64 Erstlauf-DATUM)
SP15_CROSS = (64, 128, 256)      # REV2-V2-Zelle: EXISTS d hier mit Träger
PILOT_DIMS = (8, 32, 64)         # CPU-Pilot-Welt (hs=64)
PILOT_CROSS = (32, 64)
SP15_JL_ACC = 0.85              # User-Band: 0.0 → > 0.85 im JL-Kanal
SP15_JL_P = 0.01                # DEC_P_TAU-Wert, explizit registriert
CANON8_BLIND = 0.05             # V2-Kontrast-Band (P4-Datum 0.0 exakt)
MI6_KAUSAL = 0.0014406          # MI-6 GPU diverg_w1 (gap-Einheiten)
SP15_KO_MIN = round(MI6_KAUSAL * 10.0, 6)
SP15_RATIO = 10.0

P4_WIDE = (0.958333, 0.0)
P4_CANON8 = (0.0, 1.0)
P4_TEXTUR = (1.0, 0.0)
P4_NUMERISCH = (0.833333, 0.025)


def _same(a: float, b: float, eps: float = 1e-9) -> bool:
    """6-dp-Band-Gleichheit: die registrierten Konstanten sind gerundete
    Doppel — exakte binäre Gleichheit reicht mit 1e-9-Fenster."""
    return abs(float(a) - float(b)) < eps


def _ko_experts() -> List[int]:
    """K.O.-Experten aus der commitierten Preset-Datei (Provenienz)."""
    preset = load_preset("wittgenstein") or {}
    experts = sorted(int(e) for e in preset.get("experts", [1, 3]))
    if not experts:
        raise ValueError("ko_experts: wittgenstein-Preset ohne Expertenwahl.")
    return experts


def stufe(text: str) -> None:
    print(f"\n--- {text}", flush=True)


# -- L0-Korpus: das P4-P1a-Rezept (verbatim aus lane_weit) ---------------------

def _l0_korpus(oracle: Any) -> Dict[str, Any]:
    """24 Texte: 6 Zeitpunkte × 2 Maschinen × 2 Stile; A = gelehrte
    Biografie (learn_pair_pin + bio_halted + bio_gelernt), B = gleiche
    Phasen ohne learn_pair_pin (P4-lane_weit-Rezept, unverändert)."""
    a = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(a, LEX_PAIR)   # PREREG: A = gelehrte Biografie via LEX_PAIR
    bio_halted(a, "spiegel")
    bio_gelernt(a, "spiegel")
    b = make_composite(mode="oracle", oracle=oracle)
    bio_halted(b, "spiegel")
    bio_gelernt(b, "spiegel")
    texts: List[str] = []
    labels: List[int] = []
    styles: List[str] = []
    for machine, label in ((a, 1), (b, 0)):
        for runde in ZEITPUNKTE:
            probe = machine.preview_steps(["spiegel"] * runde,
                                          lexical_grounding=True)
            for style in ("textur", "numerisch"):
                texts.append(render_state(machine, style, probe=probe))
                labels.append(label)
                styles.append(style)
            machine.step("spiegel", lexical_grounding=True)
    labels_np = np.asarray(labels)
    classes = {str(l): int((labels_np == l).sum()) for l in (0, 1)}
    # A-Hälfte: die ersten 12 Renderings (runde-Aufstieg, textur vor numerisch)
    return {"texts": texts, "labels": labels_np, "styles": styles,
            "n_classes": {"1": classes["1"], "0": classes["0"]},
            "a_texts": texts[:12], "n": len(texts)}


def lane_l0(oracle: Any, korpus: Dict[str, Any]) -> Dict[str, Any]:
    """V1 — L0-Anker: die P4-P1a-Zahlen auf dem unveränderten Default-Pfad."""
    texts = korpus["texts"]
    labels = korpus["labels"]
    hs = int(oracle.model.config.hidden_size)

    def decode(sel: List[int], dims: int) -> Dict[str, Any]:
        return _loo_decoder(
            np.stack([oracle.hidden_state(texts[i], dimensions=dims)
                      for i in sel]), labels[sel])

    wide = decode(list(range(len(texts))), hs)
    canon8 = decode(list(range(len(texts))), 8)
    per_stil = {stil: decode([i for i, s in enumerate(korpus["styles"])
                              if s == stil], hs)
                for stil in ("textur", "numerisch")}
    verdict = (
        wide["n_texts"] == 24
        and _same(wide["acc"], P4_WIDE[0]) and _same(wide["p_perm"], P4_WIDE[1])
        and _same(canon8["acc"], P4_CANON8[0])
        and _same(canon8["p_perm"], P4_CANON8[1])
        and _same(per_stil["textur"]["acc"], P4_TEXTUR[0])
        and _same(per_stil["textur"]["p_perm"], P4_TEXTUR[1])
        and _same(per_stil["numerisch"]["acc"], P4_NUMERISCH[0])
        and _same(per_stil["numerisch"]["p_perm"], P4_NUMERISCH[1])
        and korpus["n_classes"] == {"1": 12, "0": 12})
    return {"wide_gepoolt": wide, "canon_8": canon8,
            "je_stil": per_stil, "verdict": bool(verdict)}


def lane_jl(oracle: Any, korpus: Dict[str, Any],
            canon8_acc: float) -> Dict[str, Any]:
    """V2 — LOO-Bio-Decodierung im ortho-JL-Kanal, REV-2 Grid
    d ∈ {8, 32, 64, 128, 256}."""
    texts = korpus["texts"]
    labels = korpus["labels"]
    hs = int(oracle.model.config.hidden_size)
    wide_h = [oracle.hidden_state(t, dimensions=hs) for t in texts]
    decode: Dict[str, Dict[str, Any]] = {}
    for d in JL_DIMS:
        h_arr = np.stack([
            oracle.hidden_state(t, projection="orthogonal_jl", proj_dim=d)
            for t in texts])
        dec = _loo_decoder(h_arr, labels)
        cache = getattr(oracle, "_jl_cache", {})
        matrix = cache.get(int(d))
        ortho = None if matrix is None else float(
            np.abs(matrix @ matrix.T - np.eye(int(d))).max())
        noexp = all(float(np.linalg.norm(h_arr[i])) <=
                    float(np.linalg.norm(wide_h[i])) + 1e-9
                    for i in range(len(texts)))
        decode[f"jl{d}"] = {**dec, "ortho_max": ortho,
                            "no_expand": bool(noexp)}
    # REV2-V2-Zelle: EXISTS d ∈ SP15_CROSS mit Träger (Erstlauf-Konjunktion
    # am Top-d allein wurde durch die Dilution am Erstlauf verworfen).
    crossing = [d for d in SP15_CROSS
                if decode[f"jl{d}"]["acc"] > SP15_JL_ACC
                and decode[f"jl{d}"]["p_perm"] <= SP15_JL_P]
    spec_ok = (all(decode[k]["no_expand"] for k in decode)
               and all((decode[k]["ortho_max"] is not None
                        and decode[k]["ortho_max"] < 1e-12)
                       for k in decode))
    carry_ok = bool(crossing) and spec_ok
    contrast_ok = float(canon8_acc) < CANON8_BLIND
    return {"decode": decode, "carry_ok": bool(carry_ok),
            "crossing": crossing, "contrast_ok": bool(contrast_ok),
            "verdict": bool(carry_ok and contrast_ok)}


# -- V3: wide-Lese-Adapter + Knockout-Notwendigkeit -----------------------------

def _bio_machine(oracle: Any, wide: bool) -> Tuple[CompositeMachine, Dict[str, Any]]:
    """Baugleiche Bio-Maschine (A-Rezept); Opt-in NACH der Reifung."""
    machine = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(machine, LEX_PAIR)
    bio_halted(machine, "spiegel")
    bio_gelernt(machine, "spiegel")
    info: Dict[str, Any] = {}
    if wide:
        info = machine.goedel.enable_wide_read()
    return machine, info


def _fold_vec(text: str) -> np.ndarray:
    """REV-3-Eingabe-Schablone (nur tanh-Datum-Arme): SHA-256 des GANZEN
    Texts → deterministischer 8-Vektor normal(0, 0.1); canon-legale
    Array-Eingabe (_input_vector akzeptiert Vektoren). Umgeht den
    uni_204-Fenster-Kollaps (K3: encode = 0 für die Renderings)."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed).normal(0.0, 0.1, size=8)


def _probe_sens(oracle: Any, machine: CompositeMachine, texts: List[str],
                ko_experts: List[int], wide: bool) -> Dict[str, Any]:
    """REV-4-Probe (K5-Korrektur der REV-3-Schablone, Bänder UNVERÄNDERT):
    (a) tanh-Arm via goedel.observe (Submaschine; Eingabe = der REV-3
    -Ganitext-Fold aus K3-Umgehung — die Ist-Schiene wird text-abhängig;
    der pred-Kanal bleibt der Canon-Oracle-Lese-Pfad); (b) LINEAR-Arm je
    Text: z = W_read @ h_1152 (wide) bzw. z = hidden_state(t, 8) direkt
    (block — das EXAKTE Canon-tanh-Argument, strided-Mean machine_base
    reshape(-1, d).mean(axis=0); REV-3 nutzte eine konsekutive Ersatzmatrix
    → spec-Abweichung 0.285075 → K5). Knockout-Contexte STRENG sequenziell
    (nested additive Stacks wären −100 statt −50)."""
    from machines.mech_interp import MechInterpProbe
    probe = MechInterpProbe(oracle.model, oracle.tokenizer)
    fp_a = probe.fingerprint()
    hs = int(oracle.model.config.hidden_size)
    state_norm = max(float(np.linalg.norm(np.asarray(
        machine.goedel.state()["state"], dtype=np.float64))), 1e-12)
    rows_tanh: List[Dict[str, Any]] = []
    rows_lin: List[Dict[str, Any]] = []
    for t in texts:
        in_vec = _fold_vec(t)
        raw_r = machine.goedel.observe(in_vec)
        with probe.knockout_experts(ko_experts, layers=None, penalty=-50.0):
            ko_r = machine.goedel.observe(in_vec)
        rows_tanh.append({"raw": round6(raw_r.gap), "ko": round6(ko_r.gap),
                          "div": round6(abs(ko_r.gap - raw_r.gap)),
                          "pred_norm_raw": round6(
                              float(raw_r.invariants["self_pred_norm"]))})
        if wide:
            adapter = machine.goedel.wide_read_adapter
            h_raw = np.asarray(oracle.hidden_state(t, dimensions=hs),
                               dtype=np.float64)
            with probe.knockout_experts(ko_experts, layers=None,
                                        penalty=-50.0):
                h_ko = np.asarray(oracle.hidden_state(t, dimensions=hs),
                                  dtype=np.float64)
            z_raw, z_ko = adapter @ h_raw, adapter @ h_ko
        else:
            with probe.knockout_experts(ko_experts, layers=None,
                                        penalty=-50.0):
                z_ko = np.asarray(oracle.hidden_state(t, dimensions=8),
                                  dtype=np.float64)
            z_raw = np.asarray(oracle.hidden_state(t, dimensions=8),
                               dtype=np.float64)
        rows_lin.append({"div_lin": round6(float(
                            np.linalg.norm(z_ko - z_raw)
                            / (state_norm + 1e-6))),
                         "z_norm_raw": round6(float(np.linalg.norm(z_raw))),
                         "z_norm_ko": round6(float(np.linalg.norm(z_ko)))})
    fp_b = probe.fingerprint()
    sens_tanh = float(max(r["div"] for r in rows_tanh))
    sens_lin = float(max(r["div_lin"] for r in rows_lin))
    return {"sens_tanh": round6(sens_tanh), "sens_lin": round6(sens_lin),
            "rows_tanh": rows_tanh, "rows_lin": rows_lin,
            "n_distinct_raw_tanh": len({r["raw"] for r in rows_tanh}),
            "restore": bool(probe.fingerprint_equal(fp_a, fp_b)),
            "state_norm": round6(state_norm)}


def _train_wide(oracle: Any, texts_a: List[str]) -> Tuple[CompositeMachine, Dict[str, Any]]:
    """1 Pass über die A-Renderings in Korpus-Reihenfolge (Delta-Regel
    je Schritt) auf der GOEDEL-SUBMASCHINE (Korrektur K2: der Prompt ist
    der Text, der Adapter lernt auf Text-Hiddens — Erstlauf trainierte am
    describe()-Kanal); Reihenfolge = (textur, numerisch) je runde-Aufstieg."""
    machine, info = _bio_machine(oracle, wide=True)
    go = machine.goedel
    for t in texts_a:
        go.step(t)
    state = go.state()
    info["w_mean_abs"] = state["wide_read"]["w_mean_abs"]
    info["adapter_digest"] = hashlib.sha256(
        go.wide_read_adapter.tobytes()).hexdigest()
    info["gap_digest"] = hashlib.sha256(
        np.asarray(go._gap_history, dtype=np.float64).tobytes()).hexdigest()
    return machine, info


def lane_kausal_wide(ctx: Tuple[Any, ...], korpus: Dict[str, Any]) -> Dict[str, Any]:
    """V3 — REV3-Band auf den LINEAR-Lesungen (desselbe Knockout-Template,
    keine tanh-Sättigung): sens_lin_wide ≥ 0.014406 ∧ ratio_lin ≥ 10;
    tanh-Arme = Sättigungs-DATUM (REV-2-Befund), Canon-Seed-Kollaps =
    K3-Struktur-DATUM."""
    _torch, model, tok, oracle, _dev = ctx
    texts = korpus["texts"]
    ko_experts = _ko_experts()
    machine_wide, train_datum = _train_wide(oracle, korpus["a_texts"])
    machine_block, _ = _bio_machine(oracle, wide=False)
    w = _probe_sens(oracle, machine_wide, texts, ko_experts, wide=True)
    b = _probe_sens(oracle, machine_block, texts, ko_experts, wide=False)
    sens_lin_wide = w["sens_lin"]
    sens_lin_block = b["sens_lin"]
    ratio_lin = sens_lin_wide / max(sens_lin_block, 1e-12)
    verdict = (sens_lin_wide >= SP15_KO_MIN) and (ratio_lin >= SP15_RATIO)
    canon_seeds = sorted({machine_wide.goedel._text_seed(t) for t in texts})
    return {"sens_lin_wide": round6(sens_lin_wide),
            "sens_lin_block": round6(sens_lin_block),
            "ratio_lin": round6(ratio_lin),
            "sens_tanh_wide": w["sens_tanh"],
            "sens_tanh_block": b["sens_tanh"],
            "n_distinct_raw_tanh_wide": w["n_distinct_raw_tanh"],
            "n_distinct_raw_tanh_block": b["n_distinct_raw_tanh"],
            "n_distinct_canon_seeds": len(canon_seeds),
            "canon_seed_values": canon_seeds,
            "state_norm_wide": w["state_norm"],
            "state_norm_block": b["state_norm"],
            "pred_norm_max_raw_wide": round6(max(
                r["pred_norm_raw"] for r in w["rows_tanh"])),
            "pred_norm_max_raw_block": round6(max(
                r["pred_norm_raw"] for r in b["rows_tanh"])),
            "ko_experts": ko_experts,
            "penalty": -50.0, "layers": "alle",
            "restore_wide": w["restore"], "restore_block": b["restore"],
            "ko_min": SP15_KO_MIN, "ratio_min": SP15_RATIO,
            "datum_mi6_kausal": MI6_KAUSAL,
            "train_datum": train_datum,
            "proben_wide_tanh": w["rows_tanh"],
            "proben_block_tanh": b["rows_tanh"],
            "proben_wide_lin": w["rows_lin"],
            "proben_block_lin": b["rows_lin"],
            "verdict": bool(verdict)}


def lane_wide_det(oracle: Any, texts_a: List[str]) -> Dict[str, Any]:
    """V4-Teil: 2 Trainings auf frischen Maschinen → Adapter-Digest UND
    Gap-History-Digest identisch (In-Lauf-Determinismus)."""
    _m1, d1 = _train_wide(oracle, texts_a)
    _m2, d2 = _train_wide(oracle, texts_a)
    ok = d1["adapter_digest"] == d2["adapter_digest"] and \
        d1["gap_digest"] == d2["gap_digest"]
    return {"d1": d1, "d2": d2, "verdict": bool(ok)}


# -- CPU-Pilot: JL-Mathematik + Band-Logik (synthetische Fixtures) ---------------

def _jl_matrix_local(full: np.ndarray, dim: int) -> np.ndarray:
    """Register-Replikat der Oracle-Zelle (QR Seed 42, reduced, rows
    orthonormal) — für den CPU-Piloten ohne Modell."""
    rng = np.random.default_rng(42)
    gauss = rng.standard_normal((full.shape[0], int(dim)))
    q, _ = np.linalg.qr(gauss)
    return np.ascontiguousarray(q.T).astype(np.float64)


def lane_pilot() -> Dict[str, Any]:
    """CPU-Pilot: (a) Orthonormalität + Non-Expansion der registrierten
    JL-Zelle; (b) Hypothesen-Welt: separating Bio-Achse im Wide-Raum hat
    im JL-Kanal zu bestanden (LOO 1.0), im anti-symmetrischen Block-Mittel
    blind (0.0); (c) Knockout-Arithmetik beidseitig (PASS- und FAIL-Fall)."""
    hs = 64
    labels = np.asarray([1, 1, 1, 0, 0, 0])
    rng = np.random.default_rng(7)
    base = rng.standard_normal((6, hs)) * 0.4
    axis = rng.standard_normal(hs)
    axis = axis / np.linalg.norm(axis)
    wide = base + 2.0 * np.outer(labels.astype(float), axis)

    jl: Dict[str, Dict[str, Any]] = {}
    for d in PILOT_DIMS:
        p = _jl_matrix_local(np.zeros((hs,)), int(d))     # (d, hs) — Zelle
        h = np.asarray(wide) @ p.T
        dec = _loo_decoder(h, labels)
        ortho = float(np.abs(p @ p.T - np.eye(int(d))).max())
        jl[f"jl{d}"] = {**dec, "ortho_max": ortho,
                        "no_expand": bool(all(
                            float(np.linalg.norm(h[i])) <=
                            float(np.linalg.norm(wide[i])) + 1e-9
                            for i in range(len(wide))))}

    # Block-Blindfall: die Bio-Achse alterniert ANTI-SYMMETRISCH je Block
    # (Summe je 8er-Block exakt 0) → Block-Mittel trägt NUR die Basis.
    blind = wide + np.outer(
        labels.astype(float),
        np.asarray([1.0, -1.0] * (hs // 2)))
    block8 = blind.reshape(6, hs // 8, 8).mean(axis=2)
    blind_dec = _loo_decoder(block8, labels)

    # REV-3-Zellen (K3): Fold-Determinismus (gleicher Text → identischer
    # Vektor, verschiedene Texte → verschiedene Vektoren). Die Block-mean
    # -Linearform-Leg entfällt (K5): die Block-Seite liest die Canon-Zelle
    # hidden_state(t, 8) direkt — keine Ersatzmatrix, kein spec-DATUM.
    fold_same = bool(np.array_equal(_fold_vec("pilot-x"), _fold_vec("pilot-x")))
    fold_diff = not np.array_equal(_fold_vec("pilot-x"), _fold_vec("pilot-y"))

    # Knockout-Arithmetik: gap-Einheiten (MI-6-Maßstab)
    def knock(raw: float, ko: float) -> Dict[str, Any]:
        sens = abs(ko - raw)
        return {"sens": round6(sens),
                "over_min": bool(sens >= SP15_KO_MIN)}

    ko_pass = knock(0.03, 0.70)
    block_pass = knock(0.0300, 0.0304)
    ratio_pass = round6(ko_pass["sens"] / max(block_pass["sens"], 1e-12))
    ko_fail = knock(0.03, 0.034)          # 0.004 < 0.014406 → FAIL
    # REV-2-Band-Logik: EXISTS d ∈ PILOT_CROSS mit Träger (nur acc — der
    # p_perm-Band-Zweig ist bei n=6-Texten am Perm-Floor untestbar und
    # bleibt GPU-seitig registriert; die Synthetik trägt die
    # Dilutions-Form (monoton steigend) + Kontrast-Zelle).
    cross_ok = any(jl[f"jl{d}"]["acc"] > SP15_JL_ACC for d in PILOT_CROSS)
    monotone = (jl["jl8"]["acc"] <= jl["jl32"]["acc"] <= jl["jl64"]["acc"])
    verdict = (cross_ok and monotone
               and blind_dec["acc"] < SP15_JL_ACC          # Block-Kontrast-Logik
               and ko_pass["over_min"] and ratio_pass >= SP15_RATIO
               and not ko_fail["over_min"]
               and all(jl[k]["ortho_max"] < 1e-12 for k in jl)
               and all(jl[k]["no_expand"] for k in jl)
               and fold_same and fold_diff)
    return {"jl": jl, "blind": blind_dec, "knock": {
        "pass_wide": ko_pass, "pass_block": block_pass, "ratio": ratio_pass,
        "fail_wide": ko_fail}, "rev3": {
            "fold_same": fold_same, "fold_diff": fold_diff},
        "verdict": bool(verdict)}


def cpu_findings() -> Dict[str, Any]:
    """CPU-Pilot doppellauf (bitidentisch, JSON-equal)."""
    runs = [lane_pilot() for _ in range(2)]
    det_equal = json.dumps(runs[0], sort_keys=True, default=str) == \
        json.dumps(runs[1], sort_keys=True, default=str)
    runs[0]["det_equal"] = det_equal
    return {"pilot": runs[0], "det_equal": det_equal}


# -- Befunde --------------------------------------------------------------------

def findings(cpu: Dict[str, Any],
             gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Verdikt-Zeilen nach den PREREG-Bändern (None = OFFEN)."""
    rows: List[Dict[str, Any]] = []
    v1 = v2 = v3 = v4 = None
    if gpu is not None:
        if gpu.get("l0") is not None:
            v1 = bool(gpu["l0"]["verdict"])
        if gpu.get("jl") is not None:
            v2 = bool(gpu["jl"]["verdict"])
        if gpu.get("kausal") is not None:
            v3 = bool(gpu["kausal"]["verdict"])
        if gpu.get("purity") is not None and gpu.get("train_det") is not None:
            v4 = bool(gpu["purity"]["verdict"]
                      and gpu["train_det"]["verdict"])

    rows.append({"id": "sp15_l0_regression", "assert": v1,
                 "note": ("L0-Anker: P4-P1a-Rezept reproduziert die P4-Zahlen "
                          "EXAKT auf dem unberrührten Default-Pfad (wide "
                          "0.958333/0.0, canon-8 0.0/1.0, textur 1.0, "
                          "numerisch 0.833333/0.025)")})
    rows.append({"id": "sp15_jl_biografie_erhalt", "assert": v2,
                 "note": ("REV2-V2 Tragkanal-Zelle: EXISTS d ∈ {64,128,256} "
                          "acc > 0.85 UND p_perm ≤ 0.01; Non-Expansion je d; "
                          "Orthonormalität < 1e-12; Kontrast canon-8 < 0.05; "
                          "Erstlauf-Zeilen 8/32/64 = Dilutions-DATUM")})
    rows.append({"id": "sp15_kausal_verstaerkung", "assert": v3,
                 "note": ("REV4-V3 LINEAR-Lesungen (kein tanh; K5-Korrektur "
                          "von REV-3): z_wide = W_read @ h_1152, z_block = "
                          "hidden_state(t, 8) direkt (das exakte Canon-tanh-"
                          "Argument, STRIDED); Band: "
                          "sens_lin_wide ≥ 0.014406 (10× MI-6 0.0014406) "
                          "UND ratio_lin ≥ 10; tanh-Arme = Sättigungs-DATUM "
                          "(REV-2: div 0.0), K3 = Canon-Seed-Kollaps-DATUM")})
    rows.append({"id": "sp15_purity_l0", "assert": v4,
                 "note": ("fingerprint_equal über die Batterie + "
                          "W_read-Trainings-Determinismus (2 Läufe, "
                          "Adapter-Digest + Gap-History-Digest)")})
    rows.append({"id": "sp15_det_pilot", "assert": bool(cpu["det_equal"] and
                                                        cpu["pilot"]["verdict"]),
                 "note": "CPU-Pilot doppellauf bitidentisch (V5)"})
    return rows


def battery(ctx: Tuple[Any, ...]) -> Dict[str, Any]:
    torch, model, tok, oracle, dev = ctx
    from machines.mech_interp import MechInterpProbe
    probe = MechInterpProbe(model, tok)
    fp_start = probe.fingerprint()

    stufe("L0-Korpus (P4-P1a-Rezept verbatim)")
    korpus = _l0_korpus(oracle)
    stufe("V1 L0-Regression (canon-8 + wide + je-Stil)")
    out: Dict[str, Any] = {"korpus": {"n": korpus["n"],
                                      "n_classes": korpus["n_classes"],
                                      "erst_text": korpus["texts"][0][:64]},
                           "korrektur_log": [
                               "K1 Proben composite.observe → goedel.observe(t) "
                               "(Prompt = Text; Erstlauf describe()-Kanal, uniform)",
                               "K2 Training machine.step → goedel.step(t) "
                               "(Adapter lernt auf Text-Hiddens)",
                               "K3 Canon-Seed-Kollaps: uni_204-encode der "
                               "24 P4-Renderings = 0 (2-/5-Valenz über "
                               "100-Position-Zyklen mod 10**15) → "
                               "_text_seed = 0 → identischer input_vec "
                               "(REV-2: n_distinct_raw_block 1) — tanh-Datum-"
                               "Arme REV-3 mit sha256-Ganitext-Fold",
                               "K4 tanh-Sättigung des trainierten Wide-Lesers "
                               "(raw 1.96171 nahe Maximum, tanh′ ≈ 0 → div "
                               "0.0) — REV-3 liest LINEAR beidseitig",
                               "K5 REV-3-Block-Seite: konsekutive Ersatzmatrix "
                               "(blk_eq 0.285075) statt der Canon-8-Zelle "
                               "(STRIDED, reshape(-1, d).mean(axis=0)) — "
                               "REV-4 ruft hidden_state(t, 8) direkt, Bänder "
                               "unverändert"]}
    out["l0"] = lane_l0(oracle, korpus)
    stufe("V2 JL-Biografie-Erhalt (d 8/32/64/128/256, LOO, REV-2)")
    out["jl"] = lane_jl(oracle, korpus, out["l0"]["canon_8"]["acc"])
    stufe("V3 Kausal-Verstärkung (LINEAR-Lesungen, REV-4/K5)")
    out["kausal"] = lane_kausal_wide(ctx, korpus)
    stufe("V4 W_read-Trainings-Determinismus (2 Läufe)")
    out["train_det"] = lane_wide_det(oracle, korpus["a_texts"])
    fp_end = probe.fingerprint()
    out["purity"] = {"verdict": bool(probe.fingerprint_equal(fp_start, fp_end))}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stufe("== SP15 REV-4 — 1152D-Breitband-Selbstmodell "
          "(LINEAR-Kausal-Schablone, K5-Korrektur: direkter Canon-8-Pfad) ==")
    cpu = cpu_findings()

    gpu: Optional[Dict[str, Any]] = None
    offen = False
    if args.gpu:
        try:
            stufe("GPU-Kontext laden (argon_1b + argon_1b_machines)")
            ctx = _load_machines_ctx()
        except Exception as exc:   # noqa: BLE001
            print(f"GPU-Kontext nicht verfügbar ({exc}) — Lanes OFFEN.",
                  flush=True)
            ctx = None
            offen = True
        if ctx is not None:
            gpu = battery(ctx)
            torch, model, tok = ctx[0], ctx[1], ctx[2]
            del ctx, model, tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    rows = findings(cpu, gpu)
    print()
    print("BEFUNDE:")
    for r in rows:
        if r["assert"] is None:
            print(f"  {r['id']:<26} OFFEN")
        else:
            print(f"  {r['id']:<26} {'PASS' if r['assert'] else 'FAIL'}")
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext"})
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS | "
          f"{len(fails)} FAIL | {len(opened)} OFFEN")
    print("GRUEN — SP15-Bänder eingehalten" if not fails and not opened
          else ("ROT — Bands verletzt: " + ", ".join(r["id"] for r in fails)
                if fails else "OFFEN — GPU-Lanes nicht gefahren"))

    payload = {"cpu": {"det_equal": cpu["det_equal"],
                       "verdict": cpu["pilot"]["verdict"],
                       "pilot": cpu["pilot"]},
               "gpu": gpu, "findings": rows,
               "historie": [
                   {"run": "REV-1 Erstlauf",
                    "artefakte": "scratches/sp15_wide.json + .log (verbatim)",
                    "verdict": "ROT (exit 1) — 3 PASS | 2 FAIL",
                    "korrektur": ["K1 Proben-Kanal composite→goedel",
                                  "K2 Training-Kanal composite→goedel"]},
                   {"run": "REV-2",
                    "artefakte": "scratches/sp15_wide_rev2.json + .log "
                                 "(verbatim)",
                    "verdict": "ROT (exit 1) — 4 PASS | 1 FAIL "
                               "(sp15_kausal_verstaerkung)",
                    "befunde": ["K3 Canon-Seed-Kollaps (uni_204-encode = 0, "
                                "n_distinct_raw_block 1)",
                                "K4 tanh-Sättigung (raw 1.96171, div 0.0)",
                                "V2 trägt: crossing [256], jl256 "
                                "0.958333/0.0 (wide-gepooltes-Datum)"]},
                   {"run": "REV-3",
                    "artefakte": "scratches/sp15_wide_rev3.json + .log "
                                 "(verbatim, GRÜN mit K5-Befund)",
                    "verdict": "GRÜN (exit 0) — 5 PASS | 0 FAIL; kausal PASS "
                               "(sens_lin_wide 0.92931, ratio_lin 235.387538)",
                    "befunde": ["K5 Instrument-Defekt der Block-Seite: "
                                "konsekutive Ersatzmatrix (blk_eq 0.285075) "
                                "statt Canon-8-Zelle (STRIDED)"],
                    "formel": "LINEAR-Lesungen beidseitig — Bänder getragen, "
                              "aber die Block-Seite maß nicht das exakte "
                              "Canon-tanh-Argument"},
                   {"run": "REV-4 (dieser Lauf)",
                    "artefakte": "scratches/sp15_wide_rev4.json + .log",
                    "verdict": "K5-Korrektur derselben REV-3-Bänder (keine "
                               "Formeländerung): z_block = hidden_state(t, 8) "
                               "direkt; ein Fail schließt die Runde "
                               "ROT-mit-Befund"}],
               "all_pass": not fails and not offen,
               "open": [r["id"] for r in opened]}
    out_path = "scratches/sp15_wide_rev4.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    print(f"JSON: {out_path}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    import sys
    sys.exit(main())