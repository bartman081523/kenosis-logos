"""
smoke_mech_interp_interventions.py — MI-3: Kausale Interventionen
(steer_band + knockout_experts), Hinreichendheit/Notwendigkeit getrennt.

PREREG (P2-Freeze, fixiert VOR der ersten Messung; Bänder nie geweitet;
OFFEN ≠ PASS; exit 0 alle PASS / 1 FAIL / 2 sonst; ohne --gpu bleiben die
Modell-Lane OFFEN gebucht — kein stiller Fallback).

Canon-Anker (voll gelesen 2026-10-26):
 - machine_base.py:229-267   oracle.steer: Preset-Arm = copy_ der GELERNTEN
                             Bias (Voll-Steering); dict-Arm = fill_(-steer)
                             + g[e]=+steer (SET, alle Schichten); Restore
                             copy_ in finally.
 - machine_base.py:239-271   predict_value (greedy Ziffern, "unknown").
 - machine_base.py:273-322   predict_scale_0_9 / SP3-Folge:
                             M_digit = Σ_d P_full(id(d)); tau=0.01; Ziffern
                             je ein einziges Token, Resolution-ValueError.
 - machine_base.py:276-284   SP13 Restore-Audit (bias clones + equal).
 - Presets auf Disk (gelesen): argon_1b_machines/machine_presets/
   {goedel[2,7], turing[6,0], wittgenstein[3,1], combined}.json — gelernte
   expert_bias flach 208 (26x8), steer 50.0.
 - modeling_argon.py:188-190 globale MLA-Schichten {5, 11, 17, 23}.
 - mech_interp.py:63-69      MI-1-Bänder: Edge {0,1,2,3,24,25},
                             Core {4..23} (Training-Hook-Bereich), plus
                             GLOBAL=(5,11,17,23).
 - MI-2-P4-Daten (commits d1fc355/cb46509, mi2_logit_lens_p4.json):
   Paris-Kette ℓ16 MoE +0.00699 -> 0.00703; ℓ17G Attn +0.68591 -> 0.69294,
   MoE +0.14071 -> 0.83366; Space-ℓ11G Attn +0.73470 / MoE -0.32335;
   ℓ5-Prune asymmetrisch (Ziffer -0.033, Space +0.0139).

Registrierte Theorie-Stufen (vier Felder je Verdict siehe unten):

V1 mi3_intervention_purity — Reversibilität der Sonde (EINHEIT C3).
   (a) Behauptung: die Sonde verändert den Träger NUR innerhalb der
       Interventions-Blöcke. (b) Begründung: C3 (copy_-Muster) + SP13-
       Restore-Audit. (c) Vorhersage: nach der Serie fingerprint_equal
       (expert_bias + gate.weight, alle 26 Schichten) UND canon_tv(P_final
       nach der Serie, P_final vor) == 0.0. (d) Falsifikator: Fingerprint-
       Mismatch ODER tv > 0.
   Serien-Bedingungen (jede im eigenen Block): steer_band({1,3}, EDGE),
   steer_band({1,3}, CORE), steer_band({1,3}, GLOBAL), knockout {3,7}@CORE,
   knockout {0,1,4}@CORE.

V2 mi3_core_vs_edge_sufficiency — Hinreichendheit (EINHEIT C1) am SP3-
   Stummkanal. (a) Behauptung: das Digit-Kanal-Öffnen (SP13: raw dumm,
   Voll-Steering {1,3} öffnet) liegt im CORE-BAND (4..23), nicht am EDGE.
   (b) Begründung: SP13 (smoke_sprachspiel.py, wittgenstein-Preset-Öffnen)
   + MI-1 Kernprofil (prose-L11-Anomalie im Core). (c) Vorhersage:
   ΔM_full > 0 UND ΔM_core ≥ 0.8 × ΔM_full UND ΔM_core > ΔM_edge
   strikt — ΔM_x = M_x − M_raw; M = M_digit am GENERIC_PROMPT (SP13-
   Literal), Steerings als dict-Direktsteuerung (SET, {1,3}) auf dem
   jeweiligen Band. (d) Falsifikator: (b)- oder (c)-Verletzung. Premise-
   Bruch: M_raw ≥ tau=0.01 (SP-Canon tau) -> OFFEN "Premise nicht
   repliziert" (volle Daten bleiben persistent).

V3 mi3_domain_knockout_specificity — Notwendigkeit (EINHEIT C1+C2,
   Double Dissociation). (a) Behauptung: die Domänen-Spezialisten tragen
   kausale Last domänenspezifisch. (b) Begründung: MI-1-Kernprofil
   (prose E3/E7 88.8%, code E4/E0/E1; Training-Hook-Bereich 4..23).
   (c) Vorhersage: KO {3,7} im Core stört prose stärker als KO {0,1,4}
   UND KO {0,1,4} stört code stärker als KO {3,7} — beide strikt (Mittel
   über die je 3 registrierten Texte; per-Text TVs persistent).
   (d) Falsifikator: eine der beiden Dissociationen verletzt.
   math_logic-Texte laufen als Datums-Zusatz (out-of-Predicate, wie
   MI-2 [Datums-Zusatz]; kein Band geweitet).

V4 mi3_paris_circuit_resilience — Welt-Wissen-Schutz. (a) Behauptung:
   das Fakten-Recall-Top-1 überlebt komplette Domänen-Steerings und die
   Spezialisten-Knockouts. (b) Begründung: MI-2-P4 — das Recall-Ereignis
   sitzt als großes Attn-Halbschritt in ℓ17 (nicht in den Domain-
   Spezialisten-Schichten allein). (c) Vorhersage: top1 == 9079 in jeden
   der 2+3 Bedingungen: KO {3,7}@Core, KO {0,1,4}@Core, Preset-Steering
   goedel/turing/wittgenstein (oracle.steer, GELERNTE Bias). (d)
   Falsifikator: jede Bedingung's-Argmax ≠ 9079.

V5 mi3_layer17_paris_localization — (a) Behauptung: der Löwenanteil des
   Recall-Sprungs wird am ℓ17-Halbschritt getragen und die ℓ17-KO
   stört die Antwort-Koordinate stärker als ℓ10-KO. (b) Begründung:
   MI-2-P4-Verdikt (attn@ℓ17 0.68591 vs moe 0.14071) + ℓ17-Attribution
   driver_is_global=True. (c) Vorhersage: (A-Datum) ΔP_attn(17)/(ΔP_attn
   (17)+ΔP_moe(17)) > 0.5; (B-kausal) Σ_e |Δp_e(9079)|(ℓ17) > Σ_e
   |Δp_e(9079)|(ℓ10) strikt (8 Einzel-Experten-KOs pro Schicht).
   (d) Falsifikator: (A) Anteil ≤ 0.5 ODER (B) ℓ10 ≥ ℓ17.

V6 mi3_prose_l11_anomaly_probe — der Outlier-Verdikt. (a) Behauptung:
   die prose-Anomalie-Paar {4,7} (MI-1: E3 2.3%/E4 44.6% an prose-ℓ11)
   trägt kausale Last in GENAU ℓ11. (b) Begründung: MI-1-L11-Anomalie
   (E4 44.6% an prose am ℓ11) + die MI-1-Entropie-Delle quer 6/6 (e11).
   (c) Vorhersage: KO {4,7} in EXAKT ℓ11 verschiebt die Ausgabeverteilung
   (canon_tv) auf den 3 prose-Texten stärker als KO {1,2} — strikt.
   (d) Falsifikator: Mittel{k4,7} ≤ Mittel{k1,2}.

Registrierte Proben-Texte (MI-1-smoke_mech_interp_tomography.py:146-191
GENAU kopiert — P1-reuse, kein neues Textmaterial):
 DOMAINS je 3 Texte auf code/math_logic/prose/goedel/turing/wittgenstein.
Registrierte Prompt-Literale:
 GENERIC_PROMPT = "Validity of a symbolic experiment batch, "
                   "single digit per ten percent: "
 PARIS_TEXT = "The capital of France is", target " Paris".

Konventionen: TV auf Fraction-Skala [0,1] (canon_tv); M_digit-Masse
(sp3-canon); P3 Roh-Felder je eval vollständig in JSON; P4 Einstufung
nach registrierten Bändern. P1: MI-1-/MI-2-Dateien unangetastet.

KORREKTUR-LOG (offen gebucht):
 - 2026-10-26, Lauf 1 (GPU, 41 s): Klassifikations-Schluessel-Bug — die
   evals buchen ihre Ergebnis-Felder unter Kurz-Schuesseln (purity,
   sufficiency, domain_knockout, paris_resilience, l17_localization,
   l11_anomaly), die Klassifikation pruefte aber Mitgliedschaft unter den
   registrierten Lang-Namen -> alle 6 OFFEN trotz vollstaendiger Messung.
   Fix: RESULT_KEYS-Mapping (registrierter Name -> Ergebnis-Schluessel).
   Der Lauf-1-Roh-Datensatz blieb persistent (P3) — der Fehler war rein
   klassifikatorisch, die Messfelder (fingerprint/tvs/massen) sind gueltig.
   Lauf 1: exit 2 GELB; Lauf 2 = Klassifikation ueber unveranderte
   Messfelder (Determinismus verifizierbar an resolve + l17-Datum).

Run lines:
  CPU: python -m scratches.smoke_mech_interp_interventions       (exit 2;
       schreibt KEIN Ergebnis-JSON ohne explizites --json — P3-Schutz)
  GPU: python -m scratches.smoke_mech_interp_interventions --gpu (exit 0/1;
       Default-JSON = scratches/mi3_interventions.json, P3-Roh-Felder)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from machines.mech_interp import (
    MechInterpProbe, apply_band_steer, apply_expert_knockout, canon_tv,
    layer_bands,
)

# -- Registrierte Konstanten --------------------------------------------------
STEER_AMPLITUDE = 50.0
KO_PENALTY = -50.0
M_DIGIT_TAU = 0.01          # SP-Canon tau (machine_base.py:285-286)
SUFF_RATIO_CORE = 0.8       # Vorhersage (c): ΔM_core ≥ 0.8 × ΔM_full (V2)
MAX_LENGTH = 128
N_LAYERS = 26               # Argon-1B (Canon)

_EDGE_LIST, _CORE_LIST = layer_bands(N_LAYERS)
BAND_EDGE = tuple(_EDGE_LIST)        # (0, 1, 2, 3, 24, 25)
BAND_CORE = tuple(_CORE_LIST)        # (4, ..., 23)
BAND_GLOBAL = (5, 11, 17, 23)        # modeling_argon.py:188-190 (layer % 6 == 5)

EX_WITT = (1, 3)            # SP-Canon wittgenstein-Preset-Experten
KO_PROSE = (3, 7)           # MI-1 prose-Monopolisten
KO_CODE = (0, 1, 4)         # MI-1 code-Spezialisten
KO_ANOM_A = (4, 7)          # MI-1-L11-Outlier-Paar (Verdikt 6)
KO_ANOM_B = (1, 2)          # Kontroll-Paar (Verdikt 6)

PRESETS = ("goedel", "turing", "wittgenstein")

GENERIC_PROMPT = ("Validity of a symbolic experiment batch, "
                  "single digit per ten percent: ")
PARIS_TEXT = "The capital of France is"

DOMAINS: Dict[str, List[str]] = {
    "code": [
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
        "    pivot = items[0]\n    return quicksort([x for x in items[1:] if x < pivot])"
        " + [pivot] + quicksort([x for x in items[1:] if x >= pivot])",
        "for i in range(10):\n    buffer[i] = buffer[i] * 2 % 255\n    print(buffer[i])",
        "class Node:\n    def __init__(self, value, next=None):\n"
        "        self.value = value\n        self.next = next",
    ],
    "math_logic": [
        "For all natural numbers n, the sum 1 + 2 + ... + n equals n(n+1)/2.",
        "If P implies Q and Q is false, then P is false. Proof: assume P, then Q, contradiction.",
        "Jede endliche Gruppe mit Primzahlordnung ist zyklisch.",
    ],
    "prose": [
        "Der warme Sommerwind trug in der Abenddämmerung den Duft von Kiefern "
        "über den stillen Bergsee.",
        "The warm summer breeze carried the scent of pine trees across the quiet "
        "mountain lake at dusk.",
        "Oma erzählt ihre Geschichte vom alten Garten, der jedes Jahr im Mai "
        "in Blüten stand.",
    ],
    "goedel": [
        "Die Gödelnummer eines Formelausdrucks entsteht aus Primfaktoren: "
        "2^1 · 3^2 · 5^1 kodiert die Zeichenfolge (x).",
        "Jede konsistente formale Theorie, in der Arithmetik ausdrückbar ist, "
        "enthält wahre, aber unbeweisbare Sätze.",
        "Ein Satz, der seine eigene Unbeweisbarkeit behauptet, ist wahr, "
        "wenn die Theorie widerspruchsfrei bleibt.",
    ],
    "turing": [
        "Eine Turingmaschine liest das Symbol, schreibt das neue Symbol, bewegt "
        "den Kopf nach links oder rechts und wechselt den Zustand.",
        "The transition table maps (state, read) to (write, move, next): "
        "(s0, '0') -> ('1', R, s1).",
        "Die Halteproblementcheidung ist unmöglich, denn jede Entscheidersimulation "
        "wäre selbst eine Turingmaschine.",
    ],
    "wittgenstein": [
        "Ein Wort hat Bedeutung im Gebrauch, nicht im Wörterbuch — das "
        "Sprachspiel gibt der Zeichenfolge ihre Rolle.",
        "Die Grenzen meiner Sprache bedeuten die Grenzen meiner Welt.",
        "Zu einem Satz gehört erst die Grammatik des Spiels: 'Ziegel!' ist "
        "Anruf, Befehl, Antwort — je nach Zug im Spiel.",
    ],
}


def resolve_token_ids_exact(tokenizer, target: str,
                            candidates: Tuple[str, ...]) -> List[int]:
    """MI-2-P4-Auflösung (Oberflächen-Exakt, gebucht 2026-10-26):
    len(encoded) == 1 UND decode([id]) == target EXAKT (kein lstrip);
    Mehr-Token-Kandidaten abgewiesen, bevor ein Vertreter entsteht."""
    ids: List[int] = []
    for cand in candidates:
        encoded = tokenizer.encode(cand, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        decoded_first = tokenizer.decode([encoded[0]], skip_special_tokens=True)
        if decoded_first == target:
            ids.append(int(encoded[0]))
    return ids


# -- CPU-Pilot (PILOT-Labels, promovieren keine Verdicts — R2) ---------------
def pilot_rows() -> List[Tuple[str, str, str]]:
    import torch as _torch

    rows: List[Tuple[str, str, str]] = []

    # pilot_band_math: SET-Semantik des echten Helfers an Stub-Buffer.
    buf = _torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    apply_band_steer(buf, (1, 3), 50.0)
    ok_band = _torch.equal(buf, _torch.tensor(
        [-50.0, 50.0, -50.0, 50.0, -50.0, -50.0, -50.0, -50.0]))

    # pilot_knockout_math: ADDITIVE Semantik des echten Helfers.
    buf2 = _torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
    apply_expert_knockout(buf2, (0, 2), -50.0)
    ok_knock = _torch.equal(buf2, _torch.tensor(
        [-48.5, 2.5, -46.5, 4.5, 5.5, 6.5, 7.5, 8.5]))

    # pilot_restore_math: C3-Runde (clone -> mutate -> copy_ -> equal).
    orig = buf.clone()
    apply_expert_knockout(buf, (5,), -50.0)
    buf.copy_(orig)
    ok_restore = _torch.equal(buf, orig)
    ok_restore = ok_restore and not _torch.equal(buf, orig * 0 + 1.0)

    # pilot_tv_math: canon_tv-Identität/Differenz.
    a = _torch.softmax(_torch.zeros(8), dim=0)
    b = _torch.softmax(_torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0]), dim=0)
    ok_tv = canon_tv(a, a) == 0.0 and canon_tv(a, b) > 0.0

    # pilot_select_math: Bänder + Experten-Sets konsistent.
    ok_select = (
        len(BAND_EDGE) == 6 and len(BAND_CORE) == 20
        and BAND_GLOBAL == (5, 11, 17, 23)
        and all(0 <= e < 8 for e in EX_WITT)
        and all(0 <= e < 8 for e in KO_PROSE + KO_CODE)
        and set(KO_ANOM_A + KO_ANOM_B) <= set(range(8)))

    rows.append(("pilot_band_math", "PILOT" if ok_band else "BROKEN",
                 f"band-template: {buf.tolist() if not ok_band else 'ok'}"))
    rows.append(("pilot_knockout_math", "PILOT" if ok_knock else "BROKEN",
                 "additiv: ok" if ok_knock else f"knockout: {buf2.tolist()}"))
    rows.append(("pilot_restore_math", "PILOT" if ok_restore else "BROKEN",
                 f"C3-Runde: {'bitidentisch' if ok_restore else 'Mismatch'}"))
    rows.append(("pilot_tv_math", "PILOT" if ok_tv else "BROKEN",
                 f"tv(a,a)=0, tv(a,b)>0: {ok_tv}"))
    rows.append(("pilot_select_math", "PILOT" if ok_select else "BROKEN",
                 "Baender+Sets registriert ok" if ok_select else "Fehler"))
    return rows


# -- GPU-Lane -----------------------------------------------------------------
def eval_purity(probe: MechInterpProbe, oracle: Any, fp_before: Dict[str, Any],
                p_before: torch.Tensor, results: Dict[str, Any]) -> None:
    """V1 — Serien-Interventionen, NACH der Serie fingerprint + Kontroll-TV."""
    # Serien-Bedingungen (P2 registriert): jede in eigenem Block.
    with probe.steer_band(EX_WITT, BAND_EDGE):
        pass
    with probe.steer_band(EX_WITT, BAND_CORE):
        pass
    with probe.steer_band(EX_WITT, BAND_GLOBAL):
        pass
    with probe.knockout_experts(KO_PROSE, BAND_CORE):
        pass
    with probe.knockout_experts(KO_CODE, BAND_CORE):
        pass
    fp_after = probe.fingerprint()
    equal = probe.fingerprint_equal(fp_before, fp_after)
    p_after = probe.last_probs(GENERIC_PROMPT)
    tv = canon_tv(p_before, p_after)
    results["purity"] = {
        "fingerprint_equal": bool(equal),
        "control_tv": tv,
        "tv_zero": bool(tv == 0.0),
        "series": ["steer_band(EX_WITT, EDGE)", "steer_band(EX_WITT, CORE)",
                   "steer_band(EX_WITT, GLOBAL)",
                   "knockout(KO_PROSE, CORE)", "knockout(KO_CODE, CORE)"],
    }


def eval_sufficiency(probe: MechInterpProbe, oracle: Any,
                     results: Dict[str, Any]) -> None:
    """V2 — SP3-Stummkanal am GENERIC_PROMPT; Steerings als dict-Direkt-
    steuerung (SET, {1,3}) über die Bänder (oracle.steer dict-Arm-Canon)."""
    m_raw = probe.digit_mass(GENERIC_PROMPT)
    with oracle.steer({"experts": list(EX_WITT), "steer": STEER_AMPLITUDE}):
        m_full = probe.digit_mass(GENERIC_PROMPT)
    with probe.steer_band(EX_WITT, BAND_CORE):
        m_core = probe.digit_mass(GENERIC_PROMPT)
    with probe.steer_band(EX_WITT, BAND_EDGE):
        m_edge = probe.digit_mass(GENERIC_PROMPT)
    dm_full = m_full["M_digit"] - m_raw["M_digit"]
    dm_core = m_core["M_digit"] - m_raw["M_digit"]
    dm_edge = m_edge["M_digit"] - m_raw["M_digit"]
    results["sufficiency"] = {
        "m_raw": m_raw, "m_full": m_full, "m_core": m_core, "m_edge": m_edge,
        "dm_full": dm_full, "dm_core": dm_core, "dm_edge": dm_edge,
        "premise_raw_dumb": bool(m_raw["unknown"]),
        "full_opens": bool(dm_full > 0.0),
        "core_ratio": (dm_core / dm_full) if dm_full > 0.0 else None,
        "core_sufficient": bool(dm_full > 0.0
                                and dm_core >= SUFF_RATIO_CORE * dm_full),
        "edge_smaller": bool(dm_core > dm_edge),
    }


def eval_domain_knockout(probe: MechInterpProbe, results: Dict[str, Any]) -> None:
    """V3 — Double Dissociation im Core-Band; TVs per Text (EINHEIT T1/R3)."""
    texts = DOMAINS["prose"] + DOMAINS["code"] + DOMAINS["math_logic"]
    labels = (["prose"] * 3) + (["code"] * 3) + (["math_logic"] * 3)
    raw_p = [probe.last_probs(t) for t in texts]
    tv_prose_ko: Dict[str, List[float]] = {"ko_prose": [], "ko_code": [],
                                           "ko_anom_a": [], "ko_anom_b": []}
    with probe.knockout_experts(KO_PROSE, BAND_CORE) as _:
        ko_p1 = [probe.last_probs(t) for t in texts]
    with probe.knockout_experts(KO_CODE, BAND_CORE):
        ko_p2 = [probe.last_probs(t) for t in texts]
    with probe.knockout_experts(KO_ANOM_A, (11,)):
        ko_p3 = [probe.last_probs(t) for t in texts]
    with probe.knockout_experts(KO_ANOM_B, (11,)):
        ko_p4 = [probe.last_probs(t) for t in texts]
    rows: List[Dict[str, Any]] = []
    for i, (t, label, p0, p1, p2, p3, p4) in enumerate(
            zip(texts, labels, raw_p, ko_p1, ko_p2, ko_p3, ko_p4)):
        tv1, tv2, tv3, tv4 = (canon_tv(p0, p1), canon_tv(p0, p2),
                              canon_tv(p0, p3), canon_tv(p0, p4))
        rows.append({"text": t, "domain": label, "tv_ko_prose": tv1,
                     "tv_ko_code": tv2, "tv_ko_anom_a": tv3,
                     "tv_ko_anom_b": tv4})
        print(f"  [row {i}] {label}: ko_prose={tv1:.4f} ko_code={tv2:.4f}"
              f" anomal_k47={tv3:.4f} kontrolle_k12={tv4:.4f}")
        if label == "prose":
            tv_prose_ko["ko_prose"].append(tv1)
            tv_prose_ko["ko_code"].append(tv2)
            tv_prose_ko["ko_anom_a"].append(tv3)
            tv_prose_ko["ko_anom_b"].append(tv4)

    def _mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    prose_kopro = _mean(tv_prose_ko["ko_prose"])
    prose_kocode = _mean(tv_prose_ko["ko_code"])
    code_kopro = _mean([r["tv_ko_prose"] for r in rows if r["domain"] == "code"])
    code_kocode = _mean([r["tv_ko_code"] for r in rows if r["domain"] == "code"])
    results["domain_knockout"] = {
        "rows_9": rows,   # 9 Rows (3x prose/code/math_logic)
        "prose_mean_ko_prose": prose_kopro,
        "prose_mean_ko_code": prose_kocode,
        "code_mean_ko_prose": code_kopro,
        "code_mean_ko_code": code_kocode,
        "dissociation_prose": bool(prose_kopro > prose_kocode),
        "dissociation_code": bool(code_kocode > code_kopro),
    }


def eval_paris_resilience(probe: MechInterpProbe, oracle: Any,
                          paris_id: int, results: Dict[str, Any]) -> None:
    """V4 — top1 == 9079 unter 5 registrierten Bedingungen."""
    conditions: Dict[str, int] = {}
    with probe.knockout_experts(KO_PROSE, BAND_CORE):
        conditions["ko_prose@core"] = int(probe.last_probs(PARIS_TEXT).argmax().item())
    with probe.knockout_experts(KO_CODE, BAND_CORE):
        conditions["ko_code@core"] = int(probe.last_probs(PARIS_TEXT).argmax().item())
    for preset in PRESETS:
        with oracle.steer(preset):
            conditions[f"preset_{preset}"] = int(
                probe.last_probs(PARIS_TEXT).argmax().item())
    stable = all(v == paris_id for v in conditions.values())
    results["paris_resilience"] = {
        "paris_id": paris_id,
        "conditions_top1": conditions,
        "stable": bool(stable),
    }


def eval_l17_localization(probe: MechInterpProbe, paris_id: int,
                          results: Dict[str, Any]) -> None:
    """V5 — (A) Sublayer-Datum auf 9077 (P4-Referenz); (B) Einzel-Experten-
    KOs in ℓ17 vs ℓ10 (16 Forwards, per-Experte persistiert)."""
    sub = probe.sublayer_logits(PARIS_TEXT, {"id": paris_id, "str": " Paris"})
    r17 = sub["rows"][17]
    da, dm = r17["dp_attn"], r17["dp_moe"]
    share = da / (da + dm) if (da + dm) != 0.0 else 0.0

    def _sum_abs_dp(layer: int) -> Dict[str, Any]:
        deltas: Dict[str, float] = {}
        raw = probe.last_probs(PARIS_TEXT)[paris_id].item()
        for e in range(8):
            with probe.knockout_experts((e,), (layer,)):
                ko = probe.last_probs(PARIS_TEXT)[paris_id].item()
            deltas[str(e)] = abs(ko - raw)
        return {"total": float(sum(deltas.values())), "per_expert": deltas,
                "p_raw": raw}

    s17 = _sum_abs_dp(17)
    s10 = _sum_abs_dp(10)
    results["l17_localization"] = {
        "sublayer_17": {"dp_attn": da, "dp_moe": dm, "attn_share": share,
                        "anchor_tv": sub["anchor_tv"],
                        "telescope_broken": sub["telescope_broken"]},
        "ko_total_17": s17, "ko_total_10": s10,
        "l17_lion": bool(share > 0.5),
        "l17_stronger": bool(s17["total"] > s10["total"]),
    }


def eval_l11_anomaly(probe: MechInterpProbe, results: Dict[str, Any]) -> None:
    """V6 — KO {4,7} vs {1,2} in GENAU ℓ11 auf den 3 prose-Texten."""
    texts = DOMAINS["prose"]
    raw_p = [probe.last_probs(t) for t in texts]
    with probe.knockout_experts(KO_ANOM_A, (11,)):
        ko_a = [probe.last_probs(t) for t in texts]
    with probe.knockout_experts(KO_ANOM_B, (11,)):
        ko_b = [probe.last_probs(t) for t in texts]
    tv_a = [canon_tv(p, q) for p, q in zip(raw_p, ko_a)]
    tv_b = [canon_tv(p, q) for p, q in zip(raw_p, ko_b)]
    mean_a = float(sum(tv_a) / len(tv_a))
    mean_b = float(sum(tv_b) / len(tv_b))
    results["l11_anomaly"] = {
        "tv_k47": tv_a, "tv_k12": tv_b,
        "mean_k47": mean_a, "mean_k12": mean_b,
        "outlier_causal": bool(mean_a > mean_b),
    }


def run_gpu(checkpoint_base: str, checkpoint_machines: str,
            results: Dict[str, Any], findings: List[Tuple[str, str, str]]) -> None:
    from machines.verify_machines import _load_machines_model

    model, tokenizer = _load_machines_model(checkpoint_base, checkpoint_machines)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = MechInterpProbe(model, tokenizer, device=device,
                            max_length=MAX_LENGTH)
    from machines.machine_base import ArgonOracle
    oracle = ArgonOracle(model, tokenizer, device=device)

    # Auflösung (Oberflächen-Exakt, keine Vertreter):
    paris_ids = resolve_token_ids_exact(tokenizer, " Paris", (" Paris",))
    results["resolve"] = {"paris": paris_ids}
    if len(paris_ids) != 1:
        findings.append((
            "mi3_resolve_target",
            "OFFEN",
            f"Paris-Auflösung nicht eindeutig ({paris_ids}) — OFFEN ist "
            "kein PASS; sonstiges gemessen und persistent (keine Band-"
            "Erfindung)."))
        paris_id = paris_ids[0] if paris_ids else -1
    else:
        paris_id = paris_ids[0]

    fp_before = probe.fingerprint()
    p_before = probe.last_probs(GENERIC_PROMPT)

    for name, fn in (
            ("mi3_intervention_purity",
             lambda: eval_purity(probe, oracle, fp_before, p_before, results)),
            ("mi3_core_vs_edge_sufficiency",
             lambda: eval_sufficiency(probe, oracle, results)),
            ("mi3_domain_knockout_specificity",
             lambda: eval_domain_knockout(probe, results)),
            ("mi3_paris_circuit_resilience",
             lambda: eval_paris_resilience(probe, oracle, paris_id, results)),
            ("mi3_layer17_paris_localization",
             lambda: eval_l17_localization(probe, paris_id, results)),
            ("mi3_prose_l11_anomaly_probe",
             lambda: eval_l11_anomaly(probe, results)),
    ):
        try:
            fn()
        except Exception as exc:      # noqa: BLE001 — BROKEN-Buchung
            results.setdefault("broken", {})[name] = f"{type(exc).__name__}: {exc}"
            findings.append((
                name, "BROKEN",
                f"GPU-Lane brach: {type(exc).__name__}: {exc} "
                "(kein stiller Fallback; sonstiges gemessen bleibt)."))


# -- classification + main ----------------------------------------------------
# Registrierter Verdict-Name -> Ergebnis-Schluessel der evals
# (KORREKTUR-LOG Lauf 1: Klassifikation pruefte sonst Mitgliedschaft
# unter Lang-Namen, waehrend die evals unter Kurz-Schluesseln buchen).
RESULT_KEYS = {
    "mi3_intervention_purity": "purity",
    "mi3_core_vs_edge_sufficiency": "sufficiency",
    "mi3_domain_knockout_specificity": "domain_knockout",
    "mi3_paris_circuit_resilience": "paris_resilience",
    "mi3_layer17_paris_localization": "l17_localization",
    "mi3_prose_l11_anomaly_probe": "l11_anomaly",
}


def classify(name: str, results: Dict[str, Any]) -> Tuple[str, str]:
    """P4-Einstufung je registrierten Bänder (kein Re-Decide, kein
    Band-Kompromiss)."""
    if name == "mi3_intervention_purity":
        p = results["purity"]
        return ("PASS", f"fingerprint={p['fingerprint_equal']} "
                        f"control_tv={p['control_tv']!r}") if (
            p["fingerprint_equal"] and p["tv_zero"]) else (
            "FAIL", f"fingerprint={p['fingerprint_equal']} "
                        f"control_tv={p['control_tv']!r}")

    if name == "mi3_core_vs_edge_sufficiency":
        p = results["sufficiency"]
        if p["premise_raw_dumb"] is False:
            return ("OFFEN",
                    f"Premise nicht repliziert: M_raw={p['m_raw']['M_digit']:.4f} "
                    f"(≥ tau={M_DIGIT_TAU}) — volle Daten persistent.")
        if not p["full_opens"]:
            return ("OFFEN",
                    f"ΔM_full={p['dm_full']:.4f} ≤ 0 — Steering öffnet den "
                    "Digit-Kanal hier nicht (keine Band-Erfindung); Daten "
                    "persistent.")
        if p["core_sufficient"] and p["edge_smaller"]:
            return ("PASS",
                    f"ΔM_full={p['dm_full']:.4f} core_ratio="
                    f"{p['core_ratio']:.3f} edge={p['dm_edge']:.4f}")
        return ("FAIL",
                f"ΔM_full={p['dm_full']:.4f} core_ratio="
                f"{p['core_ratio']:.3f} core_sufficient="
                f"{p['core_sufficient']} edge_smaller={p['edge_smaller']}")

    if name == "mi3_domain_knockout_specificity":
        p = results["domain_knockout"]
        ok = p["dissociation_prose"] and p["dissociation_code"]
        note = (f"prose: ko_prose={p['prose_mean_ko_prose']:.4f} > "
                f"ko_code={p['prose_mean_ko_code']:.4f}; "
                f"code: ko_code={p['code_mean_ko_code']:.4f} > "
                f"ko_prose={p['code_mean_ko_prose']:.4f}")
        return ("PASS", note) if ok else ("FAIL", note)

    if name == "mi3_paris_circuit_resilience":
        p = results["paris_resilience"]
        note = ("top1-Bedingungen: "
                + ", ".join(f"{k}={v}" for k, v in p["conditions_top1"].items())
                + f" (paris_id={p['paris_id']})")
        return ("PASS", note) if p["stable"] else ("FAIL", note)

    if name == "mi3_layer17_paris_localization":
        p = results["l17_localization"]
        note = (f"attn_share(17)={p['sublayer_17']['attn_share']:.3f} "
                f"Σ|Δp|(17)={p['ko_total_17']['total']:.4f} vs "
                f"Σ|Δp|(10)={p['ko_total_10']['total']:.4f}")
        if p["l17_lion"] and p["l17_stronger"]:
            return ("PASS", note)
        return ("FAIL", note)

    if name == "mi3_prose_l11_anomaly_probe":
        p = results["l11_anomaly"]
        note = (f"Mittel ko{{4,7}}(ℓ11)={p['mean_k47']:.4f} vs "
                f"ko{{1,2}}(ℓ11)={p['mean_k12']:.4f}")
        return ("PASS", note) if p["outlier_causal"] else ("FAIL", note)

    return ("OFFEN", "unregistriertes Verdict — kein stiller Fallback.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MI-3: Kausale Interventionen (steer_band + knockout)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--checkpoint_base", default="./argon_1b")
    ap.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    ap.add_argument("--json", default=None,
                    help="Ergebnis-Pfad; Default: GPU->MI-3-Datei, "
                         "CPU-Lane schreibt OHNE --json nichts "
                         "(P3-Schutz, MI-2-P4-Muster)")
    args = ap.parse_args()

    t0 = time.time()
    findings: List[Tuple[str, str, str]] = list(pilot_rows())

    data: Dict[str, Any] = {"pilot": [
        {"name": n, "status": s, "note": note} for n, s, note in findings]}

    gpu_broken = False
    if args.gpu:
        results: Dict[str, Any] = {}
        try:
            run_gpu(args.checkpoint_base, args.checkpoint_machines,
                    results, findings)
        except Exception as exc:      # noqa: BLE001 — BROKEN-Buchung
            gpu_broken = True
            findings.append((
                "gpu_broken", "BROKEN",
                f"GPU-Lane brach vor der Messung: {type(exc).__name__}: "
                f"{exc} (kein stiller Fallback)."))
        else:
            data["gpu"] = results

    verdicts = [f for f in findings if not f[0].startswith("pilot_")]
    classified: List[Tuple[str, str, str]] = []
    registered = ("mi3_intervention_purity", "mi3_core_vs_edge_sufficiency",
                  "mi3_domain_knockout_specificity",
                  "mi3_paris_circuit_resilience",
                  "mi3_layer17_paris_localization",
                  "mi3_prose_l11_anomaly_probe")
    if args.gpu and not gpu_broken:
        for name in registered:
            if name in {f[0] for f in findings if f[1] == "BROKEN"}:
                note = next(f[2] for f in findings
                            if f[0] == name and f[1] == "BROKEN")
                classified.append((name, "BROKEN", note))
            elif RESULT_KEYS[name] not in results:
                classified.append((
                    name, "OFFEN",
                    "GPU-Lane lieferte kein Datum — OFFEN ist kein PASS"))
            else:
                classified.append((name, *classify(name, results)))
    else:
        for name in registered:
            classified.append((
                name, "OFFEN",
                "GPU-Lane nicht gefahren oder Bruch gebucht (OFFEN ist "
                "kein PASS)."))
    findings.extend(classified)

    has_fail = any(s == "FAIL" for _, s, _ in classified)
    all_pass = all(s == "PASS" for _, s, _ in classified)

    for n, s, note in findings:
        print(f"[{s:6s}] {n}: {note}")
    print("-" * 88)
    counts = [sum(1 for _, s, _ in classified if s == k) for k in
              ("PASS", "OFFEN", "FAIL")]
    print(f"Verdikte: {counts[0]} PASS | {counts[1]} OFFEN | {counts[2]} FAIL"
          f"  ({time.time() - t0:.1f} s)")
    if has_fail:
        print("ROT — Falsifikator ausgelöst; keine Kürzung, volle Daten "
              "persistent.")
    elif all_pass and not gpu_broken:
        print("GRÜN — alle 6 Verdikte PASS; die Kausalität ist gemessen.")
    else:
        print("GELB — offen (GPU-Lane nicht gefahren oder Bruch gebucht; "
              "OFFEN ist kein PASS).")

    payload = {
        "all_pass": bool(all_pass and not has_fail and not gpu_broken),
        "open": [f[0] for f in classified if f[1] == "OFFEN"],
        "findings": [{"name": f[0], "status": f[1], "note": f[2]}
                     for f in findings],
        "data": data,
    }
    json_path = args.json if args.json is not None else \
        ("scratches/mi3_interventions.json" if args.gpu else None)
    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(f"JSON: {json_path}")
    else:
        print("JSON: — (CPU-Lane schreibt ohne --json nichts)")
    if has_fail:
        return 1
    return 0 if (all_pass and not gpu_broken) else 2


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(main())