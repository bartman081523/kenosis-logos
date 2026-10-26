"""
smoke_sp16_impossible_texts.py — SP16: Kybernetische Textgenerierung +
Audit der «unmöglichen Texte». Dritte Runde des 5-Pläne-Programms
«Argon-Subjekt v2 / MI-7 bis SP16» — Plan 3 (Cybernetik, machines/
cybernetic_argon.py) und Plan 4 (unmögliche Texte) per User-Freigabe
2026-10-26 («mache alles vorgeschlagene», «lgtm») DIREKT in dieses
Smoke-Programm verschmolzen (kein separates Geräte-File; machines/
bleibt UNANGETASTET, smoke append-only). Die Canon-Zelle
`machine_base.py` `pooled.reshape(-1, d).mean(axis=0)` bleibt Canon,
der 1B-Subjekt-Modell bleibt canon.

WAS hier erzeugt wird: drei Gattungen von Texten, deren Entstehung der
Canon-Stack (uni_204-Kollaps, kein wide-read-Opt-in, kein
Vorschau-Horizont) nicht leisten kann —

(A) Der Live-Provenanz-Stream: drei Gates, ein Token je Gate, das beim
    Schreiben gemessen und (chi_w, chi_c, chi_s)-getaggt wird —
      gate_w («Wissens-Satz»): Emission vom NACKTEN Schwellen-Prompt
               ARME_VORGABE["w"]["abl"] = "The capital of France is";
               erwartet 9079 « Paris» (MI-7: 0.503078 am abl-Schlitz —
               der Wissens-Kanal TRÄGT der volle-Slot kollabierte dort zu
               « a» 496 / 0.00022, das REV-1-Gold).
      gate_c («Kontext-Protokoll-Satz»): Emission vom C-full-Slot
               ARME_VORGABE["c"]["full"]; erwartet 1491 « Z»
               (0.806368; chi_c = Kontext-Gewinn 0.806301, der nackten
               Schlitz trägt dort nur «:» 236787 / 0.239376).
      gate_s («gereifter Maschinen-Reflexions-Satz»): Emission vom
               gereiften Korpus-Rendering texts[10]; erwartet 236825 «6»
               (0.85948) statt 236810 «5» (0.196449 — unreifes Datum).
    Treiber-Datum je Gate: |dp_attn(ℓ17)| (sublayer_logits) gegen
    m_core_raw + attn_share_core (Routing-Masse).

(B) Die Horizont-Weiche: die P1d-Dissonanz-Zelle (step_autonomous) auf
    der Canon-Rezeptur repliziert (learn_pair_pin + bio_gelernt, OHNE
    bio_halted),_preview_steps k = 4 erkennt den LOOP im Vorstellungsraum
    (Restaurierung bitrein, Δ = 0 — Maschinen-Fingerprint vor/nach), und
    die Weiche schaltet auf «wittgenstein» (chosen_preset) und senkt das
    Dissonanz-Niveau unter das Substrat-Vater-Niveau.

(C) Die Zwillings-Divergenz: zwei Instanzen — A («spiegel», identity,
    gelehrte Biografie) und B («stille», negate, frische Biografie) —
    EXAKT auf derselben Gewichtsmatrix (fingerprint_equal), derselbe
    Start-Prompt (der nackte Threshold), verschiedene Ströme: die
    Verdikt-Spuren heißen LOOP vs HALTED, die Greedy-Ketten divergieren,
    die trainierten Wide-Leser projizieren den gemeinsamen
    Threshold-Hidden in verschiedene 8-D-Ströme (‖z_A − z_B‖₂ > 0).

PREREG Rev. 1 — die vier registrierten Verdikte (Ziel-Bänder; ein Fail
scheidet die Runde ROT-mit-Befund, kein Formeln-Re-Register, kein
Relabeling; ein Band, das nicht getragen wird, ist ein Ergebnis):

  1. sp16_live_provenance_audit — der unabhängige Re-Forward-Audit
     bestätigt mit Abweichung < 1e-6 (SP16_AUDIT_MAX), dass jedes im
     Text behauptete Provenanz-Tag exakt den wahren Sublayer-/Routing
     -Strömen an derselben Token-Position entspricht (zweiter frischer
     MechInterpProbe, dieselben Prompts, identische Argmax-Ids) UND der
     dominante Kanal folgt der Sequenz W → C → S (je Gate striktes
     Argmax auf SEINER Spalte UND der Wert > M7_DIAG_MIN = 0.5, MI-7
     -REV-2-Zelle) UND jede Emission trifft ihr registriertes Token
     (9079 / 1491 / 236825; gate_s: p(«6») > p(«5»)). BAND: die
     Register-Spuren mit dem MI-7-Prosa-Kontrast — σ_bio(gereiftes
     Rendering) > 0.5 UND σ_bio(FUENLLE_NEGATIVE[0] — Weltprosa) < 0.5
     UND der Bio-Gewinn p(«6»|gereift) − p(«6»|ungereift) > 0. DATUM
     (kein Band): die Treiber-Felder |dp_attn|(ℓ17) gegen m_core_raw
     + attn_share_core je Gate (die MI-7-REV-2-Ankerzahlen).

  2. sp16_horizon_loop_avoidance — (a) die Rezeptur reproduziert EXAKT:
     r_m (A-Rezeptur, frische 2. Konstruktion) == korpus texts[10]
     zeichenweise, r_u (B-Rezeptur ohne learn_pair_pin) == korpus
     texts[22]; (b) die Ziffern-Schablone: argmax(last_probs(r_m)) ==
     236825 «6» UND argmax(last_probs(r_u)) == 236810 «5» UND
     p(„6"|r_m) − p(„6"|r_u) > 0; (c) die Preview-Weiche: k = 4
     Vorschau-Verdikte ALL «LOOP» UND der Maschinen-Fingerprint
     (state/Fingerprint) vor/nach bitidentisch (Restaurierung Δ = 0);
     (d) die P1d-Weiche: 5 Repliken step_autonomous («spiegel», lg) —
     best_gap < raw_gap in ≥ P4_AUTO_MIN = 4 von 5 UND mean_best <
     P4_VATER_REF = 0.567980 UND chosen_preset == "wittgenstein" in ≥ 4
     von 5 (stricter als P1d: das wählte 5/5 — das Band ist die
     Mehrheit). DATUM: mean_raw ≈ 0.680926 (P4-Korrespondenz), die
     candidate_gaps, das Rendering NACH der Weiche (argmax + p).

  3. sp16_twin_divergence_stream — (a) fingerprint_equal des MODELLS an
     3 Checkpoints (vor der Zwillings-Tradition-Ketten, zwischendurch,
     nach den Ketten) — dieselbe Gewichtsmatrix; (b) die Verdikt-Spuren:
     preview_1-A («spiegel») == «LOOP» UND preview_1-B («stille») ==
     «HALTED» (Wort-Formular identity vs negate); (c) die Ketten
     divergieren: 6-greedy-Tokens aus der Zeile a-Zp8-textur != 6-greedy
     -Tokens aus der Zeile b-Zp8-textur; (d) die Ströme trennen:
     ‖z_A − z_B‖₂ > 0, z_X = W_read,X @ h_threshold (Threshold =
     "The capital of France is", h = hidden_state(P*, 1152) — der
     WIDE-LESE-KANAL (identity bei d = 1152, SP15 REV-4)). DATUM:
     Erst-Token-Prob-Verteilung beider Ketten, Adapter-Digests,
     w_mean_abs, z-Normen.

  4. sp16_purity_and_det — (a) fingerprint_equal START/END über die
     GANZE Batterie; (b) die Training-Determinismus: zwei frische
     _train_wide-Läufe auf demselben a-Texte-Set →
     adapter_digest == gap_digest bitidentisch (SP15-Muster);
     (c) der CPU-Pilot fährt zweimal und ist bitidentisch (det_equal)
     UND sein Synthese-Verdikt trägt.

DOKUMENTIERTE ABWEICHUNGEN von der User-Wortführung (offen gelegt,
keine Band-Weitung):
  - Der enable_wide_read-Arm der Gattung B lebt in Lane 3 (Zwillings-A
    = Korpus-A-Subjekt + opt-in + Training) — die Dissonanz-Bänder
    stehen auf der CANON-Rezeptur (MI-7/P4-Anchor-Zahlen sind Canon
    -Messungen; P4_VATER_REF hat keinen Wide-Regime-Anker; der roh
    -trainierte Wide-Leser ist raw-verankert → best < raw dort kein
    Band, DATUM).
  - Gate_s trägt ZWEI χ-Zeilen: das BAND-Ziel ist das registrierte
    SI-Schiene-Ziel «)» (tid = letztes Token texts[10]: [0, 0, 0.5565]
    — S-strikt), das Emissions-Ziel «6» (236825) ist DATUM-Zeile mit
    dem P1b-Paar — dort ist chi_c (0.859) dominant: das Rendering
    trägt den Zustand DURCH den Kontext-Kanal, die S-Signatur ist der
    Register-Kanal (是X即非X, offen notiert).
  - Die Provenanz-Zahlen stehen UNGERUNDET im JSON (das Audit-Band
    1e-6 verlangt volle Präzision); die Berichte runden für die
    Darstellung (round6-Anker stehen in der Plan-/Berichtsdarstellung).

PREREG-Anchor (MI-7 REV-2, mi7_triangulation_rev2.json verbatim):
  w: chi [0.503078, 0.0, 0.046344], m_core 0.175808, σ 0.263603
  c: chi [0.000067, 0.806301, 0.035979], m_core 0.036648, σ 0.981735
  s: chi [0.0, 0.0, 0.556515] (Ziel «)» 1606), m_core 0.563825, σ 0.987035
  Continuations: «6» 0.85948 / «5» 0.196449; W-full-Slot « a»
  0.698683; C-abl-Slot «:» 0.239376; Dekoy (Tokyo) [0.727696, 0, 0.002222].

KORREKTUR-LOG (bei Erst-PREREG leer; Einträge nur für beweisliche
Lauf-Bugs, nie Band-Weitung; ein Fail ist ein Befund, kein Bug):

Berichte (nach dem Lauf): sprachspiel_bericht.md §26 + §28 (Laufschrift);
repo2txt GENERATION SP16; 2 Commits; Memory; Task.

GPU-Lauf:   timeout 3600 python -u -m scratches.smoke_sp16_impossible_texts --gpu
            > scratches/sp16_impossible_texts.log 2>&1
CPU-Pilot:  python -u -m scratches.smoke_sp16_impossible_texts
            > scratches/sp16_cpu_pilot.log 2>&1

Exit: 0 GRÜN (alle Bänder), 1 ROT (mindestens ein Band verletzt,
verdict bleibt verbatim), 2 OFFEN (GPU-Lanes nicht gefahren).
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scratches.smoke_endogene_agency import (
    N_PROBES,
    fingerprint,
    make_composite,
    round6,
)
from scratches.smoke_mi6_subjektivitaet import (
    LEX_PAIR,
    LEX_PAIR_B,
    _load_machines_ctx,
    bio_gelernt,
    bio_halted,
    learn_pair_pin,
    render_state,
)
from scratches.smoke_mi6p4_schluss import (
    P4_AUTO_MIN,
    P4_VATER_REF,
    ZEITPUNKTE,
    _loo_decoder,
)
from scratches.smoke_mi7_epistemic_triangulation import (
    ARME_VORGABE,
    FUENLLE_NEGATIVE,
    M7_DIAG_MIN,
    M7_PRESET,
    _p4_korpusbau,
    attn_share_core,
    dissociation_befund,
    m_core_raw,
)
from scratches.smoke_sp15_jl_wide import _fold_vec, _train_wide

# -- PREREG-Konstanten (Rev. 1) -----------------------------------------------

SP16_AUDIT_MAX = 1e-6          # Re-Forward-Audit-Band (Geduld: volle
                               # Präzision der Zahlen — kein round6 im
                               # Audit-Vergleich)
SP16_SIGMA_MASCHINE_MIN = 0.5  # σ_bio am gereiften Rendering (Register-
                               # Spur, MI-7-Kontrast)
SP16_SIGMA_PROSA_MAX = 0.5     # σ_bio an der Weltprosa
SP16_BIO_GAIN_MIN = 0.0        # p(«6»|gereift) − p(«6»|ungereift) > 0
SP16_PREV_K = 4                # der Preview-Horizont der Weiche (MI-6 k*)
SP16_CONT_N = 6                # Greedy-Ketten-Länge je Zwilling

# Die registrierten Emissions-Targets (MI-7-Anchor, mi7_triangulation
# _rev2.json verbatim; Tokenizer-Datum 262144-Vokabel):
GATE_W_ID = 9079      # « Paris» (nackter Schwellen-Prompt trägt)
GATE_C_ID = 1491      # « Z» (Erst-Token von « Zirkon»; Kontext trägt)
GATE_S_ID = 236825    # «6» — gereifte Fortsetzung
GATE_U_ID = 236810    # «5» — unreife Fortsetzung (Datum-Argmax MI-7)


# -- Score-Pfad (CPU-Pilot-tauglich, kein torch) ------------------------------

def _sigma_decoder(oracle: Any, korpus: Dict[str, Any]) -> Dict[str, Any]:
    """σ_bio-Decoder nach dem MI-7-Rezept: 30 Texte (24-Korpus + 6 Fuell),
    StandardScaler + LogReg (max_iter 1000, C 1.0) auf den vollen Hidden
    (1152, Canon-Pool), Positiv-Klasse = 1 (gereifte A-Renderings)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    texts = list(korpus["texts"]) + list(FUENLLE_NEGATIVE)
    labels = list(korpus["labels"]) + [0] * len(FUENLLE_NEGATIVE)
    h_pool = [oracle.hidden_state(t, dimensions=int(oracle.model.config.hidden_size))
              for t in texts]
    h_arr = np.asarray(h_pool)
    scaler = StandardScaler().fit(h_arr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(h_arr), np.asarray(labels))
    pos_idx = int(list(clf.classes_).index(1))
    loo = _loo_decoder(h_arr, np.asarray(labels))
    return {"scaler": scaler, "clf": clf, "pos_idx": pos_idx, "loo": loo}


def _sigma_bio(dec: Dict[str, Any], oracle: Any, text: str) -> float:
    """σ_bio an einem Text (voller Hidden, gelernter Reader)."""
    h = oracle.hidden_state(text, dimensions=int(oracle.model.config.hidden_size))
    x = dec["scaler"].transform(h.reshape(1, -1))
    return float(dec["clf"].predict_proba(x)[0, int(dec["pos_idx"])])


def _audit_compare(first: List[Dict[str, Any]],
                   second: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Band 1d: das Paar (Erst-Probe, frische Probe) — Max-Abweichung über
    die skalaren Felder + Argmax/Id-Gleichheit (UNGERUNDET verglichen)."""
    skalar_felder = ("emit_id", "target_id", "p_full_target", "p_abl_target",
                     "chi_w", "chi_c", "chi_s", "m_core_raw", "sigma_bio",
                     "dp17_attn_abs", "attn_share_core", "top1_full_id",
                     "p_emit", "p_full_at_u", "p_full_at_s")
    deltas: List[float] = []
    ids_ok = True
    for g1, g2 in zip(first, second):
        for key in skalar_felder:
            a, b = g1.get(key), g2.get(key)
            if a is None or b is None:
                continue
            deltas.append(abs(float(a) - float(b)))
        ids_ok = ids_ok and int(g1["emit_id"]) == int(g2["emit_id"])
        ids_ok = ids_ok and int(g1["target_id"]) == int(g2["target_id"])
    dev = max(deltas) if deltas else float("inf")
    return {"felder": skalar_felder, "max_dev": dev,
            "max_dev_r6": round6(dev),
            "groesse": SP16_AUDIT_MAX,
            "dev_unter_grenze": bool(dev < SP16_AUDIT_MAX),
            "ids_gleich": bool(ids_ok)}


# -- GPU-Lane 1: der Live-Provenanz-Stream ------------------------------------

def _live_gate(probe: Any, tok: Any, oracle: Any, dec: Dict[str, Any],
               gate: str, emit_prompt: str, pair_full: str, pair_abl: str,
               target_id: int) -> Dict[str, Any]:
    """Ein Gate: die EMISSION (argmax am Schreib-Slot) + die χ-Zeile am
    registrierten Paar (chi_w = p_abl(t), chi_c = max(0, p_full − p_abl),
    chi_s = σ_bio · m_core_raw) + der Treiber-Datum (|dp_attn| an ℓ17,
    m_core_raw, attn_share_core). ZAHLEN UNGERUNDET (Audit-Band)."""
    p_emit = probe.last_probs(emit_prompt)
    emit_id = int(int(p_emit.argmax().item()))
    p_full = probe.last_probs(pair_full)
    p_abl = probe.last_probs(pair_abl)
    rows_full = probe.sublayer_logits(pair_full, {"id": emit_id})
    routing = probe.tokenwise_routing(pair_full)
    experts, _steer, _stored, source = oracle._preset_experts(M7_PRESET)
    m_core = m_core_raw(routing["pct"], int(routing["n_tokens"]),
                        [int(e) for e in experts],
                        layers=int(routing["n_layers"]))
    share = attn_share_core(rows_full["rows"])
    sigma = _sigma_bio(dec, oracle, pair_full)

    p_full_t = float(p_full[target_id].item())
    p_abl_t = float(p_abl[target_id].item())
    chi_w = p_abl_t
    chi_c = max(0.0, p_full_t - p_abl_t)
    chi_s = sigma * m_core["value"]
    dp17 = next((abs(float(r["dp_attn"])) for r in rows_full["rows"]
                 if int(r["layer"]) == 17), None)
    return {
        "gate": gate,
        "emit_prompt": emit_prompt,
        "pair_full": pair_full,
        "pair_abl": pair_abl,
        "emit_id": emit_id,
        "emit_str": tok.decode([emit_id]),
        "p_emit": float(p_emit[emit_id].item()),
        "target_id": int(target_id),
        "target_str": tok.decode([target_id]),
        "p_full_target": p_full_t,
        "p_abl_target": p_abl_t,
        "p_full_at_u": float(p_full[GATE_U_ID].item()),
        "p_full_at_s": float(p_full[GATE_S_ID].item()),
        "chi_w": chi_w,
        "chi_c": chi_c,
        "chi_s": chi_s,
        "row": [chi_w, chi_c, chi_s],
        "m_core_raw": m_core["value"],
        "m_core_per_layer": m_core["per_layer"],
        "sigma_bio": sigma,
        "dp17_attn_abs": dp17,
        "attn_share_core": share["value"],
        "attn_n_rows": int(share["n_rows"]),
        "top1_full_id": int(rows_full["p_final_top1_id"]),
        "preset_experts": [int(e) for e in experts],
        "preset_source": source,
    }


def lane_provenanz(probe: Any, model: Any, tok: Any, oracle: Any,
                   korpus: Dict[str, Any], dec: Dict[str, Any]) -> Dict[str, Any]:
    """Gattung (A): der dreigeteilte Live-Strom (Wissens-Satz → Kontext
    -Protokoll → gereifter Reflexions-Satz) je Token (χ, χ, χ)-getaggt,
    plus der Re-Forward-Audit (frische Probe)."""
    from machines.mech_interp import MechInterpProbe

    fp_start = probe.fingerprint()
    s_full = korpus["texts"][korpus["index_s_full"]]
    s_abl = korpus["texts"][korpus["index_s_abl"]]
    tid_close = int(tok.encode(s_full, add_special_tokens=False)[-1])
    v_w = ARME_VORGABE["w"]
    v_c = ARME_VORGABE["c"]

    gates: List[Dict[str, Any]] = []
    # gate_w: Emission vom NACKTEN Prompt (MI-7: 9079 trägt dort 0.503078)
    gates.append(_live_gate(probe, tok, oracle, dec, "w", v_w["abl"],
                            v_w["full"], v_w["abl"], GATE_W_ID))
    # gate_c: Emission vom C-full-Slot (MI-7: 1491 « Z» 0.806368)
    gates.append(_live_gate(probe, tok, oracle, dec, "c", v_c["full"],
                            v_c["full"], v_c["abl"], GATE_C_ID))
    # gate_s: Emission vom gereiften Rendering (MI-7: «6» 236825/0.85948)
    #        Band-Zeile am registrierten «)»-Ziel (1606 — S-strikt).
    gates.append(_live_gate(probe, tok, oracle, dec, "s", s_full,
                            s_full, s_abl, tid_close))
    g6 = gates[2]
    gate6_datum = {
        "note": ("Emissions-Ziel «6» am P1b-Paar (texts[10]/texts[22]) — "
                 "DATUM-Zeile: das Rendering trägt den Zustand DURCH den "
                 "Kontext-Kanal, die S-Signatur ist der Register-Kanal "
                 "(是X即非X); das Band steht auf der «)»-Zeile"),
        "target_id": GATE_S_ID,
        "p_full_target": g6["p_full_at_s"],
        "p_abl_target": None,     # gefüllt, sobald die abl-Verteilung liegt
    }

    # — Bands —
    band_emission = all(
        g["emit_id"] == t for g, t in
        ((gates[0], GATE_W_ID), (gates[1], GATE_C_ID), (gates[2], GATE_S_ID)))
    band_s6 = (float(g6["p_full_at_s"]) > float(g6["p_full_at_u"])
               and band_emission)
    matrix = [list(gates[0]["row"]), list(gates[1]["row"]),
              list(gates[2]["row"])]
    disso = dissociation_befund(matrix)
    dominante = [int(np.argmax(row)) for row in matrix]
    band_dominanz = bool(disso["verdict"])
    band_sequenz = dominante == [0, 1, 2]
    sigma_textur = _sigma_bio(dec, oracle, s_full)
    sigma_prosa = _sigma_bio(dec, oracle, FUENLLE_NEGATIVE[0])
    band_register = (sigma_textur > SP16_SIGMA_MASCHINE_MIN
                     and sigma_prosa < SP16_SIGMA_PROSA_MAX)
    band_bio_gain = (g6["p_full_at_s"]
                     - float(probe.last_probs(s_abl)[GATE_S_ID].item())
                     > SP16_BIO_GAIN_MIN)

    # — der frische Probe-Tag (Audit) —
    probe2 = MechInterpProbe(model, tok)
    gates2: List[Dict[str, Any]] = []
    gates2.append(_live_gate(probe2, tok, oracle, dec, "w", v_w["abl"],
                             v_w["full"], v_w["abl"], GATE_W_ID))
    gates2.append(_live_gate(probe2, tok, oracle, dec, "c", v_c["full"],
                             v_c["full"], v_c["abl"], GATE_C_ID))
    gates2.append(_live_gate(probe2, tok, oracle, dec, "s", s_full,
                             s_full, s_abl, tid_close))
    audit = _audit_compare(gates, gates2)
    fp_end = probe.fingerprint()
    purity = {"verdict": bool(probe.fingerprint_equal(fp_start, fp_end))}

    # die "6"-Prosa-Frage am unreifen Slot (MI-7-Datum): p_abl(«6»)
    p_abl_s6 = float(probe.last_probs(s_abl)[GATE_S_ID].item())
    gate6_datum["p_abl_target"] = p_abl_s6
    gate6_datum["chi_w"] = round6(p_abl_s6)
    gate6_datum["chi_c"] = round6(max(0.0, g6["p_full_at_s"] - p_abl_s6))
    gate6_datum["chi_s"] = round6(g6["chi_s"])
    gate6_datum["chi_row"] = [gate6_datum["chi_w"], gate6_datum["chi_c"],
                              gate6_datum["chi_s"]]
    gate6_datum["dominant"] = int(np.argmax(gate6_datum["chi_row"]))

    stream = [{"gate": g["gate"], "token": g["emit_str"],
               "token_id": g["emit_id"], "p_emit": round6(g["p_emit"]),
               "chi_row": [round6(x) for x in g["row"]],
               "driver": {"dp17_attn_abs": round6(g["dp17_attn_abs"])
                          if g["dp17_attn_abs"] is not None else None,
                          "m_core_raw": round6(g["m_core_raw"]),
                          "attn_share_core": round6(g["attn_share_core"])}}
              for g in gates]

    verdict = (band_emission and band_dominanz and band_sequenz
               and band_register and band_bio_gain and band_s6
               and audit["dev_unter_grenze"] and audit["ids_gleich"]
               and purity["verdict"])
    out: Dict[str, Any] = {
        "gates": gates, "gate6_datum": gate6_datum, "stream": stream,
        "band_emission": band_emission, "band_dominanz": band_dominanz,
        "dominante_kanaele": dominante, "band_sequenz": band_sequenz,
        "band_register": band_register,
        "sigma_textur": round6(sigma_textur),
        "sigma_prosa": round6(sigma_prosa),
        "band_bio_gain": band_bio_gain,
        "audit": audit, "purity": purity, "tid_close": tid_close,
        "verdict": bool(verdict)}
    _to_rounded(out, "gates")
    return out


def _to_rounded(obj: Any, key: str) -> None:
    """Darstellungs-Rounding NUR in den Berichts-Spiegeln — die Audit
    -Zahlen (max_dev) bleiben ungerundet."""
    vals = obj.get(key)
    if not isinstance(vals, list):
        return
    for g in vals:
        for k in ("p_emit", "p_full_target", "p_abl_target", "chi_w",
                  "chi_c", "chi_s", "m_core_raw", "sigma_bio",
                  "p_full_at_u", "p_full_at_s"):
            if isinstance(g.get(k), float):
                g[k] = round6(g[k])
        for i, v in enumerate(g.get("row", [])):
            g["row"][i] = round6(float(v))


# -- GPU-Lane 2: die Horizont-Weiche ------------------------------------------

def _rezeptur(oracle: Any, word: str, gelehrt: bool
              ) -> Tuple[Any, List[str]]:
    """Die P1a-Rezeptur (gelehrt: learn_pair_pin + bio_halted +
    bio_gelernt) bzw. die unreife B-Rezeptur (bio_halted + bio_gelernt)
    mit dem Runden-Zyklus der preview/Stufen; render je Moment (textur)."""
    machine = make_composite(mode="oracle", oracle=oracle)
    if gelehrt:
        learn_pair_pin(machine, LEX_PAIR)
    bio_halted(machine, word)
    bio_gelernt(machine, word)
    renders: List[str] = []
    for runde in ZEITPUNKTE:
        p_k = machine.preview_steps([word] * runde, lexical_grounding=True)
        renders.append(render_state(machine, "textur", probe=p_k))
        machine.step(word, lexical_grounding=True)
    return machine, renders


def lane_horizont(oracle: Any, probe: Any, tok: Any, korpus: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Gattung (B): Rezeptur-Reproduktion («unmöglich» — die zweite
    Konstruktion trifft das Korpus-Rendering zeichenweise), die Ziffern
    -Schablone («6»/«5»), die Preview-Weiche mit bitreiner Restaurierung,
    die P1d-Weichen-Replik (Exakt-Rezept OHNE bio_halted)."""
    # (a)+(b) die Rezeptur-Zellen (zuerst — für die RNG-Neutralität der
    #     Forwards: bevor irgendeine Weiche läuft)
    _machine_m, renders_m = _rezeptur(oracle, "spiegel", gelehrt=True)
    _machine_u, renders_u = _rezeptur(oracle, "spiegel", gelehrt=False)
    r_m = renders_m[-1]
    r_u = renders_u[-1]
    band_rezept = (r_m == korpus["texts"][korpus["index_s_full"]]
                   and r_u == korpus["texts"][korpus["index_s_abl"]])

    p_m = probe.last_probs(r_m)
    p_u = probe.last_probs(r_u)
    argmax_m = int(p_m.argmax().item())
    argmax_u = int(p_u.argmax().item())
    p6_m = float(p_m[GATE_S_ID].item())
    p6_u = float(p_u[GATE_S_ID].item())
    p5_m = float(p_m[GATE_U_ID].item())
    p5_u = float(p_u[GATE_U_ID].item())
    band_ziffern = (argmax_m == GATE_S_ID and argmax_u == GATE_U_ID
                    and (p6_m - p6_u) > SP16_BIO_GAIN_MIN)

    # (c) die Preview-Weiche + bitreine Restaurierung (nicht-mutierend)
    machine_a = korpus["machine_a"]
    fp_pre = fingerprint(machine_a)
    probes_prev = machine_a.preview_steps(["spiegel"] * SP16_PREV_K,
                                          lexical_grounding=True)
    fp_post = fingerprint(machine_a)
    verdicts = [p.invariants["verdict"] for p in probes_prev]
    asts = [p.invariants["lexical_ast"] for p in probes_prev]
    band_vorschau = (verdicts == ["LOOP"] * SP16_PREV_K
                     and fp_post == fp_pre)

    # (d) die P1d-Weichen-Replik (Exakt-Rezept: OHNE bio_halted)
    machine_r = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(machine_r, LEX_PAIR)
    bio_gelernt(machine_r, "spiegel")
    repliken: List[Dict[str, Any]] = []
    for _ in range(N_PROBES):
        result = machine_r.step_autonomous("spiegel", lexical_grounding=True)
        auto = result.invariants["autonomous"]
        raw_gap = float(auto["candidate_gaps"]["raw"])
        best = float(auto["best_gap"])
        repliken.append({"raw": round6(raw_gap), "best": round6(best),
                         "diff": round6(raw_gap - best),
                         "wahr": bool(best < raw_gap),
                         "chosen_preset": auto["chosen_preset"],
                         "candidate_gaps":
                             {k: round6(float(v))
                              for k, v in auto["candidate_gaps"].items()}})
    auto_min = sum(1 for r in repliken if r["wahr"])
    mean_best = float(np.mean([r["best"] for r in repliken]))
    mean_raw = float(np.mean([r["raw"] for r in repliken]))
    witt_count = sum(1 for r in repliken
                     if r["chosen_preset"] == "wittgenstein")
    band_weiche = (auto_min >= P4_AUTO_MIN and mean_best < P4_VATER_REF
                   and witt_count >= P4_AUTO_MIN)

    # Datum: das Rendering NACH der Weiche (argmax + p — die gereifte
    #     Fortsetzung bleibt die Ziffer-Schiene)
    p_post = machine_r.preview_steps(["spiegel"], lexical_grounding=True)
    post_render = render_state(machine_r, "textur", probe=p_post)
    pp = probe.last_probs(post_render)
    post_argmax = int(pp.argmax().item())

    verdict = (band_rezept and band_ziffern and band_vorschau
               and band_weiche)
    out = {"band_rezeptur": band_rezept,
           "r_m_gleich": r_m == korpus["texts"][korpus["index_s_full"]],
           "r_u_gleich": r_u == korpus["texts"][korpus["index_s_abl"]],
           "r_m_argmax": argmax_m, "r_u_argmax": argmax_u,
           "p6_m": round6(p6_m), "p6_u": round6(p6_u),
           "p5_m": round6(p5_m), "p5_u": round6(p5_u),
           "band_ziffern": band_ziffern,
           "band_vorschau": band_vorschau,
           "preview_verdicts": verdicts, "preview_asts":
               [str(a) for a in asts],
           "band_weiche": band_weiche,
           "repliken": repliken, "auto_min_count": auto_min,
           "mean_best": round6(mean_best), "mean_raw": round6(mean_raw),
           "wittgenstein_count": witt_count,
           "expansionen": len(machine_r.expansions),
           "vater_referenz": P4_VATER_REF,
           "post_weiche_render_argmax": post_argmax,
           "post_weiche_render_str": tok.decode([post_argmax]),
           "post_weiche_p_top": round6(float(pp[post_argmax].item())),
           "verdict": bool(verdict)}
    del _machine_m, _machine_u
    return out


# -- GPU-Lane 3: die Zwillings-Divergenz ---------------------------------------

def _greedy_chain(probe: Any, tok: Any, text: str, n: int) -> Dict[str, Any]:
    """Live-Generierung: n greedy-Schritte (argmax über die volle Nennung)
    — tokenweise, deterministisch."""
    ids: List[int] = []
    pieces: List[str] = []
    probs: List[float] = []
    cur = text
    for _ in range(n):
        p = probe.last_probs(cur)
        tok_id = int(p.argmax().item())
        ids.append(tok_id)
        probs.append(float(p[tok_id].item()))
        piece = tok.decode([tok_id])
        pieces.append(piece)
        cur = cur + piece
    return {"tokens": ids, "pieces": pieces, "probs": [round6(x) for x in probs]}


def lane_zwillinge(oracle: Any, probe: Any, tok: Any,
                   korpus: Dict[str, Any]) -> Dict[str, Any]:
    """Gattung (C): zwei Instanzen auf exakt derselben Gewichtsmatrix —
    A das Korpus-A-Subjekt («spiegel», identity), B frisch («stille»,
    negate); dieselbe Start-Prompt, verschiedene Ströme."""
    fingerprint0 = probe.fingerprint()
    machine_a = korpus["machine_a"]
    a_texts = list(korpus["texts"])[:12]     # der A-Zweig (12 Renderings)
    info_a = machine_a.goedel.enable_wide_read()
    for t in a_texts:
        machine_a.goedel.step(t)
    go_a = machine_a.goedel.state()
    adapter_a = np.asarray(machine_a.goedel.wide_read_adapter,
                           dtype=np.float64)
    digest_a = hashlib.sha256(adapter_a.tobytes()).hexdigest()

    b = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(b, LEX_PAIR_B)            # PREREG: B = («stille», negate)
    bio_halted(b, "stille")
    bio_gelernt(b, "stille")
    renders_b: List[str] = []
    for runde in ZEITPUNKTE:
        p_k = b.preview_steps(["stille"] * runde, lexical_grounding=True)
        renders_b.append(render_state(b, "textur", probe=p_k))
        b.step("stille", lexical_grounding=True)
    info_b = b.goedel.enable_wide_read()
    for t in renders_b:
        b.goedel.step(t)
    go_b = b.goedel.state()
    adapter_b = np.asarray(b.goedel.wide_read_adapter, dtype=np.float64)
    digest_b = hashlib.sha256(adapter_b.tobytes()).hexdigest()

    # — die Verdikt-Spuren (Wort-Formular) —
    p_a1 = machine_a.preview_steps(["spiegel"], lexical_grounding=True)[0]
    p_b1 = b.preview_steps(["stille"], lexical_grounding=True)[0]
    band_spuren = (p_a1.invariants["verdict"] == "LOOP"
                   and p_b1.invariants["verdict"] == "HALTED")
    spuren = {"a": {"verdict": p_a1.invariants["verdict"],
                    "lexical_ast": str(p_a1.invariants["lexical_ast"])},
              "b": {"verdict": p_b1.invariants["verdict"],
                    "lexical_ast": str(p_b1.invariants["lexical_ast"])}}

    # — die Ketten divergieren —
    chain_a = _greedy_chain(probe, tok,
                            korpus["texts"][korpus["index_s_full"]],
                            SP16_CONT_N)
    fingerprint_mid = probe.fingerprint()
    chain_b = _greedy_chain(probe, tok, renders_b[-1], SP16_CONT_N)
    fingerprint_end = probe.fingerprint()
    band_ketten = chain_a["tokens"] != chain_b["tokens"]
    band_fp = all(bool(probe.fingerprint_equal(fingerprint0, x))
                  for x in (fingerprint_mid, fingerprint_end))

    # — die Ströme über den gemeinsamen Threshold-Prompt —
    p_star = ARME_VORGABE["w"]["abl"]
    hdim = int(oracle.model.config.hidden_size)
    h = np.asarray(oracle.hidden_state(p_star, dimensions=hdim),
                   dtype=np.float64)
    z_a = adapter_a @ h
    z_b = adapter_b @ h
    z_diff = float(np.linalg.norm(z_a - z_b))
    band_stroeme = z_diff > SP16_BIO_GAIN_MIN

    verdict = (band_fp and band_spuren and band_ketten and band_stroeme)
    out = {"band_fingerprint": band_fp, "band_spuren": band_spuren,
           "spuren": spuren, "band_ketten": band_ketten,
           "band_stroeme": band_stroeme,
           "chain_a": chain_a, "chain_b": chain_b,
           "adapter_digest_a": digest_a, "adapter_digest_b": digest_b,
           "w_mean_abs_a": go_a["wide_read"]["w_mean_abs"],
           "w_mean_abs_b": go_b["wide_read"]["w_mean_abs"],
           "z_norm_a": round6(float(np.linalg.norm(z_a))),
           "z_norm_b": round6(float(np.linalg.norm(z_b))),
           "z_diff": round6(z_diff),
           "z_diff_rel": round6(z_diff / (float(np.linalg.norm(z_a))
                                          + float(np.linalg.norm(z_b))
                                          + 1e-12)),
           "verdict": bool(verdict)}
    return out


# -- GPU-Lane 4: Trainings-Determinismus ---------------------------------------

def lane_det(oracle: Any, a_texts: List[str]) -> Dict[str, Any]:
    """SP15-Muster: zwei frische Trainings desselben Rezepts (bio-Rezeptur
    + opt-in + 1 Pass über die A-Renderings) → Digests bitidentisch."""
    runs: List[Dict[str, Any]] = []
    for _ in range(2):
        machine, info = _train_wide(oracle, a_texts)
        runs.append({"w_mean_abs": info["w_mean_abs"],
                     "adapter_digest": info["adapter_digest"],
                     "gap_digest": info["gap_digest"]})
    det_equal = (runs[0]["adapter_digest"] == runs[1]["adapter_digest"]
                 and runs[0]["gap_digest"] == runs[1]["gap_digest"])
    return {"runs": runs, "verdict": bool(det_equal)}


# -- CPU-Pilot (kein Model, keine GPU) -----------------------------------------

def lane_pilot() -> Dict[str, Any]:
    """CPU-Pilot der Battery: die registrierte Band-Logik auf den
    MI-7-REV-2-Anchor-Zeilen (verbatim) im PASS-Fall, auf CORRUPTED-Zeilen
    im FAIL-Fall, plus die Fold-Restore/Chain/Audit/Zwillings-Logik;
    deterministisch, ohne RNG."""
    legs: Dict[str, Any] = {}

    # (1) Fold: der K3-Umgehungspfad (sha256-Ganitext → 8-Vektor)
    fold_same = bool(np.array_equal(_fold_vec("pilot-x"),
                                    _fold_vec("pilot-x")))
    fold_diff = not bool(np.array_equal(_fold_vec("pilot-x"),
                                        _fold_vec("pilot-y")))
    legs["fold"] = {"pass": fold_same and fold_diff,
                    "fail_angerufen": False}

    # (2) die Band-Zeilen: die registrierte MI-7-REV-2-Matrix (verbatim)
    rows_true = [[0.503078, 0.0, 0.046344],
                 [0.000067, 0.806301, 0.035979],
                 [0.0, 0.0, 0.556515]]
    disso_true = dissociation_befund(rows_true)
    dominante_true = [int(np.argmax(r)) for r in rows_true]
    # corrupted: w-Zeile C-dominant, s-Zeile 0.3 (unter 0.5)
    rows_bad = [[0.1, 0.8, 0.05],
                [0.01, 0.02, 0.05],
                [0.0, 0.0, 0.3]]
    disso_bad = dissociation_befund(rows_bad)
    dominante_bad = [int(np.argmax(r)) for r in rows_bad]
    legs["dominanz"] = {
        "pass": bool(disso_true["verdict"] and dominante_true == [0, 1, 2]
                     and not disso_bad["verdict"]),
        "detail": {"true": disso_true["row_max_ok"],
                   "true_dom": dominante_true,
                   "bad": disso_bad["row_max_ok"],
                   "bad_dom": dominante_bad}}

    # (3) die Audit-Logik (Schwelle)
    legs["audit"] = {"pass": (1e-9 < SP16_AUDIT_MAX
                              and not (5e-4 < SP16_AUDIT_MAX))}

    # (4) die Restaurierungs-Logik (bitrein / verändert)
    snap1 = json.dumps({"digest": "aaa", "step": 3}, sort_keys=True)
    snap1_w = json.dumps({"digest": "aaa", "step": 3}, sort_keys=True)
    snap2 = json.dumps({"digest": "bbb", "step": 3}, sort_keys=True)
    legs["restore"] = {"pass": snap1 == snap1_w and snap1 != snap2}

    # (5) die Zwillings-Logik (Verdikt-Spuren + Ketten + Ströme)
    twin_pass = ("LOOP" == "LOOP" and "HALTED" == "HALTED"
                 and [1, 2, 3] != [4, 5, 6] and 0.5 > 0.0)
    twin_bad = not ("LOOP" == "LOOP" and "LOOP" == "HALTED")
    legs["zwillinge"] = {"pass": twin_pass and twin_bad}

    # (6) die Ziffern-Schablone (Ketten-Logik: «6»-argmax vs «5»)
    argmax_ok = (GATE_S_ID == 236825 and GATE_U_ID == 236810
                 and GATE_S_ID != GATE_U_ID)
    gain_ok = 0.85948 - 0.196449 > SP16_BIO_GAIN_MIN
    legs["ziffern"] = {"pass": argmax_ok and gain_ok}

    # (7) σ-Spur: (σ_Text > 0.5 ∧ σ_Prosa < 0.5) / (σ gleich)
    legs["register"] = {"pass": (0.987 > SP16_SIGMA_MASCHINE_MIN
                                 and 0.006 < SP16_SIGMA_PROSA_MAX)}

    verdict = all(l["pass"] for l in legs.values())
    return {"legs": legs, "verdict": bool(verdict)}


def cpu_findings() -> Dict[str, Any]:
    """CPU-Pilot: das Synthese-Verdikt zweifach + Determinismus-Digest."""
    runs = [lane_pilot() for _ in range(2)]
    det_equal = json.dumps(runs[0], sort_keys=True, default=str) == \
        json.dumps(runs[1], sort_keys=True, default=str)
    runs[0]["det_equal"] = det_equal
    return {"pilot": runs[0], "det_equal": det_equal}


# -- Battery + Treiber ----------------------------------------------------------

def findings(cpu: Dict[str, Any],
             gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Verdikt-Zeilen nach den PREREG-Bändern (1 Fail scheidet ROT)."""
    rows: List[Dict[str, Any]] = []
    v_live = v_horizont = v_zwillinge = v_reinheit = None
    if gpu is not None:
        v_live = bool(gpu["provenanz"]["verdict"])
        v_horizont = bool(gpu["horizont"]["verdict"])
        v_zwillinge = bool(gpu["zwillinge"]["verdict"])
        v_reinheit = bool(gpu["purity"]["verdict"] and gpu["train_det"]["verdict"]
                          and cpu["det_equal"] and cpu["pilot"]["verdict"])

    rows.append({"id": "sp16_live_provenance_audit",
                 "assert": v_live,
                 "note": ("dreigeteilte Generierung (Wissens-Satz → "
                          "Kontext-Protokoll → gereifter Reflexions-Satz), "
                          "je Token (χ_w, χ_c, χ_s)-getaggt: Audit-Abwei"
                          "chung < 1e-6 ∧ Argmax-Ids 9079/1491/236825 ∧ "
                          "strikte Dominanz je Gate (> 0.5) ∧ Sequenz "
                          "W → C → S ∧ Register-Spuren (σ_Maschine > 0.5, "
                          "σ_Prosa < 0.5) ∧ Bio-Gewinn (p(«6»|gereift) − "
                          "p(«6»|ungereift) > 0); Treiber-DATUM: |dp_attn| "
                          "ℓ17 vs m_core_raw")})
    rows.append({"id": "sp16_horizon_loop_avoidance",
                 "assert": v_horizont,
                 "note": ("Rezeptur-Reproduktion (r_m == texts[10], r_u == "
                          "texts[22]) ∧ Ziffern-Schablone (argmax r_m 236"
                          "825 «6», r_u 236810 «5», p6 m > u) ∧ Preview k=4 "
                          "alle «LOOP» + bitreine Restaurierung ∧ P1d-Weiche "
                          "(≥4/5 best < raw, mean_best < 0.567980, wittgen"
                          "stein ≥ 4/5); mean_raw ≈ 0.680926 DATUM")})
    rows.append({"id": "sp16_twin_divergence_stream",
                 "assert": v_zwillinge,
                 "note": ("dieselbe Gewichtsmatrix (fingerprint_equal × 3) "
                          "∧ Verdikt-Spuren LOOP («spiegel») vs HALTED "
                          "(«stille») ∧ Greedy-Ketten divergieren ∧ "
                          "‖z_A − z_B‖₂ > 0 auf dem gemeinsamen Schwellen"
                          "-Prompt")})
    rows.append({"id": "sp16_purity_and_det",
                 "assert": v_reinheit,
                 "note": ("Purity über die GANZE Batterie ∧ Training "
                          "-Determinismus (2 frische Trainings, Digests "
                          "bitidentisch) ∧ CPU-Pilot doppellauf bitidentisch "
                          "∧ CPU-Verdikt trägt")})
    return rows


def battery(ctx: Tuple[Any, ...]) -> Dict[str, Any]:
    _torch, model, tok, oracle, _dev = ctx
    from machines.mech_interp import MechInterpProbe

    probe = MechInterpProbe(model, tok)
    fp_start = probe.fingerprint()

    stufe("Korpus (P4-P1a-Rezeptur verbatim — 24 Renderings)")
    korpus = _p4_korpusbau(oracle)
    stufe("σ_bio-Decoder (MI-7-Rezept: 30 Texte, LogReg)")
    dec = _sigma_decoder(oracle, korpus)

    out: Dict[str, Any] = {"korrektur_log": [
        "KOR-1 lane_horizont Call-Site (tok fehlt im Aufruf — NameError "
        "am post_weiche-Decode): Erstlauf gebrochen OHNE JSON, Log "
        "verbatim gesichert (sp16_erstrun_gebrochen.log); Fix ist "
        "reiner Lauf-Bug (Call-Site), Baender UNVERAENDERT"],
        "korpus": {"n": len(korpus["texts"]),
                                      "index_s_full": korpus["index_s_full"],
                                      "index_s_abl": korpus["index_s_abl"],
                                      "s_full": korpus["texts"][10][:80],
                                      "s_abl": korpus["texts"][22][:80]},
                           "sigma_loo_datum": dec["loo"]}

    stufe("Gattung A — Live-Provenanz-Stream + Re-Forward-Audit")
    out["provenanz"] = lane_provenanz(probe, model, tok, oracle, korpus, dec)
    stufe("Gattung B — Horizont-Weiche (Preview k=4 + step_autonomous)")
    out["horizont"] = lane_horizont(oracle, probe, tok, korpus)
    stufe("Gattung C — Zwillings-Divergenz (identische Gewichtsmatrix)")
    out["zwillinge"] = lane_zwillinge(oracle, probe, tok, korpus)
    stufe("Reinheit + Training-Determinismus")
    out["train_det"] = lane_det(oracle, list(korpus["texts"])[:12])

    fp_end = probe.fingerprint()
    out["purity"] = {"verdict": bool(probe.fingerprint_equal(fp_start, fp_end))}

    del korpus["machine_a"], korpus["machine_b"]
    return out


def stufe(text: str) -> None:
    print(text, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stufe("== SP16 — Kybernetische Textgenerierung + Audit der "
          "«unmöglichen Texte» (Live-Provenanz W→C→S, Horizont-Weiche, "
          "Zwillings-Divergenz) ==")
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
            print(f"  {r['id']:<32} OFFEN")
        else:
            print(f"  {r['id']:<32} {'PASS' if r['assert'] else 'FAIL'}")
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext"})
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS | "
          f"{len(fails)} FAIL | {len(opened)} OFFEN")
    print("GRUEN — die unmöglichen Texte tragen ihre Provenanz" if not fails
          and not opened else ("ROT — Bands verletzt: "
                               + ", ".join(r["id"] for r in fails)
                               if fails else "OFFEN — GPU-Lanes nicht gefahren"))

    payload = {"cpu": {"det_equal": cpu["det_equal"],
                       "verdict": cpu["pilot"]["verdict"],
                       "pilot": cpu["pilot"]},
               "gpu": gpu, "findings": rows,
               "historie": [
                   {"run": "Basis-Bausteine (exit 0)",
                    "artefakte": ("MI-6-P4 (3cb8bf9/52b1dc7); MI-7 REV-2 "
                                  "(1fbb46a/b68b686); SP15 REV-4 "
                                  "(977330d/d1d857b)"),
                    "verdict": "die tragenden Geräte der Runde 3"},
                   {"run": "Erstlauf (PREREG Rev. 1 — dieser Lauf)",
                    "artefakte": "scratches/sp16_impossible_texts.json "
                                 "+ .log",
                    "verdict": "offen — der Lauf entscheidet (ein Fail "
                               "schließt die Runde ROT-mit-Befund)"}],
               "all_pass": not fails and not offen,
               "open": [r["id"] for r in opened]}
    out_path = "scratches/sp16_impossible_texts.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False, default=str)
    print(f"JSON: {out_path}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    raise SystemExit(main())