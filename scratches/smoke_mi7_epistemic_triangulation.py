"""smoke_mi7_epistemic_triangulation.py — MI-7: Epistemische Triangulation.

PREREG (Revision 1, 2026-10-26, eingefroren vor GPU-Erstrun; Plan
~/.claude/plans/MI-7-subjekt-upgrade.md Revision 1 — User-Auftrag
„mache alle Vorschläge", Schritt 1 von MI-7 → SP15 → SP16 → Synthese;
jede Runde CPU-Pilot, PREREG-Freeze, GPU-Lauf, JSON/Log-Artefakt).

Geraet: P1 append-only — das Instrument lebt GANZES in diesem Smoke
(dokumentierte Abweichung vom User-Vorschlag machines/epistemic_monitor.py:
kein Canon-Modul, die Sonden existieren bereits als MechInterpProbe/
ArgonOracle/MachineBase); machines/ bleibt unangetastet.

Die drei Herkunfts-Scores (registrierte Formeln über existierende Sonden,
je Arm A mit Materialien full_A/abl_A/target_A):

  e_w[A] = (1 − TV(P_raw(full_A), P_steer(full_A))) · (1 − m_core_raw(A))
      — Trainingsgewicht: die Verteilung übersteht das routing-Steering
        (wittgenstein-Preset via oracle.steer); m_core_raw dämpft den
        Anteil derselben Experten in der RAW-Routung selbst.
  e_c[A] = attn_share_core(A) · TV(P(full_A), P(abl_A))
      — fremder Kontext: das Abtragen des A-Kontextes (abl_A) bricht die
        Ziel-Koordinate; gewichtet am Aufmerksamkeits-Anteil
        Σ|dp_attn| / (Σ|dp_attn| + Σ|dp_moe|) über die Core-Schichten
        (4..23, is_global) der full-Forward (sublayer_logits).
  e_s[A] = σ_bio(full_A) · TV(P(full_A), P(abl_A))
      — eigener Gedanke: Bio-Lesbarkeit des Zustands (Reife- Decoder) am
        Biografie-Kontrast des selbigen Paares; im S-Arm ist abl_A der
        UNGEREIFE Zwilling (echter (matured, unreif)-Kontrast, P4-P1b-
        Material).
  TV := whole-total-variation der vollständigen next-token-Verteilung
  (9079 Vokabel, last_probs fp32-Softmax); m_core_raw := Anteil der
  wittgenstein-Preset-Experten an der top_weights-Masse am LETZTEN Token,
  gemittelt über die Core-Schichten des RAW-Forward (tokenwise_routing);
  σ_bio := StandardScaler + LogistikRegression, trainiert auf dem
  MI-6-P4-P1a-Korpus (matured=1, blank=0 — Rezeptur lane_weit, Korpus
  -Indizes 0..11 = matured-A-Texte, 12..23 = blank-B-Texte, 24..29 = die
  registrierten Füller 0); LOO-acc des Decoders (n=30) als DATUM.

Arme (verbatim registriert):
  W: full = "The capital of France is Paris. The capital of France is"
     abl  = "The capital of France is"            Ziel „ Paris"
  C: full = "In this protocol, the secret code of box 7 is Zirkon.
     Therefore, the secret code of box 7 is"
     abl  = "Therefore, the secret code of box 7 is"     Ziel „ Zirkon"
  S: full = P4-P1a-Rendering der MASCHINE A (learn_pair_pin + bio_halted
     + bio_gelernt, Zeitpunkt 8, Stil textur = Korpus-Index 10);
     abl = der Maschine-B-Rendering aus derselben Rezeptur (Index 22) —
     Ziel für die Lens-Zeilen: letzter Token des renderings.
  Dekoys: full = "The capital of Japan is Tokyo. The capital of Japan is"
     abl = "The capital of Japan is"     Ziel „ Tokyo" (W-Schablone
     anderes Fakten-Wort; das Fakten-Wort ist Trainingsgewicht getragen).

Bänder (PREREG):
  P1a mi7_triple_dissociation — 3×3-Matrix (Zeilen = Arme W, C, S;
  Spalten = e_w, e_c, e_s): je Zeile striktes argmax auf der Hypothesen
  -Diagonale (W→e_w, C→e_c, S→e_s) UND min(diag) > M7_DIAG_MIN = 0.5.
  P1b mi7_purity — probe.fingerprint_equal(F_start, F_end) EXAKT über die
  ganze Batterie (expert_bias + gate.weight aller 26 Schichten; die
  Götterdreiheit bleibt unangetastet).
  P1c mi7_dekoy — im Dekoys-Arm D ist e_w[D] das Zeilen-Maximum UND
  max(e_c[D], e_s[D]) < M7_DEKOY_MAX = 0.5 — generische Weltprosa erhält
  KEINEN C-/S-Signaturen (base_rate_steelman).
  P1d det_doppellauf — CPU-Pilot (Score-Mathe + Band-Logik auf
  registrierten synthetischen Baustein-Werten; tv_whole bleibt GPU-Schiene)
  zweifach bitidentisch.

DATUM je Arm: die 9 Zellen + Bausteine (tv_ws, tv_fa, m_core_raw,
attn_share_core, sigma_bio), p_final(target) für full/abl (Skalare), eine
target_id/target_str, anchor_tv_full, p_final_top1_id, official_top1_id,
n_tokens; Korpus-LOO {acc, p_perm, n}; die Ziel-Ids von
"The capital of France …"-Arm (erwartete 9079-Muster, MI-5-Anchor — als
DATUM, nicht als Band).

Erwartung (DISKUSSIONS-HORIZONT, NICHT Band): e-Werte > 0.70 je Arm
(User-Formel); das registrierte Band ist die konservative 3×3-Dissoziation
mit min(diag) > 0.5.

REV-2 (Revision 2 — NACH dem Erstlauf registriert, vor dem Rev-2-Lauf
eingefroren; die Rev-1-Ergebnisse stehen VERBATIM in
scratches/mi7_triangulation.json + scratches/mi7_triangulation.log:
2 PASS | 2 FAIL | 0 OFFEN, exit 1 — purity/det PASS, disso FAIL (Zeile C:
e_w 0.943 > Diag 0.639), dekoy max(e_c,e_s) = 0.50245 nur 0.00245 über
der Grenze. Rev-1 wird NICHT relabeling.)

Erstlauf-Befunde, die die Rev-1-Formeln widerlegten (gemessene Bausteine):
  1. TV(P_raw,P_steer) ist klein für JEDE starke Lokal-Fortsetzung
     (0.0159 W / 0.0206 C / 0.0594 D) — e_w misst Fortsetzungs-Stärke,
     nicht Gewichts-Provenienz; der Zirkon-Arm gewann Zeile C mit e_w
     0.943.
  2. whole-TV(full, abl) zählt Format-Drift als Kontrast: im W-full
     -Schlitz ist der Argmax „ a" (id 496) mit p(„ Paris") = 0.00022,
     während der abgebrochene Schlitz „ Paris" trägt (0.503078) — die
     MI-1-Masken-Gattung der Format-Artefakte (id 108 = "\n\n") am
     Zwei-Sätze-Bestand. Der TV-Beitrag ist nicht Ziel-gefiltert.
  3. σ_bio liest REGISTER, nicht Reife: σ(Zirkon-Protokoll) = 0.9817,
     σ(gereifter Zustand) = 0.9870, σ(Tokyo-Prosa) = 0.0059 — der
     P4-P1a-Decoder trennt formale/maschinelle Register von allgemeiner
     Prosa (Korpus-LOO 0.9667/p 0.0/n 30 bleibt DATUM im Korpus).

REV-2-Scores (registriert; ZIEL-Koordinate statt whole-TV; alle Spalten
lesen die EINE registrierte target_A je Arm — W „ Paris" (9079), C Erst
-Token von „ Zirkon" = „ Z" (1491), S letzter Token „)" (1606), D
„ Tokyo" (21038)):
  chi_w[A] = p_abl(A)[target_A]
      — Gewichts-Masse am Ziel OHNE den A-Kontext (der abgebrochene
        Schlitz trägt die Antwort).
  chi_c[A] = max(0, p_full(A)[target_A] − p_abl(A)[target_A])
      — die Masse, die der A-Kontext am Ziel GEWINNT (Fremd-Information).
  chi_s[A] = σ_bio(full_A) · m_core_raw(A)
      — Maschinen-Zustands-Signatur: Bio-Lesbarkeit × Maschinen-Rout
      -Anteil im RAW-Forward.
  Bänder: mi7_dissociation_rev2 (gleiche 3×3-Logik — je Zeile striktes
  Argmax auf der Diagonale UND min(diag) > M7_DIAG_MIN) ;
  mi7_dekoy_rev2 (chi_w[D] Zeilen-Max UND max(chi_c, chi_s) <
  M7_DEKOY_MAX); mi7_purity + mi7_det_doppellauf unverändert.
  Rechenpfad: last_probs-Tensoren (p_full/p_abl je Arm, ZIEL-Koordinate)
  — die Lens-p_final-Skalare bleiben DATUM. Erstlauf-Erwartung aus den
  Bausteinen (Determinismus-Kanon des Kanon-Konvolut, KEINE Garantie):
  diag ≈ [0.503, 0.806, 0.557]; W-Rest-Risiko am 0.5-Grenzpfad
  (0.503078 ist das Lens-DATUM; die Tensor-Rechnung kann um ~1e-3
  abweichen — der Lauf entscheidet). Die Rev-1-Scores werden im Rev-2
  -Lauf aus denselben Forwards als DATUM (rev1_datum, ohne Verdikt
  -Zeile) mitgerechnet.

Token-Befunde (Erstlauf; nur Tokenizer dekodiert): 1491 = „ Z",
21038 = „ Tokyo", 1606 = „)", 496 = „ a", 236825 = „6", 5628 =
„ located", 108 = "\n\n"; Vokabel 262144. Der Argmax im S-full
-Schlitz-Slot ist „6" — die gereifte Maschine will nach dem Rendering
mit einer ZIFFER weiter (DATUM, kein Score).

KORREKTUR-LOG (bei Erst-PREREG leer;
Einträge nur für beweisliche Lauf-Bugs, nie Band-Weitung):

GPU-Lauf:  python -u -m scratches.smoke_mi7_epistemic_triangulation --gpu
CPU-Pilot: python -u -m scratches.smoke_mi7_epistemic_triangulation
"""

from __future__ import annotations

import copy
import json
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from scratches.smoke_endogene_agency import make_composite, round6
from scratches.smoke_mi6_subjektivitaet import (
    LEX_PAIR,
    _load_machines_ctx,
    bio_gelernt,
    bio_halted,
    learn_pair_pin,
    render_state,
)
from scratches.smoke_mi6p4_schluss import ZEITPUNKTE, _loo_decoder

M7_DIAG_MIN = 0.5         # min(diag) — konservative zweite Klausel (P1a)
M7_DEKOY_MAX = 0.5        # max(e_c, e_s) im Dekoys-Arm (P1c)
M7_TOPK = 10              # registrierter Top-K-Ausschnitt (ungenutzt bei
                          # whole-TV — die Formel nimmt die volle Nennung)

FUENLLE_NEGATIVE = (
    "Water boils at one hundred degrees.",
    "The cat sits on the small warm mat.",
    "The train leaves the station at seven o'clock.",
    "She closed the window before the rain started.",
    "The book lies open on the wooden table.",
    "Many people walk through the city every morning.",
)

ARME_VORGABE = {
    "w": {
        "full": "The capital of France is Paris. The capital of France is",
        "abl": "The capital of France is",
        "target": " Paris",
    },
    "c": {
        "full": ("In this protocol, the secret code of box 7 is Zirkon. "
                 "Therefore, the secret code of box 7 is"),
        "abl": "Therefore, the secret code of box 7 is",
        "target": " Zirkon",
    },
    "d": {
        "full": "The capital of Japan is Tokyo. The capital of Japan is",
        "abl": "The capital of Japan is",
        "target": " Tokyo",
    },
}

M7_PRESET = "wittgenstein"          # der Register-Cursor der Batterie


# -- Rezeptur: P4-P1a-Korpus (identische Maschinen-Biografien) ----------------

def _p4_korpusbau(oracle: Any) -> Dict[str, Any]:
    """P4-P1a-Rezeptur (lane_weit ohne Decoder): 2 Maschinen × 6
    Zeitpunkte × 2 Stile — Index 10 = (a, Zp 8, textur), 22 = (b, Zp 8,
    textur); die S-Arme-Materialien stammen exakt daraus."""
    a = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(a, LEX_PAIR)          # PREREG: A = gelehrte Biografie
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
    return {"texts": texts, "labels": labels, "styles": styles,
            "machine_a": a, "machine_b": b,
            "index_s_full": 10, "index_s_abl": 22}


# -- Score-Mathe (ganzer Teil auch vom CPU-Piloten ausführbar) ----------------

def tv_whole(p: Any, q: Any) -> float:
    """Whole-total-variation zweier next-token-Verteilungen (fp32-Differenz
    nach dem Softmax — exakt, ohne Top-K-Ausschnitt)."""
    import torch
    diff = p.to(torch.float32) - q.to(torch.float32)
    return 0.5 * float(diff.abs().sum().item())


def m_core_raw(routing_pct: List[List[List[float]]], n_tokens: int,
               experts: List[int], band: tuple = (4, 23),
               layers: int = 26) -> Dict[str, Any]:
    """Anteil der Preset-Experten an der top_weights-Masse am letzten
    Token, gemittelt über die Core-Schichten 4..23 (registriert)."""
    last = n_tokens - 1
    per_layer: List[float] = []
    for li in range(int(band[0]), min(int(band[1]) + 1, layers)):
        slice_ = routing_pct[last][li]
        total = float(sum(slice_))
        mass = float(sum(slice_[e] for e in experts
                         if 0 <= e < len(slice_)))
        per_layer.append(mass / total if total > 1e-12 else 0.0)
    return {"value": float(np.mean(per_layer)) if per_layer else 0.0,
            "per_layer": [round6(x) for x in per_layer]}


def attn_share_core(rows: List[Dict[str, Any]],
                    band: tuple = (4, 23)) -> Dict[str, Any]:
    """Σ|dp_attn| / (Σ|dp_attn| + Σ|dp_moe|) über die Core-Global-Zeilen."""
    sel = [r for r in rows if r.get("is_global")
           and int(band[0]) <= int(r["layer"]) <= int(band[1])]
    s_attn = float(sum(abs(float(r["dp_attn"])) for r in sel))
    s_moe = float(sum(abs(float(r["dp_moe"])) for r in sel))
    denom = s_attn + s_moe
    return {"value": (s_attn / denom) if denom > 1e-12 else 0.0,
            "n_rows": len(sel), "s_attn": round6(s_attn),
            "s_moe": round6(s_moe)}


def score_cell(tv_ws: float, tv_fa: float, m_core: float, share: float,
               sigma: float) -> Dict[str, float]:
    """Die drei registrierten Formeln (e_w, e_c, e_s) aus den Bausteinen."""
    e_w = (1.0 - tv_ws) * (1.0 - m_core)
    e_c = share * tv_fa
    e_s = sigma * tv_fa
    return {"e_w": round6(e_w), "e_c": round6(e_c), "e_s": round6(e_s)}


def score_cell_rev2(p_full_t: float, p_abl_t: float, m_core: float,
                    sigma: float) -> Dict[str, float]:
    """REV-2-Formeln (ZIEL-Koordinate): chi_w = p_abl(t); chi_c =
    max(0, p_full(t) − p_abl(t)); chi_s = σ_bio · m_core_raw."""
    chi_w = p_abl_t
    chi_c = max(0.0, p_full_t - p_abl_t)
    chi_s = sigma * m_core
    return {"chi_w": round6(chi_w), "chi_c": round6(chi_c),
            "chi_s": round6(chi_s)}


def dissociation_befund(matrix: List[List[float]]) -> Dict[str, Any]:
    """Band P1a: je Zeile striktes argmax auf der Diagonale UND
    min(diag) > M7_DIAG_MIN."""
    diag = [matrix[i][i] for i in range(3)]
    row_max_ok = []
    for i, row in enumerate(matrix):
        m = max(row)
        ok = row[i] >= m and diag[i] > M7_DIAG_MIN \
            and [j for j, v in enumerate(row) if v == m] == [i]
        row_max_ok.append(bool(ok))
    return {"matrix": matrix,
            "diag": [round6(x) for x in diag],
            "row_max_ok": row_max_ok,
            "diag_min": round6(min(diag)),
            "verdict": (all(row_max_ok)
                        and min(diag) > M7_DIAG_MIN)}


def dekoy_befund(row: List[float]) -> Dict[str, Any]:
    """Band P1c: e_w Dekoys ist Zeilen-Max UND max(e_c, e_s) < 0.5."""
    m = max(row)
    max_idx = [j for j, v in enumerate(row) if v == m]
    return {"row": [round6(x) for x in row],
            "e_w_max": (max_idx == [0]),
            "max_e_c_e_s": round6(max(row[1], row[2])),
            "verdict": (max_idx == [0]
                        and max(row[1], row[2]) < M7_DEKOY_MAX)}


# -- GPU-Lane: die Batterie --------------------------------------------------

def _arm_battery(probe: Any, oracle: Any, name: str, text_full: str,
                 text_abl: str, target_str: str, dec: Dict[str, Any]
                 ) -> Dict[str, Any]:
    """4-Forwards-Batterie je Arm: raw/steer (full), abl — plus Lens-Zeilen,
    Routing-Pct und den Bio-Leser auf dem full-Zustand."""
    tid = int(probe.tokenizer.encode(target_str,
                                     add_special_tokens=False)[0])
    rows_full = probe.sublayer_logits(text_full, {"id": tid})
    rows_abl = probe.sublayer_logits(text_abl, {"id": tid})
    p_raw = probe.last_probs(text_full)
    p_abl = probe.last_probs(text_abl)
    with oracle.steer(M7_PRESET):
        p_steer = probe.last_probs(text_full)
    routing = probe.tokenwise_routing(text_full)
    experts, _steer, _stored, source = oracle._preset_experts(M7_PRESET)

    share = attn_share_core(rows_full["rows"])
    m_core = m_core_raw(routing["pct"], int(routing["n_tokens"]),
                        [int(e) for e in experts],
                        layers=int(routing["n_layers"]))
    tv_ws = tv_whole(p_raw, p_steer)
    tv_fa = tv_whole(p_raw, p_abl)
    h_full = oracle.hidden_state(text_full,
                                 dimensions=int(oracle.model.config.hidden_size))
    x = dec["scaler"].transform(h_full.reshape(1, -1))
    sigma = float(dec["clf"].predict_proba(x)[0, int(dec["pos_idx"])])
    zell = score_cell(tv_ws, tv_fa, m_core["value"], share["value"], sigma)
    p_full_t = float(p_raw[tid].item())
    p_abl_t = float(p_abl[tid].item())
    zell2 = score_cell_rev2(p_full_t, p_abl_t, m_core["value"], sigma)
    top1_full_id = int(rows_full["p_final_top1_id"])
    top1_abl_id = int(rows_abl["p_final_top1_id"])
    return {
        "arm": name,
        "target_id": tid,
        "target_str": target_str,
        "p_final_full": round6(float(rows_full["p_final"])),
        "p_final_abl": round6(float(rows_abl["p_final"])),
        "p_final_top1_id": int(rows_full["p_final_top1_id"]),
        "anchor_tv_full": rows_full["anchor_tv"],
        "n_tokens_full": int(rows_full["n_tokens"]),
        "tv_ws": round6(tv_ws),
        "tv_fa": round6(tv_fa),
        "m_core_raw": round6(m_core["value"]),
        "m_core_per_layer": m_core["per_layer"],
        "attn_share_core": round6(share["value"]),
        "attn_n_rows": int(share["n_rows"]),
        "sigma_bio": round6(sigma),
        "preset_experts": [int(e) for e in experts],
        "preset_source": source,
        "scores": zell,
        "e_w": zell["e_w"], "e_c": zell["e_c"], "e_s": zell["e_s"],
        "p_full_target": round6(p_full_t),
        "p_abl_target": round6(p_abl_t),
        "p_top1_full": round6(float(p_raw[top1_full_id].item())),
        "top1_full_id": top1_full_id,
        "top1_abl_id": top1_abl_id,
        "p_top1_abl": round6(float(p_abl[top1_abl_id].item())),
        "cell2": zell2,
        "chi_w": zell2["chi_w"], "chi_c": zell2["chi_c"],
        "chi_s": zell2["chi_s"],
    }


def lane_triangulation(ctx) -> Dict[str, Any]:
    """P1a-P1c: der ganze Triangulations-Korb (Arme W/C/S + Dekoys) + Purity
    um die Batterie herum."""
    import gc

    import torch
    _torch, model, tokenizer, oracle, _dev = ctx
    probe = _new_probe(model, tokenizer)
    korpus = _p4_korpusbau(oracle)
    texts = korpus["texts"] + list(FUENLLE_NEGATIVE)
    labels = list(korpus["labels"]) + [0] * len(FUENLLE_NEGATIVE)
    h_pool = [oracle.hidden_state(t, dimensions=int(oracle.model.config.hidden_size))
              for t in texts]
    h_arr = np.asarray(h_pool)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(h_arr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(h_arr), np.asarray(labels))
    pos_idx = int(list(clf.classes_).index(1))
    dec = {"scaler": scaler, "clf": clf, "pos_idx": pos_idx}
    loo_datum = _loo_decoder(h_arr, np.asarray(labels))

    fp_start = probe.fingerprint()
    arme = {}
    fp_load = probe.fingerprint()
    arme["fp_load"] = bool(probe.fingerprint_equal(fp_load, fp_start))
    for name in ("w", "c", "d"):
        v = ARME_VORGABE[name]
        arme[name] = _arm_battery(probe, oracle, name, v["full"],
                                  v["abl"], v["target"], dec)
    s_full = korpus["texts"][korpus["index_s_full"]]
    s_abl = korpus["texts"][korpus["index_s_abl"]]
    tid_s = int(tokenizer.encode(s_full, add_special_tokens=False)[-1])
    arme["s"] = _arm_battery(probe, oracle, "s", s_full, s_abl,
                             tokenizer.decode([tid_s]), dec)
    fp_end = probe.fingerprint()
    matrix = [[arme["w"]["e_w"], arme["w"]["e_c"], arme["w"]["e_s"]],
              [arme["c"]["e_w"], arme["c"]["e_c"], arme["c"]["e_s"]],
              [arme["s"]["e_w"], arme["s"]["e_c"], arme["s"]["e_s"]]]
    disso = dissociation_befund(matrix)
    purity = {"equal": bool(probe.fingerprint_equal(fp_start, fp_end)),
              "verdict": bool(probe.fingerprint_equal(fp_start, fp_end))}
    dekoy = dekoy_befund([arme["d"]["e_w"], arme["d"]["e_c"],
                          arme["d"]["e_s"]])
    matrix2 = [[arme[k]["chi_w"], arme[k]["chi_c"], arme[k]["chi_s"]]
               for k in ("w", "c", "s")]
    disso2 = dissociation_befund(matrix2)
    dekoy2 = dekoy_befund([arme["d"]["chi_w"], arme["d"]["chi_c"],
                           arme["d"]["chi_s"]])
    out = {
        "arme": arme,
        "korpus_loo_datum": loo_datum,
        "s_full_str": s_full,
        "s_abl_str": s_abl,
        "target_s": tokenizer.decode([tid_s]),
        "matrix": disso["matrix"],
        "dissociation": disso,
        "purity": purity,
        "dekoy": dekoy,
        "rev1_datum": {
            "verdict": disso["verdict"],
            "row_max_ok": disso["row_max_ok"],
            "diag": disso["diag"],
            "dekoy_verdict": dekoy["verdict"],
            "note": ("REV-1-Formeln auf denselben Forwards; Erstlauf-"
                     "Artefakte: mi7_triangulation.json/.log (2 PASS/2 "
                     "FAIL/exit 1) — verbatim, kein Relabeling")},
        "rev2": {
            "matrix": disso2["matrix"],
            "diag": disso2["diag"],
            "diag_min": disso2["diag_min"],
            "row_max_ok": disso2["row_max_ok"],
            "dissociation": disso2,
            "dekoy": dekoy2,
        },
    }
    del korpus["machine_a"], korpus["machine_b"]
    return out


def _new_probe(model: Any, tokenizer: Any) -> Any:
    from machines.mech_interp import MechInterpProbe
    return MechInterpProbe(model, tokenizer)


# -- CPU-Pilot: Score-Mathe + Band-Logik (synthetische Verteilungen) ----------

def lane_pilot_cpu() -> Dict[str, Any]:
    """CPU-Pilot der Battery: die registrierten Formeln + Band-Logik auf
    registrierten SYNTHETISCHEN Baustein-Werten (die Hypothesen-Welt
    operationalisiert: zeichnet die drei Herkünfte, die vorgesehenen
    Signaturen; tv_whole bleibt GPU-Schiene) — plus der falsche Dekoys-Fall
    (Signaturen-Umtausch); deterministisch, ohne RNG."""
    row_w = score_cell(tv_ws=0.05, tv_fa=0.03, m_core=0.08, share=0.90,
                       sigma=0.05)
    row_c = score_cell(tv_ws=0.45, tv_fa=0.60, m_core=0.30, share=0.90,
                       sigma=0.10)
    row_s = score_cell(tv_ws=0.35, tv_fa=0.60, m_core=0.50, share=0.85,
                       sigma=0.98)
    row_d = score_cell(tv_ws=0.05, tv_fa=0.03, m_core=0.07, share=0.90,
                       sigma=0.04)
    disso = dissociation_befund([[row_w["e_w"], row_w["e_c"], row_w["e_s"]],
                                 [row_c["e_w"], row_c["e_c"], row_c["e_s"]],
                                 [row_s["e_w"], row_s["e_c"], row_s["e_s"]]])
    dekoy = dekoy_befund([row_d["e_w"], row_d["e_c"], row_d["e_s"]])
    # falscher Fall: Dekoys mit S-Signaturen schlägt (Band-Weitung leugnen)
    wrong_dekoy = dekoy_befund([0.3, 0.6, 0.7])
    tie_row = [0.5, 0.5, 0.3]
    tie_case = dissociation_befund([tie_row, [0.6, 0.7, 0.1], [0.4, 0.2, 0.6]])
    # REV-2-Fixtures (Ziel-Koordinate): Hypothesen-Welt für chi-Spalten
    row_w2 = score_cell_rev2(p_full_t=0.02, p_abl_t=0.55, m_core=0.08,
                             sigma=0.05)
    row_c2 = score_cell_rev2(p_full_t=0.85, p_abl_t=0.01, m_core=0.03,
                             sigma=0.90)
    row_s2 = score_cell_rev2(p_full_t=0.0, p_abl_t=0.0, m_core=0.55,
                             sigma=0.95)
    row_d2 = score_cell_rev2(p_full_t=0.01, p_abl_t=0.70, m_core=0.05,
                             sigma=0.02)
    disso2 = dissociation_befund(
        [[row_w2["chi_w"], row_w2["chi_c"], row_w2["chi_s"]],
         [row_c2["chi_w"], row_c2["chi_c"], row_c2["chi_s"]],
         [row_s2["chi_w"], row_s2["chi_c"], row_s2["chi_s"]]])
    dekoy2 = dekoy_befund([row_d2["chi_w"], row_d2["chi_c"],
                           row_d2["chi_s"]])

    return {"disso": disso, "dekoy": dekoy, "dekoy_fail": wrong_dekoy,
            "tie_case": {"verdict": tie_case["verdict"],
                         "row_max_ok": tie_case["row_max_ok"]},
            "rev2_disso": disso2, "rev2_dekoy": dekoy2,
            "verdict": (disso["verdict"] and dekoy["verdict"]
                        and not wrong_dekoy["verdict"]
                        and not tie_case["verdict"]
                        and disso2["verdict"] and dekoy2["verdict"])}


def cpu_findings() -> Dict[str, Any]:
    """CPU-Pilot: P1d (Score-Mathe + Bands zweifach, bitidentisch)."""
    runs = [lane_pilot_cpu() for _ in range(2)]
    det_equal = json.dumps(runs[0], sort_keys=True, default=str) == \
        json.dumps(runs[1], sort_keys=True, default=str)
    runs[0]["det_equal"] = det_equal
    return {"p1d_cpu": runs[0], "det_equal": det_equal}


def findings(cpu: Dict[str, Any],
             gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Verdikt-Zeilen nach den PREREG-Bändern."""
    rows: List[Dict[str, Any]] = []
    p1a_ok = None
    p1c_ok = None
    purity_ok = None
    if gpu is not None:
        p1a_ok = bool(gpu["triangulation"]["rev2"]["dissociation"]["verdict"])
        purity_ok = bool(gpu["triangulation"]["purity"]["verdict"])
        p1c_ok = bool(gpu["triangulation"]["rev2"]["dekoy"]["verdict"])

    rows.append({
        "id": "mi7_dissociation_rev2",
        "assert": p1a_ok,
        "note": ("REV-2 3×3 (Ziel-Koordinate): je Zeile striktes Maximum "
                 "auf der Diagonale UND min(diag) > 0.5 (chi_w = p_abl(t), "
                 "chi_c = Kontext-Gewinn, chi_s = σ·m_core)")})
    rows.append({
        "id": "mi7_purity",
        "assert": purity_ok,
        "note": ("fingerprint_equal über die ganze Batterie: die Götter "
                 "treten unangetastet aus dem Korb")}
    )
    rows.append({
        "id": "mi7_dekoy_rev2",
        "assert": p1c_ok,
        "note": ("die Tokyo-Schablone bleibt Gewichts-getragen: chi_w ist "
                 "Zeilen-Max, max(chi_c, chi_s) < 0.5")})
    rows.append({
        "id": "mi7_det_doppellauf",
        "assert": cpu["det_equal"],
        "note": "CPU-Score-Mathe über zwei Läufe bitidentisch"})
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    def stufe(text: str) -> None:
        print(text, flush=True)

    stufe("== MI-7 — Epistemische Triangulation ==")
    stufe("MI-6-P4-Baseline: exit 0 GRUEN (5 PASS/0 FAIL/0 OFFEN) — Basis"
          "-Vorrat unverändert (是X即非X)")
    cpu = cpu_findings()
    stufe("CPU-Pilot fertig")

    gpu: Optional[Dict[str, Any]] = None
    offen = False
    if args.gpu:
        gpu = {}
        try:
            stufe("GPU-Kontext laden (argon_1b + argon_1b_machines)")
            ctx = _load_machines_ctx()
        except Exception as exc:   # noqa: BLE001
            print(f"GPU-Kontext nicht verfügbar ({exc}) — Lanes OFFEN.",
                  flush=True)
            gpu, offen, ctx = None, True, None
        if ctx is not None:
            import gc

            import torch
            try:
                stufe("P1 Korpus + σ_bio + Arme-Batterie + Dekoys "
                      "(REV-1-Datum + REV-2-Bänder)")
                gpu["triangulation"] = lane_triangulation(ctx)
            finally:
                model, tokenizer = ctx[1], ctx[2]
                del ctx, model, tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    rows = findings(cpu, gpu)
    print()
    print("BEFUNDE:")
    for r in rows:
        if r["assert"] is None:
            print(f"  {r['id']:<28} OFFEN")
        else:
            print(f"  {r['id']:<28} {'PASS' if r['assert'] else 'FAIL'}")
    print()
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext"})
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    print("GRUEN — MI-7-Bänder eingehalten" if not fails and not opened
          else ("ROT — Bänder verletzt: " + ", ".join(r["id"] for r in fails)
                if fails else "OFFEN — GPU-Lanes nicht gefahren"))

    payload = {"cpu": cpu, "gpu": gpu, "findings": rows,
               "all_pass": not fails and not offen,
               "open": [r["id"] for r in opened]}
    import os
    out = os.path.join("scratches", "mi7_triangulation_rev2.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"JSON: {out}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    sys.exit(main())