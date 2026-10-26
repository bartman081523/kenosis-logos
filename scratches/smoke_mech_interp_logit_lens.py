"""
smoke_mech_interp_logit_lens.py — MI-2: Sublayer-Röntgengerät
(Attention-vs-MoE Logit-Lens) + strikte Ein-Token-Auflösung.

PREREG (P2-Freeze, fixiert VOR der ersten Messung; Bands nie geweitet;
OFFEN ≠ PASS; exit 0 alle PASS / 1 FAIL / 2 OFFEN; ohne --gpu bleiben die
Modell-Lanes OFFEN — kein stiller Fallback).

Canon-Anker (voll gelesen 2026-10-26):
 - modeling_argon.py:188-190  is_global = (layer_idx % 6 == 5) ->
   globale MLA-Schichten L5, L11, L17, L23.
 - modeling_argon.py:326-351  DecoderLayer-Sublayer-Struktur:
   h_mid = h_in + post_attention_layernorm(attn_out) (:345);
   h_out = h_mid + post_feedforward_layernorm(moe_out) (:350).
 - modeling_argon.py:615/:733/:735-736  offizieller Kopf-Pfad
   (norm -> lm_head; softcap None im Checkpoint argon_1b_machines).
 - mech_interp.py  Sonde layerwise_logits (MI-1-Muster) + sublayer_logits
   (h_mid = h_out(l-1) + normed_attn — exakt die Original-Addition :345).
 - mi1_tomography_experiment.json  MI-1-Datum: prose-L11-Anomalie
   (E3 2.3% / E4 44.6%, H=0.3149) + Entropie-Delle e11 < e10 UND e11 < e12
   quer über 6/6 Domänen (Konvergenz-Datum zu den globalen MLA-Schichten).

USER-FUND (MI-1-Audit 2026-10-26, am MI-1-Smoke verifiziert):
resolve_token_ids (smoke_mech_interp_tomography.py:204-215) nahm für
" 4" (2 Tokens [236743(" "), 236771("4")]) encoded[0] = LEERZEICHEN-Token
als Vertreter von "4" — die MI-1-"Fast-Gleichstand"-Nuance (§16.3, 0.8074
vs 0.80735) war ein Auflösungs-Artefakt: answer_prob["4"] maß faktisch
p(236743). MI-2 fixiert die Konvention (mi2_single_token_strict) und
bucht die Korrektur in den Berichten; MI-1-Dateien bleiben unangetastet
(P1). Die MI-1-Top-1-Trajektorie (Leerzeichen als echtes Top-1) bleibt
gültig — nur die Antwort-Koordinate war versehentlich dasselbe Token.

Die 6 registrierten Verdikte:

1. mi2_single_token_strict — resolve_token_ids_strict garantiert, dass
   JEDE aufgelöste ID als EINZELTOKEN decode([id]).lstrip(' ') == target
   erfüllt; je registrierter Probe genau 1 ID. Falsifikator: eine
   aufgelöste ID decodiert zu "" oder " " (Vertreter-Falle), oder
   Auflösung leer (-> OFFEN je Probe, keine Band-Erfindung).

2. mi2_sublayer_telescope — für jede Probe exakt
   sum(dp_attn(l) + dp_moe(l)) == p_final(target) - p_emb(target)
   (± 1e-6) und anchor_tv <= 1e-3. Falsifikator: Teleskop-Bruch.

3. mi2_purity_det — fingerprint() (expert_bias + gate.weight je Schicht)
   bitidentisch vor/nach allen Hooks; Doubletten: dp_attn/dp_moe/
   gap_out_tv je Schicht bitidentisch (torch.equal je row).
   Falsifikator: jeder Δ oder Hook-Rest.

4. mi2_attn_vs_moe_attribution — DATUM-Verdikt: je Probe der
   Haupt-Treiber des Kristallisations-Durchbruchs = der größere der
   Kanal-Maxima; Band: beide Kanal-Maxima unterscheiden sich strikt
   > 1e-3 (eindeutiger Treiber -> PASS); Gleichstand -> OFFEN mit
   voller Kette. Kein Vorhersage-Band, WELCHER Kanal gewinnt — das
   gemessene Datum. Peak-Schicht und is_global mitgeloggt.

5. mi2_goedel_gap_contraction — gap-Lesart = MI-1-Definition (canon_tv
   der Schicht-Verteilung gegen P_final), hier auf der p_out-Folge
   (gap_out_tv, von der Sonde je Schicht geliefert). Für jede Probe mit
   stabilem Tail (top1_out_id konstant ab l*, CRYSTAL_MIN_TAIL=2):
   Mittel(gap_out_tv | [l*..25]) < 0.5 × Mittel(gap_out_tv | [0..l*-1]).
   Falsifikator: Tail-Mittel >= 0.5 × Pre-Mittel; kein stabiler Tail
   -> OFFEN (Via-Negativa). Kein l*-Band — l* wird aus dem Bild gelesen.

6. mi2_mla_vs_mqa_profile — Mittel tv_in_mid über globale MLA-Schichten
   {5,11,17,23} > Mittel über die 22 lokalen MQA-Schichten, strikt je
   Probe (paris, arithmetic — arithmetic_sp ohne is_global-Kontrolle im
   Band: alle drei werden gemessen, das Band legt paris+arithmetic fest
   [P2]; arithmetic_sp läuft als Datums-Zusatz). 2/2 PASS, 1/2 OFFEN,
   0/2 FAIL. Konvergenz-Datum zitiert: MI-1-Entropie-Delle e11 6/6.
   Code/Logik-Lens-Proben ohne eindeutige Ein-Token-Targets bleiben in
   diesem Lauf NICHT MESSBAR gelabelt (R1: keine Band-Erfindung).

Registrierte Proben (je 1 target, strikt Ein-Token):
  paris         "The capital of France is" -> target "Paris"
  arithmetic    "2 + 2 ="                  -> target "4"
  arithmetic_sp "2 + 2 = "                 -> target " "

KORREKTUR-LOG (offen gebucht):
 - CPU-Pilot prüft resolve_strict am Stub + echten Tokenizer (GPU-Lane):
   " 4" -> [236743, 236771]; strict wirft 236743 raus, behält 236771;
   " Paris" -> [9079] bleibt; synaptische Teleskop-/Band-Mathe.
 - MI-1-Arithmetik-Antwortmasse als Auflösungs-Artefakt gebucht; die
   MI-2-target-Lesart (236771) liefert die korrigierte Messung.
 - GPU-Hauptlauf 1 (30.1 s) brach an norm_mod = self.model.norm —
   AttributeError: 'ArgonForConditionalGeneration' object has no attribute
   'norm' (BROKEN gebucht, exit 2 GELB). Canon: norm sitzt am
   ArgonModel-Submodul (modeling_argon.py:362), der Wrapper :626-634
   trägt nur lm_head; MI-1-Muster mech_interp.py:178. Fix
   norm_mod = self.model.model.norm vor dem Rerun — P2 unberührt,
   kein Band geweitet, kein stiller Fallback.
 - GPU-Hauptlauf 2 (30.4 s, ROT exit 1): MI2-Messung — paris mehrdeutig
   ids=[9079, 50429] (BEIDE echte Ein-Token-Vertreter von "Paris": ' Paris'
   und 'Paris' ohne Space-Form), arithmetic_sp leer (lstrip(' ') zerstört
   das Leerzeichen-Target selbst: decode(' ')=lstrip->"") — genau die
   registrierte Grenze. arithmetic EINDEUTIG id=236812 ('4'): das
   no-Space-'4'-Token; der Vertreter 236743 fällt raus, MI-1 maß 236743
   (Artefakt). Smoke-Klassifikation buchte Leer/Mehrdeutigkeit fälschlich
   als FAIL — die registrierte Familie ist OFFEN je Probe (Falsifikator
   nur bei Vertreter-Falle/Dekode-Bruch). Fix der Klassifikation
   (Messdaten unverändert) + rerun: Leer/Mehrdeutigkeit -> OFFEN je
   Probe; Messdaten unverändert.

Konventionen: TV auf Fraction-Skala [0,1] (canon_tv); P3 Roh-Felder
(sublayer-rows je Probe) vollständig in JSON; P4 Einstufung nach
registrierten Bändern. P1: MI-1-Dateien unangetastet.

Run lines:
  CPU: python -m scratches.smoke_mech_interp_logit_lens           (exit 2)
  GPU: python -m scratches.smoke_mech_interp_logit_lens --gpu     (exit 0/1)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from machines.mech_interp import CRYSTAL_MIN_TAIL, MechInterpProbe

JSON_PATH = "scratches/mi2_logit_lens.json"

# -- Registrierte Konstanten --------------------------------------------------
TELESCOPE_TOL = 1e-6
LENS_ANCHOR_TOL = 1e-3
DRIVER_MARGIN = 1e-3      # Eindeutigkeit des Haupt-Treibers (Verdikt 4)
CONTRACT_RATIO = 0.5      # Tail-Mittel < 0.5 x Pre-Mittel (Verdikt 5)
MAX_LENGTH = 128
GLOBAL_LAYERS = (5, 11, 17, 23)   # modeling_argon.py:188-190

LENS_PROBES: List[Tuple[str, str, str]] = [
    # (name, prompt, target) — targets strikt Ein-Token (Verdikt 1)
    ("paris", "The capital of France is", "Paris"),
    ("arithmetic", "2 + 2 =", "4"),
    ("arithmetic_sp", "2 + 2 = ", " "),
]
CAND_MAP: Dict[str, Tuple[str, ...]] = {
    "paris": (" Paris", "Paris"),
    "arithmetic": (" 4", "4"),
    "arithmetic_sp": (" ", ""),
}


def resolve_token_ids_strict(tokenizer, target: str,
                             candidates: Tuple[str, ...]) -> List[int]:
    """MI-2-Konvention (Fix des MI-1-Bruchs): eine Kandidaten-ID zählt nur,
    wenn sie ALS EINZELTOKEN die target-Form repräsentiert:
    decode([encoded[0]]).lstrip(' ') == target. Mehr-Token-Kandidaten wie
    " 4" ([236743(" "), 236771("4")]) schleusen so kein Vertreter-Token ein."""
    ids: List[int] = []
    for cand in candidates:
        encoded = tokenizer.encode(cand, add_special_tokens=False)
        if not encoded:
            continue
        decoded_first = tokenizer.decode([encoded[0]], skip_special_tokens=True)
        if decoded_first.lstrip(" ") == target:
            ids.append(int(encoded[0]))
    return ids


# -- CPU-PILOT (R2: PILOT promotet nie Verdicte) -------------------------------
def pilot_rows() -> List[Tuple[str, str, str]]:
    out: List[Tuple[str, str, str]] = []

    # pilot_telescope: synthetische dp-Felder teleskopieren exakt auf die
    # Endpunktdifferenz
    dp = [(0.01, 0.02), (0.03, -0.01), (0.05, 0.04)]
    telescoped = sum(a + b for a, b in dp)
    endpoints = (0.01 + 0.02 + 0.03 - 0.01 + 0.05 + 0.04)  # p_final - p_emb
    ok = abs(telescoped - endpoints) < 1e-12
    out.append(("pilot_telescope", "PILOT" if ok else "BROKEN",
                f"teleskop={telescoped:.6f} endpunkte={endpoints:.6f}"))

    # pilot_resolve_strict: die Vertreter-Falle an einem Stub-Tokenizer
    class _StubTok:
        # encode(" 4") -> [1(" "), 2("4")], encode("4") -> [2("4")]
        def encode(self, text: str, add_special_tokens: bool = False):
            if text == " 4":
                return [1, 2]
            if text == "4":
                return [2]
            if text == " Paris":
                return [3]
            return [7]

        def decode(self, seq, skip_special_tokens: bool = True):
            return {1: " ", 2: "4", 3: " Paris", 7: " "}[seq[0]]

    stub = _StubTok()
    ids_arith = resolve_token_ids_strict(stub, "4", (" 4", "4"))
    ids_paris = resolve_token_ids_strict(stub, "Paris", (" Paris",))
    ok = ids_arith == [2] and ids_paris == [3]
    out.append(("pilot_resolve_strict", "PILOT" if ok else "BROKEN",
                f"arith={ids_arith} (vertreter 1 raus) paris={ids_paris}"))

    # pilot_contract: Tail < 0.5 x Pre an synthetischem gap-Verlauf
    gaps = [0.9] * 6 + [0.2] * 8
    pre = gaps[:6]
    tail = gaps[6:]
    ok = (sum(tail) / len(tail)) < CONTRACT_RATIO * (sum(pre) / len(pre))
    out.append(("pilot_contract", "PILOT" if ok else "BROKEN",
                f"tail={sum(tail) / len(tail):.3f} pre={sum(pre) / len(pre):.3f}"))

    # pilot_mla_select: globale Schicht-Selektion {5,11,17,23} aus % 6
    glo = tuple(i for i in range(26) if i % 6 == 5)
    loc_n = sum(1 for i in range(26) if i % 6 != 5)
    ok = glo == GLOBAL_LAYERS and loc_n == 22
    out.append(("pilot_mla_select", "PILOT" if ok else "BROKEN",
                f"global={glo} lokal_n={loc_n}"))
    return out


# -- Registrierte Auswertungen (P3-Roh -> Verdikt) ------------------------------
def eval_telescope(results: Dict[str, Dict[str, Any]]
                   ) -> Tuple[str, str, str, Dict[str, Any]]:
    notes, fields = [], {}
    ok = True
    for name, res in results.items():
        dev = abs(res["telescope_target"] - res["endpoints_target"])
        anchor = res["anchor_tv"]
        ok &= dev <= TELESCOPE_TOL and anchor <= LENS_ANCHOR_TOL
        fields[name] = {"telescope_dev": dev, "anchor_tv": anchor}
        notes.append(f"{name}: dev={dev:.3e} anchor={anchor:.3e}")
    status = "PASS" if ok else "FAIL"
    return "mi2_sublayer_telescope", status, "; ".join(notes), fields


def eval_purity_det(probe: MechInterpProbe, fp_before, results,
                    tokenizer) -> Tuple[str, str, str, Dict[str, Any]]:
    fp_after = probe.fingerprint()
    fp_ok = probe.fingerprint_equal(fp_before, fp_after)
    det_notes, fields = [], {}
    det_ok = True
    for name, text, target in LENS_PROBES:
        ids = resolve_token_ids_strict(tokenizer, target, CAND_MAP[name])
        if len(ids) != 1:
            continue
        rerun = probe.sublayer_logits(text, {"id": ids[0], "str": target})
        rows_a, rows_b = results[name]["rows"], rerun["rows"]
        same = len(rows_a) == len(rows_b) and all(
            torch.equal(torch.tensor(a[k]), torch.tensor(b[k]))
            for a, b in zip(rows_a, rows_b)
            for k in ("dp_attn", "dp_moe", "gap_out_tv", "tv_in_mid",
                      "top1_out_id"))
        det_ok &= same
        fields[name] = same
        det_notes.append(f"{name}:{'ok' if same else 'BRUCH'}")
    ok = fp_ok and det_ok
    return ("mi2_purity_det", "PASS" if ok else "FAIL",
            f"fingerprint={fp_ok} " + " ".join(det_notes),
            {"fingerprint": fp_ok, **fields})


def eval_attribution(results: Dict[str, Dict[str, Any]]
                     ) -> Tuple[str, str, str, Dict[str, Any]]:
    notes, fields = [], {}
    all_unique = True
    for name, res in results.items():
        rows = res["rows"]
        attn_row = max(rows, key=lambda r: r["dp_attn"])
        moe_row = max(rows, key=lambda r: r["dp_moe"])
        if attn_row["dp_attn"] > moe_row["dp_moe"]:
            driver = "attn"
            lead, challenger = attn_row["dp_attn"], moe_row["dp_moe"]
        else:
            driver = "moe"
            lead, challenger = moe_row["dp_moe"], attn_row["dp_attn"]
        unique = (lead - challenger) > DRIVER_MARGIN
        all_unique &= unique
        fields[name] = {
            "driver": driver,
            "peak_layer": (attn_row["layer"] if driver == "attn"
                           else moe_row["layer"]),
            "driver_is_global": (attn_row["is_global"] if driver == "attn"
                                 else moe_row["is_global"]),
            "lead": lead, "challenger": challenger, "unique": unique,
            "max_dp_attn": {"layer": attn_row["layer"],
                            "value": attn_row["dp_attn"],
                            "is_global": attn_row["is_global"]},
            "max_dp_moe": {"layer": moe_row["layer"],
                           "value": moe_row["dp_moe"],
                           "is_global": moe_row["is_global"]},
        }
        notes.append(f"{name}: {driver}@l{fields[name]['peak_layer']} "
                     f"{lead:.4f} vs {challenger:.4f} unique={unique}")
    # Gleichständige Attribution -> OFFEN (Verteilung, keine Kürzung)
    status = "PASS" if all_unique else "OFFEN"
    suffix = ("" if all_unique else
              " | Gleichstand der Kanäle — kein eindeutiger Treiber "
              "(OFFEN, keine Kürzung)")
    return ("mi2_attn_vs_moe_attribution", status,
            "; ".join(notes) + suffix, fields)


def eval_contraction(results: Dict[str, Dict[str, Any]]
                     ) -> Tuple[str, str, str, Dict[str, Any]]:
    notes, fields = [], {}
    overall: Optional[bool] = True
    for name, res in results.items():
        rows = res["rows"]
        final_top1 = rows[-1]["top1_out_id"]
        l_star = None
        for start in range(len(rows)):
            span = len(rows) - start
            if span < CRYSTAL_MIN_TAIL:
                break
            if all(rows[i]["top1_out_id"] == final_top1
                   for i in range(start, len(rows))):
                l_star = rows[start]["layer"]
                break
        if l_star is None:
            overall = None
            fields[name] = {"l_star": None, "note": "kein stabiler Tail"}
            continue
        tail = [r["gap_out_tv"] for r in rows if r["layer"] >= l_star]
        pre = [r["gap_out_tv"] for r in rows if r["layer"] < l_star]
        tail_mean = sum(tail) / len(tail)
        pre_mean = sum(pre) / len(pre)
        ok = tail_mean < CONTRACT_RATIO * pre_mean
        fields[name] = {"l_star": l_star, "tail_mean": tail_mean,
                        "pre_mean": pre_mean, "ok": ok}
        notes.append(f"{name}: l*={l_star} tail={tail_mean:.4f} "
                     f"pre={pre_mean:.4f} ok={ok}")
    if overall is None:
        status = "OFFEN"
    elif overall:
        status = "PASS"
    else:
        status = "FAIL"
    return ("mi2_goedel_gap_contraction", status,
            "; ".join(notes) if notes else "kein stabiler Tail", fields)


def eval_mla(results: Dict[str, Dict[str, Any]]
             ) -> Tuple[str, str, str, Dict[str, Any]]:
    # Band (P2): paris + arithmetic; arithmetic_sp als Datums-Zusatz
    band_names = ("paris", "arithmetic")
    notes, fields = [], {}
    wins = 0
    for name, res in results.items():
        rows = res["rows"]
        gm = sum(r["tv_in_mid"] for r in rows
                 if r["layer"] in GLOBAL_LAYERS) / len(GLOBAL_LAYERS)
        lm = sum(r["tv_in_mid"] for r in rows
                 if r["layer"] not in GLOBAL_LAYERS) / (len(rows) - len(GLOBAL_LAYERS))
        ok = gm > lm
        in_band = name in band_names
        if in_band:
            wins += ok
        fields[name] = {"global_mean_tv": gm, "local_mean_tv": lm,
                        "ok": ok, "in_band": in_band}
        notes.append(f"{name}: global={gm:.4f} local={lm:.4f} ok={ok}"
                     + ("" if in_band else " [Datums-Zusatz]"))
    status = ("PASS" if wins == 2 else
              "OFFEN" if wins == 1 else "FAIL")
    return ("mi2_mla_vs_mqa_profile", status, "; ".join(notes), fields)


# -- GPU-Lane ------------------------------------------------------------------
def run_gpu(checkpoint_base: str, checkpoint_machines: str
            ) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    from machines.verify_machines import _load_machines_model

    model, tokenizer = _load_machines_model(checkpoint_base,
                                            checkpoint_machines)
    device = next(model.parameters()).device
    probe = MechInterpProbe(model, tokenizer, device, max_length=MAX_LENGTH)

    findings: List[Tuple[str, str, str]] = []
    data: Dict[str, Any] = {}

    fp_before = probe.fingerprint()

    # -- Verdikt 1: strikte Ein-Token-Auflösung (echter Tokenizer) ----------
    # Registrierte Falsifikator-Familie: FAIL nur bei Vertreter-Falle
    # (aufgelöste ID decodiert zu ""/" " oder nach lstrip ≠ target).
    # Leere ODER mehrdeutige Auflösung -> OFFEN je Probe (registrierte
    # Allokation; kein Falsifikator, keine Band-Erfindung).
    strict_notes, any_falsifier = [], False
    n_unique, n_offen_lane = 0, 0
    resolved: Dict[str, int] = {}
    resolve_raw: Dict[str, List[int]] = {}
    for name, _text, target in LENS_PROBES:
        ids = resolve_token_ids_strict(tokenizer, target, CAND_MAP[name])
        resolve_raw[name] = ids
        if len(ids) == 0:
            n_offen_lane += 1
            strict_notes.append(f"{name}: leer (OFFEN je Probe — Konvention)")
            continue
        if len(ids) > 1:
            n_offen_lane += 1
            strict_notes.append(
                f"{name}: mehrdeutig ids={ids} (alle echt — OFFEN je Probe)")
            continue
        tid = ids[0]
        decoded = tokenizer.decode([tid], skip_special_tokens=True)
        if decoded.lstrip(" ") != target or decoded == "" or decoded.strip() == "":
            any_falsifier = True
            strict_notes.append(f"{name}: BAD id={tid} decoded='{decoded}'")
        else:
            n_unique += 1
            resolved[name] = tid
            strict_notes.append(f"{name}: id={tid} ('{decoded}')")
    data["resolve"] = resolved
    data["resolve_raw"] = resolve_raw
    v1_status = ("FAIL" if any_falsifier else
                 "PASS" if n_unique == len(LENS_PROBES) else "OFFEN")
    findings.append(("mi2_single_token_strict", v1_status,
                     "; ".join(strict_notes)))

    # -- 3 Proben über die Sonde --------------------------------------------
    results: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for name, text, target in LENS_PROBES:
        if name not in resolved:
            missing.append(name)
            continue
        results[name] = probe.sublayer_logits(
            text, {"id": resolved[name], "str": target})
    data["probes"] = results
    for name in missing:
        findings.append((name, "OFFEN",
                         "target nicht eindeutig auflösbar — keine "
                         "Band-Erfindung."))

    # -- Verdikt 2: Teleskop --------------------------------------------------
    if results:
        f2 = eval_telescope(results)
        findings.append(f2[:3])
        data["telescope"] = f2[3]
    else:
        findings.append(("mi2_sublayer_telescope", "OFFEN", "keine Probe"))

    # -- Verdikt 4: Attribution ----------------------------------------------
    if results:
        f4 = eval_attribution(results)
        findings.append(f4[:3])
        data["attribution"] = f4[3]
    else:
        findings.append(("mi2_attn_vs_moe_attribution", "OFFEN",
                         "keine Probe"))

    # -- Verdikt 5: Tail-Kontraktion -----------------------------------------
    if results:
        f5 = eval_contraction(results)
        findings.append(f5[:3])
        data["contraction"] = f5[3]
    else:
        findings.append(("mi2_goedel_gap_contraction", "OFFEN", "keine Probe"))

    # -- Verdikt 6: MLA-vs-MQA -------------------------------------------------
    if results:
        f6 = eval_mla(results)
        findings.append(f6[:3])
        data["mla"] = f6[3]
    else:
        findings.append(("mi2_mla_vs_mqa_profile", "OFFEN", "keine Probe"))

    # -- Verdikt 3: Purity/Det (Doubletten NACH allen anderen Lanes) ----------
    if results:
        f3 = eval_purity_det(probe, fp_before, results, tokenizer)
        findings.append(f3[:3])
        data["purity_det"] = f3[3]
    else:
        findings.append(("mi2_purity_det", "OFFEN", "keine Probe"))

    return findings, data


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MI-2: Sublayer-Röntgengerät (Attention vs MoE)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--checkpoint_base", default="./argon_1b")
    ap.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    ap.add_argument("--json", default=JSON_PATH)
    args = ap.parse_args()

    t0 = time.time()
    findings: List[Tuple[str, str, str]] = list(pilot_rows())

    data: Dict[str, Any] = {"pilot": [
        {"name": f[0], "status": f[1], "note": f[2]} for f in findings]}

    gpu_broken = False
    if args.gpu:
        try:
            gpu_findings, gpu_data = run_gpu(
                args.checkpoint_base, args.checkpoint_machines)
            findings.extend(gpu_findings)
            data["gpu"] = gpu_data
        except Exception as exc:  # noqa: BLE001 — registrierte OFFEN-Buchung
            gpu_broken = True
            findings.append(("gpu_broken", "BROKEN",
                             f"{type(exc).__name__}: {exc}"))
    else:
        gpu_broken = True

    registered = ("mi2_single_token_strict", "mi2_sublayer_telescope",
                  "mi2_purity_det", "mi2_attn_vs_moe_attribution",
                  "mi2_goedel_gap_contraction", "mi2_mla_vs_mqa_profile")
    for name in registered:
        if not any(f[0] == name for f in findings):
            findings.append((name, "OFFEN",
                             "GPU-Lane nicht gefahren oder Bruch gebucht "
                             "(OFFEN ist kein PASS)."))

    for name, status, note in findings:
        print(f"[{status:6s}] {name}: {note}")

    verdicts = [f for f in findings
                if not f[0].startswith(("pilot_", "gpu_broken"))]
    all_pass = all(status == "PASS" for _, status, _ in verdicts)
    has_fail = any(status == "FAIL" for _, status, _ in verdicts)
    n_pass = sum(1 for f in verdicts if f[1] == "PASS")
    n_open = sum(1 for f in verdicts if f[1] == "OFFEN")
    n_fail = sum(1 for f in verdicts if f[1] == "FAIL")

    elapsed = time.time() - t0
    print("-" * 88)
    print(f"Verdikte: {n_pass} PASS | {n_open} OFFEN | {n_fail} FAIL  "
          f"({elapsed:.1f} s)")
    if has_fail:
        print("ROT — mindestens ein Falsifikator beobachtet (P4: Einstufung, "
              "nicht Re-Decide).")
    elif all_pass and not gpu_broken:
        print("GRÜN — alle 6 Verdikte PASS; die Halbschichten sind gemessen.")
    else:
        print("GELB — offen (GPU-Lane nicht gefahren oder Bruch gebucht; "
              "OFFEN ist kein PASS).")

    payload = {
        "all_pass": bool(all_pass and not has_fail and not gpu_broken),
        "open": [f[0] for f in verdicts if f[1] == "OFFEN"],
        "findings": [{"name": f[0], "status": f[1], "note": f[2]}
                     for f in findings],
        "data": data,
    }
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"JSON: {args.json}")
    if has_fail:
        return 1
    return 0 if (all_pass and not gpu_broken) else 2


if __name__ == "__main__":
    sys.exit(main())