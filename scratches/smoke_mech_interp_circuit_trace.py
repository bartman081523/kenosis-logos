"""
smoke_mech_interp_circuit_trace.py — MI-4: Token-Level Circuit Tracing
& Sequenz-Dissoziation (Chimaeren-Texte, tokenweise Routing-Oszillographie).

PREREG (P2 — vor Messung registriert; falsifizierbare Praedikate):
  1. mi4_tokenwise_subspace_switch — je Chimaere (BEIDE noetig), im
     CORE-Band 4..23 (layer_bands(26)): die mittleren top_weights der
     Prosa-Experten {3,7} fallen strikt ueber die Umschaltstelle
     (share{3,7}(t<ts) > share{3,7}(t>=ts)), die der Code-Experten
     {0,1,4} steigen strikt (share{0,1,4}(t>=ts) > share{0,1,4}(t<ts)),
     und die mittlere Router-Entropie (ent[T][26], Canon-Router-Entropie,
     moe_layers.py:73-75) steigt strikt ueber ts (mean-ent(t>=ts) >
     mean-ent(t<ts) — Capacity-Gate-Oeffnung, :85-88). t_switch = gemeiname
     Token-Praefixlaenge von ids(Text) vs ids(Praefix). PASS = (a)∧(b)∧(c)
     je BEIDER Texte. Falsifikator: eine Relation verletzt.
  2. mi4_tokenwise_domain_dissociation — SEQ-TV je Token-Position
     (t=1..T-2, canon_tv auf [T,V]-Softmax token_probs) raw vs KO im
     CORE-Band: Prosa-Haelfte (t<ts): TV(ko{3,7}) > TV(ko{0,1,4}) strikt;
     Code-Haelfte (t>=ts): TV(ko{0,1,4}) > TV(ko{3,7}) strikt. PASS =
     (a)∧(b) je BEIDER Texte (Prose→Code UND Prose→Math). Datums-Zusatz:
     SEQ-TV-Profile je Token persistent (volle Felder, T1/R3).
     Falsifikator: eine Relation verletzt.

Canon-Belege: moe_layers.py:56-93 (ArgonEntropyAdaptiveRouter.forward —
top_weights aus RAW-Scores, Re-Norm :86, capacity_gate=clamp(H/ln8, .25,
1.0) :85-88, low_entropy_mask H<1.8 → 2. Experte x0.15 :79-84);
machines/mech_interp.py: tokenwise_routing (:470) → pct[T][26][8] mit
ent[T][26], token_probs (:462), knockout_experts (:389, ADDITIV im
CORE-Streifen, copy_-Restore C3), canon_tv (:53), assemble_pct (:562).

KORREKTUR-LOG (append-only; kein Eintrag bisher — GPU-Buchungen folgen).

Run:
  python scratches/smoke_mech_interp_circuit_trace.py            # CPU-Lane
      (5 PILOT-R2-Piloten + OFFEN-Verdikte, exit 2, schreibt kein JSON)
  python scratches/smoke_mech_interp_circuit_trace.py --gpu      # GPU-Lane
      (Default-JSON scratches/mi4_circuit_trace.json; Budget ~12 Forwards)
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List, Tuple

import torch

from machines.mech_interp import (MechInterpProbe, assemble_pct, canon_tv,
                                  layer_bands)

N_LAYERS = 26
MAX_LENGTH = 128
_EDGE_LIST, _CORE_LIST = layer_bands(N_LAYERS)
BAND_CORE: Tuple[int, ...] = tuple(_CORE_LIST)          # 4..23
BAND_EDGE: Tuple[int, ...] = tuple(_EDGE_LIST)

# Registrierte Chimaeren (User-Audit 1:1) und t_switch-Praefixe:
CHIMERA_PROSE_CODE = ("Der warme Sommerwind wehte über den See, während "
                      "das Programm lief: def quicksort(arr): "
                      "return [x for x in arr]")
CHIMERA_PROSE_MATH = ("Oma erzählte im alten Garten ihre Geschichte, "
                      "doch plötzlich galt: Für alle n in N gilt "
                      "sum_{k=1}^n k = n(n+1)/2.")
PREFIX_PROSE_CODE = ("Der warme Sommerwind wehte über den See, "
                     "während das Programm lief: ")
PREFIX_PROSE_MATH = ("Oma erzählte im alten Garten ihre Geschichte, "
                     "doch plötzlich galt: ")

KO_SET_PROSE: Tuple[int, ...] = (3, 7)      # Prosa-Träger (MI-1/MI-3-Canon)
KO_SET_CODE: Tuple[int, ...] = (0, 1, 4)    # Code-Träger (MI-1/MI-3-Canon)
KO_PENALTY = -50.0   # MI-3-Canon (knockout_experts-ADDITIV-Arm)


def common_prefix_len(a: List[int], b: List[int]) -> int:
    """Gemeiname Token-Praefixlaenge (t_switch-Marker; Plan-Risiko 2)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def share_set(pct: List[List[List[float]]], t_lo: int, t_hi: int,
              core: Tuple[int, ...], experts: Tuple[int, ...]) -> float:
    """Mittlere top_weights-Masse des Experten-Sets ueber Positions- und
    Schicht-Fenster (je Token/Schicht summiert der Set-Mitglieder; die
    top_weights eines (t,l)-Schritts summieren zum capacity_gate <= 1)."""
    acc = 0.0
    steps = 0
    for t in range(t_lo, t_hi):
        for li in core:
            acc += sum(pct[t][li][e] for e in experts)
            steps += 1
    if steps == 0:
        raise ValueError("leeres Fenster")
    return acc / steps


def mean_ent(ent: List[List[float]], t_lo: int, t_hi: int,
             core: Tuple[int, ...]) -> float:
    acc = 0.0
    steps = 0
    for t in range(t_lo, t_hi):
        for li in core:
            acc += ent[t][li]
            steps += 1
    if steps == 0:
        raise ValueError("leeres Fenster")
    return acc / steps


# -- PILOT-R2 (CPU-Lane: reine Mathematik-Piloten, promovieren nie Verdikte) --

def pilot_rows() -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []

    # (1) assemble_pct am Stub (echte Canon-Assembly, pilotierbar ohne Modell):
    ti0 = torch.tensor([[3, 1], [5, 0], [7, 2]])
    tw0 = torch.tensor([[0.5, 0.25], [0.4, 0.2], [0.45, 0.05]])
    ti1 = torch.tensor([[2, 6], [4, 6], [1, 2]])
    tw1 = torch.tensor([[0.3, 0.2], [0.6, 0.3], [0.5, 0.5]])
    e0 = torch.tensor([0.8, 1.0, 1.2])
    e1 = torch.tensor([1.4, 1.6, 1.8])
    res = assemble_pct([(0, ti0, tw0, e0), (1, ti1, tw1, e1)], 8, 2)
    shuffled = assemble_pct([(1, ti1, tw1, e1), (0, ti0, tw0, e0)], 8, 2)
    incomplete_raises = False
    try:
        assemble_pct([(1, ti1, tw1, e1)], 8, 2)
    except ValueError:
        incomplete_raises = True
    ok = (res["n_tokens"] == 3 and res["n_layers"] == 2 and res["n_experts"] == 8
          and abs(res["pct"][0][0][3] - 0.5) < 1e-6
          and abs(res["pct"][0][0][1] - 0.25) < 1e-6
          and abs(res["pct"][1][0][5] - 0.4) < 1e-6
          and res["pct"][2][1][1] == 0.5
          and res["pct"][0][0][4] == 0.0
          and abs(res["ent"][2][1] - 1.8) < 1e-6
          and res["pct"] == shuffled["pct"]
          and incomplete_raises)
    rows.append(("pilot_assemble_pct",
                 "PILOT" if ok else "PILOT-FAIL",
                 f"pct[0/0][3]={res['pct'][0][0][3]} ent[2][1]={res['ent'][2][1]} "
                 f"shuffled-gleich={res['pct'] == shuffled['pct']} "
                 f"incomplete->ValueError={incomplete_raises}"))

    # (2) t_switch-Praefix-Logik:
    cases = (([1, 2, 3, 9], [1, 2, 3, 5, 6], 3),
             ([5], [5], 1),
             ([], [1], 0),
             ([7, 8, 9], [8, 9, 9], 0))
    ok = all(common_prefix_len(a, b) == want for a, b, want in cases)
    rows.append(("pilot_prefix_math", "PILOT" if ok else "PILOT-FAIL",
                 f"prae-laengen={[common_prefix_len(a, b) for a, b, _ in cases]}"
                 f" erwartet={[w for _, _, w in cases]}"))

    # (3) Capacity-Gate-Formel (moe_layers.py:85-88) am Stub:
    ln8 = torch.log(torch.tensor(8.0))
    gate_low = torch.clamp(torch.tensor(0.3149) / ln8, min=0.25, max=1.0)
    gate_open = torch.clamp(torch.tensor(1.35) / ln8, min=0.25, max=1.0)
    damped = torch.tensor([0.85, 0.15])
    renorm = damped / damped.sum() * gate_low
    ok = (abs(gate_low.item() - 0.25) < 1e-6
          and abs(gate_open.item() - 1.35 / ln8.item()) < 1e-6
          and abs(renorm.sum().item() - gate_low.item()) < 1e-6)
    rows.append(("pilot_gate_math", "PILOT" if ok else "PILOT-FAIL",
                 f"clamp-Floor={gate_low.item():.4f}, offene Stute="
                 f"{gate_open.item():.4f}, Re-Norm-summe="
                 f"{renorm.sum().item():.4f}=gate"))

    # (4) SEQ-TV-Formel (canon_tv ueber Positionen 1..T-2):
    a = torch.softmax(torch.tensor([[1.0, 2.0, 3.0],
                                    [3.0, 1.0, 2.0],
                                    [0.5, 0.5, 4.0]]), dim=-1)
    b = torch.softmax(torch.tensor([[1.0, 2.0, 3.0],
                                    [1.0, 3.0, 2.0],
                                    [0.5, 0.5, 4.0]]), dim=-1)
    manual = canon_tv(a[1], b[1])
    ok = 0.0 < manual <= 1.0 and canon_tv(a[0], b[0]) == 0.0 \
        and canon_tv(a[2], b[2]) == 0.0
    rows.append(("pilot_seqtv_math", "PILOT" if ok else "PILOT-FAIL",
                 f"tv(a1,b1)={manual:.4f} in (0,1); Positionen 0/2 identisch "
                 f"→ tv=0"))

    # (5) Auswahl/Konstanten (KO-Sets, Band, Chimaeren-/Praefix-Verbatim):
    ok = (set(KO_SET_PROSE) <= set(range(8)) and set(KO_SET_CODE) <= set(range(8))
          and not set(KO_SET_PROSE) & set(KO_SET_CODE)
          and BAND_CORE == tuple(range(4, 24))
          and ": " in CHIMERA_PROSE_CODE and ": " in CHIMERA_PROSE_MATH
          and CHIMERA_PROSE_CODE.startswith(PREFIX_PROSE_CODE)
          and CHIMERA_PROSE_MATH.startswith(PREFIX_PROSE_MATH)
          and PREFIX_PROSE_CODE.endswith(": ")
          and PREFIX_PROSE_MATH.endswith(": ")
          and KO_PENALTY < 0)
    rows.append(("pilot_select_math", "PILOT" if ok else "PILOT-FAIL",
                 f"core={len(BAND_CORE)} Schichten, prose-Set={KO_SET_PROSE}, "
                 f"code-Set={KO_SET_CODE}, praefixe verbatim-inklusive"))
    return rows


# -- GPU-Lane ------------------------------------------------------------------

def eval_switch(probe: MechInterpProbe, results: Dict[str, Any]) -> None:
    per_text: List[Dict[str, Any]] = []
    routing_store: Dict[str, Any] = {}
    for name, text, prefix in (
            ("CHIMERA_PROSE_CODE", CHIMERA_PROSE_CODE, PREFIX_PROSE_CODE),
            ("CHIMERA_PROSE_MATH", CHIMERA_PROSE_MATH, PREFIX_PROSE_MATH)):
        tr = probe.tokenwise_routing(text)
        pct, ent, T = tr["pct"], tr["ent"], tr["n_tokens"]
        ids_full = probe._encode(text)[0].tolist()
        ids_pre = probe._encode(prefix)[0].tolist()
        t_switch = common_prefix_len(ids_full, ids_pre)
        if not (1 <= t_switch < T):
            raise RuntimeError(f"t_switch degenerat ({t_switch}/{T}) — "
                               "keine Band-Erfindung")
        before, after = (0, t_switch), (t_switch, T)
        row = {
            "text": name,
            "T": T,
            "t_switch": t_switch,
            "share37_before": share_set(pct, *before, BAND_CORE, KO_SET_PROSE),
            "share37_after": share_set(pct, *after, BAND_CORE, KO_SET_PROSE),
            "share014_before": share_set(pct, *before, BAND_CORE, KO_SET_CODE),
            "share014_after": share_set(pct, *after, BAND_CORE, KO_SET_CODE),
            "mean_ent_before": mean_ent(ent, *before, BAND_CORE),
            "mean_ent_after": mean_ent(ent, *after, BAND_CORE),
        }
        row["ok_a"] = row["share37_before"] > row["share37_after"]
        row["ok_b"] = row["share014_after"] > row["share014_before"]
        row["ok_c"] = row["mean_ent_after"] > row["mean_ent_before"]
        row["token_before_switch"] = probe.tokenizer.decode(
            [ids_full[t_switch - 1]])
        row["token_at_switch"] = probe.tokenizer.decode([ids_full[t_switch]])
        per_text.append(row)
        routing_store[name] = {"t_switch": t_switch, "T": T,
                               "pct": pct, "ent": ent}
    results["switch"] = {
        "per_text": per_text,
        "both_texts": all(r["ok_a"] and r["ok_b"] and r["ok_c"]
                          for r in per_text),
    }
    results["routing"] = routing_store     # Voll-Felder (T1/R3, Datums-Zusatz)


def eval_dissociation(probe: MechInterpProbe,
                      results: Dict[str, Any]) -> None:
    per_text: List[Dict[str, Any]] = []
    for name, text, prefix in (
            ("CHIMERA_PROSE_CODE", CHIMERA_PROSE_CODE, PREFIX_PROSE_CODE),
            ("CHIMERA_PROSE_MATH", CHIMERA_PROSE_MATH, PREFIX_PROSE_MATH)):
        p_raw = probe.token_probs(text)                    # [T, V] fp32
        ids_full = probe._encode(text)[0].tolist()
        ids_pre = probe._encode(prefix)[0].tolist()
        t_switch = common_prefix_len(ids_full, ids_pre)
        T = p_raw.shape[0]
        if not (2 <= t_switch <= T - 2):
            raise RuntimeError(f"Haelften degenerat (t_switch={t_switch}, "
                               f"T={T}) — keine Band-Erfindung")
        with probe.knockout_experts(KO_SET_PROSE, layers=list(BAND_CORE),
                                    penalty=KO_PENALTY):
            p_ko_prose = probe.token_probs(text)
        with probe.knockout_experts(KO_SET_CODE, layers=list(BAND_CORE),
                                    penalty=KO_PENALTY):
            p_ko_code = probe.token_probs(text)
        tvs_prose = [canon_tv(p_raw[t], p_ko_prose[t]) for t in range(T)]
        tvs_code = [canon_tv(p_raw[t], p_ko_code[t]) for t in range(T)]
        binn = list(range(1, T - 1))
        prose_pos = [t for t in binn if t < t_switch]
        code_pos = [t for t in binn if t >= t_switch]
        row = {
            "text": name,
            "T": T,
            "t_switch": t_switch,
            "prose_positions": len(prose_pos),
            "code_positions": len(code_pos),
            "mean_tv_prose_set": sum(tvs_prose[t] for t in prose_pos)
                                 / len(prose_pos),
            "mean_tv_code_set": sum(tvs_code[t] for t in prose_pos)
                                / len(prose_pos),
            "mean_tv_prose_set_codehalf": sum(tvs_prose[t] for t in code_pos)
                                          / len(code_pos),
            "mean_tv_code_set_codehalf": sum(tvs_code[t] for t in code_pos)
                                         / len(code_pos),
            "tv_profile_ko_prose": tvs_prose,   # Voll-Felder (Datums-Zusatz)
            "tv_profile_ko_code": tvs_code,
        }
        row["ok_a"] = row["mean_tv_prose_set"] > row["mean_tv_code_set"]
        row["ok_b"] = (row["mean_tv_code_set_codehalf"]
                       > row["mean_tv_prose_set_codehalf"])
        per_text.append(row)
    results["dissociation"] = {
        "per_text": per_text,
        "both_texts": all(r["ok_a"] and r["ok_b"] for r in per_text),
    }


def run_gpu(checkpoint_base: str, checkpoint_machines: str,
            results: Dict[str, Any],
            findings: List[Tuple[str, str, str]]) -> None:
    from machines.verify_machines import _load_machines_model

    model, tokenizer = _load_machines_model(checkpoint_base,
                                            checkpoint_machines)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = MechInterpProbe(model, tokenizer, device=device,
                            max_length=MAX_LENGTH)

    for name, fn in (
            ("mi4_tokenwise_subspace_switch",
             lambda: eval_switch(probe, results)),
            ("mi4_tokenwise_domain_dissociation",
             lambda: eval_dissociation(probe, results)),
    ):
        try:
            fn()
        except Exception as exc:      # noqa: BLE001 — BROKEN-Buchung
            results.setdefault("broken", {})[name] = \
                f"{type(exc).__name__}: {exc}"
            findings.append((
                name, "BROKEN",
                f"GPU-Lane brach: {type(exc).__name__}: {exc} "
                "(kein stiller Fallback; sonstiges gemessen bleibt)."))


# -- classification + main ----------------------------------------------------
# Registrierter Verdict-Name -> Ergebnis-Schluessel (MI-3-KORREKTUR-Muster).
RESULT_KEYS = {
    "mi4_tokenwise_subspace_switch": "switch",
    "mi4_tokenwise_domain_dissociation": "dissociation",
}


def classify(name: str, results: Dict[str, Any]) -> Tuple[str, str]:
    """P4-Einstufung je registrierten Praedikaten (kein Re-Decide)."""
    if name == "mi4_tokenwise_subspace_switch":
        p = results["switch"]
        rows = "; ".join(
            f"{r['text']}@ts={r['t_switch']}/{r['T']}: "
            f"{{3,7}} {r['share37_before']:.3f}→{r['share37_after']:.3f} "
            f"{{0,1,4}} {r['share014_before']:.3f}→{r['share014_after']:.3f} "
            f"ent {r['mean_ent_before']:.3f}→{r['mean_ent_after']:.3f}"
            for r in p["per_text"])
        return ("PASS", rows) if p["both_texts"] else ("FAIL", rows)

    if name == "mi4_tokenwise_domain_dissociation":
        p = results["dissociation"]
        rows = "; ".join(
            f"{r['text']}@ts={r['t_switch']}: prose-H TV "
            f"{r['mean_tv_prose_set']:.4f} vs {r['mean_tv_code_set']:.4f}; "
            f"code-H TV {r['mean_tv_code_set_codehalf']:.4f} vs "
            f"{r['mean_tv_prose_set_codehalf']:.4f}"
            for r in p["per_text"])
        return ("PASS", rows) if p["both_texts"] else ("FAIL", rows)

    return ("OFFEN", "unregistriertes Verdict — kein stiller Fallback.")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="MI-4: Token-Level Circuit Tracing (Chimaeren)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--checkpoint_base", default="./argon_1b")
    ap.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    ap.add_argument("--json", default=None,
                    help="Ergebnis-Pfad; Default: GPU->MI-4-Datei, "
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

    classified: List[Tuple[str, str, str]] = []
    registered = ("mi4_tokenwise_subspace_switch",
                  "mi4_tokenwise_domain_dissociation")
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
        print("GRÜN — beide MI-4-Verdikte PASS; die Umschaltung ist "
              "tokenweise gemessen.")
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
        ("scratches/mi4_circuit_trace.json" if args.gpu else None)
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