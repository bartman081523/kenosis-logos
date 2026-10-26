"""
scratches/smoke_mech_interp_tomography.py
=========================================
MI-1 — 26-Layer-Röntgengerät: schichtweise Routing-Tomographie
(26×8-Matrix je Domäne) + Logit-Lens über alle 26 Schichten.
Spur B (Mechanistische Interpretierbarkeit), MI-1 von MI-1..MI-4.
Die Spur-A-Dateien (SP1..SP13) bleiben unangetastet.

PREREG-REGISTRIERUNG (P2-Freeze — fixiert vor jeder Messung, 2026-10-26)
-----------------------------------------------------------------------
Canon-Anker:
 - machines/mech_interp.py:1-40            Sonden-API (Canon-Ankerliste)
 - moe_layers.py:56-93                     Router: (top_indices, top_weights, router_entropy)
 - moe_layers.py:128                       gate-Aufruf im MoELayer
 - modeling_argon.py:601-611               Schicht-Schleife / all_entropies
 - modeling_argon.py:615                   h = self.norm(h)
 - modeling_argon.py:733-736               lm_head + final_softcap
                                           (Checkpoint: softcap = None)
 - train_moe_router_specialization.py:239  Kanon-Pct: ungewichtete
                                           Assignment-Bincounts -> %
 - train_moe_router_specialization.py:292  Kanon-TV: 0.5*sum|a-b|
 - train_moe_router_specialization.py:341  Trainings-Hook-Bereich 4<=idx<=23
 - machines/verify_machines.py:339-356     _load_machines_model (bf16/CUDA)
 - scratches/verify_machines_stage5.log    AGGREGAT-Datum: je Domäne nur
                                           EIN 8-Vektor + EIN H über alle
                                           26 Schichten — MI-1 bricht auf.
 - scratches/endogene_agency_gpu.log:6-10  SP13: Preset-Trennung im
                                           Composite-Gap nur Δ≈0.0023 —
                                           symbolische Schicht verdeckt
                                           den Schaltkreis (Motiv).

Domänen-Samples (REGISTRIERT, je 3 Festtexte, max_length=128):
 code, math_logic, prose, goedel, turing, wittgenstein (Konstante unten).
Lens-Proben (REGISTRIERT):
 P1 "The capital of France is"  -> Antwort-Label "Paris"
 P2 "2 + 2 ="                   -> Antwort-Label "4"
 Token-Äquivalenz nach lstrip(" "); nicht auflösbar -> OFFEN (kein
 stiller Fallback), Trajektorie wird trotzdem persistiert.

Bänder (REGISTRIERT, layer_bands(26)):
 EDGE = {0,1,2,3,24,25}   CORE = {4..23}   (Trainings-Hook 4<=idx<=23).

Theorie-Stufen (vier Felder: Behauptung/Begründung/Vorhersage/Falsifikator):
 MI-1.a Tiefen-Spezialisierungs-Gesetz:
   (a) Domänen-Trennung der Router-Verteilung bricht scharf im Core auf.
   (b) Trainings-Hookbereich (train:341); expert_bias auf allen 26.
   (c) Vorhersage: CORE-Mittel-TV(ℓ) > 0.50 UND CORE > EDGE strikt.
   (d) F1 core_mean_tv <= 0.50; F2 core <= edge; F3 Zeilen aller Rows
       eines 26×8-Feldes identisch (keine Schichtauflösung).
 MI-1.b Entropie-Regime:
   (a) Hohe Auflösung korrespondiert mit abgesenkter Router-Entropie im Core.
   (b) capacity_gate-Dämpfung (moe_layers:90-92); Trainings-Ziele 0.85/1.35/1.85.
   (c) Vorhersage: je Domäne core_mittel_entropie < edge_mittel_entropie
       strikt; 6/6 -> PASS; 4-5 -> OFFEN; <=3 -> FAIL.
   (d) Falsifikator: core >= edge in der Mehrzahl der Domänen.
 MI-1.c Kristallisations-Gesetz (erste Lens-Anwendung):
   (a) Das Antwort-Label kristallisiert in einem stabilen Schicht-Fenster
       ℓ* im Core.
   (b) Residualstrom + norm + lm_head (modeling_argon:615/:733); kein Softcap.
   (c) Vorhersage: >=1 Probe mit stabilem ℓ* in CORE und aufgelöster
       Antwort-Token; stabiler TAIL >= CRYSTAL_MIN_TAIL=2 Schichten
       (machines/mech_interp.py — trivialer Ein-Schicht-Tail am Ende
       zaehlt NICHT alsKristallisation); Gap fällt nach ℓ*
       (Gap(ℓ*)>0, Gap(Ende)=0).
   (d) F1 kein stabiles Startfenster in ALLEN Proben -> OFFEN (Via-Negativa,
       Trajektorien persistiert); F2 alle stabilen ℓ* außerhalb CORE -> FAIL;
       F3 Antwort-Token nicht auflösbar -> OFFEN (kein stiller Fallback);
       F4 anchor_tv > Toleranz oder p_final-Top1 != Official-Top1 -> FAIL
       (Geräte-Bruch, fp32-Lens vs bf16-Kopf).
 MI-1.d Gerät-Korrektheit & Reinheit:
   (a) Die Sonde mißt das Feld, das sie beansprucht.
   (b) Kanon-Konvention (train:239/:292); _gate_hooks-Muster.
   (c) Vorhersage: pct-Zeilen je Schicht summieren 1.0 ± 1e-6; Parität zur
       Kanon-Messung (TV <= 1e-6); Haken entfernen sich; Träger-Fingerprint
       (expert_bias + gate.weight je Schicht, torch.equal) identisch vor/nach;
       Wiederholung (Domäne math_logic + Lens-Probe P1) bitidentisch.
   (d) F: Summen-Bruch, Paritäts-Bruch, Fingerprint-Bruch, Det-Bruch.
 EINHEIT T2 (Auflösung als eigenes Verdikt):
   je Domäne max. TV über alle Zeilen-Paare des 26×8-Feldes > 0.0 strikt;
   Aggregate (Layer-Mittelwert) werden separat als "AGGREGAT" gelabelt
   mitgereicht, tragen aber kein Dimension-Verdict (T1).

Verdikt-Register (6):
 1 mi1_canon_parity        — Parität Kanon-Pct <= 1e-6 + Zeilensummen
                             1.0±1e-6 + Feld-Vollständigkeit (26×8, Ent 26,
                             Lens 27 Rows) + n_tokens > 0.
 2 mi1_layer_resolution    — je Domäne max-Zeilenpaar-TV > 0.0 strikt.
 3 mi1_purity_det          — Fingerprint torch.equal vor/nach + Doubletten
                             (Counts/Pct/Entropie bitidentisch + Lens-Top1
                             identisch).
 4 mi1_depth_specialization— core_mean_tv > 0.50 UND core_mean_tv > edge_mean_tv.
 5 mi1_entropy_regime      — 6/6 PASS; 4-5 OFFEN; <=3 FAIL (strikt je Domäne).
 6 mi1_crystallization     — >=1 Probe mit stabilem ℓ* in CORE + Antwort
                             aufgelöst + anchor_tv <= 1e-3 + Top1-Anker;
                             Verzweigungen wie F1..F4 oben.

KORREKTUR-LOG (offene Buchung, VOR dem Hauptlauf; Registrierung R1/R2):
 - CPU-Pilot 2026-10-26, Scan 1: pilot_crystal_oscillation BRACH den
   Erst-Kristallisations-Finder — er gab den trivialen Ein-Schicht-Tail
   am Ende als ℓ*=25 zurück (Oszillation wurde als Kristallisation
   gelesen). Korrektur VOR dem Hauptlauf: stabiler TAIL >= 2 Schichten
   (CRYSTAL_MIN_TAIL in machines/mech_interp.py, registriert); der Pilot
   promotet keine Verdicts (R2).
 - GPU-Scan 1 + 2 (2026-10-26): GPU-Brueche gebucht (NameError out_layer;
   RuntimeError dtype fp32-vs-bf16 am Kopf). Korrektur VOR dem
   Hauptlauf: Lens spiegelt EXAKT den offiziellen bf16-Kopf-Pfad
   (norm -> lm_head, EINHEIT S3), Softmax auf den fp32-Logits;
   anchor_tv misst die Paritaet (Erwartung 0.0, Toleranz 1e-3).
   Offen-Verzweigung arbeitete korrekt: alle 6 Verdikte OFFEN statt
   stiller Annahme.

Konvention (gelabelte Abweichung, T3/L1):
 - Kanon-Entropie-Aggregation ist UNGEWICHTET (Mittel der Schicht-Mittel,
   train:240); unsere Entropie ist TOKEN-gewichtet (Summe/Tokenzahl).
   Parität wird NUR auf pct (exakt gleiche Zählbasis) behauptet.
 - pct als Fraction (Zeile summiert 1.0); Prozent = x100.
 TV auf Fraction-Skala [0,1] (Kanon-TV/100).

Lauf: python -m scratches.smoke_mech_interp_tomography            (CPU-Pilot)
      python -m scratches.smoke_mech_interp_tomography --gpu      (Hauptlauf)
Exit-Codes: 0 alle PASS | 1 mindestens ein FAIL | 2 OFFEN (inkl. GPU-Lane
ohne --gpu; kein stiller Fallback).
P-Protokoll: P1 append-only; P2 dies ist die Registrierung; P3 Rohdaten
(JSON, ungerundete Floats) VOR Auswertung; P4 klassifizieren, nicht neu
entscheiden.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

JSON_PATH = "scratches/mi1_tomography_experiment.json"

# -- Registrierte Konstanten -------------------------------------------------
ROW_SUM_TOL = 1e-6
CORE_TV_MIN = 0.50
LENS_ANCHOR_TOL = 1e-3
ENTROPY_PASS_N = 6      # 6/6 strikt -> PASS
ENTROPY_OFFEN_N = 4     # 4-5 -> OFFEN, <=3 -> FAIL
MAX_LENGTH = 128

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

LENS_PROBES: List[Dict[str, Any]] = [
    {"name": "paris", "text": "The capital of France is",
     "label": "Paris", "candidates": (" Paris", "Paris")},
    {"name": "arithmetic", "text": "2 + 2 =",
     "label": "4", "candidates": (" 4", "4")},
]

DOMAIN_ORDER = ("code", "math_logic", "prose", "goedel", "turing", "wittgenstein")


# -- Helfer ------------------------------------------------------------------
def resolve_token_ids(tokenizer, target: str, candidates: Tuple[str, ...]) -> List[int]:
    """Token-Äquivalenz nach lstrip(' '); keine Auflösbar -> [] (kein stiller
    Fallback — der Rufer bucht OFFEN)."""
    ids: List[int] = []
    for cand in candidates:
        encoded = tokenizer.encode(cand, add_special_tokens=False)
        if not encoded:
            continue
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        if decoded.lstrip(" ") == target:
            ids.append(int(encoded[0]))
    return ids


def aggregate_row_label(pct: "Any") -> Dict[str, Any]:
    """Layer-Mittel je Experte — NUR als AGGREGAT gelabelt (EINHEIT T1/L1,
    kein Dimension-Verdict)."""
    return {"kind": "AGGREGAT", "mean_over_layers": pct.mean(dim=0).tolist()}


def synth_rows(traj: List[int]) -> List[Dict[str, Any]]:
    """Synthetische Lens-Rows (top1_id-Trajektorie) für den CPU-Piloten."""
    return [{"kind": "layer", "layer": i, "top1_id": t}
            for i, t in enumerate(traj)]


# -- CPU-Pilot (PILOT-Labels, promovieren keine Verdicts — R2) ---------------
def pilot_math() -> List[Tuple[str, str, str]]:
    import torch

    from machines.mech_interp import canon_tv, crystallization_start, layer_bands

    rows: List[Tuple[str, str, str]] = []

    one_hot_a = torch.zeros(8); one_hot_a[0] = 1.0
    one_hot_b = torch.zeros(8); one_hot_b[5] = 1.0
    tv_self = canon_tv(one_hot_a, one_hot_a)
    tv_cross = canon_tv(one_hot_a, one_hot_b)
    rows.append(("pilot_tv_math",
                 "PILOT" if abs(tv_self) < 1e-12 and abs(tv_cross - 1.0) < 1e-12
                 else "BROKEN",
                 f"tv(id,id)={tv_self:.3e}, tv(id,other)={tv_cross:.6f}"))

    edge, core = layer_bands(26)
    ok_bands = edge == [0, 1, 2, 3, 24, 25] and core == list(range(4, 24))
    rows.append(("pilot_bands", "PILOT" if ok_bands else "BROKEN",
                 f"edge={edge}, |core|={len(core)}"))

    star = crystallization_start(synth_rows([42] * 7 + [9] * 19))
    rows.append(("pilot_crystal_at7", "PILOT" if star == 7 else "BROKEN",
                 f"star={star}"))
    alt = [9 if i % 2 == 0 else 42 for i in range(26)]
    star_alt = crystallization_start(synth_rows(alt))
    rows.append(("pilot_crystal_oscillation", "PILOT" if star_alt is None else "BROKEN",
                 f"oscillation_star={star_alt} (erwartet None)"))
    star_zero = crystallization_start(synth_rows([42] * 26))
    rows.append(("pilot_crystal_at0", "PILOT" if star_zero == 0 else "BROKEN",
                 f"star={star_zero}"))

    uniform = torch.full((8,), 0.125)
    tv_uc = canon_tv(uniform, one_hot_b)
    expect = 0.5 * (7 * 0.125 + (1.0 - 0.125))
    rows.append(("pilot_tv_uniform_onehot",
                 "PILOT" if abs(tv_uc - expect) < 1e-9 else "BROKEN",
                 f"tv={tv_uc:.6f}, erwartet {expect:.6f}"))
    return rows


# -- Tiefen-Auswertung (gebrauchter Codepfad: Pilot + Hauptlauf) -------------
def evaluate_depth(pct_by_domain: Dict[str, "Any"]) -> Dict[str, Any]:
    import torch

    from machines.mech_interp import canon_tv, layer_bands

    edge, core = layer_bands(26)
    domains = list(pct_by_domain)
    tv_by_layer: List[float] = []
    for layer in range(pct_by_domain[domains[0]].shape[0]):
        pairs = [
            canon_tv(pct_by_domain[a][layer], pct_by_domain[b][layer])
            for i, a in enumerate(domains)
            for b in domains[i + 1:]
        ]
        tv_by_layer.append(sum(pairs) / len(pairs))
    core_mean = sum(tv_by_layer[idx] for idx in core) / len(core)
    edge_mean = sum(tv_by_layer[idx] for idx in edge) / len(edge)
    per_domain_max_row_tv: Dict[str, float] = {}
    for name, mat in pct_by_domain.items():
        n = mat.shape[0]
        max_tv = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                max_tv = max(max_tv, canon_tv(mat[i], mat[j]))
        per_domain_max_row_tv[name] = max_tv
    return {
        "tv_by_layer": tv_by_layer,
        "core_mean_tv": core_mean,
        "edge_mean_tv": edge_mean,
        "band_core": core,
        "band_edge": edge,
        "per_domain_max_row_tv": per_domain_max_row_tv,
    }


def evaluate_entropy(ent_by_domain: Dict[str, "Any"]) -> Dict[str, Any]:
    import torch

    from machines.mech_interp import layer_bands

    edge, core = layer_bands(26)
    per_domain: Dict[str, Dict[str, float]] = {}
    wins = 0
    for name, ent in ent_by_domain.items():
        core_mean = float(ent[core].mean().item())
        edge_mean = float(ent[edge].mean().item())
        per_domain[name] = {"core": core_mean, "edge": edge_mean,
                            "core_lower": core_mean < edge_mean}
        if core_mean < edge_mean:
            wins += 1
    return {"per_domain": per_domain, "core_lower_count": wins}


# -- GPU-Hauptlauf -----------------------------------------------------------
def run_gpu(checkpoint_base: str, checkpoint_machines: str) -> Tuple[
        List[Tuple[str, str, str]], Dict[str, Any]]:
    """Alle 6 Verdikte; jede unerwartete Ausnahme -> OFFEN mit Fehler-Note
    (kein stiller Fallback, kein stiller Fail)."""
    import torch

    from machines.mech_interp import canon_tv, layer_bands, MechInterpProbe
    from machines.verify_machines import _load_machines_model
    from train_moe_router_specialization import measure_domain_routing

    findings: List[Tuple[str, str, str]] = []
    data: Dict[str, Any] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_machines_model(checkpoint_base, checkpoint_machines)
    probe = MechInterpProbe(model, tokenizer, device=device, max_length=MAX_LENGTH)
    edge, core = layer_bands(probe.n_layers)

    answer_targets: Dict[str, Dict[str, List[int]]] = {}
    for spec in LENS_PROBES:
        ids = resolve_token_ids(tokenizer, spec["label"], spec["candidates"])
        answer_targets[spec["name"]] = {spec["label"]: ids}

    try:
        fp_before = probe.fingerprint()

        # (ii) Domänen-Capture — volle Felder
        routing: Dict[str, Dict[str, Any]] = {}
        for name in DOMAIN_ORDER:
            routing[name] = probe.layerwise_routing(DOMAINS[name])

        # (iii) Lens-Proben
        lens: Dict[str, Dict[str, Any]] = {}
        for spec in LENS_PROBES:
            lens[spec["name"]] = probe.layerwise_logits(
                spec["text"], answer_targets=answer_targets[spec["name"]])

        # (iv) Doubletten (Det)
        det_counts = probe.layerwise_routing(DOMAINS["math_logic"])
        det_lens = probe.layerwise_logits(
            LENS_PROBES[0]["text"], answer_targets=answer_targets[LENS_PROBES[0]["name"]])

        fp_after = probe.fingerprint()

        # -- V1 canon parity (Kanon-Zählbasis exakt) -------------------------
        canon = measure_domain_routing(
            model, tokenizer, device, max_length=MAX_LENGTH,
            corpus={name: {"samples": DOMAINS[name]}
                    for name in DOMAIN_ORDER})
        parity_tvs: Dict[str, float] = {}
        row_sum_max_dev = 0.0
        for name in DOMAIN_ORDER:
            canon_pct = canon[name][0].float() / 100.0          # -> Fraction
            ours_agg = routing[name]["counts_26x8"].sum(dim=0)
            ours_agg = ours_agg / ours_agg.sum().clamp(min=1.0)
            parity_tvs[name] = canon_tv(canon_pct, ours_agg)
            row_sum_max_dev = max(
                row_sum_max_dev,
                float((routing[name]["pct_26x8"].sum(dim=1) - 1.0).abs().max().item()))
        fields_complete = (
            all(routing[name]["pct_26x8"].shape == (26, 8) for name in DOMAIN_ORDER)
            and all(len(routing[name]["entropy_26"]) == 26 for name in DOMAIN_ORDER)
            and all(routing[name]["n_tokens"] > 0 for name in DOMAIN_ORDER)
            and all(len(lens[s["name"]]["rows"]) == probe.n_layers + 1 for s in LENS_PROBES)
        )
        parity_ok = (max(parity_tvs.values()) <= 1e-6
                     and row_sum_max_dev <= ROW_SUM_TOL
                     and fields_complete)
        findings.append((
            "mi1_canon_parity", "PASS" if parity_ok else "FAIL",
            f"parity_tv_max={max(parity_tvs.values()):.3e} "
            f"row_sum_dev={row_sum_max_dev:.3e} fields={fields_complete}"))
        data["parity"] = {
            "tv_by_domain": parity_tvs,
            "row_sum_max_dev": row_sum_max_dev,
            "fields_complete": fields_complete,
        }

        # -- V2 Layer-Auflösung (T2) ----------------------------------------
        depth = evaluate_depth({n: routing[n]["pct_26x8"] for n in DOMAIN_ORDER})
        resolution_ok = all(
            v > 0.0 for v in depth["per_domain_max_row_tv"].values())
        findings.append((
            "mi1_layer_resolution", "PASS" if resolution_ok else "FAIL",
            "; ".join(f"{k}={v:.4f}"
                      for k, v in depth["per_domain_max_row_tv"].items())))
        aggregates = {n: aggregate_row_label(routing[n]["pct_26x8"])
                      for n in DOMAIN_ORDER}
        data["aggregates_labeled"] = aggregates

        # -- V3 purity + det ------------------------------------------------
        fp_ok = probe.fingerprint_equal(fp_before, fp_after)
        first_counts = routing["math_logic"]["counts_26x8"]
        det_ok = (torch.equal(first_counts, det_counts["counts_26x8"])
                  and torch.equal(routing["math_logic"]["pct_26x8"],
                                  det_counts["pct_26x8"])
                  and torch.equal(routing["math_logic"]["entropy_26"],
                                  det_counts["entropy_26"]))
        lens_rows_first = lens[LENS_PROBES[0]["name"]]["rows"]
        det_lens_ok = all(
            a["top1_id"] == b["top1_id"] and a["gap_tv"] == b["gap_tv"]
            for a, b in zip(lens_rows_first, det_lens["rows"]))
        pure_ok = fp_ok and det_ok and det_lens_ok
        findings.append((
            "mi1_purity_det", "PASS" if pure_ok else "FAIL",
            f"fingerprint={fp_ok} det_routing={det_ok} det_lens={det_lens_ok}"))
        data["purity_det"] = {"fingerprint": fp_ok, "det_routing": det_ok,
                              "det_lens": det_lens_ok}

        # -- V4 Tiefen-Spezialisierung --------------------------------------
        core_mean = depth["core_mean_tv"]
        edge_mean = depth["edge_mean_tv"]
        depth_ok = core_mean > CORE_TV_MIN and core_mean > edge_mean
        findings.append((
            "mi1_depth_specialization", "PASS" if depth_ok else "FAIL",
            f"core={core_mean:.4f} edge={edge_mean:.4f} "
            f"grenze={CORE_TV_MIN} (F1 core<=grenze, F2 core<=edge)"))
        data["depth"] = {
            "tv_by_layer": depth["tv_by_layer"],
            "core_mean_tv": core_mean, "edge_mean_tv": edge_mean,
            "peak_layer": int(max(range(26), key=lambda i: depth["tv_by_layer"][i])),
        }

        # -- V5 Entropie-Regime ---------------------------------------------
        entropy = evaluate_entropy(
            {n: routing[n]["entropy_26"] for n in DOMAIN_ORDER})
        wins = entropy["core_lower_count"]
        if wins >= ENTROPY_PASS_N:
            status, note = "PASS", f"core<edge in {wins}/6 Domänen"
        elif wins >= ENTROPY_OFFEN_N:
            status, note = "OFFEN", f"core<edge nur in {wins}/6 Domänen (4-5)"
        else:
            status, note = "FAIL", f"core<edge nur in {wins}/6 Domänen (<=3)"
        findings.append(("mi1_entropy_regime", status, note))
        data["entropy"] = entropy

        # -- V6 Kristallisation ---------------------------------------------
        unresolved = [s["name"] for s in LENS_PROBES
                      if not answer_targets[s["name"]][s["label"]]]
        violations: List[str] = []
        stars: Dict[str, Optional[int]] = {}
        for spec in LENS_PROBES:
            name = spec["name"]
            info = lens[name]
            stars[name] = probe.crystallization_start(info["rows"])
            if info["anchor_tv"] > LENS_ANCHOR_TOL:
                violations.append(f"{name}: anchor_tv={info['anchor_tv']:.4f}")
            if info["p_final_top1_id"] != info["official_top1_id"]:
                violations.append(
                    f"{name}: lens_top1={info['p_final_top1_str']!r} != "
                    f"official={info['official_top1_str']!r}")
        if unresolved:
            status = "OFFEN"
            note = (f"Antwort-Token nicht auflösbar für {unresolved} "
                    f"(kein stiller Fallback; Trajektorien persistiert)")
        elif violations:
            status = "FAIL"
            note = "; ".join(violations)
        elif not any(st is not None for st in stars.values()):
            status = "OFFEN"
            note = ("kein stabiles Startfenster in allen Proben "
                    "(Via-Negativa; Trajektorien persistiert)")
        elif any(stars[n] is not None and stars[n] in core for n in stars):
            status = "PASS"
            note = ("; ".join(f"{n}: ℓ*={stars[n]}"
                      for n in stars if stars[n] is not None)
                    + f" | core={core[0]}..{core[-1]}")
        else:
            status = "FAIL"
            note = ("stabile ℓ* außerhalb des Core-Bands in allen Proben: "
                    + "; ".join(f"{n}={stars[n]}" for n in stars if stars[n] is not None))
        findings.append(("mi1_crystallization", status, note))
        data["lens"] = {
            name: {
                "rows": lens[name]["rows"],
                "l_star": stars[name],
                "l_star_in_core": (stars[name] in core
                                   if stars[name] is not None else False),
                "p_final_top1_str": lens[name]["p_final_top1_str"],
                "official_top1_str": lens[name]["official_top1_str"],
                "anchor_tv": lens[name]["anchor_tv"],
                "answer_prob_final": {
                    spec["label"]: lens[name]["rows"][-1].get("answer_prob", {}).get(
                        spec["label"], 0.0)
                    for spec in LENS_PROBES if spec["name"] == name},
            }
            for spec in LENS_PROBES for name in [spec["name"]]
        }

        # -- Felder persistieren (P3 Roh vor Auswertung) ---------------------
        data["domains"] = {
            n: {
                "pct_26x8": routing[n]["pct_26x8"].tolist(),
                "pct_percent_26x8": routing[n]["pct_percent_26x8"].tolist(),
                "entropy_26": routing[n]["entropy_26"].tolist(),
                "n_tokens": routing[n]["n_tokens"],
            }
            for n in DOMAIN_ORDER
        }
        data["paris_trajektory_note"] = (
            "full rows in lens[name]['rows']: top1-Trajektorie + gap_tv + answer_prob")
    except Exception as exc:  # Ausnahme = Instrument-Bruch -> OFFEN, kein Fallover
        for verdict in ("mi1_canon_parity", "mi1_layer_resolution",
                        "mi1_purity_det", "mi1_depth_specialization",
                        "mi1_entropy_regime", "mi1_crystallization"):
            if not any(f[0] == verdict for f in findings):
                findings.append((verdict, "OFFEN",
                                 f"GPU-Bruch: {type(exc).__name__}: {exc}"))
    return findings, data


# -- Main --------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true",
                        help="Hauptlauf auf dem trainierten Checkpoint")
    parser.add_argument("--checkpoint_base", default="./argon_1b")
    parser.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    parser.add_argument("--json", default=JSON_PATH)
    args = parser.parse_args()

    t0 = time.time()
    findings = pilot_math()
    data: Dict[str, Any] = {"pilot": [
        {"name": f[0], "status": f[1], "note": f[2]} for f in findings]}

    gpu_broken = False
    if args.gpu:
        gpu_findings, gpu_data = run_gpu(
            args.checkpoint_base, args.checkpoint_machines)
        findings.extend(gpu_findings)
        data["gpu"] = gpu_data
    else:
        gpu_broken = True
        for verdict in ("mi1_canon_parity", "mi1_layer_resolution",
                        "mi1_purity_det", "mi1_depth_specialization",
                        "mi1_entropy_regime", "mi1_crystallization"):
            findings.append((verdict, "OFFEN",
                             "GPU-Lane OFFEN (ohne --gpu nicht gefahren; "
                             "kein stiller Fallback)"))
        data["gpu"] = {}

    print("=" * 88)
    print("MI-1 — 26-LAYER-RÖNTGERÄT (Routing-Tomographie + Logit-Lens)")
    print("=" * 88)
    for name, status, note in findings:
        print(f"[{status:6s}] {name}: {note}")

    verdicts = [f for f in findings if not f[0].startswith("pilot_")]
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
        print("GRÜN — alle 6 Verdikte PASS; das 26-Layer-Feld ist gemessen.")
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