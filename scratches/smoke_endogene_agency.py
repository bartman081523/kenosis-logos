"""
smoke_endogene_agency.py — SP13: kontrafaktisches Probehandeln & endogene
Agency (die Maschine erprobt ihre Subspaces selbst; Mauer 2, Nordstern-
Nachricht 2026-10-26, Task 2).

Canon: composite.py:219-301 (SP13-Sektion: observe :220-230, _snapshot
:232-241, _restore_snapshot :243-250, step_autonomous :252-301);
turing_machine.py:323-327 (SP13-Extension: "axiom_pool" :324 — expand_basis
pop(0) :149 und innerer Expand :241-246 laufen auch in Proben — und
"last_verdict" :325);
wittgenstein_machine.py:507-525 (SP13-Extension: noise_level/_schedule_index
— der S2/S3-Schedule mutiert noise_level je Batch :313-316 und zählt
_schedule_index :338 — plus das Lexikon, learn_pair ERSETZT Einträge
:232-233);
machine_base.py:502-516 (observe-Rahmen), :392-421 (steer — @contextmanager
mit exakter Bias-Restaurierung im finally), :541-545 (digest);
goedel_machine.py:163-172 (oracle-mode _prediction via oracle.hidden_state);
verify_machines.py:355-399 (GPU-Muster: _load_machines_model + ArgonOracle
+ steer-by-domain + Bias-Restore-Verifikation).

PREREG (P2-Freeze — fixiert vor dem Erstrun; Canon-Audit 2026-10-26 im
Plan ~/.claude/plans/sp13-endogene-agency.md: BaseMachine.observe existiert
bereits, TM-Snapshot fehlte axiom_pool [User-Fund], WM-Snapshot fehlte
noise_level/_schedule_index/lexicon).

Kernthese (User-Nordstern, Task 2): die Maschine handelt nicht passiv —
sie ERPROBT jeden Kandidaten-Subspace kontrafaktisch (nicht-mutierende
Proben) und wählt das Preset mit minimalem Composite-Gap. Ohne Oracle
(CPU) degeneriert die Wahl zum Tie (alle Proben identisch) — DATUM, kein
Band; die Wahl-Aussage ist eine GPU-Aussage (argon_1b_machines, oracle).

VERDICT-MAP (Bands fixiert VOR dem Erstrun):
  sp13_l0_anker — Zero-Regression: die kommittete SP11-lane (importiert)
    fields_ok/first_detect_ok/deterministic; die kommittete SP12-l0-lane
    (importiert) True/True/True; der SP12-Zwilling reproduziert die
    KOMMITTETE lexical_grounding_experiment.json FELDWEISE (verdicts a/b,
    basis_a/b, basis_b_sizes, expansions a/b, resolver_direct, batch-
    Gleichheiten — 10 Feldechecks). Falsifikator: jede abweichende Serie.
  sp13_snapshot_purity — observe()-Nebenwirkungsfreiheit: (CPU) N=5
    composite.observe("spiegel", lexical_grounding=True) im gelehrten
    Lexikon/LOOP-Trägheitszustand — Bitidentität des Vollfingerprint
    (digest, 3 RNG-States, axiom_pool, basis, expansions, noise_level,
    _schedule_index, lexicon-Vektoren, self_model, input_log,
    experiment_history, generated_rules, halt_verdicts); die Proben SEHEN
    LOOP (lexical_ast ('identity','x') je Probe) und fressen den Pool
    NICHT. Plus (CPU) 3 Proben im leeren Lexikon: alle None/HALTED.
    (GPU) dieselbe Purity unter steer-Rotation (goedel/turing/wittgenstein/
    combined/plain je Probe) inkl. expert_bias-Tensoren (26 Gate-Layer,
    torch.equal). Falsifikator: jede Nicht-Identität.
  sp13_det — Determinismus der autonomen Wahl: zwei Biografien
    (6 HALTED-Schritte → Lexikon-Spiel → WM-RNG-Pin → 6 step_autonomous
    -Schritte, seed 42) bitidentisch über chosen_preset-Sequenz,
    candidate_gaps je Schritt, gap-Serie, endgültigen Digest — CPU UND GPU.
    Falsifikator: jede Abweichung.
  sp13_endogenous_switch — (GPU-only) echte Wahl: jede der 12 Schritte
    hat ENTARTETE candidate_gaps zu vermeiden (max-minus-min > 0 je
    Schritt) UND der Anteil der Schritte mit chosen_preset != "raw"
    ist >= 0.8. Reaktion auf HALTED vs LOOP (Phase-Wechsel im Wortprogramm)
    wird als DATUM dokumentiert. Falsifikator: Degeneration ODER
    raw-Anteil > 0.2.
  sp13_gap_dominance — (GPU-only) Kohärenz-Gewinn: je autonomem Schritt
    gilt result.gap == best_gap == min(candidate_gaps) (EXAKT), und der
    Mittelwert der 12 autonomen Gaps liegt STRIKT unter dem Mittelwert des
    ungesteuerten raw-Zwillings (dieselbe Biografie ohne autonomy). Der
    CPU-Lauf dokumentiert die Tie-Degeneration als DATUM (alle Gaps
    gleich, chosen == "raw"). Falsifikator: Gap-Verletzung je Schritt
    ODER nicht strikt kleinerer Mittelwert.
  sp13_causal_loop_closed — Kopplung SP12 x SP13: nach dem Lexikon-Spiel
    sieht observe() den Wort-erzeugten LOOP voraus und restauriert
    verlustfrei (axiom_pool/basis/expansions identisch); der reale Schritt
    unter step_autonomous vollzieht die Expansion 3 -> 5:
    invariants.lexical_ast == "('identity', 'x')", verdict LOOP,
    expansions == ['step1:LOOP'], invariants["autonomous"] protokolliert
    Wahl + candidate_gaps. (GPU: zusätzlich die Wahl UNTER den Proben am
    resolvierten Schritt.) Falsifikator: Pool-Differenz nach der Vorhersage
    ODER ausbleibende Expansion ODER fehlendes autonomous-Protokoll.

Lanes: cpu (immer): l0_anker, snapshot_purity, det, causal_loop (CPU-
Varianten); gpu (hinter --gpu / SP13_GPU=1): die vollständigen 6 Verdikte
inkl. steer-Rotation und Wahl-Aussagen. Exit-Codes: 0 = alle gefahrenen
Verdikte PASS; 1 = FAIL vorhanden; 2 = OFFEN (GPU-Lanes angefordert,
aber nicht gefahren — kein stiller Fallback). Die Wahl-Aussagen (switch/
dominance) sind NUR die GPU-Aussagen; der CPU-Teil dokumentiert die Tie.

Run (CPU):  python -u -m scratches.smoke_endogene_agency
Run (GPU):  python -u -m scratches.smoke_endogene_agency --gpu
            (Checkpoints ./argon_1b + ./argon_1b_machines, Muster
            verify_machines.py:355-399)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.composite import CompositeMachine
from machines.machine_base import MachineConfig
from scratches.smoke_lexical_grounding import (
    LEX_PAIR,
    POOL6,
    ROUNDS,
    lane_l0_anchor as sp12_lane_l0,
    lane_zwilling,
)
from scratches.smoke_paradigm_ursprung import lane_l0_anker  # kommittete SP11-lane

SEED = 42
STEPS_PHASE = 6          # 6 HALTED-Schritte + 6 autonome LOOP-Schritte = 12
N_PROBES = 5
CANDIDATES = (None, "goedel", "turing", "wittgenstein", "combined")
SP12_JSON = "scratches/lexical_grounding_experiment.json"
LOG_PATH = "scratches/endogene_agency_experiment.log"
JSON_PATH = "scratches/endogene_agency_experiment.json"
GPU_STEER_ROTATION = ("goedel", "turing", "wittgenstein", "combined", None)


def round6(x: Any) -> Any:
    return round(float(x), 6) if x is not None else None


def make_composite(mode: str = "in_loop",
                   oracle: Any = None) -> CompositeMachine:
    """Frische Composite seed 42 (in_loop oder oracle, Canon-Pool POOL6)."""
    return CompositeMachine(MachineConfig(seed=SEED, mode=mode,
                                          params=dict(POOL6)),
                            oracle=oracle)


def learn_and_pin(machine: CompositeMachine) -> List[Dict[str, Any]]:
    """SP12-Spiel + WM-RNG-Pinning (Arm-Schicht, dokumentiert)."""
    protocol = machine.wittgenstein.lexicon_game(
        [LEX_PAIR], rounds=ROUNDS, noise_level=0.0)
    machine.wittgenstein._rng = np.random.default_rng(SEED)
    protocol = [dict(entry, resonance_after=round6(entry["resonance_after"]))
                for entry in protocol]
    return protocol


def fingerprint(machine: CompositeMachine) -> Dict[str, Any]:
    """Vollfingerprint des Composite-Zustands (JSON-stabil)."""
    t, w, g = machine.turing, machine.wittgenstein, machine.goedel
    lex = w.tokenizer.lexicon
    return {
        "digest": machine.digest(),
        "rng": {name: json.dumps(gen._rng.bit_generator.state, sort_keys=True)
                for name, gen in (("goedel", g), ("turing", t),
                                  ("wittgenstein", w))},
        "axiom_pool": list(t.axiom_pool),
        "basis": list(t.basis),
        "expansions": list(machine.expansions),
        "step": machine._step_index,
        "noise_level": w.noise_level,
        "schedule_index": w._schedule_index,
        "lexicon": {k: np.asarray(v).tolist() for k, v in lex.items()},
        "self_model": np.abs(g.self_model).sum().item(),
        "input_log_len": len(g.input_log),
        "experiment_history_len": len(w.experiment_history),
        "generated_rules_len": len(w.generated_rules),
        "halt_verdicts": list(t.halt_verdicts),
        "transition_count": t._transition_count,
        "trace_len": len(machine._trace),
        "gap_history": [round6(x) for x in machine._gap_history],
    }


def lane_l0() -> Dict[str, Any]:
    """Zero-Regression: SP11-lane + SP12-lane (beide importiert) + der
    SP12-Zwilling FELDWEISE gegen die KOMMITTETE JSON."""
    l0 = lane_l0_anker()
    l0_re = lane_l0_anker()
    s12 = sp12_lane_l0()
    with open(SP12_JSON, "r", encoding="utf-8") as fh:
        committed = json.load(fh)["twin"]
    twin = lane_zwilling()
    fields_equal = {
        "verdicts_a": twin["verdicts_a"] == committed["verdicts_a"],
        "verdicts_b": twin["verdicts_b"] == committed["verdicts_b"],
        "basis_a": twin["basis_a"] == committed["basis_a"],
        "basis_b": twin["basis_b"] == committed["basis_b"],
        "basis_b_sizes": twin["basis_b_sizes"] == committed["basis_b_sizes"],
        "expansions_a": twin["expansions_a"] == committed["expansions_a"],
        "expansions_b": twin["expansions_b"] == committed["expansions_b"],
        "resolver_direct": twin["resolver_direct"]
        == committed["resolver_direct"],
        "batches_ground_plain": twin["batches_equal_grounded_plain"]
        == committed["batches_equal_grounded_plain"],
        "batches_a_b": twin["batches_equal_a_b"]
        == committed["batches_equal_a_b"],
    }
    return {"sp11": {"fields_ok": bool(l0.get("fields_ok")),
                     "first_detect_ok": bool(l0.get("first_detect_ok")),
                     "deterministic": l0 == l0_re},
            "sp12_lane": {"fields_ok": bool(s12["fields_ok"]),
                          "first_detect_ok": bool(s12["first_detect_ok"]),
                          "deterministic": s12["deterministic"]},
            "sp12_twin_fields": fields_equal}


def lane_purity_cpu() -> Dict[str, Any]:
    """Proben im gelehrten Lexikon/LOOP-Trägheitszustand (N=5) und im
    leeren Lexikon (N=3): Bitidentität je Probe."""
    machine = make_composite()
    learn_and_pin(machine)
    machine.step("spiegel", lexical_grounding=True)         # LOOP-Trägheit
    fp0 = fingerprint(machine)
    b_state: Dict[str, Any] = {"probes": [], "purity": True, "diffs": []}
    for i in range(N_PROBES):
        probe = machine.observe("spiegel", lexical_grounding=True)
        fpN = fingerprint(machine)
        same = fpN == fp0
        b_state["purity"] = b_state["purity"] and same
        b_state["probes"].append({"n": i + 1,
                                  "verdict": probe.invariants["verdict"],
                                  "lexical_ast": probe.invariants["lexical_ast"],
                                  "purity": same})
        if not same:
            b_state["diffs"].append({"n": i + 1, "keys": sorted(
                k for k in fp0 if fp0[k] != fpN[k])})
    plain = make_composite()
    fpA = fingerprint(plain)
    a_state: Dict[str, Any] = {"probes": [], "purity": True}
    for i in range(3):
        probe = plain.observe("spiegel", lexical_grounding=True)
        same = fingerprint(plain) == fpA
        a_state["purity"] = a_state["purity"] and same
        a_state["probes"].append({"n": i + 1,
                                  "verdict": probe.invariants["verdict"],
                                  "lexical_ast": probe.invariants["lexical_ast"],
                                  "purity": same})
    return {"b_state": b_state, "a_state": a_state,
            "axiom_pool_unchanged": b_state["probes"][0]["purity"]}


def cpu_bio(steps_phase: int = STEPS_PHASE) -> Tuple[CompositeMachine, Dict[str, Any]]:
    """Biografie: 6 ungesteuerte Schritte -> Lexikon-Spiel -> Pin ->
    6 autonome Schritte (CPU: Tie-Degeneration)."""
    machine = make_composite()
    halted: List[str] = [machine.step("spiegel",
                                      lexical_grounding=False).invariants["verdict"]
                         for _ in range(steps_phase)]
    learn_and_pin(machine)
    auto_rows: List[Dict[str, Any]] = []
    for _ in range(steps_phase):
        result = machine.step_autonomous("spiegel",
                                         candidates=CANDIDATES,
                                         lexical_grounding=True)
        auto_rows.append({"verdict": result.invariants["verdict"],
                          "lexical_ast": result.invariants["lexical_ast"],
                          "chosen_preset": result.invariants["autonomous"]
                          ["chosen_preset"],
                          "candidate_gaps": {k: round6(v) for k, v in
                                             result.invariants["autonomous"]
                                             ["candidate_gaps"].items()},
                          "gap": round6(result.gap)})
    meta = {"halted_verdicts": halted, "autonomous": auto_rows}
    return machine, meta


def lane_det_cpu() -> Dict[str, Any]:
    m1, bio1 = cpu_bio()
    m2, bio2 = cpu_bio()
    equal = bio1 == bio2
    return {"deterministic": equal,
            "digest_equal": m1.digest() == m2.digest(),
            "expansions_equal": m1.expansions == m2.expansions,
            "tie_datum": all(r["chosen_preset"] == "raw"
                             for r in bio1["autonomous"])}


def lane_causal_cpu() -> Dict[str, Any]:
    """SP12 x SP13 (CPU): Voraus-Sicht + verlustfreie Restaurierung +
    reale Expansion 3 -> 5."""
    machine = make_composite()
    learn_and_pin(machine)
    fp0 = fingerprint(machine)
    probe = machine.observe("spiegel", lexical_grounding=True)
    fp1 = fingerprint(machine)
    seen = {"probe_verdict": probe.invariants["verdict"],
            "probe_lexical_ast": probe.invariants["lexical_ast"],
            "pool_restored": fp1["axiom_pool"] == fp0["axiom_pool"],
            "basis_restored": fp1["basis"] == fp0["basis"],
            "expand_restored": fp1["expansions"] == fp0["expansions"],
            "fingerprint_restored": fp1 == fp0}
    real = machine.step_autonomous("spiegel", candidates=CANDIDATES,
                                   lexical_grounding=True)
    return {"seen": seen,
            "real": {"verdict": real.invariants["verdict"],
                     "lexical_ast": real.invariants["lexical_ast"],
                     "basis_after": len(machine.turing.state()["basis"]),
                     "expansions": list(machine.expansions),
                     "autonomous": result_autonomous(real)}}


def result_autonomous(result: Any) -> Dict[str, Any]:
    """RAW-Fließkommawerte (die EXAKT-Bänder vergleichen ungerundete
    Floats; gerundete Sichten nur für die Anzeige)."""
    au = result.invariants.get("autonomous", {})
    return {"chosen_preset": au.get("chosen_preset"),
            "best_gap": au.get("best_gap"),
            "candidate_gaps": dict(au.get("candidate_gaps", {}))}


# -- GPU-Lanes ---------------------------------------------------------------

def gpu_context(checkpoint_base: str, checkpoint_machines: str):
    """Argon-Orakel-Kontext (Muster verify_machines.py:355-399)."""
    import torch

    from machines.machine_base import ArgonOracle
    from machines.verify_machines import _load_machines_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_machines_model(checkpoint_base, checkpoint_machines)
    oracle = ArgonOracle(model, tokenizer, device=device)
    return torch, model, tokenizer, oracle, device


def bias_fingerprint(torch: Any, model: Any) -> List[Any]:
    return [layer.moe.gate.expert_bias.data.cpu().clone()
            for layer in model.model.layers]


def bias_equal(torch: Any, before: List[Any], after: List[Any]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(before, after))


def lane_purity_gpu(ctx) -> Dict[str, Any]:
    """Purity UNTER steer-Rotation: je Probe ein anderer Kontext (incl.
    plain); Maschinen-Fingerprint + expert_bias-Tensoren bitidentisch."""
    torch, model, _tok, oracle, _dev = ctx
    machine = make_composite(mode="oracle", oracle=oracle)
    learn_and_pin(machine)
    machine.step("spiegel", lexical_grounding=True)
    fp0 = fingerprint(machine)
    b0 = bias_fingerprint(torch, model)
    probes: List[Dict[str, Any]] = []
    purity = True
    bias_ok = True
    for i, preset in enumerate(GPU_STEER_ROTATION):
        label = "raw" if preset is None else preset
        if preset is None:
            probe = machine.observe("spiegel", lexical_grounding=True)
        else:
            with oracle.steer(preset):
                probe = machine.observe("spiegel", lexical_grounding=True)
        fpN = fingerprint(machine)
        same = fpN == fp0
        b_ok = bias_equal(torch, b0, bias_fingerprint(torch, model))
        purity = purity and same
        bias_ok = bias_ok and b_ok
        probes.append({"n": i + 1, "preset": label,
                       "verdict": probe.invariants["verdict"],
                       "lexical_ast": probe.invariants["lexical_ast"],
                       "purity": same, "bias_restored": b_ok})
        if not same:
            probes[-1]["diff_keys"] = sorted(k for k in fp0 if fp0[k] != fpN[k])
    return {"probes": probes, "purity": purity, "expert_bias_restored": bias_ok}


def gpu_bio(ctx, autonomous: bool) -> Tuple[CompositeMachine, Dict[str, Any]]:
    """GPU-Biografie: 6 ungesteuerte Schritte -> Lexikon-Spiel -> Pin ->
    6 Schritte als step_autonomous (oder als plain, raw-Zwilling)."""
    _torch, _model, _tok, oracle, _dev = ctx
    machine = make_composite(mode="oracle", oracle=oracle)
    halted: List[str] = [machine.step("spiegel",
                                      lexical_grounding=False).invariants["verdict"]
                         for _ in range(STEPS_PHASE)]
    learn_and_pin(machine)
    rows: List[Dict[str, Any]] = []
    for _ in range(STEPS_PHASE):
        if autonomous:
            result = machine.step_autonomous("spiegel", candidates=CANDIDATES,
                                             lexical_grounding=True)
            au = result_autonomous(result)
        else:
            result = machine.step("spiegel", lexical_grounding=True)
            au = None
        rows.append({"verdict": result.invariants["verdict"],
                     "lexical_ast": result.invariants["lexical_ast"],
                     "autonomous": au, "gap": float(result.gap)})
    meta = {"halted_verdicts": halted, "loop_phase": rows}
    return machine, meta


def lane_autonomy_gpu(ctx) -> Dict[str, Any]:
    """Die Wahl-Aussagen (switch + dominance) + det-Doppellauf GPU."""
    m1, bio1 = gpu_bio(ctx, autonomous=True)
    m2, bio2 = gpu_bio(ctx, autonomous=True)
    det = bio1 == bio2 and m1.digest() == m2.digest()
    rows = bio1["loop_phase"]
    spread_all = True
    non_raw = 0
    dominance_exact = True
    for r in rows:
        au = r["autonomous"]
        values = [float(v) for v in au["candidate_gaps"].values()]
        # EXAKT (ungerundete Floats): spread > 0 und
        # result.gap == best_gap == min(candidate_gaps).
        spread_all = spread_all and (max(values) - min(values) > 0.0)
        if au["chosen_preset"] != "raw":
            non_raw += 1
        if not (r["gap"] == au["best_gap"]
                and r["gap"] == min(values)):
            dominance_exact = False
    _twin, raw_bio = gpu_bio(ctx, autonomous=False)
    raw_gaps = [r["gap"] for r in raw_bio["loop_phase"]]
    auto_gaps = [r["gap"] for r in rows]
    mean_auto = sum(auto_gaps) / len(auto_gaps)
    mean_raw = sum(raw_gaps) / len(raw_gaps)
    dominant = mean_auto < mean_raw
    return {"det_doppellauf": det,
            "spread_pro_schritt": spread_all,
            "non_raw_anteil": non_raw / len(rows),
            "mean_auto": mean_auto, "mean_raw": mean_raw,
            "dominant_strikt": dominant,
            "gap_dominance": dominance_exact and dominant,
            "bio": bio1, "raw_bio": raw_bio}


def lane_causal_gpu(ctx) -> Dict[str, Any]:
    """SP12 x SP13 (GPU): Voraus-Sicht + Expansion 3 -> 5 UNTER der Wahl."""
    _torch, _model, _tok, oracle, _dev = ctx
    machine = make_composite(mode="oracle", oracle=oracle)
    learn_and_pin(machine)
    fp0 = fingerprint(machine)
    probe = machine.observe("spiegel", lexical_grounding=True)
    fp1 = fingerprint(machine)
    seen = {"probe_verdict": probe.invariants["verdict"],
            "probe_lexical_ast": probe.invariants["lexical_ast"],
            "fingerprint_restored": fp1 == fp0}
    real = machine.step_autonomous("spiegel", candidates=CANDIDATES,
                                   lexical_grounding=True)
    return {"seen": seen,
            "real": {"verdict": real.invariants["verdict"],
                     "lexical_ast": real.invariants["lexical_ast"],
                     "basis_after": len(machine.turing.state()["basis"]),
                     "expansions": list(machine.expansions),
                     "autonomous": result_autonomous(real)}}


def findings(l0: Dict[str, Any], purity: Dict[str, Any], det: Dict[str, Any],
             causal: Dict[str, Any], gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ok_l0 = (l0["sp11"]["fields_ok"] and l0["sp11"]["first_detect_ok"]
             and l0["sp11"]["deterministic"]
             and l0["sp12_lane"]["fields_ok"]
             and l0["sp12_lane"]["first_detect_ok"]
             and l0["sp12_lane"]["deterministic"]
             and l0["sp12_twin_fields"] is not None
             and all(l0["sp12_twin_fields"].values()))
    rows.append({"id": "sp13_l0_anker",
                 "data": {"sp11": l0["sp11"], "sp12_lane": l0["sp12_lane"],
                          "twin_fields": l0["sp12_twin_fields"]},
                 "assert": ok_l0,
                 "note": ("Zero-Regression über importierte committete Lanes "
                          "+ feldeidentischer SP12-Zwilling")})

    ok_purity = (purity["b_state"]["purity"]
                 and purity["a_state"]["purity"]
                 and all(p["verdict"] == "LOOP" for p in purity["b_state"]["probes"])
                 and all(p["lexical_ast"] == "('identity', 'x')"
                         for p in purity["b_state"]["probes"])
                 and all(p["verdict"] == "HALTED"
                         for p in purity["a_state"]["probes"]))
    if gpu and "purity" in gpu:
        gp = gpu["purity"]
        ok_purity = ok_purity and gp["purity"] and gp["expert_bias_restored"]
    rows.append({"id": "sp13_snapshot_purity",
                 "data": {"b_state_probes": purity["b_state"]["probes"],
                          "a_state_purity": purity["a_state"]["purity"],
                          "gpu_purity": (gpu["purity"]["purity"]
                                         and gpu["purity"]["expert_bias_restored"])
                          if gpu and "purity" in gpu else None},
                 "assert": ok_purity,
                 "note": ("Vorblick sieht LOOP, fressen der axiom_pool NICHT "
                          "(User-Fund Turing-Snapshot); GPU: Bias-Tensoren "
                          "unter steer-Rotation restauriert")})

    ok_det = det["deterministic"] and det["digest_equal"]
    if gpu and "autonomy" in gpu:
        ok_det = ok_det and gpu["autonomy"]["det_doppellauf"]
    rows.append({"id": "sp13_det",
                 "data": {"cpu": {"deterministic": det["deterministic"],
                                  "digest_equal": det["digest_equal"],
                                  "tie_datum": det["tie_datum"]},
                          "gpu_det": gpu["autonomy"]["det_doppellauf"]
                          if gpu and "autonomy" in gpu else None},
                 "assert": ok_det,
                 "note": ("Autonome Läufe bitidentisch; CPU dokumentiert die "
                          "Tie-Degeneration als DATUM")})

    causal_ok = (causal["seen"]["probe_verdict"] == "LOOP"
                 and causal["seen"]["probe_lexical_ast"] == "('identity', 'x')"
                 and causal["seen"]["fingerprint_restored"]
                 and causal["real"]["verdict"] == "LOOP"
                 and causal["real"]["lexical_ast"] == "('identity', 'x')"
                 and causal["real"]["basis_after"] == 5
                 and causal["real"]["expansions"] == ["step1:LOOP"])
    if gpu and "causal" in gpu:
        g = gpu["causal"]
        causal_ok = causal_ok and g["seen"]["fingerprint_restored"] \
            and g["seen"]["probe_verdict"] == "LOOP" \
            and g["real"]["basis_after"] == 5 \
            and g["real"]["expansions"] == ["step1:LOOP"] \
            and g["real"]["autonomous"]["chosen_preset"] is not None
    rows.append({"id": "sp13_causal_loop_closed",
                 "data": {"cpu_seen": causal["seen"],
                          "cpu_real": causal["real"],
                          "gpu_seen": gpu["causal"]["seen"]
                          if gpu and "causal" in gpu else None,
                          "gpu_real": gpu["causal"]["real"]
                          if gpu and "causal" in gpu else None},
                 "assert": causal_ok,
                 "note": ("Wort -> Programm -> Voraus-Sicht -> Expansion 3->5 "
                          "unter der Wahl (SP12 x SP13 gekoppelt)")})

    if gpu and "autonomy" in gpu:
        au = gpu["autonomy"]
        ok_switch = au["spread_pro_schritt"] and au["non_raw_anteil"] >= 0.8
        rows.append({"id": "sp13_endogenous_switch",
                     "data": {"non_raw_anteil": au["non_raw_anteil"],
                              "spread_pro_schritt": au["spread_pro_schritt"]},
                     "assert": ok_switch,
                     "note": ("Echte Wahl statt Einheits-Preset; HALTED/LOOP-"
                              "Reaktion als Datums in bio1 geliefert")})
        ok_dom = au["gap_dominance"]
        rows.append({"id": "sp13_gap_dominance",
                     "data": {"mean_auto": au["mean_auto"],
                              "mean_raw": au["mean_raw"],
                              "det_doppellauf": au["det_doppellauf"]},
                     "assert": ok_dom,
                     "note": ("g_auto(n) == best_gap == min(candidate_gaps) "
                              "je Schritt exakt; Mittelwert strikt unter raw")})
    else:
        rows.append({"id": "sp13_endogenous_switch",
                     "data": {}, "assert": None,
                     "note": "GPU-Lane OFFEN (ohne --gpu nicht gefahren)"})
        rows.append({"id": "sp13_gap_dominance",
                     "data": {}, "assert": None,
                     "note": "GPU-Lane OFFEN (ohne --gpu nicht gefahren)"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true",
                        help="GPU-Lanes (argon_1b_machines oracle-mode) fahren")
    parser.add_argument("--checkpoint_base", default="./argon_1b")
    parser.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    args = parser.parse_args()
    env_gpu = os.environ.get("SP13_GPU", "") in ("1", "true", "True")
    want_gpu = args.gpu or env_gpu

    print("== SP13 — Endogene Agency: kontrafaktisches Probehandeln ==")
    print("Canon: composite.py:219-301, turing_machine.py:323-327,")
    print("       wittgenstein_machine.py:507-525, machine_base.py:502-516")
    print("Prereg: N_PROBES=5, STEPS_PHASE=6+6, seed 42; Wahl = argmin Composite-Gap")
    print(f"Modus: {'CPU+GPU' if want_gpu else 'CPU'}")

    l0 = lane_l0()
    purity = lane_purity_cpu()
    det = lane_det_cpu()
    causal = lane_causal_cpu()

    gpu: Optional[Dict[str, Any]] = None
    offen = False
    if want_gpu:
        try:
            ctx = gpu_context(args.checkpoint_base, args.checkpoint_machines)
        except Exception as exc:   # noqa: BLE001 — OFFEN, kein stiller Fallback
            print(f"GPU-Kontext nicht verfügbar ({exc}) — GPU-Lanes OFFEN.")
            gpu, offen = None, True
        else:
            gpu = {"purity": lane_purity_gpu(ctx),
                   "autonomy": lane_autonomy_gpu(ctx),
                   "causal": lane_causal_gpu(ctx)}

    rows = findings(l0, purity, det, causal, gpu)

    print()
    print("BEFUNDE:")
    for r in rows:
        if r["assert"] is None:
            print(f"  {r['id']:<26} OFFEN")
        else:
            print(f"  {r['id']:<26} {'PASS' if r['assert'] else 'FAIL'}")
    print()
    print("NOTIZEN:")
    for r in rows:
        print(f"  - {r['id']}: {r['note']}")

    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext", "note": "GPU-Kontext fehlgeschlagen"})
    print()
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN"
          f"{', unverwartet' if fails else ''}")
    print("CPU-Tie-Datum: ohne Oracle sind alle Kandidaten-Gaps identisch;")
    print("die Wahl-Aussagen sind die GPU-Aussagen (oracle-mode).")
    print(f"JSON: {JSON_PATH}")

    payload = {"l0": l0, "purity": purity, "det": det, "causal": causal,
               "gpu": gpu, "findings": rows,
               "all_pass": not fails and not opened, "open": [r["id"] for r in opened]}
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)

    if fails:
        print("ROT — Bands verletzt (Daten, kein Band-Schieben): "
              + ", ".join(r["id"] for r in fails))
        return 1
    if opened:
        print("OFFEN — GPU-Lanes nicht gefahren (kein stiller Fallback).")
        return 2
    print("GRUEN — Endogene Agency: die Maschine waehlt ihren Subspace selbst.")
    return 0


if __name__ == "__main__":
    sys.exit(main())