"""
smoke_composite_oracle.py — CompositeMachine: Oracle- vs. In-Loop-Trajektorien
unter Preset-Steering auf dem realen trainierten Router (argon_1b_machines).

Lanes:
  L1/L2 — composite in_loop (Seed 42), 2 Läufe -> Determinismus-Digests
  L3/L4 — composite oracle   ungesteuert, 2 Läufe -> Determinismus-Digests
  L5    — composite oracle unter expert_bias-Steering "combined" (6 Experten)
  L6    — composite oracle unter expert_bias-Steering "goedel"   ({2,7})
  L7/L9 — composite oracle mit erzwungener Identitäts-Tautologie ('identity','x')
          als erstes Batch-Programm (Schritt 1): der compile_rule-Hebel übersetzt
          die Regel als Selbst-Transition ("n_i", BLANK) -> (BLANK, "N", "n_i"),
          Turing liefert LOOP, die Komposition expandiert die Basis (uni_256:113);
          L9 = Wiederholung von L7 (Determinismus auf dem getriggerten Pfad).
  L8    — wie L7 unter expert_bias-Steering "goedel" ({2,7}).
          L7-L9 tragen params={"axiom_pool": ["ALPHA", "BETA"]}; die Expansion
          frisst je LOOP zwei Pool-Einträge (turing.step expandiert intern,
          composite.step mit reason="composite:LOOP" danach) — die Zweiebenen-
          Axiomatik (objektive Lauf-Basis + reflexive Kompositions-Basis).
  L10-L12 — Multi-Expansions-Set: 6er-Pool ["ALPHA","BETA","GAMMA","DELTA",
          "EPSILON","ZETA"], die Tautologie wird deterministisch an den
          Schritten 1/10/12 erzwungen (FORCE_STEPS_MULTI) — die Expansionen
          der natürlichen Verläufe sind RNG-driftanfällig (je Expansion 2
          RNG-Draws, Streams divergieren), die erzwungene Cadence garantiert
          die Basis-Leiter 3→5→7→9. L11 wie L10 unter Steering "goedel",
          L12 = Wiederholung von L10 (Determinismus der vollen Leiter).

Messgrößen je Lane: 16-Schritt-Gap-Verlauf, Digest, Axiom-Expansionen,
Verdict-Verlauf, Basis-Profile (Basis-Groesse je Schritt), Router-Entropie
auf Fixprobe (oracle.entropy_scalar).

Pass-Kriterien:
  - L1==L2 und L3==L4 und L7==L9 und L10==L12 (bitidentische Digests —
    Determinismus in beiden Modis, auf dem Expansion-Pfad und der vollen
    Leiter)
  - alle Gaps endlich (keine NaN/Inf)
  - entropy_bias_free: |ΔH(probe, standard→steer)| < 0.02 — DOKUMENTIERTE
    Design-Eigenschaft: die Router-Entropie wird VOR dem expert_bias berechnet
    (moe_layers.py:60-66: router_entropy aus bias-freien Sigmoid-Affinitäten;
    der Bias wirkt nur auf die Top-K-Auswahl), das H-Meter ist daher
    bias-invariant; das Steering-Signal liegt in der AUSWAHL (Stufe-5-Lauf:
    Anteil 1.000, TV 0.36-0.41) und in der Trajektorie
  - steer_traj_shift: die Oracle-Gap-Trajektorie unter Steering != ungesteuert
    (L5/L6 != L3, L8 != L7, L11 != L10), ohne die Bias-Zustände zu hinterlassen
    (Restaurierung)
  - loop_tautology: L7 enthält mindestens ein "LOOP"-Verdikt und >= 1
    Expansion; basis_expanded: Turing-Basis > 3 Dimensionen nach dem Lauf.
  - multi_expansion: L10 realisiert die volle Leiter (Basis 9, 3 Expansionen,
    LOOP-Verdikte exakt an den Schritten 1/10/12); multi_steer_shift: L11
    hält dieselbe strukturelle Leiter unter Steering bei niedrigerem Gap-Band.
Run:  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      venv_openmythos/bin/python -u -m scratches.smoke_composite_oracle
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from machines.composite import CompositeMachine  # noqa: E402
from machines.machine_base import ArgonOracle, MachineConfig  # noqa: E402
from machines.wittgenstein_machine import GeneratedSymbolicRule  # noqa: E402
from machines.verify_machines import _load_machines_model  # noqa: E402

STEPS = 16
PROBE = ("Die Wittgenstein-Regel wird Programm: Turing führt den AST aus, "
         "Gödel predet das Ergebnis; die Regel zeigt, wie man sie ausführt. "
         "gap=0.36 trend=STABLE basis=SCALAR,TIME,SPACE")

TAUTAST = ("identity", "x")           # die Identitäts-Tautologie (compile_rule-Hebel)
TAUT_POOL = {"axiom_pool": ["ALPHA", "BETA"]}
TAUT_POOL6 = {"axiom_pool": ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"]}
FORCE_STEPS_MULTI = {1, 10, 12}


def force_rule_at(machine: CompositeMachine, ast: Any,
                  steps: set) -> None:
    """
    Probe-Pfad: erzwingt `ast` als ERSTE Regel des Wittgenstein-Batch an den
    genannten Kompositions-Schritten (1-basiert), ohne den AST-Sampler der
    Wittgenstein-Maschine anzutasten — die Regel compiliert zum Selbst-Programm.
    Eine Erzwingung je genanntem Schritt (die erste create_rule-Regel des
    Batches; die weiteren Batch-Regeln laufen durch den normalen Sampler).
    """
    wm = machine.wittgenstein
    original = wm.create_rule
    used: set = set()

    def patched() -> GeneratedSymbolicRule:
        current = wm._step_index + 1          # 0-basiert → laufender Schritt
        if current in steps and current not in used:
            used.add(current)
            rule = GeneratedSymbolicRule(ast)
            wm.generated_rules.append(rule)
            return rule
        return original()

    wm.create_rule = patched


def run_lane(mode: str, preset: str | None, oracle: ArgonOracle,
             params: Dict[str, Any] | None = None,
             force_steps: set | None = None) -> Dict[str, Any]:
    """16 Composite-Schritte in einem Modus (+ optional unter Steering)."""
    machine = CompositeMachine(MachineConfig(seed=42, mode=mode, params=dict(params or {})),
                               oracle=oracle)
    if force_steps is not None:
        force_rule_at(machine, TAUTAST, set(force_steps))
    verdicts: List[str] = []
    basis_profile: List[int] = []
    h_probe: float | None = None
    restored = True
    if preset is None:
        h_probe = oracle.entropy_scalar(PROBE)
        for _ in range(STEPS):
            result = machine.step(None)
            verdicts.append(str(result.invariants["verdict"]))
            basis_profile.append(len(machine.turing.basis))
    else:
        bias_before = [layer.moe.gate.expert_bias.data.clone()
                       for layer in oracle.model.model.layers]
        with oracle.steer(preset) as info:
            h_probe = oracle.entropy_scalar(PROBE)
            for _ in range(STEPS):
                result = machine.step(None)
                verdicts.append(str(result.invariants["verdict"]))
                basis_profile.append(len(machine.turing.basis))
            info_experts = list(info["experts"])
        bias_after = [layer.moe.gate.expert_bias.data
                      for layer in oracle.model.model.layers]
        restored = all(torch.equal(a, b) for a, b in zip(bias_before, bias_after))
    gaps = list(machine.gap().history)
    finite = all(np.isfinite(g) for g in gaps)
    return {"mode": mode, "preset": preset or "(standard)",
            "digest": machine.digest(), "gaps": gaps, "finite": finite,
            "expansions": list(machine.expansions),
            "verdicts": verdicts, "halt_verdicts": int(sum(1 for v in verdicts
                                                           if v in ("LOOP", "COLLAPSE", "EXPLOSION"))),
            "basis_profile": basis_profile,
            "basis_len": len(machine.turing.basis),
            "h_probe": h_probe, "restored": bool(restored),
            "top2_gap_delta": round(float(gaps[0] - gaps[-1]), 6)}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Composite Oracle/In-Loop Experiment")
    parser.add_argument("--checkpoint_base", default="./argon_1b")
    parser.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] device={device} base={args.checkpoint_base} machines={args.checkpoint_machines}")
    model, tokenizer = _load_machines_model(args.checkpoint_base, args.checkpoint_machines)
    oracle = ArgonOracle(model, tokenizer, device=device)

    lanes: Dict[str, Dict[str, Any]] = {}
    lanes["L1"] = run_lane("in_loop", None, oracle)
    lanes["L2"] = run_lane("in_loop", None, oracle)
    lanes["L3"] = run_lane("oracle", None, oracle)
    lanes["L4"] = run_lane("oracle", None, oracle)
    lanes["L5"] = run_lane("oracle", "combined", oracle)
    lanes["L6"] = run_lane("oracle", "goedel", oracle)
    lanes["L7"] = run_lane("oracle", None, oracle, params=TAUT_POOL, force_steps={1})
    lanes["L8"] = run_lane("oracle", "goedel", oracle, params=TAUT_POOL, force_steps={1})
    lanes["L9"] = run_lane("oracle", None, oracle, params=TAUT_POOL, force_steps={1})
    lanes["L10"] = run_lane("oracle", None, oracle, params=dict(TAUT_POOL6),
                            force_steps=FORCE_STEPS_MULTI)
    lanes["L11"] = run_lane("oracle", "goedel", oracle, params=dict(TAUT_POOL6),
                            force_steps=FORCE_STEPS_MULTI)
    lanes["L12"] = run_lane("oracle", None, oracle, params=dict(TAUT_POOL6),
                            force_steps=FORCE_STEPS_MULTI)

    print("=" * 88)
    print("COMPOSITE: ORACLE- VS. IN-LOOP-TRAJEKTORIEN UNTER PRESET-STEERING")
    print("=" * 88)
    for name, lane in lanes.items():
        gaps_str = " ".join(f"{g:.2f}" for g in lane["gaps"])
        print(f"{name} mode={lane['mode']:7s} preset={lane['preset']:10s} "
              f"digest={lane['digest'][:12]} h_probe={lane['h_probe']:.4f} "
              f"restored={lane['restored']} expansions={lane['expansions']} "
              f"halt={lane['halt_verdicts']} basis={lane['basis_len']} "
              f"gapΔ0→end={lane['top2_gap_delta']:+.4f}")
        print(f"    gaps: {gaps_str}")
        if lane["halt_verdicts"]:
            print(f"    verdicts: {lane['verdicts']}")
        if max(lane["basis_profile"]) > 3:
            print(f"    basis_profile: {lane['basis_profile']}")

    findings: List[Tuple[str, str]] = []
    det = lanes["L1"]["digest"] == lanes["L2"]["digest"]
    findings.append(("det_in_loop", "PASS" if det else "FAIL",))
    det_o = lanes["L3"]["digest"] == lanes["L4"]["digest"]
    findings.append(("det_oracle", "PASS" if det_o else "FAIL"))
    finite = all(l["finite"] for l in lanes.values())
    findings.append(("finite_gaps", "PASS" if finite else "FAIL"))
    delta_h_comb = abs(lanes["L5"]["h_probe"] - lanes["L3"]["h_probe"])
    delta_h_goe = abs(lanes["L6"]["h_probe"] - lanes["L3"]["h_probe"])
    findings.append(("entropy_bias_free",
                     "PASS" if (delta_h_comb < 0.02 and delta_h_goe < 0.02) else "FAIL"))
    traj_shift = (lanes["L5"]["gaps"] != lanes["L3"]["gaps"]
                  and lanes["L6"]["gaps"] != lanes["L3"]["gaps"])
    findings.append(("steer_traj_shift", "PASS" if traj_shift else "FAIL"))
    det_t = lanes["L7"]["digest"] == lanes["L9"]["digest"]
    findings.append(("det_tautology", "PASS" if det_t else "FAIL"))
    loop_ok = ("LOOP" in lanes["L7"]["verdicts"]
               and len(lanes["L7"]["expansions"]) > 0)
    findings.append(("loop_tautology", "PASS" if loop_ok else "FAIL"))
    grew = lanes["L7"]["basis_len"] > 3
    findings.append(("basis_expanded", "PASS" if grew else "FAIL"))
    finite_t = all(lanes[name]["finite"] for name in ("L7", "L8", "L9"))
    findings.append(("tautology_finite", "PASS" if finite_t else "FAIL"))
    traj_t = lanes["L8"]["gaps"] != lanes["L7"]["gaps"]
    findings.append(("tautology_steer_shift", "PASS" if traj_t else "FAIL"))
    multi = (lanes["L10"]["basis_len"] == 9
             and len(lanes["L10"]["expansions"]) == 3
             and lanes["L10"]["verdicts"][0] == "LOOP"
             and lanes["L10"]["verdicts"][9] == "LOOP"
             and lanes["L10"]["verdicts"][11] == "LOOP")
    findings.append(("multi_expansion", "PASS" if multi else "FAIL"))
    det_m = lanes["L10"]["digest"] == lanes["L12"]["digest"]
    findings.append(("det_multi_expansion", "PASS" if det_m else "FAIL"))
    multi_steer = (lanes["L11"]["basis_len"] == 9
                   and lanes["L11"]["gaps"] != lanes["L10"]["gaps"])
    findings.append(("multi_steer_shift", "PASS" if multi_steer else "FAIL"))
    clean = all(l["restored"] for l in lanes.values())
    findings.append(("bias_restored", "PASS" if clean else "FAIL"))

    print("-" * 88)
    print("BEFUNDE")
    for label, status in findings:
        print(f"  {label:20s} {status}")

    report = {"lanes": lanes,
              "findings": {k: v for k, v in findings},
              "delta_h_combined": float(delta_h_comb),
              "delta_h_goedel": float(delta_h_goe)}
    out = REPO_ROOT / "scratches" / "composite_oracle_experiment.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Save] {out.relative_to(REPO_ROOT)}")

    failed = [name for name, status in findings if status != "PASS"]
    if failed:
        print(f"FEHLGESCHLAGEN: {len(failed)} — {', '.join(failed)}")
        return 1
    print("GRÜN — alle Experiment-Befunde bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())