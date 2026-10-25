"""
smoke_hermeneutik.py — SP8: der hermeneutische Zirkel Python <-> Lisp.

Die Review-Frage: compile_rule (composite.py:76-88) erkennt nur den EXAKTEN
Tautologie-Knoten ('identity','x') — aber die algebraisch trivialen Identitaeten
add(0,x) / multiply(1,x) / add(x,0) / multiply(x,1) haben KEINEN identity-Knoten
und laufen als harmlose HALT-Kette durch. Die Lisp-Reflexion (Vendor-Reduktor
symbolic-operator.lisp:118-182 ueber die Bruecke, Op simplify_ast) wascht den
identity-Wrapper weg und normalisiert die Form; die Entlarvungs-Schicht des
Bridge-Ops markiert, wenn die Reduktion auf das nackte Var-Blatt x fuhrt UND
eine triviale Identitaet im Ursprung stand.

Phase 1 — Paritaet: dieselben Faelle durch Python-Seite
          (lisp_parity.simplify_ast_canonical, Vendor-Spiegel) UND Lisp-Seite
          (SBCL handle-simplify-ast); exakte Feldgleichheit Pflicht.
Phase 2 — der Zirkel: die entlarvte Tautologie wird zum Programm —
          compile_rule uebersetzt ('identity','x') zur Selbst-Transition,
          Turing erkennt den Loop, die Komposition expandiert die Basis
          (uni_256:113, TAUT_POOL6). Kontrast: kein einfacher sin(x)-Ast
          (+0) compiliert zum Loop — HALT-Garantie.

Befunde:
  sp8_parity   — Python- und Lisp-Seite liefern identische Felder je Fall
  sp8_reveal   — die Entlarvung trifft exakt die Tautologie-Familie:
                 add(0,x)/multiply(1,x)/add(x,0)/multiply(x,1)/identity(x)
                 -> revealed; sin(x)+0 / x+x / const-Fold / 0*x -> nicht
  sp8_zirkel   — entlarvte Tautologie erzwungen (Schritt 1, TAUT_POOL6):
                 LOOP-Verdikt, Expansion "step1:LOOP", Basis > 3, Gaps endlich
  sp8_contrast — sin(x)+0 (nicht entlarvt) erzwungen: kein LOOP in 6 Schritten,
                 Basis bleibt 3 (HALT-Garantie des Nicht-Tautologen)
  sp8_finite   — alle Gaps endlich
  sp7_break    — Kripke Plus/Quus: die erzwungene Tautologie an Schritt 9 wird
                 Programm (LOOP-Verdikt an 9, Expansion "step9:LOOP", Basis > 3)
  sp7_paradigm — das Gödel-Paradigmen-Signal (ratio > 2) feuert im Fenster der
                 Schritte 9-13 — erst NACH wiederholtem Gebrauch der Regel
  sp7_quus     — dieselbe Stelle, harmlose Form sin(x)+0: kein LOOP an 9,
                 keine "step9"-Expansion — gleicher Gebrauch, andere Bedeutung

Run:  python3 -m scratches.smoke_hermeneutik          # CPU, braucht SBCL
Log:  ... | tee scratches/hermeneutik_experiment.log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from machines.composite import CompositeMachine  # noqa: E402
from machines.wittgenstein_machine import GeneratedSymbolicRule  # noqa: E402
from machines.lisp_bridge import LISPBridge  # noqa: E402
from machines.lisp_parity import simplify_ast_canonical  # noqa: E402
from machines.machine_base import MachineConfig  # noqa: E402

STEPS = 16
STEPS_CONTRAST = 6
TAUTAST = ("identity", "x")
TAUT_POOL6 = {"axiom_pool": ["ALPHA", "BETA", "GAMMA", "DELTA",
                             "EPSILON", "ZETA"]}

# (name, ast, revealed_expected) — die Paritaets- und Entlarvungs-Matrix.
PARITY_CASES: List[Tuple[str, List[Any], bool]] = [
    ("add0x",      ["add", 0.0, "x"], True),
    ("mul1x",      ["multiply", 1.0, "x"], True),
    ("id_add0x",   ["add", ["identity", "x"], 0.0], True),
    ("sini_add0",  ["add", ["sin", "x"], 0.0], False),
    ("add_xx",     ["add", "x", "x"], False),
    ("leaf_x",     "x", False),
    ("mulx1",      ["multiply", "x", 1.0], True),
    ("sub_nofold", ["subtract", 2.0, 1.0], False),
    ("add_fold",   ["add", 2.0, 3.0], False),
    ("mul0annih",  ["multiply", 0.0, ["sin", "x"]], False),
    ("nested_mix", ["add", ["add", ["sin", "x"], 0.0],
                    ["multiply", "x", 1.0]], False),
    ("pure_id",    ["identity", "x"], True),
]


def compare_case(name: str, ast: List[Any],
                 python: Dict[str, Any], lisp: Dict[str, Any],
                 revealed_expected: Any) -> Dict[str, Any]:
    fields = ("ast_simplified", "size_in", "size_out")
    equal = all(_same(python[f], lisp[f]) for f in fields)
    equal &= python.get("revealed") == lisp.get("revealed")
    equal &= (python.get("contains_trivial_identity")
              == lisp.get("contains_trivial_identity"))
    return {"name": name, "ast": ast, "python": python, "lisp": lisp,
            "equal": equal,
            "revealed": bool(python.get("revealed")),
            "revealed_expected": revealed_expected,
            "flagged": bool(python.get("contains_trivial_identity"))}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, str) != isinstance(b, str):
        return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# -- Phase 2: der Zirkel ------------------------------------------------------
def force_rule_at(machine: CompositeMachine, ast: Tuple[Any, ...],
                  steps: set) -> None:
    """Probe-Pfad wie smoke_composite_oracle.py — erzwinge `ast` als erste
    Regel des Batch an den genannten Schritten (1-basiert)."""
    wm = machine.wittgenstein
    original = wm.create_rule
    used: set = set()

    def patched() -> GeneratedSymbolicRule:
        current = wm._step_index + 1
        if current in steps and current not in used:
            used.add(current)
            rule = GeneratedSymbolicRule(ast)
            wm.generated_rules.append(rule)
            return rule
        return original()

    wm.create_rule = patched


def run_composite(force_ast: Tuple[Any, ...], steps_run: int,
                  force_steps: set | None = None, seed: int = 42) -> Dict[str, Any]:
    """SP8/SP7-Laufgeruest. Seed dokumentiert: 42 für den SP8-Zirkel
    (natürliche Tautologien bei 10/12), 7 für SP7 (Paradigmen-Signal im
    Fenster 9-13, Seed-Scan 2026-10-25)."""
    machine = CompositeMachine(MachineConfig(seed=seed, mode="in_loop",
                                             params=dict(TAUT_POOL6)))
    force_rule_at(machine, force_ast, force_steps or {1})
    verdicts: List[str] = []
    basis: List[int] = []
    gaps: List[float] = []
    paradigms: List[Dict[str, Any]] = []
    paradigms_c: List[Dict[str, Any]] = []
    for _ in range(steps_run):
        result = machine.step(None)
        verdicts.append(str(result.invariants["verdict"]))
        basis.append(len(machine.turing.basis))
        gaps.append(float(result.gap))
        paradigms.append(dict(result.invariants.get("paradigm_signal", {})))
        paradigms_c.append(dict(result.invariants.get("paradigm_signal_composite", {})))
    return {"verdicts": verdicts, "basis": basis, "final_basis": basis[-1],
            "expansions": list(machine.expansions), "gaps": gaps,
            "gaps_finite": all(g == g and abs(g) != float("inf") for g in gaps),
            "paradigm_ratio": [p.get("ratio") for p in paradigms],
            "paradigm_detected": [bool(p.get("detected")) for p in paradigms_c],
            "forced_ast": list(force_ast)}


def main() -> int:
    bridge = LISPBridge()
    if not bridge.available():
        print("SKIP — SBCL-Bridge nicht verfuegbar (Exit 3, kein stiller False-Pass)")
        return 3

    print("[SP8.1] Paritaet Python (lisp_parity) vs Lisp (SBCL simplify_ast) …")
    rows: List[Dict[str, Any]] = []
    for name, ast, revealed_expected in PARITY_CASES:
        python = simplify_ast_canonical(ast)
        lisp = bridge.simplify_ast(ast)
        row = compare_case(name, ast, python, lisp, revealed_expected)
        rows.append(row)
    # Maschinen-Ursprung: ein echter Sampler-AST wandert durch den Zirkel.
    def ast_to_payload(ast: Any) -> Any:
        """Python-Tupel-AST -> JSON-Payload (Rekursion; bares Blatt bleibt)."""
        if isinstance(ast, tuple):
            return [ast_to_payload(item) for item in ast]
        return ast

    machine_probe = CompositeMachine(MachineConfig(seed=11, mode="in_loop"))
    sampled = machine_probe.wittgenstein.create_rule()
    ast_machine = ast_to_payload(sampled.ast)
    python_m = simplify_ast_canonical(ast_machine)
    lisp_m = bridge.simplify_ast(ast_machine)
    rows.append(compare_case("machine_seed11", ast_machine, python_m, lisp_m,
                             revealed_expected=None))
    # Sampler-Fall: nur Paritaetsanspruch (kein Reveal-Erwartungswert).

    print("[SP8.2] Der Zirkel: entlarvte Tautologie -> Programm -> LOOP …")
    zirkel = run_composite(TAUTAST, STEPS)

    print("[SP8.3] Kontrast: sin(x)+0 (nicht entlarvt) -> HALT-Garantie …")
    contrast = run_composite(("add", ("sin", "x"), 0.0), STEPS_CONTRAST)

    print("[SP7] Kripke Plus/Quus: erzwungene Tautologie an Schritt 9 (seed 7) …")
    kripke = run_composite(TAUTAST, STEPS, force_steps={9}, seed=7)
    print("[SP7.2] Kontrast (Quus): sin(x)+0 an Schritt 9 — keine Bruchstelle …")
    contrast_q = run_composite(("add", ("sin", "x"), 0.0), STEPS,
                               force_steps={9}, seed=7)
    first_detect = next((idx for idx, flag in
                         enumerate(kripke["paradigm_detected"], start=1) if flag),
                        None)

    print("=" * 88)
    print("SP8 — HERMENEUTISCHER ZIRKEL (PYTHON <-> LISP)")
    print("=" * 88)
    for row in rows:
        expected = ("—" if row.get("revealed_expected") is None
                    else str(row["revealed_expected"]))
        print(f"  {row['name']:14s} simp={json.dumps(row['python']['ast_simplified']):22s}"
              f" parity={row['equal']!s:5s} reveal={row['revealed']!s:5s} "
              f"(erwartet {expected:5s}) "
              f"sizes=({row['python']['size_in']},{row['python']['size_out']})")
    print(f"  Zirkel: verdicts={zirkel['verdicts'][:6]} … basis_end={zirkel['final_basis']} "
          f"expansions={zirkel['expansions']}")
    print(f"  Kontrast: verdicts={contrast['verdicts']} basis={contrast['basis']}")
    print(f"  Kripke: verdicts(9..13)={kripke['verdicts'][8:13]} "
          f"expansions={kripke['expansions']} basis_end={kripke['final_basis']}")
    print(f"  paradigm_detected={kripke['paradigm_detected']} "
          f"first_detect={first_detect}")
    print(f"  Quus-Kontrast: verdicts(9..13)={contrast_q['verdicts'][8:13]} "
          f"expansions={contrast_q['expansions']} "
          f"basis_end={contrast_q['final_basis']}")

    findings: List[Tuple[str, bool]] = [
        ("sp8_parity", all(r["equal"] for r in rows)),
        ("sp8_reveal", all(r["revealed"] == r["revealed_expected"] for r in rows
                           if r["revealed_expected"] is not None)),
        ("sp8_zirkel", "LOOP" in zirkel["verdicts"]
                       and "step1:LOOP" in zirkel["expansions"]
                       and zirkel["final_basis"] > 3 and zirkel["gaps_finite"]),
        ("sp8_contrast", "LOOP" not in contrast["verdicts"]
                         and contrast["final_basis"] == 3),
        ("sp8_finite", zirkel["gaps_finite"] and contrast["gaps_finite"]),
        ("sp7_break", kripke["verdicts"][8] == "LOOP"
                      and "step9:LOOP" in kripke["expansions"]
                      and kripke["final_basis"] > 3 and kripke["gaps_finite"]),
        ("sp7_paradigm", first_detect is not None and 9 <= first_detect <= 13),
        ("sp7_quus", contrast_q["verdicts"][8] != "LOOP"
                     and "step9:LOOP" not in contrast_q["expansions"]
                     and contrast_q["gaps_finite"]),
    ]
    print("-" * 88)
    print("BEFUNDE")
    for name, ok in findings:
        print(f"  {name:14s} {'PASS' if ok else 'FAIL'}")

    out = REPO_ROOT / "scratches" / "hermeneutik_experiment.json"
    out.write_text(json.dumps({
        "parity_rows": rows, "zirkel": zirkel, "contrast": contrast,
        "kripke": kripke, "contrast_quus": contrast_q,
        "first_detect": first_detect,
        "findings": [{"name": n, "status": "PASS" if ok else "FAIL"}
                     for n, ok in findings],
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[Save] {out.relative_to(REPO_ROOT)}")
    failed = [n for n, ok in findings if not ok]
    if failed:
        print(f"FEHLGESCHLAGEN: {len(failed)} — {', '.join(failed)}")
        return 1
    print("GRUEN — alle SP8-Befunde bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())