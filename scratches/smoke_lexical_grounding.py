"""
smoke_lexical_grounding.py — SP12: Lexical Grounding — das gelernte Wort
wird Programm (Semantische Kausalität; erster Stein des geschlossenen
Subjekt-Zyklus, Nordstern-Nachricht 2026-10-26, Task 1).

Canon: composite.py:108-140 (step / text-Fluss :110 / Bridging :120-124),
:66-92 (identity-Tautologie -> Selbst-Transition -> LOOP),
turing_machine.py:144-157 (expand_basis: EIN Term je Call) und :241-246
(Turing INNERER Expand je Halt-Fall => +2 Terme je LOOP-Schritt; Canon-Beleg
SP7-L0 final_basis 7 über 2 LOOP-Expansionen),
wittgenstein_machine.py:49-50 (SymbolicOperator.UNARY/BINARY),
:152-246 (embed_word lexikon-first :176, resonance :214, learn_pair :218,
word_formula :163), :342-381 (lexicon_game: Lernen nur bei success >= 0.5),
composite.py (SP12-Edits: step(obj, lexical_grounding=False) +
WittgensteinMachine.resolve_lexical_ast — reine Lexikon-Lesung, kein RNG).

PREREG (P2-Freeze — fixiert vor dem Erstrun; Vor-Smokes 2026-10-26 sind im
Plan ~/.claude/plans/sp12-lexical-grounding.md dokumentiert — Änderungs-
notizen 1-3: Klang-Dach formel_max 0.9831, L0-Regression grün, +2-Expansions-
Mechanik).

Kernthese (User-Nordstern, Task 1): sozial gelernte Wörter übersetzen sich
in AST-Programme. Der Resolver ist reine Lexikon-Lesung: Kandidaten sind
AUSCHLIESSLICH Lexikon-gelernte Operator-Wörter (Klang-vs-Gebrauch-Scheidung
— der SH-72-Fallback löst NIE auf, siehe Klang-Dach-Datum), tau = 0.85
(User-Prereg), UNARY -> (op, 'x'), BINARY -> (op, 'x', 'x').

VERDICT-MAP (Bands fixiert VOR dem Erstrun):
  sp12_l0_anker — machines-Edit-Regression: die kommittete
    lane_l0_anker-Funktion (aus dem committeten SP11-Skript importiert)
    liefert fields_ok + first_detect_ok; das Doppel-Replay ist bitidentisch.
    Falsifikator: jede abweichende Feld-Serie.
  sp12_det — Zwilling-Doppel-Läufe (Arm A ×2, Arm B ×2) bitidentisch.
    Falsifikator: jede Abweichung.
  sp12_t1_semantik — (a) alle 5 Lernrunden learned=True (noise_level=0.0,
    Canon-Arg lexicon_game für Lern-Sicherheit); (b) resonance_after[4] >=
    0.85; (c) resolve_lexical_ast("spiegel") == ('identity', 'x');
    (d) NACH-Lexikon-Kandidaten == ["identity"]; (e) B-Schritt-1
    invariants.lexical_ast == "('identity', 'x')".
    Falsifikator: jede Bandverletzung (kein tau-Schieben).
  sp12_t2_rng_reinheit — generierte Batches TRIPLE-identisch (Arm A
    grounded, Arm A' plain, Arm B mit WM-RNG-Pinning nach den 5 Lernrunden):
    batch_asts + rule_names + w_series je Schritt gleich. Die Resolution
    ist reine Lexikon-Lesung — sie konsumiert KEINE RNG-Züge.
    Falsifikator: jede Batch-Differenz.
  sp12_t3_scheidung — (a) Arm A lexical_ast == None an JEDEM Schritt
    (kein Klang-Auslösen über den SH-72-Fallback); (b) Arm A == Arm A'
    (opt-in-Default-Invarianz im leeren Lexikon); (c) Klang-Dach als DATEN
    (formel_max/formel_argmax — SP2-Messkonvention, kein Band).
    Falsifikator: Verletzung von (a) oder (b).
  sp12_t4_verdikt — Arm A: ALLE Verdikte HALTED, final_basis == 3,
    expansions == []; Arm B: Verdikte[0:2] == [LOOP, LOOP]; Basis nach
    Schritt 1 == 5 (User-Pass „Basis auf 5"), nach Schritt 2 == 7
    (+2 je LOOP: Turing-innerer Expand :241-246 + Composite :132-134);
    expansions[0:2] == ['step1:LOOP', 'step2:LOOP']; Schritte 3-6 als
    DATEN (mögliche Pool-Erschöpfung wird dokumentiert, kein Band).
    Falsifikator: Arm-A-Expansion, Arm-B-Verdikt != LOOP am resolvierten
    Schritt, oder fehlende Basis-Differenz.

Lanes: l0_anker (Regression), det, zwilling (A/A'/B, seed 42, 6 Schritte,
obj="spiegel" je Schritt). CPU-only; kein Maschinen-Code-Zugriff über die
harness-Ebene hinaus (die SP12-Extension der Maschinen ist committet und
durch lane_l0_anker regressions-geprüft).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

from machines.composite import CompositeMachine
from machines.machine_base import MachineConfig
from machines.wittgenstein_machine import SymbolicOperator
from scratches.smoke_paradigm_ursprung import lane_l0_anker

SEED = 42
STEPS = 6
TAU = 0.85
ROUNDS = 5
LEX_PAIR = ("spiegel", "identity")
POOL6 = {"axiom_pool": ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
                        "ZETA"]}
LOG_PATH = "scratches/lexical_grounding_experiment.log"
JSON_PATH = "scratches/lexical_grounding_experiment.json"


def round6(x: Any) -> Any:
    return round(float(x), 6) if x is not None else None


def make_composite() -> CompositeMachine:
    """Frischer composite seed 42 (in_loop, Canon-Pool POOL6)."""
    return CompositeMachine(MachineConfig(seed=SEED, mode="in_loop",
                                          params=dict(POOL6)))


def run_lexical(machine: CompositeMachine, word: str, steps: int,
                lexical: bool = True) -> Dict[str, Any]:
    """Verlauf mit je-Schritt-Attribution: lexikale Resolution, generierter
    Batch-AST (RNG-Marker für T2), Verdikt, Basis-Größe, Sub-Gaps."""
    rows: List[Dict[str, Any]] = []
    wm = machine.wittgenstein
    for step in range(1, steps + 1):
        rules_before = len(wm.generated_rules)
        st = machine.step(word, lexical_grounding=lexical)
        inv = st.invariants
        batch_ast = (str(wm.generated_rules[rules_before].ast)
                     if len(wm.generated_rules) > rules_before else None)
        t_state = machine.turing.state()
        rows.append({
            "step": step,
            "step_index": st.step,
            "lexical_word": inv["lexical_word"],
            "lexical_ast": inv["lexical_ast"],
            "batch_ast": batch_ast,
            "rule_name": inv["rule_name"],
            "verdict": inv["verdict"],
            "transitions": int(inv["transitions"]),
            "basis_size": len(t_state["basis"]),
            "gap": round6(st.gap),
            "w_gap": round6(inv["submachines"]["wittgenstein"]["gap"]),
            "t_gap": round6(inv["submachines"]["turing"]["gap"]),
            "g_gap": round6(inv["submachines"]["goedel"]["gap"]),
        })
    return {"rows": rows,
            "verdicts": [r["verdict"] for r in rows],
            "lexical_asts": [r["lexical_ast"] for r in rows],
            "batch_asts": [r["batch_ast"] for r in rows],
            "rule_names": [r["rule_name"] for r in rows],
            "w_series": [r["w_gap"] for r in rows],
            "basis_sizes": [r["basis_size"] for r in rows],
            "final_basis": rows[-1]["basis_size"],
            "expansions": list(machine.expansions)}


def resonance_profile(machine: CompositeMachine) -> Dict[str, Any]:
    """Klang-Raum-Messung (DATA — Klang-Dach, SP2-Konvention) plus die
    NACH-Spiel-Lexikon-Resonanz."""
    tok = machine.wittgenstein.tokenizer
    word = LEX_PAIR[0]
    k, idx, kind = tok.word_formula(word)
    base = tok.embed_word(word)
    formel = {op: round6(tok.resonance(base, tok.embed_word(op)))
              for op in SymbolicOperator.UNARY + SymbolicOperator.BINARY}
    lexikon_ops = [op for op in SymbolicOperator.UNARY + SymbolicOperator.BINARY
                   if op in tok.lexicon]
    lexikon = {op: round6(tok.resonance(tok.embed_word(word),
                                        tok.embed_word(op)))
               for op in lexikon_ops}
    return {
        "spiegel_formula": {"k": k, "idx": idx, "kind": kind},
        "formel_achsen": formel,
        "formel_max": max(formel.values()),
        "formel_argmax": max(formel, key=formel.get),
        "lexikon_woerter": sorted(tok.lexicon.keys()),
        "lexikon_kandidaten": lexikon_ops,
        "lexikon_resonanzen": lexikon,
    }


def arm_a_grounded() -> Dict[str, Any]:
    """Arm A: Wort OHNE gelernte Bedeutung — Resolver None (kein Klang-
    Auslösen), normale Kette."""
    machine = make_composite()
    run = run_lexical(machine, "spiegel", STEPS, lexical=True)
    run["profile"] = resonance_profile(machine)
    return run


def arm_a_plain() -> Dict[str, Any]:
    """Arm A': OHNE lexical_grounding — opt-in-Default-Invarianz."""
    machine = make_composite()
    return run_lexical(machine, "spiegel", STEPS, lexical=False)


def arm_b_after_learning() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Arm B: 5 Lernrunden (noise_level=0.0 — Canon-Arg für Lern-Sicherheit),
    DANACH WM-RNG-Pinning (Arm-Schicht: default_rng(42) — identisch zum
    reset-Body), dann der Verlauf: das gelernte Wort wird Programm."""
    machine = make_composite()
    protocol = machine.wittgenstein.lexicon_game(
        [LEX_PAIR], rounds=ROUNDS, noise_level=0.0)
    profile_b = resonance_profile(machine)
    machine.wittgenstein._rng = np.random.default_rng(SEED)
    run = run_lexical(machine, "spiegel", STEPS, lexical=True)
    return run, {"protocol": protocol,
                 "resonance_after": [e["resonance_after"] for e in protocol],
                 "learned": [e["learned"] for e in protocol],
                 "profile_b": profile_b}


def lane_l0_anchor() -> Dict[str, Any]:
    """machines-Edit-Regression über die kommittete SP11-Lane (die
    committete Feldvergleich-Mechanik, unverändert)."""
    l0 = lane_l0_anker()
    l0_re = lane_l0_anker()
    return {"result": l0,
            "fields_ok": bool(l0.get("fields_ok")),
            "first_detect_ok": bool(l0.get("first_detect_ok")),
            "deterministic": l0 == l0_re}


def lane_det(arm_a: Dict[str, Any], arm_b_run: Dict[str, Any]) -> Dict[str, Any]:
    """Doppel-Läufe: Arm A und Arm B je frisch reproduziert, bitidentisch."""
    a_re = arm_a_grounded()
    b_re, _b_meta_re = arm_b_after_learning()
    return {"arm_a_deterministic": a_re == arm_a,
            "arm_b_deterministic": b_re == arm_b_run}


def lane_zwilling() -> Dict[str, Any]:
    a = arm_a_grounded()
    a_plain = arm_a_plain()
    b, b_meta = arm_b_after_learning()
    m_direct = make_composite()
    m_direct.wittgenstein.lexicon_game([LEX_PAIR], rounds=ROUNDS,
                                       noise_level=0.0)
    resolver_direct = m_direct.wittgenstein.resolve_lexical_ast("spiegel")
    return {
        "arm_a": a, "arm_a_plain": a_plain, "arm_b": b, "arm_b_meta": b_meta,
        "resolver_direct": str(resolver_direct),
        "verdicts_a": a["verdicts"], "verdicts_b": b["verdicts"],
        "basis_a": a["final_basis"], "basis_b": b["final_basis"],
        "basis_b_sizes": b["basis_sizes"],
        "expansions_a": a["expansions"], "expansions_b": b["expansions"],
        "batches_equal_grounded_plain": (
            a["batch_asts"] == a_plain["batch_asts"]
            and a["rule_names"] == a_plain["rule_names"]
            and a["w_series"] == a_plain["w_series"]),
        "batches_equal_a_b": (
            a["batch_asts"] == b["batch_asts"]
            and a["rule_names"] == b["rule_names"]
            and a["w_series"] == b["w_series"]),
    }


def findings(twin: Dict[str, Any], det: Dict[str, Any],
             l0: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    a = twin["arm_a"]
    apr = twin["arm_a_plain"]
    b = twin["arm_b"]
    meta = twin["arm_b_meta"]
    prof = a["profile"]

    # --- sp12_l0_anker ---
    rows.append({"id": "sp12_l0_anker",
                 "data": {"fields_ok": l0["fields_ok"],
                          "first_detect_ok": l0["first_detect_ok"],
                          "deterministic": l0["deterministic"]},
                 "assert": (l0["fields_ok"] and l0["first_detect_ok"]
                            and l0["deterministic"]),
                 "note": ("machines-Regression über die kommittete SP11-Lane — "
                          "die committeten Feld-Serien bleiben bitidentisch")})

    # --- sp12_det ---
    rows.append({"id": "sp12_det",
                 "data": {"arm_a": det["arm_a_deterministic"],
                          "arm_b": det["arm_b_deterministic"]},
                 "assert": det["arm_a_deterministic"]
                 and det["arm_b_deterministic"],
                 "note": "Zwilling-Doppel-Läufe bitidentisch"})

    # --- sp12_t1_semantik ---
    res_serie = meta["resonance_after"]
    learned_all = all(meta["learned"])
    nach_kandidaten = meta["profile_b"]["lexikon_kandidaten"]
    nach_resonanz = meta["profile_b"]["lexikon_resonanzen"]
    resolver_direct = twin["resolver_direct"]
    rows.append({"id": "sp12_t1_semantik",
                 "data": {"learned": meta["learned"],
                          "resonance_after": res_serie,
                          "resonance_final": res_serie[-1],
                          "resolver_direct": resolver_direct,
                          "nach_kandidaten": nach_kandidaten,
                          "nach_resonanzen": nach_resonanz},
                 "assert": (learned_all
                            and res_serie[-1] >= TAU
                            and resolver_direct == "('identity', 'x')"
                            and nach_kandidaten == ["identity"]
                            and float(nach_resonanz.get("identity", 0.0)) >= TAU
                            and b["rows"][0]["lexical_ast"]
                            == "('identity', 'x')"),
                 "note": ("5 Lernrunden (noise_level=0.0) — Gebrauchs-Achse "
                          "resonance_after[-1] >= tau; der Resolver löst "
                          "direkt UND im Lauf zu ('identity', 'x') auf")})

    # --- sp12_t2_rng_reinheit ---
    rows.append({"id": "sp12_t2_rng_reinheit",
                 "data": {"batches_a_plain": twin["batches_equal_grounded_plain"],
                          "batches_a_b": twin["batches_equal_a_b"]},
                 "assert": (twin["batches_equal_grounded_plain"]
                            and twin["batches_equal_a_b"]),
                 "note": ("Batch-ASTs + rule_names + w_series je Schritt "
                          "identisch in A/A'/B — die Resolution ist reine "
                          "Lexikon-Lesung (kein RNG-Zug)")})

    # --- sp12_t3_scheidung ---
    rows.append({"id": "sp12_t3_scheidung",
                 "data": {"vor_resolution": a["lexical_asts"],
                          "klang_dach_max": prof["formel_max"],
                          "klang_dach_argmax": prof["formel_argmax"],
                          "spiegel_formula": prof["spiegel_formula"]},
                 "assert": (all(x is None for x in a["lexical_asts"])
                            and twin["batches_equal_grounded_plain"]),
                 "note": ("Vor dem Spiel löst der SH-72-Fallback NICHT auf "
                          "(Klang-Dach als DATEN; die LEXIKON-Gegenwart ist "
                          "die Scheidung); opt-in-Default-Invarianz A == A'")})

    # --- sp12_t4_verdikt ---
    rows.append({"id": "sp12_t4_verdikt",
                 "data": {"verdicts_a": a["verdicts"],
                          "verdicts_b": b["verdicts"],
                          "basis_a": a["final_basis"],
                          "basis_b_sizes": b["basis_sizes"],
                          "expansions_a": a["expansions"],
                          "expansions_b": b["expansions"]},
                 "assert": (all(v == "HALTED" for v in a["verdicts"])
                            and a["final_basis"] == 3
                            and a["expansions"] == []
                            and b["verdicts"][0] == "LOOP"
                            and b["verdicts"][1] == "LOOP"
                            and b["basis_sizes"][0] == 5
                            and b["basis_sizes"][1] == 7
                            and b["expansions"][:2]
                            == ["step1:LOOP", "step2:LOOP"]),
                 "note": ("Arm B: das gelernte Wort wird Programm — "
                          "('identity','x') ist Tautologie -> LOOP -> "
                          "+2 Basis-Terme je LOOP (Turing-innerer Expand "
                          ":241-246 + Composite :132-134); Schritte 3-6 "
                          "DATEN (Pool-Mechanik)")})

    for r in rows:
        r["data"] = {k: (v if not isinstance(v, list) else v[:12])
                     for k, v in r["data"].items()}
    return rows


def main() -> int:
    print("== SP12 — Lexical Grounding: das gelernte Wort wird Programm ==")
    print("Canon: composite.py:108-140/:66-92, turing_machine.py:144-157/:241-246,")
    print("       wittgenstein_machine.py:49-50/:152-246/:342-381")
    print("Prereg tau = 0.85; 5 Lernrunden, noise_level 0.0; seed 42; 6 Schritte")

    l0 = lane_l0_anchor()
    twin = lane_zwilling()
    det = lane_det(twin["arm_a"], twin["arm_b"])
    rows = findings(twin, det, l0)

    print()
    print("BEFUNDE:")
    for r in rows:
        print(f"  {r['id']:<26} {'PASS' if r['assert'] else 'FAIL'}")
    print()
    print("NOTIZEN:")
    for r in rows:
        print(f"  - {r['id']}: {r['note']}; "
              + "; ".join(f"{k}={v}" for k, v in r["data"].items()))

    fails = [r for r in rows if not r["assert"]]
    print()
    b = twin["arm_b"]
    print("Zwilling (seed 42, 6 Schritte, obj='spiegel'):")
    print("  Arm A verdicts:", twin["verdicts_a"], "| final_basis",
          twin["basis_a"], "| expansions", twin["expansions_a"])
    print("  Arm B verdicts:", twin["verdicts_b"], "| final_basis",
          twin["basis_b"], "| expansions", twin["expansions_b"])
    print("  Arm B basis-Verlauf:", twin["basis_b_sizes"])
    print("  Arm B lexical:", b["rows"][0]["lexical_word"], "->",
          b["rows"][0]["lexical_ast"])
    print("  resonance_after:", twin["arm_b_meta"]["resonance_after"])
    print()
    print(f"BILANZ: {len(rows) - len(fails)} PASS; {len(fails)} FEHLGESCHLAGEN"
          f"{', unverwartet' if fails else ''}; 0 OFFEN")
    print(f"JSON: {JSON_PATH}")

    payload = {"l0": l0, "twin": twin, "det": det, "findings": rows,
               "all_pass": not fails, "open": []}
    with open(JSON_PATH, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False)

    if fails:
        print("ROT — Bands verletzt (Daten, kein Band-Schieben): "
              + ", ".join(r["id"] for r in fails))
        return 1
    print("GRUEN — Semantische Kausalität geschlossen (das gelernte Wort wird Programm).")
    return 0


if __name__ == "__main__":
    sys.exit(main())