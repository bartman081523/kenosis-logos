"""
turing_machine.py — Turing-Maschine: berechenbare Ordnung und ihre Halt-Fälle.

Canon-Mechanik (dokumentiert):
  - uni_256_geometric_computer.py:93-232 — Halt-Oracle der geometric computer:
      check_halting() (:165-188): |Ψ| < 0.1 → COLLAPSE; |Ψ| > 100 →
      EXPLOSION; quantisierte State-Signatur (sorted, round 1) im
      Loop-Detektor > 3 → LOOP; sonst CONTINUE.
      expand_basis(reason) (:113-131): pop(0) aus dem Canon-Pool, neue
      Dimension mit N(0, 0.1)-Seed; nach Expansion loop_detector = {} und
      COLLAPSE → SCALAR += 2.0 (:221-225); leerer Pool → Meldung.
      geometric_product_step (:133-163): SCALAR = 0.99*s + N(0,0.01) +
      0.1*(Σ val_i^2); Cross-Rotation new_val = 0.95*val + 0.2*next, tanh.
  - uni_204_corrected.py:297-315 — Selbstbezug: die Maschine liest ihre
    eigene Quelle, der Gödel-Encoder kodiert sie.

Die Turing-Maschine kombiniert beides: eine deterministische
Transitionstabelle (state, read → write, move, next, halt?) plus das
Halt-Verdikt des geometrischen Zustandsraums — die Halt-Klassen des
uni_256-Orakels ordnen einen Lauf ein; im "in_loop"-Mode bestimmt die
Argon-Router-Entropie das Tape-Budget (hohe Entropie = mehr Budget) und
LOOP/EXPLOSION führen zur Axiom-Expansion.

Oracle-Mode: Argon predet die Anzahl der verbrauchten Transitionen
(Ziffer); der Gap misst |Voraussage - Ist|/(|state| + 1e-6).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.goedel_machine import GoedelNumber
from machines.machine_base import BaseMachine, MachineConfig, StepResult

TRANSITION = Tuple[str, str, str]  # (write, move, next_state)

# uni_256:101-107 — Initial-Basis legacy uni_253/254.
DEFAULT_BASIS = ["SCALAR", "TIME", "SPACE"]


class TransitionTable:
    """Deterministische Transitionstabelle — keine versteckten Fälle."""

    def __init__(self, transitions: Dict[Tuple[str, str], TRANSITION],
                 blank: str = "0"):
        self.transitions = transitions
        self.blank = blank

    def apply(self, state: str, read: str) -> Optional[TRANSITION]:
        return self.transitions.get((state, read))


class TuringPrograms:
    """Canon-Programme für Verifikation + Korpus (klein, deterministisch)."""

    @classmethod
    def binary_increment(cls) -> TransitionTable:
        """
        Inkrement einer Binärzahl (Standard-Turing-Beispiel): Band trägt die
        Zahl am Kursor beginnend; 'seek' läuft rechts zum Blank, 'carry'
        läuft links mit Übertrag zurück, Halt nach letztem Übertrag.
        """
        blank = "▁"  # Blank unter den Band-Symbolen (kein Konflikt mit '0'/'1')
        transitions: Dict[Tuple[str, str], TRANSITION] = {
            ("seek", "0"): ("0", "R", "seek"),
            ("seek", "1"): ("1", "R", "seek"),
            ("seek", "▁"): ("▁", "L", "carry"),
            ("carry", "0"): ("1", "N", "halt"),
            ("carry", "1"): ("0", "L", "carry"),
            ("carry", "▁"): ("1", "N", "halt"),
        }
        return TransitionTable(transitions, blank=blank)

    @classmethod
    def looper(cls) -> TransitionTable:
        """Endlosschleife (für LOOP-Verdikt + Expansion-Test) — liest Blank. """
        return TransitionTable({("spin", "▁"): ("▁", "N", "spin")}, blank="▁")


class TuringMachine(BaseMachine):
    """
    Deterministische Turing-Läufe + Halt-Orakel im uni_256-Sinn; im
    in_loop-Mode bestimmt die Argon-Router-Entropie das Tape-Budget.
    """

    MACHINE_NAME = "turing"

    def __init__(self, config: MachineConfig, oracle=None):
        super().__init__(config, oracle=oracle)
        self.tape_budget = int(config.params.get("tape_budget", 32))
        self.max_transitions = int(config.params.get("max_transitions", 64))
        self.axiom_pool: List[Dict[str, Any]] = list(
            config.params.get("axiom_pool", []))  # uni_255-Candidates (extern bereitgestellt)
        self.reset(config.seed)

    # -- Lifecycle ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> None:
        seed = self.config.seed if seed is None else seed
        self._rng = np.random.default_rng(seed)
        self.basis: List[str] = list(DEFAULT_BASIS)                 # uni_256:102
        self.state_vector: Dict[str, float] = {"SCALAR": 1.0, "TIME": 0.5, "SPACE": 0.5}
        self.loop_detector: Dict[str, int] = {}                      # uni_256:110
        self.halt_verdicts: List[str] = []
        self.head = 0
        self.tape: Dict[int, str] = {}
        self.tape_state = "seek"
        self._transition_count = 0
        self._step_index = 0
        self._trace = []
        self._gap_history = []
        self._last_verdict = "CONTINUE"

    # -- uni_256-Mechanik ---------------------------------------------------
    def geometric_product_step(self) -> None:
        """uni_256:133-163 — Geometric-Product-Proxy (Deterministisch, N aus RNG)."""
        new_state: Dict[str, float] = {}
        scalar = self.state_vector.get("SCALAR", 0.0)
        new_state["SCALAR"] = scalar * 0.99 + self._rng.normal(0, 0.01)
        dims = [d for d in self.basis if d != "SCALAR"]
        for i, d1 in enumerate(dims):
            val1 = self.state_vector.get(d1, 0.0)
            new_state["SCALAR"] += val1 * val1 * 0.1
            next_d = dims[(i + 1) % len(dims)]
            val2 = self.state_vector.get(next_d, 0.0)
            new_val = val1 * 0.95 + val2 * 0.2 + self._rng.normal(0, 0.05)
            new_state[d1] = float(np.tanh(new_val))
        self.state_vector = new_state

    def check_halting(self) -> str:
        """uni_256:165-188 — COLLAPSE/EXPLOSION/LOOP/CONTINUE."""
        magnitude = sum(abs(v) for v in self.state_vector.values())
        if magnitude < 0.1:
            return "COLLAPSE"
        if magnitude > 100:
            return "EXPLOSION"
        signature = str(sorted((k, round(v, 1)) for k, v in self.state_vector.items()))
        self.loop_detector[signature] = self.loop_detector.get(signature, 0) + 1
        if self.loop_detector[signature] > 3:
            return "LOOP"
        return "CONTINUE"

    def expand_basis(self, reason: str) -> bool:
        """uni_256:113-131 — Axiom-Expansion aus dem Pool; leer → False."""
        if not self.axiom_pool:
            self._trace.append(f"expand_basis({reason}): Pool leer")
            return False
        candidate = self.axiom_pool.pop(0)
        term = candidate if isinstance(candidate, str) else candidate.get("term", "")
        if term in self.basis:
            return False
        self.basis.append(term)
        self.state_vector[term] = float(self._rng.normal(0, 0.1))
        self._trace.append(f"expand_basis({reason}) -> {term}")
        self.loop_detector = {}                                    # uni_256:221
        return True

    def run_state_machine(self, program: TransitionTable, max_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Turing-Lauf mit uni_256-Halt-Orakel: LOOP über die quantisierte
        Lauf-Signatur (state, head, Band) + Loop-Zähler > 3 (kanonische
        Mechanik :178-186, angewandt auf den Bandlauf); BUDGET = CONTINUE
        (kein Verdikt binnen Budget).
        """
        max_steps = max_steps if max_steps is not None else self.max_transitions
        steps = 0
        halted: Optional[str] = None
        tape_growth = 0
        run_signatures: Dict[str, int] = {}
        while steps < max_steps:
            read = self.tape.get(self.head, program.blank)
            transition = program.apply(self.tape_state, read)
            if transition is None:
                halted = "COLLAPSE"  # unbekannte (state, read)-Kombination → flatten
                break
            write, move, next_state = transition
            self.tape[self.head] = write
            if next_state == "halt":
                halted = "HALTED"
                break
            self.head += 1 if move == "R" else (-1 if move == "L" else 0)
            self.tape_state = next_state
            steps += 1
            tape_growth = max(tape_growth, len(self.tape))
            if tape_growth > self.tape_budget:
                halted = "EXPLOSION"
                break
            # uni_256:178-186 — quantisierte Lauf-Signatur im Lauf-Zähler.
            signature = str((self.tape_state, self.head,
                             tuple(sorted((k, v) for k, v in self.tape.items()))))
            run_signatures[signature] = run_signatures.get(signature, 0) + 1
            if run_signatures[signature] > 3:
                halted = "LOOP"
                break
        verdict = halted or "BUDGET"
        self._transition_count += steps
        return {"verdict": verdict, "transitions": steps, "head": self.head,
                "tape": {int(k): sym for k, sym in self.tape.items()},
                "tape_growth": tape_growth}

    def repeated_signature(self) -> bool:
        """uni_256:178-186 — quantisierte Signatur wiederholt > 3 Mal."""
        signature = str(sorted((k, round(v, 1)) for k, v in self.state_vector.items()))
        return self.loop_detector.get(signature, 0) > 3

    # -- Protokoll ------------------------------------------------------------
    def step(self, obj: Any) -> StepResult:
        """Ein Lauf des aktiven Programms + uni_256-Halt-Orakel."""
        program = self.active_program(obj)
        self.prepare_tape(obj, program)
        oracle_meta: Dict[str, Any] = {"oracle": None}

        # 1. Vorhersage VOR dem Lauf: in_loop — Router-Entropie setzt Budget;
        #    oracle — Argon predet die verbrauchte Transitionen-Ziffer.
        if self.mode == "in_loop" and self.oracle is not None:
            entropy = self.oracle.entropy_scalar(str(obj or "turing step"))
            budget_factor = max(0.25, min(entropy / math_ln8(), 1.0))   # Amplituden-Gate (0.25..1)
            self.max_transitions = max(8, int(budget_factor * 64))
            oracle_meta.update({"entropy": entropy, "budget_factor": budget_factor})

        if self.mode == "oracle" and self.oracle is not None:
            value = self.oracle.predict_value(
                "Transitions consumed by the next Turing run, single digit: ")
            if value == "unknown":
                # -1 ist nie eine legitime Transitionenzahl — der Gap zeigt den
                # Verfehlskanal des Orakels offen (kein stiller Fallback, Plan-Risiko 5).
                prediction = -1.0
                oracle_meta.update({"oracle": "predict_value", "note": "unknown"})
            else:
                prediction = float(value)
                oracle_meta.update({"oracle": "predict_value", "value": value})
        else:
            prediction = float(self._transition_count)

        self.geometric_product_step()
        run = self.run_state_machine(program)
        verdict = run["verdict"]
        actual = float(run["transitions"])

        # 2. uni_256-Expansion bei Halt-Fall (LOOP/COLLAPSE/EXPLOSION) —
        #    die Expansion verändert das Gap-Ist NICHT (uni_256:213-225
        #    misst sie als Intervention, nicht als Transitionen).
        if verdict in ("LOOP", "COLLAPSE", "EXPLOSION"):
            self.halt_verdicts.append(verdict)
            self.expand_basis(reason=verdict)
            if verdict == "COLLAPSE":
                self.state_vector["SCALAR"] = self.state_vector.get("SCALAR", 0.0) + 2.0

        gap = self.note_gap([prediction], [actual], [1.0 + abs(actual)])
        self._step_index += 1
        record = self.gap()
        self._trace.append(
            f"step={self._step_index} verdict={verdict} transitions={int(actual)} "
            f"prediction={prediction:.2f} gap={gap:.6f} trend={record.trend}")
        invariants = {"verdict": verdict, "transitions": int(actual), "trend": record.trend,
                      "avg": record.avg, "basis": list(self.basis),
                      "prediction": float(prediction),
                      "|Ψ|": float(sum(abs(v) for v in self.state_vector.values()))}
        return StepResult(machine=self.MACHINE_NAME, step=self._step_index, gap=gap,
                          trend=record.trend, trace=list(self._trace),
                          invariants=invariants, argon_meta=oracle_meta)

    def active_program(self, obj: Any) -> TransitionTable:
        if isinstance(obj, TransitionTable):
            return obj
        if isinstance(obj, str) and obj.strip() == "looper":
            return TuringPrograms.looper()
        return TuringPrograms.binary_increment()

    def prepare_tape(self, obj: Any, program: TransitionTable) -> None:
        """
        Jeder Schritt startet mit frischem Band: ein String wird am Kursor
        beginnend aufgetragen ('seek' startet links); None → leeres Band.
        Programme-Selektoren („looper") tragen SICH NICHT auf das Band.
        """
        if isinstance(obj, str) and obj.strip() == "looper":
            obj = None
        self.tape = {i: ch for i, ch in enumerate(obj)} if isinstance(obj, str) else {}
        self.head = 0
        first_key = next(iter(program.transitions), None)
        self.tape_state = first_key[0] if first_key else "start"

    def state(self) -> Dict[str, Any]:
        gap = self.gap()
        return {
            "machine": self.MACHINE_NAME,
            "mode": self.mode,
            "step": self._step_index,
            "basis": list(self.basis),
            "state_vector": {k: float(v) for k, v in self.state_vector.items()},
            "halt_verdicts": list(self.halt_verdicts),
            "transition_count": self._transition_count,
            "gap": {"current": gap.current, "trend": gap.trend, "avg": gap.avg},
            "gap_history": list(self._gap_history),
            "seed": self.config.seed,
        }

    def describe(self) -> str:
        gap = self.gap()
        return (f"TuringMachine(mode={self.mode}, step={self._step_index}, "
                f"gap={gap.current:.4f} {gap.trend}, basis={len(self.basis)} dims, "
                f"verdicts={list(self.halt_verdicts)})")

    def self_reference(self) -> Dict[str, Any]:
        """uni_204:297-315 — eigene Quelle + Gödelzahl."""
        source = open(__file__, encoding="utf-8").read()
        signature = GoedelNumber.structural_signature(source)
        return {"source_file": __file__,
                "goedel_number": GoedelNumber.encode(source),
                "signature": signature}

    # -- Observations-Schicht -------------------------------------------------
    def _snapshot(self) -> Dict[str, Any]:
        return {"basis": list(self.basis), "state_vector": dict(self.state_vector),
                "loop_detector": dict(self.loop_detector),
                "halt_verdicts": list(self.halt_verdicts), "head": self.head,
                "tape": dict(self.tape), "tape_state": self.tape_state,
                "transition_count": self._transition_count, "step": self._step_index,
                "trace": list(self._trace), "gap_history": list(self._gap_history),
                "max_transitions": self.max_transitions,
                # SP13: expand_basis pop(0) (:149) und innerer Expand (:241-246)
                # laufen auch in observe()-Proben — der Pool muss mit.
                "axiom_pool": list(self.axiom_pool),
                "last_verdict": self._last_verdict,
                "rng_state": self._rng.bit_generator.state}

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.basis = snapshot["basis"]
        self.state_vector = snapshot["state_vector"]
        self.loop_detector = snapshot["loop_detector"]
        self.halt_verdicts = snapshot["halt_verdicts"]
        self.head = snapshot["head"]
        self.tape = snapshot["tape"]
        self.tape_state = snapshot["tape_state"]
        self._transition_count = snapshot["transition_count"]
        self._step_index = snapshot["step"]
        self._trace = snapshot["trace"]
        self._gap_history = snapshot["gap_history"]
        self.max_transitions = snapshot["max_transitions"]
        self.axiom_pool = snapshot["axiom_pool"]
        self._last_verdict = snapshot["last_verdict"]
        self._rng.bit_generator.state = snapshot["rng_state"]


def math_ln8() -> float:
    """ln(8) — Amplituden-Gate-Faktor (capacity_gate = clamp(H/ln(n), 0.25, 1))."""
    return float(np.log(8.0))