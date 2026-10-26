"""
composite.py — Die Komposition der drei Maschinen („Götterdreiheit").

Aufbau (Plan): 1. Wittgenstein erzeugt eine AST-Regel (uni_205, template-free)
→ 2. Turing führt die Regel als Programm aus — der AST wird in eine
Transitionstabelle übersetzt (ein Zustand pro AST-Knoten; die Regel wird
Programm, Tractatus: die Regel zeigt, wie man sie ausführt) und liefert
Halt-Verdikt + verbrauchte Transitionen → 3. Gödel predet das Ergebnis des
Laufs (in_loop: self_model-Verallgemeinerung uni_241:155; oracle:
Argon-Vorhersage des symbolischen Ist) → Canon-Gap uni_241:157.

Kontradiktion/Halt-Fall (LOOP/COLLAPSE/EXPLOSION im Verdikt) → Axiom-Expansion
(expand_basis, uni_256:113); der Gap-Verlauf der Komposition liefert das
Paradigmwechsel-Signal (uni_238:213-240).

FACRM-Schichten erhalten (uni_204:331/:361): null (Rauschen 0.8, ohne
Struktur) gegen alternativ (Gödel-seed, Struktur ×0.3) — die
Wittgenstein-Maschine als Subsystem.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from machines.goedel_machine import GoedelMachine
from machines.machine_base import BaseMachine, MachineConfig, StepResult
from machines.turing_machine import TransitionTable, TuringMachine
from machines.wittgenstein_machine import WittgensteinMachine

BLANK = "▁"
GAP_EPS = 1e-6          # Gap-Guard — identisch zu goedel_machine/machine_base.


class CompositeMachine(BaseMachine):
    """
    Die vereinigte Maschine — kein neuer Gap-Kanon, sondern eine Ordnung der
    drei existierenden: Wittgenstein generiert die Regel, Turing führt sie
    aus, Gödel misst die Selbstvorhersage des gesamten Verlaufs.
    """

    MACHINE_NAME = "composite"

    def __init__(self, config: MachineConfig, oracle=None):
        super().__init__(config, oracle=oracle)
        self.goedel = GoedelMachine(config, oracle=oracle)
        self.turing = TuringMachine(config, oracle=oracle)
        self.wittgenstein = WittgensteinMachine(config, oracle=oracle)
        self.expansions: List[str] = []
        self.reset(config.seed)

    # -- Lifecycle ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        sub_seeds = [m.reset(seed) for m in (self.goedel, self.turing, self.wittgenstein)]
        self._step_index = 0
        self._trace = []
        self._gap_history = []
        self.expansions = []
        return {"submachines": sub_seeds, "seed": seed}

    # -- AST → Programm ---------------------------------------------------------
    @staticmethod
    def compile_rule(ast: Any) -> TransitionTable:
        """
        „Die Regel wird Programm": Präfix-Walk über den AST; jeder Knoten ist
        ein Zustand, ein Leseschritt ohne Bewegung; das Ende halt. Ein Blatt
        ('x'/Konstante) ohne Knoten hält sofort. Blank = ``▁``.

        Ausnahme — die Identitäts-Tautologie: ``('identity', 'x')`` wird nicht
        als Kettenglied übersetzt, sondern als Selbst-Transition
        ``("n_i", BLANK) -> (BLANK, "N", "n_i")``: die tautologische Regel
        zeigt als Programm auf sich selbst und terminiert nicht — Turing
        erkennt das LOOP-Verdikt über die Lauf-Signatur, die Komposition
        expandiert die Basis (uni_256:113). Alle anderen ASTs behalten die
        HALT-Garantie des Präfix-Walks.
        """
        nodes: List[str] = []
        tautologies: set = set()

        def walk(node: Any) -> None:
            if isinstance(node, tuple):
                index = len(nodes)
                nodes.append(str(node[0]))
                if len(node) == 2 and node[0] == "identity" and node[1] == "x":
                    tautologies.add(index)   # Tautologie: der Zustand zeigt auf sich
                    return                    # das 'x'-Blatt wird konsumiert
                if len(node) == 3:
                    walk(node[1])
                    walk(node[2])
                else:
                    walk(node[1])

        walk(ast)
        transitions: Dict[Any, Any] = {}
        if not nodes:
            transitions[("n0", BLANK)] = (BLANK, "N", "halt")
        for idx, op in enumerate(nodes):
            if idx in tautologies:
                transitions[(f"n{idx}", BLANK)] = (BLANK, "N", f"n{idx}")
            else:
                next_state = f"n{idx + 1}" if idx + 1 < len(nodes) else "halt"
                transitions[(f"n{idx}", BLANK)] = (BLANK, "N", next_state)
        return TransitionTable(transitions, blank=BLANK)

    # -- Protokoll ------------------------------------------------------------
    def step(self, obj: Any, lexical_grounding: bool = False) -> StepResult:
        text = obj if isinstance(obj, str) else None

        # 1. Wittgenstein: Regel-Batch + Gültigkeit (uni_205-Mechanik);
        #    die ERSTE Regel des Batches wird dem Turing-Lauf als Programm
        #    übergeben (deterministische Wahl, dokumentiert).
        rules_before = len(self.wittgenstein.generated_rules)
        result_w = self.wittgenstein.step(text)
        batch_rules = self.wittgenstein.generated_rules[rules_before:]
        rule = batch_rules[0] if batch_rules else None
        rule_name = rule.name if rule else "(leer)"

        # SP12 (opt-in): das gelernte Wort als Programm-Erzeuger. Die
        # Resolution ist reine Lexikon-Lesung (kein RNG-Zug, kein
        # Zustands-Mutieren) und ersetzt AUSSCHLIESSLICH das kompilierte
        # Programm dieses Schritts — der generierte Batch (Regeln,
        # Validität) bleibt unverändert im Verlauf (Herkunft dokumentiert
        # in den invariants lexical_word/lexical_ast).
        lexical_ast = None
        if lexical_grounding and text is not None:
            lexical_ast = self.wittgenstein.resolve_lexical_ast(text)

        program = self.compile_rule(
            lexical_ast if lexical_ast is not None
            else (rule.ast if rule else ("identity", "x")))

        # 2. Turing: führe die Regel als Programm aus (uni_256-Verdikt).
        result_t = self.turing.step(program)
        verdict = result_t.invariants["verdict"]
        transitions = int(result_t.invariants["transitions"])

        # 3. Gödel: das Verlauf-Outcome predet (uni_241 + Oracle).
        result_g = self.goedel.step(transitions)

        # 4. Halt-Fälle → Axiom-Expansion (uni_256:113); Paradigm-Signal
        #    (uni_238) für den Verlauf.
        if verdict in ("LOOP", "COLLAPSE", "EXPLOSION"):
            if self.turing.expand_basis(reason=f"composite:{verdict}"):
                self.expansions.append(f"step{self._step_index + 1}:{verdict}")
        signal = self.goedel.paradigm_signal()
        if signal.get("detected"):
            self.expansions.append(f"step{self._step_index + 1}:paradigm")
        signal_c = self.paradigm_signal()
        if signal_c.get("detected"):
            self.expansions.append(f"step{self._step_index + 1}:paradigm")

        gap = (result_w.gap + result_t.gap + result_g.gap) / 3.0
        self._gap_history.append(gap)
        self._step_index += 1
        record = self.gap()
        self._trace.append(
            f"step={self._step_index} rule={rule_name} verdict={verdict} "
            f"transitions={transitions} gap_mean={gap:.6f} trend={record.trend}")

        invariants = {
            "rule_name": rule_name,
            "verdict": verdict,
            "transitions": transitions,
            "lexical_word": text if lexical_grounding else None,
            "lexical_ast": (str(lexical_ast) if lexical_ast is not None
                            else None),
            "validity": result_w.invariants.get("validity"),
            "submachines": {
                "goedel": {"gap": result_g.gap, "trend": result_g.trend},
                "turing": {"gap": result_t.gap, "trend": result_t.trend},
                "wittgenstein": {"gap": result_w.gap, "trend": result_w.trend},
            },
            "paradigm_signal": signal,
            "paradigm_signal_composite": signal_c,
        }
        return StepResult(machine=self.MACHINE_NAME, step=self._step_index, gap=gap,
                          trend=record.trend, trace=list(self._trace),
                          invariants=invariants, argon_meta=result_g.argon_meta)

    def paradigm_signal(self, threshold: float = 2.0) -> Dict[str, Any]:
        """uni_238:220-240 auf der KOMPOSIT-Gap-Serie (Mittel der drei
        Sub-Gaps, siehe _gap_history): der Bruch der Komposition schlägt
        hier ein — die Basis-Expansion verändert das Programm und damit den
        Verlauf, während die Goedel-Dynamik selbst (tanh-beschränkt, lernt
        über self_model) stabil bleibt. Sp7-Kripkenstein: das Paradigmen-
        Signal markiert den Punkt, an dem „plus" in „quus" umschlägt."""
        history = self._gap_history
        if len(history) < 10:
            return {"detected": False, "ratio": None, "threshold": threshold}
        recent = float(np.mean(history[-5:]))
        baseline = (float(np.mean(history[-20:-5])) if len(history) >= 20
                    else float(np.mean(history[:-5])))
        ratio = recent / (baseline + GAP_EPS)
        return {"detected": bool(ratio > threshold), "ratio": ratio,
                "threshold": threshold}

    def state(self) -> Dict[str, Any]:
        gap = self.gap()
        return {
            "machine": self.MACHINE_NAME,
            "step": self._step_index,
            "expansions": list(self.expansions),
            "gap": {"current": gap.current, "trend": gap.trend, "avg": gap.avg},
            "gap_history": list(self._gap_history),
            "goedel": self.goedel.state(),
            "turing": self.turing.state(),
            "wittgenstein": self.wittgenstein.state(),
            "seed": self.config.seed,
        }

    def describe(self) -> str:
        gap = self.gap()
        return (f"CompositeMachine(step={self._step_index}, gap={gap.current:.4f} "
                f"{gap.trend}, expansions={len(self.expansions)}, "
                f"turing_basis={len(self.turing.basis)} dims)")

    # -- SP13 — kontrafaktisches Probehandeln & endogene Agency ----------------
    def observe(self, obj: Any, lexical_grounding: bool = False) -> StepResult:
        """Nicht-mutierende Vorschau: der machine_base.py:502-Rahmen
        (Snapshot -> step -> Restore im finally) mit der Superset-Signatur
        SP12s. Die Vorschau sieht den ganzen Schritt voraus — inklusive
        LOOP-Expansionen und Argon-Orakel-Abfragen — und kehrt bitgenau
        an den Ausgangspunkt zurück."""
        snapshot = self._snapshot()
        try:
            return self.step(obj, lexical_grounding=lexical_grounding)
        finally:
            self._restore_snapshot(snapshot)

    def _snapshot(self) -> Dict[str, Any]:
        return {
            "step": self._step_index,
            "trace": list(self._trace),
            "gap_history": list(self._gap_history),
            "expansions": list(self.expansions),
            "goedel": self.goedel._snapshot(),
            "turing": self.turing._snapshot(),
            "wittgenstein": self.wittgenstein._snapshot(),
        }

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._step_index = snapshot["step"]
        self._trace = snapshot["trace"]
        self._gap_history = snapshot["gap_history"]
        self.expansions = snapshot["expansions"]
        self.goedel._restore_snapshot(snapshot["goedel"])
        self.turing._restore_snapshot(snapshot["turing"])
        self.wittgenstein._restore_snapshot(snapshot["wittgenstein"])

    def step_autonomous(
        self,
        obj: Any,
        candidates: Sequence[Any] = (None, "goedel", "turing",
                                     "wittgenstein", "combined"),
        lexical_grounding: bool = False,
    ) -> StepResult:
        """SP13 — Endogene Agency: die Maschine erprobt die Subspace-
        Kandidaten kontrafaktisch (nicht-mutierende observe()-Proben unter
        oracle.steer; ohne Oracle ungesteuert — CPU-degeneriert) und vollzieht
        den realen, mutierenden Schritt unter dem Preset, das den Composite-
        Gap minimiert (argmin, strict <: Ties → früherer Kandidat). Das
        Programm des Schritts ist in ALLEN Proben identisch (Generierung/
        Resolution konsumieren keine steuerabhängigen Züge) — der einzige
        Kanal der Wahl ist die Orakel-Kohärenz (Gödel-Gap)."""
        candidate_gaps: Dict[str, float] = {}
        best_preset: Any = None
        best_gap = float("inf")
        for preset in candidates:
            label = "raw" if preset is None else str(preset)
            if preset is None or self.oracle is None:
                probe = self.observe(obj, lexical_grounding=lexical_grounding)
            else:
                with self.oracle.steer(preset):
                    probe = self.observe(obj,
                                         lexical_grounding=lexical_grounding)
            candidate_gaps[label] = float(probe.gap)
            if float(probe.gap) < best_gap:
                best_gap = float(probe.gap)
                best_preset = preset
        chosen_label = "raw" if best_preset is None else str(best_preset)
        if best_preset is None or self.oracle is None:
            result = self.step(obj, lexical_grounding=lexical_grounding)
        else:
            with self.oracle.steer(best_preset):
                result = self.step(obj, lexical_grounding=lexical_grounding)
        result.invariants["autonomous"] = {
            "chosen_preset": chosen_label,
            "best_gap": best_gap,
            "candidate_gaps": candidate_gaps,
        }
        return result

    # -- FACRM-Schichten (uni_204:331/:361) ------------------------------------
    def run_facrm(self, trials: int = 10, count: int = 20,
                  noise_level: float = 0.8) -> Dict[str, Any]:
        """
        Null (unstrukturiertes Rauschen) gegen alternativ (Gödel-seed, Struktur
        ×0.3) in der Wittgenstein-Teilmaschine; danach Origin-RNG/Flag wiederher.
        """
        machine = self.wittgenstein
        saved_noise = machine.noise_level
        saved_structure = machine.use_structure

        samples_null: List[float] = []
        for trial in range(trials):
            machine.noise_level = noise_level
            machine.use_structure = False
            machine._rng = np.random.default_rng(trial * 1000)   # uni_204:345-349
            batch = machine.generate_experiments(count=count)
            samples_null.append(float(np.mean([e["success_rate"] for e in batch])))

        samples_alt: List[float] = []
        for trial in range(trials):
            machine.noise_level = noise_level
            machine.use_structure = True
            goedel_seed = int((self.wittgenstein_goodel_number() + trial) % (2**31))
            machine._rng = np.random.default_rng(goedel_seed)    # uni_204:375
            batch = machine.generate_experiments(count=count)
            samples_alt.append(float(np.mean([e["success_rate"] for e in batch])))

        machine.noise_level = saved_noise
        machine.use_structure = saved_structure
        machine._rng = np.random.default_rng(self.config.seed)

        return {"null": float(np.mean(samples_null)),
                "alternative": float(np.mean(samples_alt)),
                "null_samples": samples_null,
                "alt_samples": samples_alt}

    def wittgenstein_goodel_number(self) -> int:
        """uni_204:375 — Gödelzahl der Wittgenstein-Quelle als Struktur-Seed."""
        from machines.goedel_machine import GoedelNumber
        source = open(__file__, encoding="utf-8").read()
        return GoedelNumber.encode(source)