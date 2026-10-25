"""
wittgenstein_machine.py — Wittgenstein-Maschine: Bedeutung als Gebrauch.

Canon-Mechanik (uni_205_corrected.py, direkt verifiziert 2026-10-25):
  - :220-224  SymbolicOperator: UNARY (9, kein inv), BINARY (7, mit max/min),
    CONSTANTS [0.0, 1.0, 2.0, pi, e] — der ``corrected``-Pool (die
    ``_contaminated``-Variante enthält zusaätzlich phi).
  - :227-256  apply_unary: sqrt = sqrt(abs(x)) (KEIN 1e-10), exp clip(-10,10),
    log = log(abs(x)+1e-10); apply_binary: divide a/(b+1e-10),
    power = (abs(a)+1e-10)^clip(b,-5,5); überall except -> 0.0.
  - :283-302  GeneratedSymbolicRule.evaluate: NAN/inf -> 0.0, clip ±1e6.
  - :333-353  generate_symbolic_rule (template-free AST, Tiefe 3): Blatt bei
    depth >= max_depth oder rng < 0.3; Blatt ist 'x' mit p = 0.7, sonst
    rng.choice(CONSTANTS); unär bei rng < 0.4 (uni_241's generate_rule nutzt
    0.35 — der Port folgt uni_205:341); binär sonst.
  - :362-419  generate_experiments: x ~ uniform(0.1, 5.0, 10); use_structure
    → noise ~ N(0, nl*0.3), sonst N(0, nl); tolerance = 0.5 + 0.1*mean|gt|
    (uni_204:241 nutzt dieselbe Struktur mit fester Regelmatrix);
    validity = mean(success_rate); rule_diversity = unique ASTs/total.
  - :259-281  Regelname aus der AST-Struktur (max 50 Zeichen).

Lisp-Paritätanker (machines/lisp_bridge.py):
  - geometric-tokenizer.lisp:17/:85/:98/:160 — SH-72: word -> SM-Basis-Formel
    (k = 1 + mod(idx 12), mode = floor(idx 12)), 64-Punkte-Domain [0.1, 2*pi],
    embed-word normalisiert; resonance = Dot-Produkt.

Oracle-Mode: Argon predet die Gültigkeit eines Experiments (Ziffer als
success_count-Anteil × 10, sonst "unknown" → Canon-Noise-Pfad).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.machine_base import BaseMachine, MachineConfig, StepResult

# uni_205:220-224 — korrigierter Pool (kein phi); "contaminated" zusaätzlich
# +phi für die Lisp-Brücke (utils.lisp-Konstanten).
CORRECTED_POOL = [0.0, 1.0, 2.0, float(np.pi), float(np.e)]
CONTAMINATED_POOL = CORRECTED_POOL + [(1.0 + math.sqrt(5.0)) / 2.0]  # phi


class SymbolicOperator:
    """uni_205:217-256 — Operator-Apparatur der Bildlogik-Maschine."""

    UNARY = ["identity", "negate", "square", "sqrt", "sin", "cos", "exp", "log", "abs"]
    BINARY = ["add", "subtract", "multiply", "divide", "power", "max", "min"]
    CONSTANTS = CORRECTED_POOL

    @staticmethod
    def apply_unary(op: str, x: float) -> float:
        try:
            if op == "identity":
                return x
            if op == "negate":
                return -x
            if op == "square":
                return x * x
            if op == "sqrt":
                return float(np.sqrt(abs(x)))
            if op == "sin":
                return float(np.sin(x))
            if op == "cos":
                return float(np.cos(x))
            if op == "exp":
                return float(np.exp(np.clip(x, -10, 10)))
            if op == "log":
                return float(np.log(abs(x) + 1e-10))
            if op == "abs":
                return abs(x)
            return x
        except Exception:  # uni_205:240-241
            return 0.0

    @staticmethod
    def apply_binary(op: str, a: float, b: float) -> float:
        try:
            if op == "add":
                return a + b
            if op == "subtract":
                return a - b
            if op == "multiply":
                return a * b
            if op == "divide":
                return a / (b + 1e-10)
            if op == "power":
                return float(np.power(abs(a) + 1e-10, np.clip(b, -5, 5)))
            if op == "max":
                return max(a, b)
            if op == "min":
                return min(a, b)
            return a
        except Exception:  # uni_205:255-256
            return 0.0


class GeneratedSymbolicRule:
    """uni_205:259-302 — Template-free AST-Regel; evaluate mit Canon-Fehlfällen."""

    def __init__(self, ast: Any, name: str = ""):
        self.ast = ast
        self.name = name or self._generate_name()

    def _generate_name(self) -> str:
        def describe(node: Any) -> str:
            if isinstance(node, tuple):
                if len(node) == 2:
                    return f"{node[0]}({describe(node[1])})"
                if len(node) == 3:
                    return f"({describe(node[1])} {node[0]} {describe(node[2])})"
            if node == "x":
                return "x"
            return f"{node:.2f}" if isinstance(node, float) else str(node)
        return describe(self.ast)[:50]

    def evaluate(self, x: float) -> float:
        def eval_node(node: Any) -> float:
            if isinstance(node, tuple):
                if len(node) == 2:
                    return SymbolicOperator.apply_unary(node[0], eval_node(node[1]))
                if len(node) == 3:
                    return SymbolicOperator.apply_binary(
                        node[0], eval_node(node[1]), eval_node(node[2]))
            if node == "x":
                return x
            return float(node)

        try:
            result = eval_node(self.ast)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return float(np.clip(result, -1e6, 1e6))
        except Exception:
            return 0.0


class GeometricTokenizer:
    """
    SH-72-Embedding (Lisp-Parität: geometric-tokenizer.lisp:17/:85/:98/:160):
    word -> Basis-Formel (k = 1 + mod(gematria, 12); mode = floor(...)),
    evaluiert an 64 Punkten des Domains [0.1, 2*pi] und normalisiert.
    """

    DOMAIN_POINTS = 64
    BASIS_WIDTH = 12
    SH_INDEX = 72

    def __init__(self):
        self.lexicon: Dict[str, List[float]] = {}
        idx0 = 0.1
        idx1 = 2.0 * math.pi
        self.domain = [idx0 + (i / (self.DOMAIN_POINTS - 1)) * (idx1 - idx0)
                       for i in range(self.DOMAIN_POINTS)]

    @staticmethod
    def string_gematria(word: str) -> int:
        """Summe der Char-Codes (Lisp-Parität: string-gematria)."""
        return sum(ord(ch) for ch in word)

    def basis_formula(self, idx: int) -> Tuple[int, str]:
        """get-sh-basis-formula: k = 1 + mod(idx 12); mode = floor(idx / 12)."""
        k = 1 + (idx % self.BASIS_WIDTH)
        mode = idx // self.BASIS_WIDTH
        return k, ["sin", "cos", "exp", "inv_add", "log", "square", "identity"][min(mode, 6)]

    def word_formula(self, word: str) -> Tuple[float, int, str]:
        """word-to-formula: (gematria mod 72) -> SH-Basis-Formel (k, kind)."""
        gematria = self.string_gematria(word)
        idx = gematria % self.SH_INDEX
        k, kind = self.basis_formula(idx)
        return k, idx, kind

    def embed_word(self, word: str) -> List[float]:
        """Formel an 64 Punkten evaluiert und normalisiert (embed-word :98).

        S3 (Sprachspiel-Folge): ein gelerntes Wort geht dem Klang vor —
        Lisp-Parität geometric-tokenizer.lisp:95-98: (or (gethash word lexicon)
        …) steht VOR dem SH-72-Fallback. Ein leeres Lexikon (Canon) verändert
        nichts; Gebrauch, nicht Gematria, entscheidet erst nach dem Lernen."""
        if word in self.lexicon:
            return list(self.lexicon[word])
        k, idx, kind = self.word_formula(word)
        values = []
        for x in self.domain:
            try:
                if kind == "sin":
                    value = math.sin(k * x)
                elif kind == "cos":
                    value = math.cos(k * x)
                elif kind == "exp":
                    value = math.exp(min(-x / k, 100.0))
                elif kind == "inv_add":
                    value = 1.0 / (x + k)          # Lisp: (divide 1.0d0 (add x k))
                elif kind == "log":
                    # Lisp safe-log verbatim: x <= 0 → 0.0 (nicht 1e-30-Clamp).
                    value = 0.0 if x + k <= 0 else math.log(x + k)
                elif kind == "square":
                    value = (x ** 2) / k
                else:
                    value = x
            except (ValueError, OverflowError, ZeroDivisionError):
                value = 0.0
            values.append(value if math.isfinite(value) else 0.0)
        # Lisp normalize-vector verbatim (utils.lisp:91-98): norm < 1e-9 →
        # Vektor UNVERÄNDERT (nicht nullgesetzt).
        norm = math.sqrt(sum(v * v for v in values))
        if norm < 1e-9:
            return values
        return [v / norm for v in values]

    def resonance(self, a: List[float], b: List[float]) -> float:
        """Dot-Produkt zweier (normalisierter) Embeddings = cosine."""
        return sum(x * y for x, y in zip(a, b))

    def learn_pair(self, w1: str, w2: str, alpha: float = 0.3) -> Dict[str, int]:
        """S3 — soziales Lexikon-Lernen (Tractatus -> Untersuchungen im Code):

        Zwei Wörter im gemeinsamen Gebrauch drehen ihre Embeddings auf-
        einander zu: e_new(w) = normalize((1-alpha) e(w) + alpha e(partner)).
        Die Drehung ist gegenseitig und im Lexikon persistent — ab dem
        nächsten embed_word gewinnt der Gebrauch gegen die Gematria.

        Canon-neutral: der lexikonfreie Tokenizer (Paritäts-Fixtures, SH-72-
        Proben) trifft diese Methode nie."""
        e1 = self.embed_word(w1)
        e2 = self.embed_word(w2)
        mixed_1 = [(1.0 - alpha) * a + alpha * b for a, b in zip(e1, e2)]
        mixed_2 = [(1.0 - alpha) * b + alpha * a for a, b in zip(e1, e2)]
        self.lexicon[w1] = self._normalize(mixed_1)
        self.lexicon[w2] = self._normalize(mixed_2)
        return {w1: self.string_gematria(w1), w2: self.string_gematria(w2)}

    @staticmethod
    def _normalize(values: List[float]) -> List[float]:
        """Lisp normalize-vector verbatim (utils.lisp:91-98): norm < 1e-9 →
        Vektor UNVERÄNDERT (nicht nullgesetzt)."""
        norm = math.sqrt(sum(v * v for v in values))
        if norm < 1e-9:
            return values
        return [v / norm for v in values]


class WittgensteinMachine(BaseMachine):
    """
    Generative Bildlogik-Maschine (uni_205:305-419): neue Regel-ASTs, strukturiert
    gegen unstrukturiertes Rauschen; der Gödel-informierte Seed (FACRM-Alternativ)
    senkt effective Rauschen × 0.3.
    """

    MACHINE_NAME = "wittgenstein"

    def __init__(self, config: MachineConfig, oracle=None):
        super().__init__(config, oracle=oracle)
        self.noise_level = float(config.params.get("noise_level", 0.5))
        # S2 (Ent-Sättigung): noise_schedule rotiert noise_level je Batch aus
        # einer Streuungsleiter — mit None (Canon) bleibt noise_level konstant.
        # Der schedule rotiert über generate_experiments-Aufrufe (ein Batch-Feld
        # = ein Rauschwert), damit das Ist spreet statt bei 1.00 zu kleben.
        schedule = config.params.get("noise_schedule")
        self.noise_schedule: Optional[List[float]] = (
            [float(v) for v in schedule] if schedule else None)
        self._schedule_index = 0
        self.exp_count = int(config.params.get("experiments", 10))
        self.max_depth = int(config.params.get("max_depth", 3))
        # uni_205:317-322 — use_structure (Rauschen × 0.3) ist EINE eigene
        # Schicht, unabhängig vom oracle/in_loop-Mode (uni_204 toggelt beide).
        self.use_structure = bool(config.params.get("use_structure",
                                                    self.mode == "in_loop"))
        self.tokenizer = GeometricTokenizer()
        # S1 (SP3-Folge): opt-in Lesekanal predict_scale_0_9 — das 6-Token-
        # Greedy-Echo ("100000" = "1" + "00000") ist autoregressives Nach-
        # hallen, kein Urteil. Default off: Canon-Pfad predict_value unverändert.
        self.oracle_scale = bool(config.params.get("oracle_scale", False))
        self.reset(config.seed)

    # -- Lifecycle ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        self._rng = np.random.default_rng(seed)
        self._step_index = 0
        self._trace = []
        self._gap_history = []
        self._schedule_index = 0
        self.experiment_history: List[Dict[str, Any]] = []
        self.generated_rules: List[GeneratedSymbolicRule] = []
        return {"seed": seed}

    # -- AST-Generator (uni_205:333-353) ---------------------------------------
    def _generate_ast(self, depth: int = 0) -> Any:
        if depth >= self.max_depth or self._rng.random() < 0.3:
            constants = (CONTAMINATED_POOL if self.config.pool == "contaminated"
                         else CORRECTED_POOL)
            if self._rng.random() < 0.7:
                return "x"
            return self._rng.choice(constants)
        if self._rng.random() < 0.4:
            op = self._rng.choice(SymbolicOperator.UNARY)
            return (op, self._generate_ast(depth + 1))
        op = self._rng.choice(SymbolicOperator.BINARY)
        return (op, self._generate_ast(depth + 1), self._generate_ast(depth + 1))

    def create_rule(self) -> GeneratedSymbolicRule:
        rule = GeneratedSymbolicRule(self._generate_ast(0))
        self.generated_rules.append(rule)
        return rule

    # -- Experimente (uni_205:362-419) --------------------------------------
    def generate_experiments(self, count: Optional[int] = None) -> List[Dict[str, Any]]:
        count = self.exp_count if count is None else count
        if self.noise_schedule:
            # S2: ein Aufruf = ein Rauschstufen-Batch; deterministisch rotiert.
            self.noise_level = float(self.noise_schedule[
                self._schedule_index % len(self.noise_schedule)])
        experiments: List[Dict[str, Any]] = []
        for i in range(count):
            rule = self.create_rule()
            x_values = self._rng.uniform(0.1, 5.0, size=10)
            ground_truth = [rule.evaluate(x) for x in x_values.tolist()]
            if self.use_structure:
                noise = self._rng.normal(0, self.noise_level * 0.3, size=len(x_values))
            else:
                noise = self._rng.normal(0, self.noise_level, size=len(x_values))
            predictions = [gt + n for gt, n in zip(ground_truth, noise)]
            tolerance = 0.5 + 0.1 * float(np.mean(np.abs(ground_truth)))
            successes = sum(1 for gt, pred in zip(ground_truth, predictions)
                            if abs(gt - pred) < tolerance)
            experiments.append({
                "id": i,
                "rule_name": rule.name,
                "rule_ast": str(rule.ast),
                "success_rate": successes / len(x_values),
            })
        self.experiment_history.extend(experiments)
        if self.noise_schedule:
            self._schedule_index += 1
        return experiments

    # -- Sprachspiel (S3/SP6: Gebrauch schlaegt Gematria) ----------------------
    def lexicon_game(self, pairs: List[Tuple[str, str]], rounds: int = 15,
                     alpha: float = 0.3, success_threshold: float = 0.5,
                     noise_level: Optional[float] = None) -> List[Dict[str, Any]]:
        """SP6 — das Lexikon-Spiel: je Runde ein Sprachschritt. Die Maschine
        erzeugt eine Regel, evaluiert sie an x ~ U(0.1, 5) mit Gauß-Rauschen
        (Canon-Toleranz 0.5 + 0.1*mean|gt|); gelingt der Schritt
        (success_rate >= success_threshold), lernen die beiden Wörter des
        Runden-Paars voneinander (GeometricTokenizer.learn_pair). Die Resonanz
        des Paars wandert dadurch von der Gematria-Achse (SP2: gepaart 0.31
        < unverwandt 0.36) in die Gebrauchs-Achse (> 0.75 > unverwandt).

        Canon-neutral: neue Methode, kein bestehender Pfad berührt."""
        base_noise = self.noise_level if noise_level is None else float(noise_level)
        protocol: List[Dict[str, Any]] = []
        for round_index in range(rounds):
            w_lang, w_partner = pairs[round_index % len(pairs)]
            rule = self.create_rule()
            x_values = self._rng.uniform(0.1, 5.0, size=10)
            ground_truth = [rule.evaluate(x) for x in x_values.tolist()]
            noise = self._rng.normal(0, base_noise, size=len(x_values))
            predictions = [gt + n for gt, n in zip(ground_truth, noise)]
            tolerance = 0.5 + 0.1 * float(np.mean(np.abs(ground_truth)))
            successes = sum(1 for gt, pred in zip(ground_truth, predictions)
                            if abs(gt - pred) < tolerance)
            success_rate = successes / len(x_values)
            learned = success_rate >= success_threshold
            if learned:
                self.tokenizer.learn_pair(w_lang, w_partner, alpha=alpha)
            protocol.append({
                "round": round_index,
                "pair": [w_lang, w_partner],
                "rule_name": rule.name,
                "success_rate": round(success_rate, 4),
                "learned": learned,
                "resonance_after": round(float(self.tokenizer.resonance(
                    self.tokenizer.embed_word(w_lang),
                    self.tokenizer.embed_word(w_partner))), 6),
            })
        return protocol

    # -- Protokoll (uni_241-Gap über den Gültigkeits-Anteil) ------------------
    def step(self, obj: Any) -> StepResult:
        oracle_meta: Dict[str, Any] = {"oracle": None}
        # 1. Vorhersage VOR der Auswertung: in_loop — Mittel der letzten Batches;
        #    oracle — Argon predet die Gültigkeit in Zehner-Schritten (Ziffer).
        if self.mode == "oracle" and self.oracle is not None:
            prompt = ("Validity of a symbolic experiment batch, "
                      "single digit per ten percent: ")
            if self.oracle_scale and hasattr(self.oracle, "predict_scale_0_9"):
                # S1-Kanal: ein Forward, Ziffern-Logits 0..9 vollvokabular
                # massen-geprüft — unter tau bleibt die Antwort ehrlich
                # "unknown" (Stummkanal-Befund), sonst kontinuierlich in [0,1]
                # (kein Multi-Digit-Echo, kein min(value/10)-Clip).
                value = self.oracle.predict_scale_0_9(prompt)
                if value == "unknown":
                    prediction = 0.0
                    oracle_meta.update({"oracle": "predict_scale_0_9",
                                        "note": "unknown"})
                else:
                    prediction = min(max(float(value), 0.0), 1.0)
                    oracle_meta.update({"oracle": "predict_scale_0_9",
                                        "value": value})
            else:
                value = self.oracle.predict_value(prompt)
                if value == "unknown":
                    prediction = 0.0
                    oracle_meta.update({"oracle": "predict_value", "note": "unknown"})
                else:
                    prediction = min(max(value / 10.0, 0.0), 1.0)
                    oracle_meta.update({"oracle": "predict_value", "value": value})
        else:
            history = [e["success_rate"] for e in self.experiment_history]
            prediction = float(np.mean(history[-10:])) if history else 0.0

        # 2. Auswertung (nur der neue Batch zählt als symbolisches Ist).
        batch = self.generate_experiments()
        actual = float(np.mean([e["success_rate"] for e in batch]))

        # 3. Canon-Gap (uni_241:157) auf scalarem Ist-Zustand.
        gap = self.note_gap([prediction], [actual], [1.0 + abs(actual)])
        self._step_index += 1
        record = self.gap()
        self._trace.append(
            f"step={self._step_index} validity={actual:.4f} prediction={prediction:.4f} "
            f"gap={gap:.6f} trend={record.trend}")
        invariants = {"validity": actual, "prediction": prediction,
                      "trend": record.trend, "avg": record.avg,
                      "rule_diversity": self.rule_diversity()}
        return StepResult(machine=self.MACHINE_NAME, step=self._step_index, gap=gap,
                          trend=record.trend, trace=list(self._trace),
                          invariants=invariants, argon_meta=oracle_meta)

    def state(self) -> Dict[str, Any]:
        gap = self.gap()
        return {
            "machine": self.MACHINE_NAME,
            "mode": self.mode,
            "step": self._step_index,
            "noise_level": self.noise_level,
            "use_structure": self.use_structure,
            "pool": self.config.pool,
            "gap": {"current": gap.current, "trend": gap.trend, "avg": gap.avg},
            "gap_history": list(self._gap_history),
            "rule_diversity": self.rule_diversity(),
            "rule_count": len(self.generated_rules),
            "seed": self.config.seed,
        }

    def describe(self) -> str:
        gap = self.gap()
        return (f"WittgensteinMachine(mode={self.mode}, step={self._step_index}, "
                f"gap={gap.current:.4f} {gap.trend}, rules={len(self.generated_rules)})")

    def rule_diversity(self) -> float:
        """uni_205:413-419: unique AST-Strukturen / total."""
        if len(self.generated_rules) < 2:
            return 0.0
        unique_asts = len(set(str(r.ast) for r in self.generated_rules))
        return unique_asts / len(self.generated_rules)

    # -- Observations-Schicht -------------------------------------------------
    def _snapshot(self) -> Dict[str, Any]:
        return {"step": self._step_index, "trace": list(self._trace),
                "gap_history": list(self._gap_history),
                "experiment_history": list(self.experiment_history),
                "generated_rules": list(self.generated_rules),
                "rng_state": self._rng.bit_generator.state}

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._step_index = snapshot["step"]
        self._trace = snapshot["trace"]
        self._gap_history = snapshot["gap_history"]
        self.experiment_history = snapshot["experiment_history"]
        self.generated_rules = snapshot["generated_rules"]
        self._rng.bit_generator.state = snapshot["rng_state"]