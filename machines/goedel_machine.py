"""
goedel_machine.py — Gödel-Maschine: Selbstbezug als reale Unabschließbarkeit.

Canon-Mechanik (dokumentiert, nicht neu erfunden):
  - uni_204_corrected.py:143-203 — GoedelEncoder: Primfaktor-Gödelzahl
    (100 Primzahlen, ``prime_idx = i % 100``, ``power = ord(char) % 10 + 1``,
    MOD 10**15), strukturelle Signatur (line/def/class/self_references/loop/
    hash_prefix) und Output-Vorhersage-Heuristik (def*5 + class*10 + 20).
  - uni_204_corrected.py:297-315 — Turing-Selbstbezug: Quelle liest sich
    selbst (``Path(__file__)``) und kodiert die eigene Quelle.
  - uni_241_enhanced_gtw_machine.py:140-231 — die Gap-Dynamik:
      state = standard_normal(d); self_model = standard_normal(d, d) * 0.1
      prediction   = tanh(self_model @ state)          (:155, in_loop)
      actual       = tanh(0.9*state + 0.3*sin(state) + 0.2*input
                          + 0.05*roll(state,1) + 0.02*randn(d))  (:207-213)
      gap          = ||pred - actual|| / (||state|| + 1e-6)  (Canon-Form :157)
      self_model  += 0.05*outer(err, state); *= 0.99        (:216-225)
  - uni_238_active_goedel_gap.py:220-240 — Paradigmwechsel-Detektor
    (recent = mean letzte 5; baseline = mean(gap[-20:-5]) oder mean(gap[:-5]);
    ratio = recent/(baseline+1e-6)).
  - lisp goedel-encoder.lisp:23 — Lisp-Konvention der Kodierung (max 15
    Zeichen, 20 Primzahlen, power = voller char-code, MOD 10**12) für die
    P1-Paritätsschicht (machines/lisp_bridge.py).

Oracle-Mode ("oracle"): Argon predet das symbolische Ist — das externe
Selbstmodell verallgemeinert uni_241 (Plan: „Oracle-Mode verallgemeinert
uni_241"). In-loop bleibt die klassische tanh(W @ state)-Selbstvorhersage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.machine_base import BaseMachine, GapRecord, MachineConfig, StepResult

GAP_EPS = 1e-6
MOD_GOEDEL = 10**15
MOD_LISP = 10**12


class GoedelNumber:
    """Primfaktor-Kodierung beider Konventionen (100-Primzahlen + Lisp-Brüche)."""

    PRIMES_100: List[int] = [
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
        53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
        127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197,
        199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
        283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379,
        383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
        467, 479, 487, 491, 499, 503, 509, 521, 523, 541]
    PRIMES_15: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    PRIMES_20_LISP: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                                 47, 53, 59, 61, 67, 71]

    @classmethod
    def encode(cls, source: str, max_chars: int = 1000) -> int:
        """uni_204:163-178: prime_idx = i % 100; power = ord % 10 + 1; MOD 10**15."""
        primes = cls.PRIMES_100
        encoding = 1
        for i, char in enumerate(source[:max_chars]):
            prime_idx = i % len(primes)
            power = ord(char) % 10 + 1
            encoding = (encoding * pow(primes[prime_idx], power, MOD_GOEDEL)) % MOD_GOEDEL
        return encoding

    @classmethod
    def encode_lisp(cls, source: str, max_chars: int = 15) -> int:
        """
        Lisp-Konvention (goedel-encoder.lisp:23-42): volle char-code-Exponenten
        über die ersten ``max_chars`` Zeichen, 20 Primzahlen, MOD 10**12 —
        nur ASCII-Threads (Bignum-Explosion bei Nicht-ASCII → Plan-Risiko 2).
        """
        primes = cls.PRIMES_20_LISP
        encoding = 1
        for i, char in enumerate(source[:max_chars]):
            prime_idx = i % len(primes)
            power = ord(char)
            encoding = (encoding * pow(primes[prime_idx], power, MOD_LISP)) % MOD_LISP
        return encoding

    @staticmethod
    def structural_signature(source: str) -> Dict[str, Any]:
        """uni_204:180-189: line/def/class/self_references/loop + hash_prefix 16."""
        return {
            "line_count": source.count("\n"),
            "def_count": source.count("def "),
            "class_count": source.count("class "),
            "self_references": source.count("__file__"),
            "loop_count": source.count("for ") + source.count("while "),
            "hash_prefix": hashlib.sha256(source.encode()).hexdigest()[:16],
        }

    @staticmethod
    def predict_output_signature(source: str) -> str:
        """uni_204:191-203: computable Vorhersage der Ausgabe-Textur."""
        sig = GoedelNumber.structural_signature(source)
        predicted_lines = sig["def_count"] * 5 + sig["class_count"] * 10 + 20
        predicted_phases = 4  # FACRM hat 4 Phasen
        return (f"PREDICTED: ~{predicted_lines} output lines, {predicted_phases} phases, "
                f"hash={sig['hash_prefix']}")


class GoedelMachine(BaseMachine):
    """
    8D-Selbstmodellmaschine (uni_241-Mechanik) mit Gödelzahl-Kanal
    (uni_204-Konvention) und Paradigmwechsel-Signal (uni_238:220).
    """

    MACHINE_NAME = "goedel"

    def __init__(self, config: MachineConfig, oracle=None):
        super().__init__(config, oracle=oracle)
        self.dims = config.dimensions
        self.paradigm_threshold = float(config.params.get("paradigm_threshold", 2.0))
        self.reset(config.seed)
        self.source_path = Path(__file__)

    # -- Lifecycle ------------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        self._rng = np.random.default_rng(seed)
        self.state_vec = self._rng.standard_normal(self.dims)      # uni_241:144 (Canon: "state")
        self.self_model = self._rng.standard_normal((self.dims, self.dims)) * 0.1
        self._step_index = 0
        self._trace = []
        self._gap_history = []
        self.input_log: List[Dict[str, Any]] = []
        return {"state": self.state_vec.tolist(), "seed": seed}

    # -- Intern -----------------------------------------------------------------
    def _text_seed(self, text: str) -> int:
        """Deterministischer Text→Seed (uni_205-FACRM-Trick: Gödelzahl mod 2**31)."""
        number = GoedelNumber.encode(text)
        return int(number % (2**31))

    def _input_vector(self, obj: Any) -> np.ndarray:
        """
        uni_241: None → randn*0.1 (in_loop); Array direkt; Text →
        deterministisch gefaltet (Gödelzahl-Seed); Scalar (Trace-Outcome der
        Komposition, z.B. Transitionenzahl) → konstant auf alle Dimensionen
        broadcasted (dokumentierte Zusatz-Lage, kein Canon-Anspruch).
        """
        if obj is None:
            return self._rng.standard_normal(self.dims) * 0.1
        if isinstance(obj, (list, tuple, np.ndarray)):
            vec = np.asarray(obj, dtype=np.float64)
            if vec.shape != (self.dims,):
                raise ValueError(f"input: {vec.shape} != ({self.dims},)")
            return vec
        if isinstance(obj, str):
            seed = self._text_seed(obj)
            return np.random.default_rng(seed).normal(0.0, 0.1, size=self.dims)
        if isinstance(obj, (int, float)) and not isinstance(obj, bool):
            return np.full(self.dims, float(obj))
        raise TypeError(f"input: nicht unterstützter Typ {type(obj)!r}")

    def _prediction(self, text: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """in_loop: tanh(W @ state); oracle: tanh(Argon-Hidden gefaltet)."""
        meta: Dict[str, Any] = {}
        if self.mode == "oracle" and self.oracle is not None:
            prompt = text if text else self.describe()
            hidden = self.oracle.hidden_state(prompt, dimensions=self.dims)
            prediction = np.tanh(hidden)                            # Canon-Tanh uni_241:155
            meta["oracle"] = "hidden_state"
            return prediction, meta
        prediction = np.tanh(self.self_model @ self.state_vec)
        meta["oracle"] = None
        return prediction, meta

    def _actual_dynamics(self, input_vec: np.ndarray) -> np.ndarray:
        """uni_241:207-213 — exakt: 0.9*s + 0.3*sin(s) + 0.2*in + 0.05*roll + 0.02*noise."""
        return np.tanh(0.9 * self.state_vec + 0.3 * np.sin(self.state_vec)
                       + 0.2 * input_vec + 0.05 * np.roll(self.state_vec, 1)
                       + 0.02 * self._rng.standard_normal(self.dims))

    # -- Protokoll --------------------------------------------------------
    def step(self, obj: Any) -> StepResult:
        input_vec = self._input_vector(obj)
        text = obj if isinstance(obj, str) else None
        prediction, oracle_meta = self._prediction(text)

        actual = self._actual_dynamics(input_vec)
        gap = self.note_gap(prediction, actual, self.state_vec)

        err = actual - prediction
        self.self_model += 0.05 * np.outer(err, self.state_vec)  # uni_241:220
        self.self_model *= 0.99                                  # uni_241:221
        self.state_vec = actual

        self._step_index += 1
        trend, avg = (r := self.gap()).trend, r.avg
        self._trace.append(
            f"step={self._step_index} gap={gap:.6f} trend={trend} oracle={oracle_meta['oracle']}")

        invariants = {"trend": trend, "avg": avg,
                      "paradigm": self.paradigm_signal(),
                      "self_pred_norm": float(np.linalg.norm(prediction)),
                      "actual_norm": float(np.linalg.norm(actual))}
        return StepResult(machine=self.MACHINE_NAME, step=self._step_index, gap=gap,
                          trend=trend, trace=list(self._trace),
                          invariants=invariants, argon_meta=oracle_meta)

    def state(self) -> Dict[str, Any]:
        gap = self.gap()
        return {
            "machine": self.MACHINE_NAME,
            "mode": self.mode,
            "dims": self.dims,
            "step": self._step_index,
            "state": [float(v) for v in self.state_vec],
            "self_model": [[float(v) for v in row] for row in self.self_model],
            "gap": {"current": gap.current, "trend": gap.trend, "avg": gap.avg},
            "gap_history": list(self._gap_history),
            "seed": self.config.seed,
        }

    def describe(self) -> str:
        gap = self.gap()
        return (f"GoedelMachine(mode={self.mode}, step={self._step_index}, "
                f"gap={gap.current:.4f} {gap.trend}, |W|=absorbiert "
                f"{np.mean(np.abs(self.self_model)):.4f})")

    # -- Canon-Erweiterungen ------------------------------------------------
    def paradigm_signal(self) -> Dict[str, Any]:
        """uni_238:220-240: ratio recent/baseline > threshold → Paradigmwechsel."""
        history = self._gap_history
        if len(history) < 10:
            return {"detected": False, "ratio": None, "threshold": self.paradigm_threshold}
        recent = float(np.mean(history[-5:]))
        baseline = (float(np.mean(history[-20:-5])) if len(history) >= 20
                    else float(np.mean(history[:-5])))
        ratio = recent / (baseline + GAP_EPS)
        return {"detected": bool(ratio > self.paradigm_threshold), "ratio": ratio,
                "threshold": self.paradigm_threshold}

    def self_reference(self) -> Dict[str, Any]:
        """
        uni_204:297-315 — die Maschine liest den eigenen Quellcode und kodiert
        ihn mit der 100-Primzahlen-Konvention („schreibt sich selbst").
        """
        source = self.source_path.read_text(encoding="utf-8")
        signature = GoedelNumber.structural_signature(source)
        return {
            "source_file": self.source_path.name,
            "goedel_number": GoedelNumber.encode(source),
            "signature": signature,
            "predicted_output": GoedelNumber.predict_output_signature(source),
        }

    # -- Observations-Schicht -------------------------------------------------
    def _snapshot(self) -> Dict[str, Any]:
        return {"state_vec": self.state_vec.copy(),
                "self_model": self.self_model.copy(),
                "step": self._step_index, "trace": list(self._trace),
                "gap_history": list(self._gap_history),
                "input_log": list(self.input_log),
                "rng_state": self._rng.bit_generator.state}

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self.state_vec = snapshot["state_vec"]
        self.self_model = snapshot["self_model"]
        self._step_index = snapshot["step"]
        self._trace = snapshot["trace"]
        self._gap_history = snapshot["gap_history"]
        self.input_log = snapshot["input_log"]
        self._rng.bit_generator.state = snapshot["rng_state"]

    def gap_record(self) -> GapRecord:
        return self.gap()