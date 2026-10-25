"""
machine_base.py — Götterdreiheit-Protokoll: gemeinsame Kontrakte aller Maschinen.

Alle drei Maschinen (Gödel, Turing, Wittgenstein) — und ihre Komposition —
trennen dasselbe Messkonzept: den Gödel-Gap als normierte Differenz zwischen
Selbstvorhersage und symbolischem Ist:

    apply_gap(pred, actual, state) = |pred - actual| / (|state| + 1e-6)

Canon-Belege:
  - uni_241_enhanced_gtw_machine.py:155-168   — ``prediction = tanh(W @ state)``;
    ``gap = ||pred - actual|| / (||state|| + 1e-6)``; Append in ``gap_history``.
  - uni_241_enhanced_gtw_machine.py:172-189   — Trend-Mischnung
    (recent = mean letzer 5 vs. older; 1.3 → RISING, 0.7 → FALLING, sonst
    STABLE; < 5 Einträge → UNKNOWN).
  - uni_238_active_goedel_gap.py:220-240      — Paradigmwechsel-Detektor
    (ratio recent/baseline > threshold).
  - cognito-construct.lisp:21/186-255         — Lisp-Struktur mit
    ``goedel-gap-history`` (push: erstes Element ist der neueste Gap);
    Predict-Self per Vektor-Add, Update 0.95/0.05.
  - compare_gemma3_vs_argon.py:249-260        — Steering-Muster:
    ``expert_bias.data.clone()`` vor Mutation, ``fill_(-steer)`` +
    ``+steer`` auf den gewählten Experten, ``copy_``-Restaurierung.

Der ArgonOracle ist die externe Selbstmodell-Erweiterung (Maschinen-Mode
``"oracle"``): statt (bzw. ergänzend zu) dem gelernten ``self_model`` predet
Argon das symbolische Ist aus dem antrainierten Verständnis; der Gap misst,
wie weit das Netz die Maschine aus sich heraus vorhersagt. Im Mode
``"in_loop"`` steuert die Router-Entropie die Maschinendynamik live
(Tape-Budget, Loop→Expansion). Der Mode ist per MachineConfig wählbar —
beide Haken sitzen in den Maschinen, nicht im Argon-Kern (modeling_argon.py
und moe_layers.py bleiben unangetastet).
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Canon-Gap (uni_241:157) und Trend-Mischnung (uni_241:172)
# ---------------------------------------------------------------------------
GAP_EPS = 1e-6


def apply_gap(predicted: Sequence[float], actual: Sequence[float],
              state: Sequence[float], eps: float = GAP_EPS) -> float:
    """Normierter Gödel-Gap uni_241:157: ||pred - actual|| / (||state|| + eps)."""
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    s = np.asarray(state, dtype=np.float64)
    return float(np.linalg.norm(p - a) / (np.linalg.norm(s) + eps))


def _safe_mean(values: Sequence[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def gap_trend(history: Sequence[float]) -> Tuple[str, float]:
    """Trend + Mittelwert nach uni_241:172-189 (Thresholds 1.3 / 0.7)."""
    history = list(history)
    if not history:
        return "UNKNOWN", 0.0
    current = history[-1]
    avg = _safe_mean(history[-10:])
    if len(history) < 5:
        return "UNKNOWN", avg
    recent = _safe_mean(history[-5:])
    older = _safe_mean(history[-10:-5]) if len(history) >= 10 else _safe_mean(history[:-5])
    if recent > older * 1.3:
        return "RISING", avg
    if recent < older * 0.7:
        return "FALLING", avg
    return "STABLE", avg


@dataclass
class GapRecord:
    """JSON-stabiler Gap-Zustand (uni_241 get_gap_status)."""

    current: float
    trend: str
    avg: float
    history: List[float]

    @classmethod
    def from_history(cls, history: Sequence[float]) -> "GapRecord":
        history = list(history)
        trend, avg = gap_trend(history)
        return cls(current=history[-1] if history else 0.0, trend=trend, avg=avg, history=history)


# ---------------------------------------------------------------------------
# Maschinen-Konfiguration / Schritt-Ergebnis (JSON-stabil)
# ---------------------------------------------------------------------------
@dataclass
class MachineConfig:
    """Konfiguration aller Maschinen; mode wählt Oracle- vs. In-Loop-Haken."""

    seed: int = 42
    mode: str = "oracle"                 # "oracle" | "in_loop"
    dimensions: int = 8                  # uni_241: state/self_model-Dimension
    pool: str = "corrected"              # uni_205: "corrected" = {0, 1, 2, pi, e} (kein phi)
    steer: float = 50.0                  # expert_bias-Amplitude (compare-Suite: 50.0)
    max_steps: int = 256
    params: Dict[str, Any] = field(default_factory=dict)

    def with_params(self, **overrides: Any) -> "MachineConfig":
        merged = dict(self.params)
        merged.update(overrides)
        return MachineConfig(
            seed=self.seed, mode=self.mode, dimensions=self.dimensions,
            pool=self.pool, steer=self.steer, max_steps=self.max_steps, params=merged,
        )


@dataclass
class StepResult:
    """Ergebnis eines Maschinenschritts — vollständig JSON-serialisierbar."""

    machine: str
    step: int
    gap: float
    trend: str
    trace: List[str]
    invariants: Dict[str, Any]
    argon_meta: Dict[str, Any]


# ---------------------------------------------------------------------------
# Lisp-Paritäts-RNG (utils.lisp:44-60) — exakt: LCG + Box-Muller
# ---------------------------------------------------------------------------
class LCGRandom:
    """
    Bitgenaue Wiederholung des Lisp-eigenen deterministischen Stroms
    (utils.lisp:44-60) für P1-Parität Seed-basierter Lisp-Ops (ast_eval etc.):

        *rng-state* 123456789; set-rng-seed -> logand seed #xFFFFFFFF
        deterministic-random(limit):
            state = mod(a*state + c, 2^32)          a=1664525, c=1013904223
            float:  (* limit (/ (float state 1.0d0) m))
            int:    (mod state limit)
        random-normal: Box-Muller aus ZWEI float(1.0d0)-Zügen, u1 < 1e-10 -> 1e-10
    """

    A = 1664525
    C = 1013904223
    M = 4294967296  # 2^32
    DEFAULT_STATE = 123456789
    _U1_CLAMP = 1e-10

    def __init__(self, seed: Optional[int] = None):
        self.state = self.DEFAULT_STATE if seed is None else (seed & 0xFFFFFFFF)

    def reset(self, seed: Optional[int] = None) -> None:
        """set-rng-seed-Äquivalent (logand seed #xFFFFFFFF)."""
        self.state = self.DEFAULT_STATE if seed is None else (seed & 0xFFFFFFFF)

    def draw(self) -> int:
        self.state = (self.A * self.state + self.C) % self.M
        return self.state

    def integer(self, limit: int) -> int:
        """(mod *rng-state* limit) — limit muss int >= 1 sein."""
        return self.draw() % int(limit)

    def float(self, limit: float = 1.0) -> float:
        """(* limit (/ (float state 1.0d0) m)) — Reihenfolge wie Lisp (Division zuerst)."""
        return float(limit) * (float(self.draw()) / float(self.M))

    def normal(self) -> float:
        """random-normal (Box-Muller): sqrt(-2 ln u1) * cos(2 pi u2)."""
        u1 = self.float(1.0)
        u2 = self.float(1.0)
        if u1 < self._U1_CLAMP:
            u1 = self._U1_CLAMP
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# ArgonOracle: externe Selbstmodell-Erweiterung (Mode "oracle")
# ---------------------------------------------------------------------------
# Expertenwahl-Doppelanker: Startvorschlag des Plans; machine_registry.json
# (Phase-A-Baseline, Schritt 5) überschreibt ihn, sobald er gemessen ist.
MACHINE_EXPERT_DEFAULTS: Dict[str, List[int]] = {
    "goedel": [2, 7],
    "turing": [0, 6],
    "wittgenstein": [1, 3],
}

N_ROUTED_EXPERTS_DEFAULT = 8


class ArgonOracle:
    """
    Liest aus Argon, was kein Symbol-Prozessors leisten kann: Zahlvorhersage
    aus den Logits, Router-Entropie der Text, verstecke Repräsentation — und
    lenkt das Routing über additiven ``expert_bias`` ohne Gewichtsupdate
    (Suite-Muster compare_gemma3_vs_argon.py:249-260, restore via ``copy_``).

    Nur eval + greedy + ``no_grad``; die Maschinen selbst sind deterministisch.
    """

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None,
                 steer: float = 50.0, n_experts: int = N_ROUTED_EXPERTS_DEFAULT):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.steer_amplitude = float(steer)  # Amplitude; Methode steer() heißt
        # unangetastet steer() — Attribut-Assign würde die Methode maskieren.
        self.n_experts = int(n_experts)
        self.model.eval()

    # -- Intern: Encoding + Gate-Entropie-Hooks (Muster: train_*:154-163) ----
    def _encode(self, text: str, max_length: int = 256) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=max_length)["input_ids"].to(self.device)

    def _gate_hooks(self, collector: Dict[str, List[torch.Tensor]]):
        def hook_fn(_module, _inp, out):
            top_indices, _, router_ent = out
            collector["counts"] = collector["counts"] + torch.bincount(
                top_indices.reshape(-1), minlength=self.n_experts).float()
            collector["entropies"].append(router_ent.float().mean())
        return [layer.moe.gate.register_forward_hook(hook_fn)
                for layer in self.model.model.layers]

    # -- Öffentliche Lesekanäle ---------------------------------------------
    @torch.no_grad()
    def predict_value(self, prompt: str, max_digits: int = 6) -> Any:
        """
        Greedy Ziffern-Lesung aus der Vorhersageverteilung: der Argmax-Next-Token
        muss ein (einstelliges) Ziffern-Token sein ("0".." 9"); sonst
        ``"unknown"`` — kein stiller Fallback (Plan-Risiko 5).
        """
        import re
        if self.model.training:
            self.model.eval()
        ids = self._encode(prompt)
        digits = ""
        cache = None
        for _ in range(max_digits):
            outputs = self.model(input_ids=ids, past_key_values=cache, use_cache=True)
            logits = outputs.logits[0, -1]
            next_id = int(torch.argmax(logits).item())
            token_str = self.tokenizer.decode([next_id], skip_special_tokens=True)
            token_str = token_str.lstrip(" ")
            if not (len(token_str) == 1 and token_str.isdigit()):
                if not digits:
                    return "unknown"
                break
            digits += token_str
            cache = outputs.past_key_values
            ids = torch.tensor([[next_id]], device=ids.device)
        if not digits:
            return "unknown"
        # Konsistenz-Schicht: die Vorhersage muss eine Zahl sein.
        if re.fullmatch(r"-?\d+", digits) is None:
            return "unknown"
        return int(digits)

    @torch.no_grad()
    def entropy_scalar(self, text: str, max_length: int = 256) -> float:
        """Mittlere Router-Entropie über alle Schichten (Muster _gate_hooks)."""
        collector: Dict[str, List[torch.Tensor]] = {
            "counts": torch.zeros(self.n_experts, device=self.device),
            "entropies": [],
        }
        ids = self._encode(text, max_length)
        handles = self._gate_hooks(collector)
        try:
            _ = self.model(ids)
        finally:
            for h in handles:
                h.remove()
        if not collector["entropies"]:
            return 0.0
        return float(torch.stack(collector["entropies"]).mean().item())

    @torch.no_grad()
    def hidden_state(self, text: str, dimensions: Optional[int] = None) -> np.ndarray:
        """
        Finaler Hidden-State (Hook an ``model.model.norm``, modeling_argon.py:615),
        Sequenz gemittelt und deterministisch auf ``dimensions`` (Default: die
        Maschinen-Dimension, 8) zusammengefallen: gemitteltes (hidden,) wird als
        (hidden/8, 8)-Streifen gemittelt — dokumentierte Reduktion, kein
        Canon-Anspruch.
        """
        d = int(dimensions) if dimensions else 8
        captured: List[torch.Tensor] = []

        def hook(_module, _inp, out):
            captured.append(out.detach())

        handle = self.model.model.norm.register_forward_hook(hook)
        try:
            ids = self._encode(text)
            _ = self.model(ids)
        finally:
            handle.remove()

        if not captured:
            raise RuntimeError("ArgonOracle.hidden_state: norm-Hook feuerte nicht.")
        pooled = captured[0].float().mean(dim=1).reshape(-1)  # (hidden,)
        if pooled.numel() % d != 0:
            raise ValueError(
                f"hidden_state: hidden={pooled.numel()} teilt nicht auf {d}.")
        return pooled.reshape(-1, d).mean(axis=0).cpu().numpy().astype(np.float64)

    # -- Steering: Preset-/Default-Anbindung (Suite-Muster) ------------------
    def _preset_experts(self, name_or_preset: Any) -> Tuple[List[int], float, Optional[List[float]], str]:
        """
        Löst einen Preset auf: dict direkt, Preset-Datei
        ``argon_1b_machines/machine_presets/<name>.json`` (falls vorhanden),
        sonst die Default-Expertenwahl; Registry (machine_registry.json)
        schlägt den Plan-Vorschlag, sobald er gemessen ist.
        """
        name = name_or_preset if isinstance(name_or_preset, str) else "combined"
        preset: Dict[str, Any] = {}
        if isinstance(name_or_preset, dict):
            preset = dict(name_or_preset)
        else:
            preset = load_preset(name) or load_registry_machine(name)

        experts = preset.get("experts") or MACHINE_EXPERT_DEFAULTS.get(
            name, list(range(self.n_experts)))
        experts = sorted(int(e) for e in experts if 0 <= int(e) < self.n_experts)
        steer = float(preset.get("steer", self.steer_amplitude))
        raw_bias = preset.get("expert_bias_values" if "expert_bias_values" in preset
                              else "expert_bias") if isinstance(preset, dict) else None
        if isinstance(raw_bias, dict):
            raw_bias = raw_bias.get("values")   # Preset-Schema: {layers, values}
        stored_bias = raw_bias if isinstance(raw_bias, (list, tuple)) else None
        source = ("preset-file" if preset else ("registry" if name != "combined" else "defaults"))
        return [int(e) for e in experts], steer, stored_bias, source

    @contextmanager
    def steer(self, name_or_preset: Any = "combined"):
        """
        Additives Routing-Steering ohne Gewichtsupdate: auf den gewählten
        Experten ``+steer``, auf alle übrigen ``-steer`` (fill_(-steer)-
        Muster); bei Preset mit gespeichertem ``expert_bias`` direkt
        ``copy_``. Die Original-Bias-Vektoren werden exakt restauriert, denn
        spezialisierte Checkpoints persistieren einen kalibrierten Bias.
        """
        experts, steer, stored_bias, source = self._preset_experts(name_or_preset)
        if not self.model.training:
            pass  # eval: Inline-Balancing bleibt inaktiv; kein Zustands-Toggle nötig
        orig_bias = [layer.moe.gate.expert_bias.data.clone()
                     for layer in self.model.model.layers]
        try:
            with torch.no_grad():
                for i, layer in enumerate(self.model.model.layers):
                    g = layer.moe.gate.expert_bias.data
                    if stored_bias is not None:
                        start = i * self.n_experts           # flat n_layers·n_experts
                        values = torch.as_tensor(
                            stored_bias[start:start + self.n_experts],
                            dtype=g.dtype, device=g.device)
                        g.copy_(values)
                        continue
                    g.fill_(-steer)
                    for e in experts:
                        g[e] = steer
            yield {"experts": experts, "steer": steer, "stored_bias": stored_bias is not None,
                   "source": source}
        finally:
            with torch.no_grad():
                for layer, g0 in zip(self.model.model.layers, orig_bias):
                    layer.moe.gate.expert_bias.data.copy_(g0)


# ---------------------------------------------------------------------------
# Registry-/Preset-Lader (Dateien werden in Schritt 4/5 geschrieben)
# ---------------------------------------------------------------------------
_PRESET_DIRS = (
    "argon_1b_machines/machine_presets",
    "machines/presets",
)


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """Liest ``machine_presets/<name>.json`` aus einem der Preset-Verzeichnisse."""
    import os
    for directory in _PRESET_DIRS:
        path = os.path.join(directory, f"{name}.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return None


def load_registry_machine(name: str) -> Optional[Dict[str, Any]]:
    """Liest eine Maschinen-Domäne aus machine_registry.json (falls vorhanden)."""
    import os
    path = "machine_registry.json"
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    entry = (registry.get("machines") or {}).get(name)
    if not isinstance(entry, dict):
        return None
    cleaned = {k: v for k, v in entry.items() if k in ("experts", "steer", "entropy_target")}
    return cleaned or None


# ---------------------------------------------------------------------------
# Basisklasse aller Maschinen (Plan: API-Symmetrie)
# ---------------------------------------------------------------------------
class BaseMachine(ABC):
    """
    Gemeinsames Protokoll: reset / step / observe / state / gap / digest /
    describe. Die Maschinen halten ihre Canon-Mechanik selbst
    (uni_204/205/238/241/256, Lisp-Canon); die Basisklasse liefert nur den
    JSON-stabilen Zustands-/Digest-/Gap-Rahmen und die Observations-Schicht
    (Snapshot -> step -> Restore).
    """

    MACHINE_NAME = "base"

    def __init__(self, config: MachineConfig, oracle: Optional[ArgonOracle] = None):
        self.config = config
        self.oracle = oracle
        self._rng = np.random.default_rng(config.seed)  # uni_*-Canon-RNG
        self._step_index = 0
        self._trace: List[str] = []
        self._gap_history: List[float] = []

    # -- Protokoll ------------------------------------------------------------
    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """Resonanzlos neu aufsetzen (Seed-Reset; gibt den Initial-State)."""

    @abstractmethod
    def step(self, obj: Any) -> StepResult:
        """Ein Maschinenschritt; mutierend; liefert Gap + Trace."""

    @abstractmethod
    def state(self) -> Dict[str, Any]:
        """JSON-stabiler Zustand (sortierte Keys im Digest)."""

    @abstractmethod
    def describe(self) -> str:
        """Menschliche Kurzbeschreibung der Maschinen-Lage."""

    def observe(self, obj: Any) -> StepResult:
        """Nicht-mutierende Beobachtung: Snapshot -> step -> Restore."""
        snapshot = self._snapshot()
        try:
            return self.step(obj)
        finally:
            self._restore_snapshot(snapshot)

    def _snapshot(self) -> Dict[str, Any]:
        raise NotImplementedError(
            f"{self.MACHINE_NAME}: observe() braucht _snapshot/_restore_snapshot.")

    def _restore_snapshot(self, snapshot: Dict[str, Any]) -> None:
        raise NotImplementedError(
            f"{self.MACHINE_NAME}: observe() braucht _snapshot/_restore_snapshot.")

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def trace(self) -> List[str]:
        return list(self._trace)

    # -- Gap-/Digest-Rahmen ---------------------------------------------------
    def note_gap(self, predicted: Sequence[float], actual: Sequence[float],
                 state: Sequence[float]) -> float:
        """Canon-Gap anwenden und in der History anhängen (Append = neuester)."""
        gap = apply_gap(predicted, actual, state)
        self._gap_history.append(gap)
        return gap

    def gap(self) -> GapRecord:
        return GapRecord.from_history(self._gap_history)

    def digest(self) -> str:
        payload = json.dumps(self.state(), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # -- Argon-Haken ----------------------------------------------------------
    def _oracle_prediction(self, prompt: str, state: Sequence[float]) -> Any:
        """
        Canon-Verallgemeinerung (uni_241:157): Argon predet das symbolische Ist.
        Numerische Vorhersage (predict_value) oder None — Maschinen entscheiden
        selbst, ob ein fehlender Oracle-Prediction-Wert den Schritt abbricht
        (in_loop-Fallback) oder als Gap-„unknown" zählt.
        """
        if self.oracle is None or self.config.mode != "oracle":
            return None
        return self.oracle.predict_value(prompt)


# ---------------------------------------------------------------------------
# Fabrik
# ---------------------------------------------------------------------------
def make_machine(name: str, config: Optional[MachineConfig] = None,
                 oracle: Optional[ArgonOracle] = None) -> BaseMachine:
    """Fabrik: "goedel" | "turing" | "wittgenstein" | "composite"."""
    if config is None:
        config = MachineConfig()
    if name == "goedel":
        from machines.goedel_machine import GoedelMachine
        return GoedelMachine(config, oracle=oracle)
    if name == "turing":
        from machines.turing_machine import TuringMachine
        return TuringMachine(config, oracle=oracle)
    if name == "wittgenstein":
        from machines.wittgenstein_machine import WittgensteinMachine
        return WittgensteinMachine(config, oracle=oracle)
    if name == "composite":
        from machines.composite import CompositeMachine
        return CompositeMachine(config, oracle=oracle)
    raise ValueError(f"make_machine: unbekannte Maschine {name!r} "
                     "(erlaubt: goedel, turing, wittgenstein, composite)")