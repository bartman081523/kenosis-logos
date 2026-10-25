"""
lisp_bridge.py — SBCL-Brücke: Python → `sbcl --script machines/lisp/run-bridge.lisp`.

Protokoll (machines/lisp/bridge-protocol.lisp):
  * EIN Auftrag = eine S-Expression-Zeile auf stdin (:op "NAME" :key value ...).
  * Antwort: Zeile Mini-JSON zwischen ``;;; RESULT-BEGIN``/``END`` — alles
    andere (Banner, Philosopher-Text) wird verworfen. *(setf *log-level*
    :none)* killt die Vendor-Logs (utils.lisp:should-log-p → NIL).
  * Fehler → {"error": ...} im Frame (Exit 0) — kein stiller Fall.
  * Aufruf `["timeout", "--kill-after=15", f"{t}s", "sbcl", "--script", ...]`
    — NIE `sleep`, NIE `sbcl --non-interactive` (das setzt --quit und
    terminiert VOR der Skriptausführung — SBCL 2.6.9-Fund).
  * ASCII-rein (Plan-Risiko 2: Bignum-Exponenten, Locale-Freiheit).
  * Kein Quicklisp, kein cl-json nötig (package.lisp: (:use #:cl)).

Parität:
  P1 bitgenau (pure functions): goedel (prin1-Kanon), gap (|actual| mit
  1e-9-Guard — anders als uni_241!), sh72 (embed/gematria/resonance),
  ast_eval (utils-LCG bitgenau seedbar). P2 strukturell: philosopher
  (sxhash-/Text-Kanäle). Spiegel: machines/lisp_parity.py;
  Fixtures: machines/lisp/parity/lisp_fixtures.json.

Usage:

    from machines.lisp_bridge import LISPBridge
    bridge = LISPBridge()
    data = bridge.call("sh72", words=["turing", "goedel"])

CLI:

    python -m machines.lisp_bridge ping
    python -m machines.lisp_bridge sh72 --words turing goedel
    python -m machines.lisp_bridge ast_eval --seed 42 --count 3
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

BRIDGE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BRIDGE_DIR / "lisp" / "run-bridge.lisp"
VENDOR_DIR = BRIDGE_DIR / "lisp" / "vendor"

MARKER_BEGIN = ";;; RESULT-BEGIN"
MARKER_END = ";;; RESULT-END"

ASCII_PROTOCOL = True  # Bridge-Payloads sind ASCII-rein ( dokumentiert)


class BridgeError(RuntimeError):
    """Brücke fehlgeschlagen (SBCL fehlt, Timeout, unparste Antwort, Op-Error)."""


def sbcl_available(script: Path = SCRIPT_PATH) -> bool:
    """SBCL + Skript vorhanden? (verify_machines: Stufe 4 → SKIP statt Fail)."""
    return shutil.which("sbcl") is not None and script.is_file()


def vendor_digest(vendor: Path = VENDOR_DIR) -> Dict[str, str]:
    "SHA-256 je Vendor-Datei (sortiert) — Provenienz/Eviction-Test der Kopie."
    digests: Dict[str, str] = {}
    for path in sorted(p for p in vendor.rglob("*") if p.is_file()):
        digests[str(path.relative_to(vendor))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
    return digests


def _escape_string(value: str) -> str:
    out = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + out + '"'


def _encode_number(value: float) -> str:
    "SBCL-READ-Kanon: float → d0-Suffix (double), int → ~D."
    if isinstance(value, bool):
        return "T" if value else "NIL"
    if isinstance(value, int):
        return str(value)
    text = repr(float(value))
    return text + "d0"                     # z. B. 1e-09 → "1e-09d0"


def encode_value(value: Any) -> str:
    "Python-Wert → S-Expression-Bruchstück (ASCII-rein)."
    if value is None:
        return "NIL"
    if isinstance(value, bool):
        return "T" if value else "NIL"
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, (int, float)):
        return _encode_number(value)
    if isinstance(value, (list, tuple)):
        return "(" + " ".join(encode_value(v) for v in value) + ")"
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            sym = ":" + str(key).replace("_", "-").upper()
            parts.append(sym + " " + encode_value(item))
        return "(" + " ".join(parts) + ")"
    raise TypeError(f"bridge value: nicht unterstützter Typ {type(value)!r}")


def encode_request(op: str, **kwargs: Any) -> str:
    ":op + KwArgs → (:op \"NAME\" :KEY value ...)."
    parts = [":op", _escape_string(op)]
    for key, value in kwargs.items():
        if value is None:
            continue
        sym = ":" + str(key).replace("_", "-").upper()
        parts.append(sym)
        parts.append(encode_value(value))
    return "(" + " ".join(parts) + ")"


def extract_frame(stdout: str) -> Dict[str, Any]:
    "Frame zwischen den Marken extrahieren und JSON-parsen."
    begin = stdout.find(MARKER_BEGIN)
    end = stdout.find(MARKER_END)
    if begin < 0 or end < 0 or end < begin:
        preview = stdout.strip()[:300] if stdout.strip() else "(leere Ausgabe)"
        raise BridgeError(f"kein RESULT-Frame in SBCL-Ausgabe: {preview!r}")
    payload = stdout[begin + len(MARKER_BEGIN):end].strip()
    if not payload:
        raise BridgeError("RESULT-Frame leer")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        raise BridgeError(f"JSON im Frame unparsbar: {e}: {payload[:200]!r}")


class LISPBridge:
    """Einer SBCL-Subprozess je Anfrage (deterministisch, kein Session-Zustand)."""

    def __init__(self, timeout: float = 45.0, script: Path = SCRIPT_PATH,
                 interpreter: str = "sbcl"):
        self.script = script
        self.interpreter = interpreter
        self.timeout = timeout

    def available(self) -> bool:
        return sbcl_available(self.script)

    def call(self, op: str, **kwargs: Any) -> Dict[str, Any]:
        """Anfrage ausführen; {"error": ...} → BridgeError (kein stiller Pass)."""
        request = encode_request(op, **kwargs)
        if not self.available():
            raise BridgeError(f"Brücke nicht verfügbar: {self.interpreter}/"
                              f"{self.script} fehlt")
        command = ["timeout", "--kill-after=15", f"{self.timeout:.0f}s",
                   self.interpreter, "--script", str(self.script)]
        try:
            proc = subprocess.run(command, input=request + "\n",
                                  capture_output=True, text=True,
                                  timeout=self.timeout + 10.0)
        except subprocess.TimeoutExpired as e:
            raise BridgeError(f"SBCL-Timeout nach {self.timeout}s (op={op})") from e
        except FileNotFoundError as e:
            raise BridgeError(f"SBCL nicht startbar: {e}") from e

        data = extract_frame(proc.stdout)
        if "error" in data:
            raise BridgeError(f"op={op}: {data['error']}")
        return data

    # -- Komfort-Ops (P1 außer philosopher — P2 strukturell) ------------------
    def ping(self) -> Dict[str, Any]:
        return self.call("ping")

    def goedel(self, expr: Union[str, list, tuple]) -> Dict[str, Any]:
        return self.call("goedel", expr=expr)

    def gap(self, pred: Union[float, Sequence[float]],
            actual: Union[float, Sequence[float]]) -> Dict[str, Any]:
        return self.call("gap", pred=pred, actual=actual)

    def sh72(self, words: Sequence[str]) -> Dict[str, Any]:
        return self.call("sh72", words=list(words))

    def ast_eval(self, seed: int = 42, count: int = 3,
                 xs: Sequence[float] = (1.0,)) -> Dict[str, Any]:
        return self.call("ast_eval", seed=int(seed), count=int(count),
                         xs=[float(x) for x in xs])

    def philosopher(self, problem: str = "self-reference", cycles: int = 2,
                    seed: int = 123456789,
                    queries: Sequence[str] = ("consciousness", "existence", "meaning"),
                    self_modify_threshold: float = 0.5) -> Dict[str, Any]:
        return self.call("philosopher", problem=problem, n_cycles=int(cycles),
                         seed=int(seed), queries=list(queries),
                         self_modify_threshold=float(self_modify_threshold))


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SBCL-Brücke (ASCII-Protokoll)")
    parser.add_argument("op", choices=["ping", "goedel", "gap", "sh72",
                                       "ast_eval", "philosopher"])
    parser.add_argument("--expr", default="hi")
    parser.add_argument("--pred", nargs="*", type=float, default=None)
    parser.add_argument("--actual", nargs="*", type=float, default=None)
    parser.add_argument("--words", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--xs", nargs="*", type=float, default=[1.0])
    parser.add_argument("--problem", default="self-reference")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    bridge = LISPBridge(timeout=args.timeout)
    if args.op == "ping":
        data = bridge.ping()
    elif args.op == "goedel":
        data = bridge.goedel(args.expr)
    elif args.op == "gap":
        if not args.pred or not args.actual:
            parser.error("gap braucht --pred/--actual")
        data = bridge.gap(args.pred if len(args.pred) > 1 else args.pred[0],
                          args.actual if len(args.actual) > 1 else args.actual[0])
    elif args.op == "sh72":
        data = bridge.sh72(args.words or ["turing"])
    elif args.op == "ast_eval":
        data = bridge.ast_eval(seed=args.seed, count=args.count, xs=args.xs)
    else:
        data = bridge.philosopher(problem=args.problem, cycles=args.cycles,
                                  seed=args.seed)
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    _cli()