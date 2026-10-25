"""
lisp_parity.py — Python-Spiegel des LISP-Kanons (machines/lisp/vendor).

Diese Schicht spiegelt die VENDOR-SEMANTIK (lisp-cognito-philosopher, MIT,
Julian 2026-02) exakt — sie ist NICHT der uni_205-Maschinenkanon (der lebt
in machines/wittgenstein_machine.py). Divergenzen sind belegt:

  - lisp symbolic-operator.lisp:10-20  *unary-operators* MIT ``inv`` (10),
    *binary-operators* OHNE max/min (5), *constants* = [0,1,2,pi,e,phi] (6,
    „contaminated"; uni_205 corrected hat 5 ohne phi).
  - :56-79 generate-random-ast: unary-Prob 0.6 (uni_205: 0.4), leaf-Kette
    dr<0.3 → dr<0.5 → (dr<0.5 → var | const dr(10.0) | const aus Liste);
    ALLE Ziehung über den utils-LCG → bitgenau seedbar (Plan-Stufe 4-P1).
  - :26-50 apply-operator: sqrt = safe-sqrt (negativ → 0.0, KEIN abs),
    exp = exp(min(x,100)) (KEIN unterer Clamp), log = safe-log (x<=0 → 0.0),
    power = expt(abs(a), min(abs(b),10)); sonst Error (Bridge fängt ihn).
  - utils.lisp:159-183 safe-helpers; :82-110 vector utils.
  - geometric-tokenizer.lisp:12-31 get-sh-basis-formula (mode 0..5, sonst
    'x — für idx < 72 unerreichbar, floor(71/12)=5), :35-43 domain
    (0.1..2π, i/63.0), :104-121 embed-word (evaluiert + normalisiert mit
    normalize-vector: norm < 1e-9 → Vektor UNVERÄNDERT, nicht null).
  - goedel-encoder.lisp:10-27 *primes* (20), goedel-encode-string max 15
    Zeichen, GANZE char-code-Exponenten, MOD 10**12; prin1-to-string quotes
    Strings → '"hi"' (4 Zeichen) → Zahl 0 (2^34·5^? ≡ 0 mod 10^12 — Canon:
    ≥3-Zeichen-Strings mit 2er/5er-Primen kollabieren fast immer; gleiches
    Phänomen wie „Gödel number: 0" im uni_204-Log :15).
  - goedel-encoder.lisp:87-150 GAP-KANON: Nenner ist |actual| mit 1e-9-Guard
    (uni_241s ‖state‖+1e-6 ist ein ANDERER Kanon — apply_gap in
    machine_base), :156-164 interpret-gap-Schwellen 0.1/0.3/0.5/0.7.

Protokollbeschränkung der Brücke: ASCII-rein (Plan-Risiko 2), P1 nur für
pure functions; sxhash-/Text-Kanäle (predict-output-signature, philosopher-
Verlauf) bleiben P2 strukturell.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple, Union

from machines.machine_base import LCGRandom

# lisp *primes* (goedel-encoder.lisp:10-11) — exakt 20.
LISP_PRIMES: Tuple[int, ...] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)

MOD_LISP = 10**12            # goedel-encode-string; mod (expt 10 12)
LISP_MAX_CHARS = 15          # Default &optional (max-chars 15)

LISP_UNARY = ["identity", "negate", "square", "sqrt", "sin", "cos",
              "exp", "log", "abs", "inv"]                       # :10-12
LISP_BINARY = ["add", "subtract", "multiply", "divide", "power"]  # :14-16
PHI = (1.0 + math.sqrt(5.0)) / 2.0
LISP_CONSTANTS = [0.0, 1.0, 2.0, math.pi, math.e, PHI]            # :18-20

DOMAIN_POINTS = 64
SH_INDEX = 72
BASIS_WIDTH = 12

EPS_LISP = 1e-9  # safe-divide / gap / normalize Guards (utils.lisp)


class LispOperatorError(ValueError):
    """evaluate-ast/apply-operator Signals (unknown op, unbound variable)."""


# -- prin1-Kanon ---------------------------------------------------------------

def prin1_string(s: str) -> str:
    """
    prin1-to-string eines SBCL-Strings: Anführungszeichen + Escapes. Die
    Brücke hält die Fixtures ASCII-rein ohne Back-Quotes → minimal-Kanon.
    """
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def goedel_encode_string(s: str, max_chars: int = LISP_MAX_CHARS) -> int:
    """goedel-encoder.lisp:19-27 — volle Exponenten, 20 Primzahlen, MOD 1e12."""
    encoding = 1
    for i, char in enumerate(s[:max_chars]):
        prime = LISP_PRIMES[i % len(LISP_PRIMES)]
        power = ord(char)
        encoding = encoding * pow(prime, power, MOD_LISP) % MOD_LISP
    return encoding


def goedel_encode(expr_canonical: str) -> int:
    """goedel-encode: erst prin1 (Anführungszeichen!), dann encode-string."""
    encoding = 1
    for i, char in enumerate(expr_canonical[:LISP_MAX_CHARS]):
        prime = LISP_PRIMES[i % len(LISP_PRIMES)]
        power = ord(char)
        encoding = encoding * pow(prime, power, MOD_LISP) % MOD_LISP
    return encoding


# -- Operatoren (apply-operator, :26-50 + utils safe-helpers) -------------------

def safe_sqrt(x: float) -> float:
    return 0.0 if x < 0 else float(math.sqrt(x))


def safe_log(x: float) -> float:
    return 0.0 if x <= 0 else float(math.log(x))


def safe_divide(a: float, b: float) -> float:
    return 0.0 if abs(b) < EPS_LISP else a / b


def apply_operator(op: str, args: Sequence[float]) -> float:
    """Lisp-apply-operator — Error auf unbekanntem Op (kein stiller Fallback)."""
    a = args[0] if len(args) > 0 else 0.0
    b = args[1] if len(args) > 1 else 0.0
    if op == "identity":
        return a
    if op == "negate":
        return -a
    if op == "square":
        return a * a
    if op == "sqrt":
        return safe_sqrt(a)
    if op == "sin":
        return float(math.sin(a))
    if op == "cos":
        return float(math.cos(a))
    if op == "exp":
        return float(math.exp(min(a, 100.0)))
    if op == "log":
        return safe_log(a)
    if op == "abs":
        return float(abs(a))
    if op == "inv":
        return safe_divide(1.0, a)
    if op == "add":
        return a + b
    if op == "subtract":
        return a - b
    if op == "multiply":
        return a * b
    if op == "divide":
        return safe_divide(a, b)
    if op == "power":
        return float(math.pow(abs(a), min(abs(b), 10.0)))
    raise LispOperatorError(f"Unknown operator: {op.upper()}")


# -- AST ( Lisp-Format: ('var',), ('const', v), (op, arg[, arg2]) ) -------------

def generate_random_ast(rng: LCGRandom, depth: int = 0, max_depth: int = 3) -> Any:
    """
    symbolic-operator.lisp:56-79 — exakte Ziehungsreihenfolge. Blattprüfung
    short-circuited (depth >= max-depth ⇒ keine Ziehung). AST-Form:
    ('var',) | ('const', value) | (op, arg) | (op, arg1, arg2).
    """
    if depth >= max_depth or rng.float(1.0) < 0.3:
        if rng.float(1.0) < 0.5:
            if rng.float(1.0) < 0.5:
                return ("var",)
            return ("const", rng.float(10.0))
        return ("const", LISP_CONSTANTS[rng.integer(len(LISP_CONSTANTS))])
    if rng.float(1.0) < 0.6:
        op = LISP_UNARY[rng.integer(len(LISP_UNARY))]
        return (op, generate_random_ast(rng, depth + 1, max_depth))
    op = LISP_BINARY[rng.integer(len(LISP_BINARY))]
    return (op, generate_random_ast(rng, depth + 1, max_depth),
            generate_random_ast(rng, depth + 1, max_depth))


def evaluate_ast(ast: Any, bindings: Dict[str, float]) -> float:
    """evaluate-ast — ('var',) → bindings['x'] (Error unbound); unbekannt → Error."""
    if isinstance(ast, tuple):
        if ast and ast[0] == "const":
            return float(ast[1])
        if ast and ast[0] == "var":
            if "x" not in bindings:
                raise LispOperatorError("Unbound variable: X")
            return float(bindings["x"])
        op, args = ast[0], [evaluate_ast(arg, bindings) for arg in ast[1:]]
        return apply_operator(op, args)
    return float(ast)


def sexpr_string(ast: Any) -> str:
    """prin1-to-string des ASTs: Uppercase-Symbole, double → '~ad0'. """
    def fmt_number(v: float) -> str:
        text = repr(float(v))
        return f"{text}d0" if "e" not in text and "E" not in text else text.replace("e", "d")

    if isinstance(ast, tuple):
        if ast and ast[0] == "var":
            return "(:VAR X)"
        if ast and ast[0] == "const":
            return f"(:CONST {fmt_number(ast[1])})"
        return ("(" + str(ast[0]).upper() + " "
                + " ".join(sexpr_string(arg) for arg in ast[1:]) + ")")
    return str(ast).upper()


def ast_size(ast: Any) -> int:
    """Knotenzahl (Vendor: 1 + Summe der Kinder) — const-Kind ist ein Atom."""
    if not isinstance(ast, tuple):
        return 1
    return sum(ast_size(arg) for arg in ast[1:]) + 1


def ast_depth(ast: Any) -> int:
    """Blatttiefe-Tiefe (Vendor-Canon): :CONST zählt als Blattebene → 0."""
    if not isinstance(ast, tuple):
        return 0
    if ast[0] == "const":
        return 0
    return 1 + max((ast_depth(arg) for arg in ast[1:]), default=0)


# -- Gap-Kanon (goedel-encoder.lisp:87-150) ------------------------------------

def lisp_gap(pred: Union[float, List[float]],
             actual: Union[float, List[float]]) -> float:
    """Nenner = |actual|-Norm mit 1e-9-Guard (anders als uni_241!).

    Vendor-Canon (goedel-encoder.lisp:96-130): Strings sind in CL vectorp →
    sie landen im VECTOR-Pfad und brechen in vector-distance ab (NUMBER-
    Fehler); der compute-string-gap-Zweig ist damit aus der Bridge NICHT
    erreichbar — hier bewusst gespiegelt als Fehler, nicht als Levenshtein.
    """
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) \
            and isinstance(pred, (int, float)) and not isinstance(pred, bool):
        magnitude = abs(float(actual))
        return 0.0 if magnitude < EPS_LISP else abs(float(pred) - float(actual)) / magnitude
    if isinstance(pred, str) or isinstance(actual, str) \
            or not isinstance(pred, (list, tuple)) or not isinstance(actual, (list, tuple)):
        raise LispOperatorError("string/mixed input: vector-distance erhält "
                                "keine NUMBERs (Vendor-Canon, tot im Reaktionsfall)")
    diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(pred, actual)))
    norm = math.sqrt(sum(y * y for y in actual))
    return 0.0 if norm < EPS_LISP else diff / norm


def interpret_gap(gap: float) -> str:
    """goedel-encoder.lisp:156-164 — Keywords verbatim (Bindestriche)."""
    if gap < 0.1:
        return "excellent-self-knowledge"
    if gap < 0.3:
        return "good-self-knowledge"
    if gap < 0.5:
        return "moderate-blind-spot"
    if gap < 0.7:
        return "significant-blind-spot"
    return "severe-blind-spot"


# -- SH-72 (geometric-tokenizer.lisp) ------------------------------------------

def sh_formula(idx: int) -> Tuple[int, str]:
    """get-sh-basis-formula :14-31 — k = 1+mod(idx 12); mode 0..5 ('x' sonst)."""
    k = 1 + (idx % BASIS_WIDTH)
    mode = idx // BASIS_WIDTH
    kind = ["sin", "cos", "exp_decay", "inv_add", "log", "square"][min(mode, 5)]
    return k, kind


def domain_vector() -> List[float]:
    """make-domain-vector :35-43 — i/63.0 (0.1..2π)."""
    start, end = 0.1, 2.0 * math.pi
    return [start + (i / 63.0) * (end - start) for i in range(DOMAIN_POINTS)]


def string_gematria(s: str) -> int:
    return sum(ord(ch) for ch in s)


def embed_word(word: str, domain: Sequence[float] = None) -> List[float]:
    """
    embed-word :104-121 + evaluate-formula :123-152 + normalize-vector —
    exakter Lisp-Spiegel (safe-guards verbatim; norm < 1e-9 → Vektor
    UNVERÄNDERT).
    """
    domain = list(domain) if domain is not None else domain_vector()
    k, kind = sh_formula(string_gematria(word) % SH_INDEX)
    values: List[float] = []
    for x in domain:
        try:
            if kind == "sin":
                value = math.sin(k * x)                                  # formula: sin (multiply k x)
            elif kind == "cos":
                value = math.cos(k * x)
            elif kind == "exp_decay":
                value = math.exp(min(-(x / k), 100.0))                   # exp (divide (negate x) k)
            elif kind == "inv_add":
                value = safe_divide(1.0, x + k)                          # divide 1.0d0 (add x k)
            elif kind == "log":
                value = safe_log(x + k)                                  # log (add x k)
            elif kind == "square":
                value = safe_divide(math.pow(abs(x), 2.0), k)            # divide (power x 2) k
            else:
                value = float(x)                                          # otherwise 'x (idx ≥ 72)
        except (ValueError, OverflowError, ZeroDivisionError):
            value = 0.0
        values.append(value if math.isfinite(value) else 0.0)
    norm = math.sqrt(sum(v * v for v in values))
    if norm < EPS_LISP:
        return values                  # Lisp normalize-vector: unverändert!
    return [v / norm for v in values]


def resonance(word1: str, word2: str, domain: Sequence[float] = None) -> float:
    """resonance :160-167 — Dot-Produkt derEmbeddings."""
    a, b = embed_word(word1, domain), embed_word(word2, domain)
    return sum(x * y for x, y in zip(a, b))


__all__ = [
    "LispOperatorError", "LISP_PRIMES", "LISP_UNARY", "LISP_BINARY",
    "LISP_CONSTANTS", "MOD_LISP", "PHI", "apply_operator", "ast_depth",
    "ast_size", "domain_vector", "evaluate_ast", "embed_word", "goedel_encode",
    "goedel_encode_string", "interpret_gap", "lisp_gap", "prin1_string",
    "generate_random_ast", "resonance", "safe_divide", "safe_log", "safe_sqrt",
    "sh_formula", "sexpr_string", "string_gematria",
]