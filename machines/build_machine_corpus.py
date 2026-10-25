"""
build_machine_corpus.py — Maschinen-Korpus (Seed 42) + Phase-A-Baselinemessung.

Stufe 2 des Trainingsplans („beibringen"): Der Korpus lehrt Argon die drei
Maschinen-Domänen — Gödel (formal, eng), Turing (ablaufend), Wittgenstein
(Sprachspiel, offen). Die Samples sind doppelt gegründet:

  1. Prosa (DE/EN): Lehrtexte zu Unvollständigkeit/Halt/Sprachspiel —
     wie die Sprachen in CALIBRATION_CORPUS, hier DE+EN je Domäne.
  2. Traces der tatsächlichen Ports (machines/*): deterministische Läufe
     mit Seed 42 erzeugen die Zahlen im Text — der Korpus lehrt die
     Maschinen-Dynamik selbst (`pred=... act=... gap=...`-Zeilenformat),
     nicht nur ihre Beschreibung.

Entropie-Ziele (Plan): goedel 1.30 (unter Code 1.35 — eng/formal),
turing 1.65, wittgenstein 1.90 (über Math 1.85 — Spiel-Mehrdeutigkeit).

Phase-A (``--phase_a``): Baseline-Routing des UNTRAINIERTEN ``argon_1b``
auf den Maschinen-Domänen messen (``_gate_hooks`` aus
``train_moe_router_specialization``), Top-2-Experten je Domäne wählen
(Kollisions-Fallback: Plan-Defaults) und alles nach
``machine_registry.json`` schreiben — die Expertenwahl, die der
Trainingslauf (Schritt 3) dann antrainiert.

Determinismus: doppelte Erzeugung muss byte-identisch sein (Prüfung
integriert), damit die Verify-Stufe 2 denselben Digest liest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from machines.machine_base import MACHINE_EXPERT_DEFAULTS, MachineConfig
from machines.goedel_machine import GoedelMachine, GoedelNumber
from machines.turing_machine import TuringMachine, TuringPrograms
from machines.wittgenstein_machine import WittgensteinMachine, GeometricTokenizer

CORPUS_PATH = Path("machines/corpus/machine_corpus.jsonl")
REGISTRY_PATH = Path("machine_registry.json")

RANDOM_SEED = 42
SAMPLES_PER_DOMAIN = 24

ENTROPY_TARGETS: Dict[str, float] = {
    "goedel": 1.30,
    "turing": 1.65,
    "wittgenstein": 1.90,
}

# ---------------------------------------------------------------------------
# Prosa (DE/EN) — Lehrtexte, 8 je Domäne (4 DE + 4 EN), bewusst < 512 Zeichen.
# ---------------------------------------------------------------------------
PROSE: Dict[str, Dict[str, List[str]]] = {
    "goedel": {
        "de": [
            "Sei F ein formal konsistentes System arithmetikfähig genug für die Nachfolgerfunktion. Die Unvollständigkeit besagt: Es gibt einen Satz G, dessen Gödelnummer zwar im System ausdrückbar ist, dessen Wahrheit aber unentscheidbar bleibt.",
            "Der Gödel-Gap misst den Abstand zwischen Selbstvorhersage und symbolischem Ist: gap = |pred - actual| / (|state| + 1e-6). Ein sinkender Gap heißt wachsende Selbstkenntnis; ein steigender Gap entlarvt blinde Flecken des Modells.",
            "Primzahlen tragen die Kodierung: Jedes Zeichen der Kanonform wird zum Exponenten der Position-Primzahl. Die Gödelzahl ist dann der Rest modulo 10^12 in der Lisp-Konvention oder 10^15 in der Python-Konvention.",
            "Die strukturelle Signatur eines Ausdrucks zählt Tiefe, Größe, Atome und Listen — die Kodierung kennt die Form, nicht den Inhalt. Ein System, das seine eigene Struktur kodiert, referenziert sein Innenleben als Objekt und bleibt dennoch konsistent.",
        ],
        "en": [
            "Let F be a formal system capable of reasoning about successor arithmetic. Incompleteness states: there is a formula G whose Goedel number is expressible in F, but whose truth cannot be decided inside F itself.",
            "The Goedel gap measures the distance between self-prediction and symbolic actuality: gap = |pred - actual| / (|state| + 1e-6). A falling gap means growing self-knowledge; a rising gap exposes blind spots of the model.",
            "Primes carry the encoding: each character of the canonical form becomes the exponent of its positional prime. The Goedel number is then the remainder modulo 10^12 in the Lisp convention or 10^15 in the Python convention.",
            "The structural signature of an expression counts depth, size, atoms and lists — the encoding knows the shape of form rather than its content. A system that encodes its own structure refers to itself as an object and stays consistent nonetheless.",
        ],
    },
    "turing": {
        "de": [
            "Eine Turing-Maschine liest, schreibt und läuft: Ein Band trägt Symbole, der Kopf wandert, die Transitionstabelle bestimmt alles Weitere. Halt verdient das Programm nur, wenn kein Zustand ihn verweigert.",
            "Die Halt-Klassen sind vier: CONTINUE heißt laufend innerhalb des Budgets, LOOP sagt eine quantisierte Signatur voraus, die öfter als dreimal wiederholt wird, COLLAPSE flacht die Zustandsmagnitude unter 0.1 ein, EXPLOSION sprengt das Band.",
            "Bei LOOP oder EXPLOSION expandiert die Basis: Ein neues Axiom (SCALAR, TIME, SPACE, ...) tritt in den Zustandsraum ein und löscht den Loop-Zähler — die Maschine vergrößert das Fenster ihrer eigenen Sprache, statt weiterzuspinnen.",
            "Universelle Maschinen simulieren jede andere: Das Inkrement-Programm trägt eine Binärzahl, 'seek' läuft rechts zum Blank, 'carry' nimmt den Übertrag zurück und hält — drei Symbole genügen, um Zählmaschinen zu bauen.",
        ],
        "en": [
            "A Turing machine reads, writes and runs: a tape carries symbols, the head travels, the transition table decides everything else. A program deserves to halt only when no state refuses to let it.",
            "There are four halt classes: CONTINUE means running inside the budget, LOOP predicts a quantized signature repeated more than three times, COLLAPSE flattens state magnitude below 0.1, EXPLOSION tears through the tape.",
            "At LOOP or EXPLOSION the basis expands: a new axiom (SCALAR, TIME, SPACE, ...) enters the state space and clears the loop counter — the machine widens the window of its own language instead of spinning on.",
            "Universal machines simulate any other: the increment program carries a binary number, 'seek' runs right to the blank, 'carry' walks the carry-bit back and halts — three symbols suffice to build counting machines.",
        ],
    },
    "wittgenstein": {
        "de": [
            "Der Tractatus sagt: Das Bild ist ein Modell der Wirklichkeit, seine Elemente stehen zu den Objekten der Lage. Die Bildlogik-Maschine erzeugt daraus Regeln als ASTs — Formen, die Wirklichkeit abbilden, statt sie zu erzählen.",
            "Später, im Sprachspiel, heißt Bedeutung: Gebrauch. Ein Wort ist kein Abbild, sondern ein Zug im Spiel — 'Zeig auf X und sage: das ist das.' Die Maschine lernt Gebrauch, indem die Formel ihre Auswertung an 64 Punkten erhält.",
            "Die SH-72-Basis mappt Gematria mod 72 auf Basisformeln: sin, cos, exp, invers additiv, Logarithmus, Quadrat — jede Formel ausgewertet auf einem Domain von 0.1 bis 2 pi. Resonanz heißt das Skalarprodukt zweier Embeddings.",
            "Ein AST macht Blätter bei Tiefe drei oder mit Wahrscheinlichkeit 0.3; unär mit 0.4, binär sonst. Die Auswertungstabelle (x, y) zeigt die Regel im Gebrauch — man weiß, was ein Satz sagt, indem man ihn anwendet.",
        ],
        "en": [
            "The Tractatus says: the picture is a model of reality, its elements matching the objects of the situation. The picture-logic machine turns this into rules as ASTs — forms depicting reality instead of narrating it.",
            "Later, in language games, meaning is use: a word is not an image but a move in a game — 'point at X and say: that is it.' The machine learns use by giving the formula its own evaluation over 64 points.",
            "The SH-72 basis maps gematria mod 72 onto basis formulas: sin, cos, exp, inverse additive, logarithm, square — each evaluated over a domain from 0.1 to 2 pi. Resonance names the dot product of two embeddings.",
            "An AST makes leaves at depth three or with probability 0.3; unary with 0.4, binary otherwise. An eval table (x, y) shows the rule in use — you know what a sentence says by applying it.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Trace-Payload-Kanäle: die Ports selbst laufen lassen (deterministisch, Seed 42).
# ---------------------------------------------------------------------------
SE_PROSE_DE = "Ausdruck als S-Expression; Selbstvorhersage und symbolisches Ist in einem Trajektorienkanal:"
SE_PROSE_EN = "Expression as an S-expression; self-prediction and symbolic actuality in one trajectory channel:"

TM_PROSE_DE = "Übergangslauf mit Halt-Orakel: Transitionen bis zum Verdikt:"
TM_PROSE_EN = "Transition run with the halt oracle: transitions until the verdict:"

WIT_X_VALUES = (0.1, 1.0, 2.0, 3.5, 5.0)
WIT_PROSE_DE = "Sprachspiel-Regel als AST; Auswertungstabelle zeigt den Gebrauch:"
WIT_PROSE_EN = "Language-game rule as an AST; the eval table shows the use:"

SE_TRAJECTORY_INPUTS = ("(add x 1)", "(sin x)", "(multiply x x)", "(subtract x 2)",
                        "(cos (add x 3.141))", "(divide x 2)", "(power x 2)", "(sqrt x)",
                        "(log (add x 1))", "(exp (negate x))", "(add (sin x) 1)",
                        "(multiply 2 x)", "(subtract (power x 2) 4)", "(divide (sin x) x)",
                        "(add x 3.141)", "(negate (cos x))")


def _fixed_rng() -> np.random.Generator:
    "Korpus-RNG (Seed 42) — nur für Interleave-Nutzung, nie für Trace-Zahlen."
    return np.random.default_rng(RANDOM_SEED)


def _goedel_payloads(count: int) -> List[Tuple[str, str]]:
    """
    Gödel-Traces: S-Expression + Gödelzahl beider Konventionen + Gap-Trajektorie
    aus dem tatsächlichen Port-Lauf. lang wechselt DE/EN je Sample.
    """
    payloads: List[Tuple[str, str]] = []
    for i in range(count):
        sexpr = SE_TRAJECTORY_INPUTS[i % len(SE_TRAJECTORY_INPUTS)]
        number = int(GoedelNumber.encode(sexpr))
        lisp = int(GoedelNumber.encode_lisp(sexpr))
        sig = GoedelNumber.structural_signature(sexpr)
        machine = GoedelMachine(MachineConfig(mode="in_loop"))
        trajectory = []
        for step in range(1, 4):
            result = machine.step(SE_TRAJECTORY_INPUTS[(i + step) % len(SE_TRAJECTORY_INPUTS)])
            inv = result.invariants
            trajectory.append(
                f"step {step}: pred={inv['self_pred_norm']:.4f} "
                f"act={inv['actual_norm']:.4f} gap={result.gap:.6f} trend={result.trend}")
        lang = "de" if i % 2 == 0 else "en"
        caption = SE_PROSE_DE if lang == "de" else SE_PROSE_EN
        payloads.append((lang, "\n".join([
            f"goedel sample {i}: {caption}",
            f"expression = {sexpr}",
            f"goedel_number(python mod 10^15) = {number}",
            f"goedel_number(lisp mod 10^12) = {lisp}",
            f"structural signature = {json.dumps(sig, sort_keys=True)}",
            "; trajectory (pred act gap):",
            *trajectory,
        ])))
    return payloads


def _turing_payloads(count: int) -> List[Tuple[str, str]]:
    """
    Turing-Läufe: Transitionstabellen + Tapesequenzen + Halt-Verdikte. Das
    obj trägt NUR Band-Symbole ('10110') oder 'looper' (aktives Programm)
    — prepare_tape spreadet die Zeichenkette wörtlich (turing_machine.py:271-282).
    """
    payloads: List[Tuple[str, str]] = []
    for i in range(count):
        is_loop = i % 2 == 1
        obj = "looper" if is_loop else format((i // 2) % 16, "04b")   # z. B. "1001"
        program = TuringPrograms.looper() if is_loop else TuringPrograms.binary_increment()
        machine = TuringMachine(MachineConfig(mode="in_loop"))
        result = machine.step(obj)
        inv = result.invariants
        rows = [f"{state} {read} -> write {write} move {move} next {next_state}"
                for (state, read), (write, move, next_state) in sorted(program.transitions.items())]
        tape_str = " ".join(f"{k}:{v}" for k, v in sorted(machine.tape.items()))
        lang = "de" if i % 2 == 0 else "en"
        caption = TM_PROSE_DE if lang == "de" else TM_PROSE_EN
        payloads.append((lang, "\n".join([
            f"turing sample {i}: {caption}",
            f"obj = {obj}",
            *rows,
            f"verdict={inv['verdict']} transitions={inv['transitions']} "
            f"prediction={inv['prediction']:.2f}",
            f"tape positions: {tape_str}",
            f"trace: {result.trace[-1]}",
        ])))
    return payloads


def _plain_node(node: Any) -> Any:
    """np.str_/np.float64-Knoten zu Plain-Python (Korpus-Anzeige, kein Canon-Zugriff)."""
    if isinstance(node, tuple):
        return tuple(_plain_node(x) for x in node)
    if isinstance(node, np.generic):
        return node.item()
    return node


def _wittgenstein_payloads(count: int) -> List[Tuple[str, str]]:
    """Wittgenstein-Payloads: AST-Regeln + (x, y)-Tabellen + SH-72-Resonanz."""
    payloads: List[Tuple[str, str]] = []
    tok = GeometricTokenizer()
    words = ("turing", "goedel", "spiel", "bild", "gebrauch", "regel", "sprache", "spielzug")
    for i in range(count):
        machine = WittgensteinMachine(MachineConfig(mode="in_loop"))
        rule = machine.create_rule()
        table = " ".join(f"x={x}:y={rule.evaluate(x):.3f}" for x in WIT_X_VALUES)
        word_a = words[i % len(words)]
        word_b = words[(i + 2) % len(words)]
        k_a, idx_a, kind_a = tok.word_formula(word_a)
        res = tok.resonance(tok.embed_word(word_a), tok.embed_word(word_b))
        lang = "de" if i % 2 == 0 else "en"
        caption = WIT_PROSE_DE if lang == "de" else WIT_PROSE_EN
        payloads.append((lang, "\n".join([
            f"wittgenstein sample {i}: {caption}",
            f"rule = {rule.name}",
            f"ast = {repr(_plain_node(rule.ast))}",
            f"eval table: {table}",
            f"sh72: word_a='{word_a}' k={k_a} idx={idx_a} kind='{kind_a}'; "
            f"word_b='{word_b}'; resonance = {res:.12f}",
            "Zeig auf das Bild und sage: das ist der Gebrauch." if lang == "de"
            else "Point at the picture and say: that is the use.",
        ])))
    return payloads


# ---------------------------------------------------------------------------
# Korpus (doppelt gebaut → identisch) + JSONL-Ausgabe.
# ---------------------------------------------------------------------------
def build_domain_rows(domain: str) -> List[Dict[str, str]]:
    """24 Zeilen je Domäne: 8 Prosa (4 DE + 4 EN) + 16 Trace-Payloads."""
    builders = {"goedel": _goedel_payloads, "turing": _turing_payloads,
                "wittgenstein": _wittgenstein_payloads}
    payload_kinds = ("trajectory", "trajectory", "trajectory", "ast_eval", "ast_eval",
                     "transitions", "trajectory", "trajectory")
    payloads = builders[domain](SAMPLES_PER_DOMAIN - sum(len(v) for v in PROSE[domain].values()))
    rows: List[Dict[str, str]] = []
    for lang in sorted(PROSE[domain]):
        for text in PROSE[domain][lang]:
            rows.append({"id": f"{domain}-{len(rows):02d}", "domain": domain,
                         "lang": lang, "kind": "prose", "text": text})
    for j, (lang, text) in enumerate(payloads):
        rows.append({"id": f"{domain}-{len(rows):02d}", "domain": domain,
                     "lang": lang, "kind": payload_kinds[j % len(payload_kinds)],
                     "text": text})
    return rows


def build_corpus_str(domains: Sequence[str] = ("goedel", "turing", "wittgenstein"),
                     ) -> str:
    """Korpus als JSONL-String — eine Zeile je Sample, Interleave domänen-robin."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["id", "domain", "lang", "kind", "text"])
    writer.writeheader()
    domain_rows = [build_domain_rows(d) for d in domains]
    for row_index in range(SAMPLES_PER_DOMAIN):
        for rows in domain_rows:
            writer.writerow(rows[row_index])
    return buf.getvalue()


def build_and_write() -> Dict[str, Any]:
    """Bauen, doppelt Build byte-identisch prüfen, schreiben, Info zurückgeben."""
    first = build_corpus_str()
    second = build_corpus_str()
    if first != second:
        raise RuntimeError("build_corpus: doppelter Build nicht byte-identisch")
    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_PATH.write_text(first, encoding="utf-8")
    rows = list(csv.DictReader(io.StringIO(first)))
    per_domain: Dict[str, int] = {}
    for row in rows:
        per_domain[row["domain"]] = per_domain.get(row["domain"], 0) + 1
    return {
        "path": str(CORPUS_PATH),
        "rows": len(rows),
        "per_domain": per_domain,
        "sha256": hashlib.sha256(first.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Phase-A: Baseline-Routing des UNTRAINIERTEN argon_1b → Expertenwahl.
# ---------------------------------------------------------------------------
CHOICE_POLICY = ("measured top-2 experts per machine domain; collision fallback "
                 "(any identical top-2 sets across domains) -> plan defaults "
                 f"{MACHINE_EXPERT_DEFAULTS} (disjoint), damit die Domänen trennbar bleiben")


def read_rows() -> List[Dict[str, str]]:
    with open(CORPUS_PATH, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """TV = 0.5 * sum |p - q| (Suite-Metrik für Routing-Separation)."""
    return float(0.5 * sum(abs(a - b) for a, b in zip(p, q)))


def measure_machine_routing(model, tokenizer, device, max_length: int = 128,
                            ) -> Dict[str, Tuple[np.ndarray, float]]:
    """
    Routing-Messung über die Maschinen-Domänen — Muster aus
    ``train_moe_router_specialization.measure_domain_routing``, hier auf das
    Maschinen-Korpus angesetzt statt auf CALIBRATION_CORPUS (nur eval + no_grad).
    """
    import torch  # lokal: Modul hält die GPU-Lane schlank

    from train_moe_router_specialization import _gate_hooks

    was_training = model.training
    model.eval()
    n_experts = model.config.n_routed_experts
    by_domain: Dict[str, List[Dict[str, str]]] = {}
    for row in read_rows():
        by_domain.setdefault(row["domain"], []).append(row)

    results: Dict[str, Tuple[np.ndarray, float]] = {}
    for domain, rows in by_domain.items():
        collector: Dict[str, Any] = {
            "counts": torch.zeros(n_experts, device=device), "entropies": []}
        handles = _gate_hooks(model, n_experts, collector)
        try:
            for row in rows:
                ids = tokenizer(row["text"], return_tensors="pt", truncation=True,
                                max_length=max_length)["input_ids"].to(device)
                # no_grad zwingend: der Collector hält Entropie-Tensoren;
                # ohne die Schicht bleiben die Autograd-Graphen am Leben
                # (OOM-Fund 2026-10-25: 11.1 GiB nach ~30 Forwards).
                with torch.no_grad():
                    # Backbone-Forward ohne lm_head: Routing-Hooks brauchen
                    # keine Logits (262144-Vokab-Spike der Head-Lane entfällt).
                    _ = model.model(ids)
        finally:
            for h in handles:
                h.remove()
        counts = collector["counts"].cpu().float().numpy()
        pct = counts / max(float(counts.sum()), 1.0) * 100.0
        mean_h = (float(torch.stack(collector["entropies"]).cpu().mean().item())
                  if collector["entropies"] else 0.0)
        results[domain] = (pct, mean_h)
    if was_training:
        model.train()
    return results


def phase_a_baseline(model_dir: str = "argon_1b") -> Dict[str, Any]:
    """
    Baseline-Routing von ``argon_1b`` auf dem Maschinen-Korpus lesen, Top-2 je
    Domäne wählen (Kollisions-Fallback: Plan-Defaults für alle), TV-Matrix dazu.
    """
    import torch

    from transformers import AutoTokenizer
    from safetensors.torch import load_file

    from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Muster aus train_moe_router_specialization.py:337-347 — explizite
    # Konstruktion + load_file (transformers 5.x ignoriert torch_dtype in
    # from_pretrained → fp32-Kopie ≈ 10.2 GB → OOM-Fund 2026-10-25).
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
        cfg_dict = json.load(handle)
    config = ArgonConfig(**cfg_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = ArgonForConditionalGeneration(config).to(
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device=device)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")),
                          strict=False)
    routing = measure_machine_routing(model, tokenizer, device)

    machines: Dict[str, Any] = {}
    top2_sets: List[frozenset] = []
    for domain, (pct, mean_h) in routing.items():
        top2 = sorted(range(len(pct)), key=lambda e: (-pct[e], e))[:2]
        top2_sets.append(frozenset(top2))
        machines[domain] = {
            "experts": list(MACHINE_EXPERT_DEFAULTS.get(domain, top2)),
            "entropy_target": ENTROPY_TARGETS[domain],
            "baseline_top2": top2,
            "baseline_pct": [round(float(v), 3) for v in pct],
            "baseline_entropy": round(mean_h, 6),
        }
    # Kollisions-Fallback: wenn zwei Domänen dasselbe Top-2 wählen, ist die
    # Baseline diffus → Plan-Defaults (disjunkt) für alle Maschinen behalten.
    collided = len(set(top2_sets)) < len(top2_sets)
    for domain, entry in machines.items():
        entry["choice_source"] = ("plan-defaults (baseline collision)" if collided
                                  else "baseline measured")
        if not collided:
            entry["experts"] = list(entry["baseline_top2"])
    tvs: Dict[str, Dict[str, float]] = {}
    domains = list(routing)
    for a in domains:
        tvs[a] = {}
        for b in domains:
            if a != b:
                # TV auf Wahrscheinlichkeitsskala (pct ist Prozent)
                tvs[a][b] = round(total_variation(np.asarray(routing[a][0]) / 100.0,
                                                  np.asarray(routing[b][0]) / 100.0), 4)
    return {
        "version": 1,
        "created": "2026-10-25",
        "checkpoint_base": model_dir,
        "checkpoint_trained": "argon_1b_machines",
        "steer": 50.0,
        "entropy_targets": ENTROPY_TARGETS,
        "machines": machines,
        "tv_between_domains": tvs,
        "choice_policy": CHOICE_POLICY,
        "corpus": {"path": str(CORPUS_PATH), "rows": len(read_rows()),
                   "sha256": hashlib.sha256(CORPUS_PATH.read_bytes()).hexdigest()},
    }


def write_registry(registry: Dict[str, Any]) -> None:
    with open(REGISTRY_PATH, "w", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=1, sort_keys=True, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase_a", action="store_true",
                        help="Baseline-Routing von argon_1b messen + Registry schreiben")
    parser.add_argument("--model_dir", default="argon_1b")
    args = parser.parse_args()

    info = build_and_write()
    print(json.dumps(info, sort_keys=True))
    if args.phase_a:
        registry = phase_a_baseline(args.model_dir)
        write_registry(registry)
        print(json.dumps({"registry": str(REGISTRY_PATH),
                          "choice": {d: e["choice_source"]
                                     for d, e in registry["machines"].items()},
                          "tv": registry["tv_between_domains"]},
                         indent=1, sort_keys=True))


if __name__ == "__main__":
    main()