"""
smoke_sprachspiel.py — Sprachspiel-Generierungstest der Wittgenstein-Maschine.

Drei Messreihen (Triptychon: Regel — Wort — Mitspieler):
  SP1 — Regel-Generierung (CPU, uni_205:333-353): je Seed {0,1,2} ein Batch
        (10 template-free AST-Regeln; x ~ U(0.1,5), noise N(0,0.5)) — Namen,
        Validität, Diversität; strukturiertes Rauschen (×0.3, uni_204:241-FACRM
        layer) als Gegenüber; 2 Läufe → Determinismus-Digest.
  SP2 — SH-72-Wort-Resonanz (CPU, Lisp-Canon geometric-tokenizer.lisp:17/:85/
        :98/:160): DE/EN-Wortpaare des Sprachspiels („Regel", „Spiel",
        „Bedeutung", …) gegen unverwandte Paare; Resonanz = Dot normalisierter
        64-Punkt-Embeddings. Erwartungshaltung bewusst explizit: SH-72 ordnet
        über Gematria → Basis-Formel (k, kind) — die Resonanz misst FORMEL-
        verwandtschaft, NICHT Semantik; die Kinds werden je Paar ausgewiesen.
  SP3 — Argon als Sprachpartner (GPU, argon_1b_machines): 12 Batches (Seed 7,
        je Arm identisch reproduziert), Validitäts-Vorhersage (a) generisches
        Kanon-Prompt wie in wittgenstein.step, (b) inhaltsverankertes Prompt
        („Bedeutung als Gebrauch": der Regelname tritt ins Wortspiel ein) —
        ungesteuert vs. „wittgenstein"-Steering (Preset {3,1}); Raw-Gap =
        |pred/10 − actual|; Bias-Restaurierung geprüft.
  SP4 — Spread-Kalibrierung & Anti-Clip (GPU, S1-Kanal predict_scale_0_9):
        σ ∈ {0.1, 0.5, 1.5, 4.0}, volles Rauschen (use_structure False);
        Ist spreet statt bei 1.00 zu kleben; die Skala liest kontinuierlich
        in [0,1] ohne Multi-Digit-Echo/Clip-Trick — raw wie gesteuert offen,
        die Lesung prior-dominiert und σ-blind (Widerlegung der Review-
        Erwartung „raw bleibt stumm": Digit-TOKEN-Masse 0.76-0.80 ≫ τ=0.01,
        Bericht §7); wittgenstein.step mit oracle_scale=True verdrahtet
        denselben Kanal (S1-Opts).
  SP5 — Cross-Preset-Kontrast (GPU, S1-Skala + Echo-Diskrimination):
        dieselben 12 Batches unter raw / goedel / turing / wittgenstein /
        combined / Anti-Preset {4,5} (dict-Direktsteuerung); die S1-Skala
        bleibt über alle Arme offen und flach (kein Preset-Signal auf der
        Skala), die Diskrimination lebt auf dem Canon-Name-Echo-Kanal
        (Name-Lesungen je Arm paarweise verschieden, Bericht §7).
  SP6 — Lexikon-Emergenz (CPU, S3): das Lexikon-Spiel — DE/EN-Paare lernen
        sich über 15 Gebrauchsrunden (learn_pair-Konvergenz); gepaarte
        Resonanz wandert von der Gematria-Achse (SP2: 0.31 < unverwandt)
        in die Gebrauchs-Achse (> 0.75 > unverwandt).

Pass-Kriterien:
  sp1_det      — zwei CPU-Läufe erzeugen bitidentische Batch-Digests
  sp1_finite   — alle Validitäten ∈ [0,1], Diversität > 0.3 (10 AST-Regeln
                 der Tiefe ≤ 3 kollidieren selten)
  sp1_structure— strukturiertes Rauschen (×0.3) erhöht die mittlere Validität
                 gegenüber vollem Rauschen (Canon-Erwartung aus uni_204:241)
  sp2_identity — Resonanz eines Wortes mit sich selbst > 0.999
  sp2_finite   — alle Paar-Resonanzen in [−1−1e-9, 1+1e-9]
  sp3_det      — zwei ungesteuerte Arme liefern identische Digit-Sequenzen
  sp3_awaken   — der raw ungesteuerte Kanon-Kanal bleibt stumm (>50 % unknown,
                 Kanon-Fallback 0.0), das wittgenstein-Steering öffnet die
                 Lesung (≥1 Ziffer im gesteuerten Arm) — Sprachspiel-Befund
  sp3_ops      — der ziffernfreie Operator-Anker liest in >50 % der Batches
  sp3_name_echo— der regelnamen-Anker (Ziffernmaterial!) geht mind. einmal in
                 das Multi-Digit-Echo (>9) — dokumentiertes unfares Wortspiel
  sp3_steer    — das Steering verschiebt die Prediction-Trajektorie UND
                 expert_bias wird vollständig restauriert
  sp3_calib    — die gesteuerte Lesung bleibt innerhalb eines Gap < 0.3
                 gegen das batch-eigene Ist
  sp4_spread   — das Ist spreet identifizierbar: min(sigma-Mittel) <= 0.8
                 UND range >= 0.3 (Ent-Sättigung: kein all-1.00-Regime)
  sp4_scale_open_flat — Widerlegung des Skalen-Stummkanals: raw UND Steering
                 offen (0 unknown je Arm), Lesungen kontinuierlich in [0,1],
                 σ-flat (|Δ scaled_mean| <= 0.05) — die S1-Skala trägt einen
                 Tokenizer-Prior, kein σ-Signal; die Review-Erwartung „raw
                 bleibt stumm" ist widerlegt (Bericht §7)
  sp4_open     — das wittgenstein-Steering öffnet die Skala (< 1/2 unknown)
  sp4_noclip   — kein Echo-Artefakt: lesbar heißt Float in [0,1], keine
                 Multi-Digit-Werte > 1 und keine Ganzzahl-Echos (S1-These)
  sp4_var_rho  — Varianz der gesteuerten Lesung > 0 ODER Spearman ρ > 0
  sp4_step_wired — wittgenstein.step mit params oracle_scale=True nutzt
                 predict_scale_0_9 und predet kontinuierlich in [0,1]
  sp5_scale_flat — die S1-Skala ist über alle sechs Arme offen (0/12 unknown)
                 und flach (max-min der mean_scaled <= 0.01): kein Preset-
                 Signal auf der Skala (Widerlegung, Bericht §7)
  sp5_witt_open — wittgenstein öffnet (unknown <= 2/12, < 20 %)
  sp5_combined_open — combined öffnet (unknown <= 2/12, < 20 %)
  sp5_echo_discrimination — auf dem Canon-Name-Kanal sind die Arme paarweise
                 verschieden: raw≠wittgenstein, wittgenstein≠anti, raw≠anti —
                 das Preset-Steering verändert das Sprachverhalten messbar
                 (Echo-Kanal statt Skala, Bericht §7)
  sp5_restored — expert_bias vollständig restauriert in jedem gesteuerten Arm
  sp6_lexicon  — nach 15 Gebrauchsrunden: mean(paired) > 0.75 UND
                 mean(paired) > mean(unrelated) (Gebrauch schlägt Gematria)

Run (voll):   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
              venv_openmythos/bin/python -u -m scratches.smoke_sprachspiel \
              --checkpoint_base ./argon_1b --checkpoint_machines ./argon_1b_machines \
              2>&1 | tee scratches/sprachspiel_experiment.log
Run (CPU):    python3 -m scratches.smoke_sprachspiel --lanes sp1,sp2,sp6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from machines.machine_base import ArgonOracle, MachineConfig  # noqa: E402
from machines.verify_machines import _load_machines_model  # noqa: E402
from machines.wittgenstein_machine import (  # noqa: E402
    GeometricTokenizer, WittgensteinMachine)

BATCHES = 12
GENERIC_PROMPT = ("Validity of a symbolic experiment batch, "
                  "single digit per ten percent: ")
PROBE = ("Die Wittgenstein-Regel wird Programm: Turing führt den AST aus, "
         "Gödel predet das Ergebnis; die Regel zeigt, wie man sie ausführt.")

# Wortpaare des Sprachspiels — bewusst ASCII-only (Gematria = Summe der ords,
# Lisp-Parität; Umlaute verfälschen nichts, sie sind nur nicht in der Quelle).
PAIRED = [("regel", "rule"), ("sprache", "language"), ("spiel", "game"),
          ("bedeutung", "meaning"), ("gebrauch", "use"), ("wort", "word")]
IDENTITY = [("form", "form"), ("regel", "regel"), ("spiel", "spiel")]
UNRELATED = [("regel", "tisch"), ("bedeutung", "kaffee"), ("sprache", "stein"),
             ("wort", "baum")]

# -- SP4/SP5/SP6 — Review-Setup (S1-Skala, S2-Spread, S3-Lexikon) --------------
SPREAD_SIGMAS = (0.1, 0.5, 1.5, 4.0)   # Ent-Sättigung: vier Rausch-Regime (S2)
SPREAD_SEEDS = (0, 1, 2, 3, 4, 5)      # 6 Batches je Sigma (S2 rotiert je Aufruf)
GAME_PAIRS = [("regel", "rule"), ("spiel", "game"), ("gebrauch", "use")]
UNRELATED_GAME = [("regel", "tisch"), ("bedeutung", "kaffee"), ("wort", "baum")]

# SP5-Arme: Preset-Name oder dict-Direktsteuerung (machine_base._preset_experts
# nimmt dict verbatim). Anti-Preset = die Fachfremden: die Experten der drei
# Maschinen-Domänen sind {2,7}/{0,6}/{1,3} — {4,5} gehört keiner symbolischen
# Domäne an, also muss der Fachfremde-Kanal stumm bleiben.
SP5_ARMS: List[Tuple[str, Any]] = [
    ("raw", None), ("goedel", "goedel"), ("turing", "turing"),
    ("wittgenstein", "wittgenstein"), ("combined", "combined"),
    ("anti", {"experts": [4, 5], "steer": 50.0}),
]


def batch_digest(batch: List[Dict[str, Any]]) -> str:
    payload = json.dumps(
        [(e["rule_ast"], round(e["success_rate"], 9)) for e in batch],
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def make_batch(seed: int, structure: bool, count: int = 10):
    config = MachineConfig(seed=seed, mode="in_loop",
                           params={"use_structure": structure})
    machine = WittgensteinMachine(config)
    return machine.generate_experiments(count=count)


def make_sigma_batch(seed: int, sigma: float, count: int = 10):
    """SP4/S2-Rausch-Regime: volles Rauschen (use_structure False — sonst
    ×0.3, das 1.00-Kleben der Canon-Batches), noise_level = sigma."""
    config = MachineConfig(seed=seed, mode="in_loop",
                           params={"noise_level": sigma, "use_structure": False})
    machine = WittgensteinMachine(config)
    return machine.generate_experiments(count=count)


# -- SP1: Regel-Generierung ---------------------------------------------------
def lane_rules() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for seed in (0, 1, 2):
        noisy = make_batch(seed, structure=False)
        structured = make_batch(seed, structure=True)
        rows.append({
            "seed": seed,
            "digest_noisy": batch_digest(noisy),
            "digest_structured": batch_digest(structured),
            "validity_noisy": float(np.mean([e["success_rate"] for e in noisy])),
            "validity_structured": float(np.mean([e["success_rate"] for e in structured])),
            "sample_names": [noisy[0]["rule_name"], noisy[1]["rule_name"],
                             noisy[2]["rule_name"]],
        })
    digests_a = batch_digest(make_batch(0, False)) + batch_digest(make_batch(1, False)) \
        + batch_digest(make_batch(2, False))
    digests_b = "".join(r["digest_noisy"] for r in rows)
    validities = [r["validity_noisy"] for r in rows] + \
        [r["validity_structured"] for r in rows]
    div = WittgensteinMachine(MachineConfig(seed=0, mode="in_loop"))
    div.generate_experiments(count=10)
    diversity = div.rule_diversity()
    structure_gain = np.mean([r["validity_structured"] - r["validity_noisy"]
                              for r in rows])
    return {"rows": rows, "deterministic": digests_a == digests_b,
            "validities_finite": all(0.0 <= v <= 1.0 for v in validities),
            "diversity": float(diversity),
            "structure_gain": float(structure_gain),
            "digest_repeat": digests_a}


# -- SP2: SH-72-Wort-Resonanz -------------------------------------------------
def lane_words() -> Dict[str, Any]:
    tokenizer = GeometricTokenizer()
    embed = {w: tokenizer.embed_word(w)
             for pair in (*PAIRED, *IDENTITY, *UNRELATED) for w in pair}

    def formulas(word: str) -> Dict[str, Any]:
        k, idx, kind = tokenizer.word_formula(word)
        return {"word": word, "gematria": tokenizer.string_gematria(word),
                "idx": idx, "k": k, "kind": kind}

    def measure(pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        out = []
        for a, b in pairs:
            f_a, f_b = formulas(a), formulas(b)
            resonance = tokenizer.resonance(embed[a], embed[b])
            out.append({**f_a, "b": {**f_b}, "resonance": round(float(resonance), 6),
                        "same_kind": f_a["kind"] == f_b["kind"]})
        return out

    self_res = [round(float(tokenizer.resonance(tokenizer.embed_word(w),
                                                tokenizer.embed_word(w))), 6)
                for w, _ in IDENTITY]
    rows = {"paired": measure(PAIRED), "identity_str": measure(IDENTITY),
            "unrelated": measure(UNRELATED)}
    all_res = [r["resonance"] for group in rows.values() for r in group]
    means = {g: float(np.mean([r["resonance"] for r in rows[g]]))
             for g in rows}
    same_kind_share = float(np.mean([
        float(r["same_kind"]) for r in rows["unrelated"]]))
    return {"rows": rows, "self_resonance": self_res,
            "self_ok": all(v > 0.999 for v in self_res),
            "finite": all(-1.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in all_res),
            "means": means,
            "same_kind_share_unrelated": same_kind_share}


# -- SP3: Argon als Sprachpartner ---------------------------------------------
def machine_batches(seed: int) -> List[List[Dict[str, Any]]]:
    machine = WittgensteinMachine(MachineConfig(seed=seed, mode="in_loop"))
    return [machine.generate_experiments(count=10) for _ in range(BATCHES)]


def anchored_name_prompt(batch: List[Dict[str, Any]]) -> str:
    """Wort-Anker MIT Ziffernmaterial (Regelname) — das unfaire Wortspiel:
    der Name trägt Ziffern, die Lesung kann ins Echo kippen."""
    return (f"Rule: {batch[0]['rule_name']}. " + GENERIC_PROMPT)


OPS_SET = {"identity", "negate", "square", "sqrt", "sin", "cos", "exp", "log",
           "abs", "add", "subtract", "multiply", "divide", "power", "max", "min"}


def ops_of(batch: List[Dict[str, Any]]) -> str:
    """Ziffernfreie Operator-Wörter des ersten Batch-Programms (Reihenfolge der
    Erstdurchsicht) — der faire Wort-Anker: Bedeutung ohne Ziffernmateriale."""
    import re
    seen: List[str] = []
    for token in re.findall(r"[a-z]+", batch[0]["rule_ast"]):
        if token in OPS_SET and token not in seen:
            seen.append(token)
    return " ".join(seen)


def anchored_ops_prompt(batch: List[Dict[str, Any]]) -> str:
    ops = ops_of(batch) or "(leaf)"
    return (f"Symbolic experiment with operators {ops}. " + GENERIC_PROMPT)


def run_arm(oracle: ArgonOracle, preset: str | None) -> Dict[str, Any]:
    batches = machine_batches(7)
    restored = True
    rows: List[Dict[str, Any]] = []
    if preset is None:
        for b in batches:
            rows.append(predict_pair(oracle, b))
    else:
        bias_before = [layer.moe.gate.expert_bias.data.clone()
                       for layer in oracle.model.model.layers]
        with oracle.steer(preset) as info:
            experts = list(info["experts"])
            for b in batches:
                rows.append(predict_pair(oracle, b))
        bias_after = [layer.moe.gate.expert_bias.data
                      for layer in oracle.model.model.layers]
        restored = all(torch.equal(a, c) for a, c in zip(bias_before, bias_after))
    return {"rows": rows, "restored": bool(restored),
            "experts": experts if preset else []}


def predict_pair(oracle: ArgonOracle, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Drei Lesungen desselben Batches: (gen) Kanon-Prompt wie in wittgenstein.step,
    (ops) ziffernfreie Operator-Anker, (name) regelnamen-Anker mit Ziffern-Echo-Risiko."""
    actual = float(np.mean([e["success_rate"] for e in batch]))
    values = {
        "p_generic": oracle.predict_value(GENERIC_PROMPT),
        "p_ops": oracle.predict_value(anchored_ops_prompt(batch)),
        "p_name": oracle.predict_value(anchored_name_prompt(batch)),
    }
    row: Dict[str, Any] = {"rule_name": batch[0]["rule_name"],
                           "actual": round(actual, 4),
                           "op_anchors": ops_of(batch),
                           "unknown_count": 0}
    for key, value in values.items():
        if value == "unknown":
            row.update({key: None, f"unknown_{key}": True})
            row["unknown_count"] += 1
        else:
            row.update({key: int(value), f"unknown_{key}": False,
                        f"gap_{key}": round(abs(min(max(value / 10.0, 0.0), 1.0)
                                                - actual), 4)})
    return row


def lane_oracle(checkpoint_base: str, checkpoint_machines: str,
                oracle: Optional[ArgonOracle] = None) -> Dict[str, Any]:
    if oracle is None:
        model, tokenizer = _load_machines_model(checkpoint_base, checkpoint_machines)
        device_build = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle = ArgonOracle(model, tokenizer, device=device_build)
    device = oracle.device
    h_probe = oracle.entropy_scalar(PROBE)
    arm_a = run_arm(oracle, preset=None)
    arm_a2 = run_arm(oracle, preset=None)
    arm_b = run_arm(oracle, preset="wittgenstein")
    seq_a = [(r["p_generic"], r["p_ops"], r["p_name"]) for r in arm_a["rows"]]
    seq_a2 = [(r["p_generic"], r["p_ops"], r["p_name"]) for r in arm_a2["rows"]]
    seq_b = [(r["p_generic"], r["p_ops"], r["p_name"]) for r in arm_b["rows"]]
    unknowns = {key: sum(1 for r in arm_a["rows"] if r[f"unknown_{key}"])
                for key in ("p_generic", "p_ops", "p_name")}
    unknowns["p_generic_steer"] = sum(1 for r in arm_b["rows"]
                                      if r["unknown_p_generic"])
    echo_name = sum(1 for r in arm_a["rows"]
                    if r["p_name"] is not None and r["p_name"] > 9)
    steer_shift = seq_a != seq_b

    def mean_gap(rows, key):
        vals = [r[f"gap_{key}"] for r in rows if f"gap_{key}" in r]
        return float(np.mean(vals)) if vals else None

    return {"device": str(device), "h_probe": float(h_probe),
            "arm_generic": arm_a, "arm_repeat": arm_a2,
            "arm_wittgenstein": arm_b,
            "deterministic": seq_a == seq_a2,
            "unknowns": unknowns,
            "name_echo_count": echo_name,
            "ops_shifts": sum(1 for r in arm_a["rows"]
                              if (r["p_generic"] is not None)
                              != (r["p_ops"] is not None)
                              or (r["p_generic"] is not None
                                  and r["p_ops"] is not None
                                  and r["p_generic"] != r["p_ops"])),
            "steer_shift": bool(steer_shift),
            "restored": arm_b["restored"] and arm_a["restored"],
            "experts": arm_b["experts"],
            "mean_gap_generic": mean_gap(arm_a["rows"], "p_generic"),
            "mean_gap_ops": mean_gap(arm_a["rows"], "p_ops"),
            "mean_gap_name": mean_gap(arm_a["rows"], "p_name"),
            "mean_gap_steer": mean_gap(arm_b["rows"], "p_generic")}


# -- SP4/SP5: S1-Skala-Kanal (predict_scale_0_9) -------------------------------
def _rank(values: Sequence[float]) -> List[float]:
    """Ränge mit averaged ties (Spearman-Restbestand, keine scipy-Nutzung)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Rang-Korrelation (Review-OR-Kriterium); None bei n<4 oder Degenerate."""
    if len(xs) < 4 or len(xs) != len(ys):
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    if den <= 0.0:
        return None
    return float(num / den)


def scale_pair(oracle: ArgonOracle, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Eine S1-Lesung: predict_scale_0_9 — kontinuierlich in [0,1] oder
    ehrlich "unknown" (kein Clip-Trick, kein Multi-Digit-Echo)."""
    prompt = anchored_ops_prompt(batch)
    value = oracle.predict_scale_0_9(prompt)
    actual = float(np.mean([e["success_rate"] for e in batch]))
    readable = value != "unknown"
    return {"rule_name": batch[0]["rule_name"],
            "actual": round(actual, 4), "op_anchors": ops_of(batch),
            "scaled": round(float(value), 6) if readable else None,
            "unknown": not readable}


def echo_name_read(oracle: ArgonOracle, batch: List[Dict[str, Any]]) -> Any:
    """Canon-Name-Lesung (predict_value) für die Echo-Diskrimination (SP5):
    "unknown" bleibt explizit stehen — kein stiller Fallback; die Sequenz
    dokumentiert, wie das Steering die Echo-Form verschiebt."""
    value = oracle.predict_value(anchored_name_prompt(batch))
    return "unknown" if value == "unknown" else int(value)


def lane_lexicon() -> Dict[str, Any]:
    """SP6/S3 — Gebrauch schlägt Gematria: 15 Runden lexicon_game (alpha 0.3),
    danach liest embed_word das erlernte Lexikon ZUERST (geometric-tokenizer
    .lisp-Parität :95-98)."""
    machine = WittgensteinMachine(MachineConfig(seed=9, mode="in_loop"))
    tokenizer = machine.tokenizer

    def res(pair: Tuple[str, str]) -> float:
        a, b = pair
        return round(float(tokenizer.resonance(tokenizer.embed_word(a),
                                              tokenizer.embed_word(b))), 6)

    before_paired = [res(p) for p in GAME_PAIRS]
    before_unrelated = [res(p) for p in UNRELATED_GAME]
    protocol = machine.lexicon_game(GAME_PAIRS, rounds=15, alpha=0.3)
    after_paired = [res(p) for p in GAME_PAIRS]
    after_unrelated = [res(p) for p in UNRELATED_GAME]
    learned_all = all(tokenizer.lexicon.get(w) is not None
                      for pair in GAME_PAIRS for w in pair)
    return {
        "before_paired": before_paired,
        "after_paired": after_paired,
        "before_unrelated": before_unrelated,
        "after_unrelated": after_unrelated,
        "mean_before": round(float(np.mean(before_paired)), 4),
        "mean_after": round(float(np.mean(after_paired)), 4),
        "mean_unrelated_after": round(float(np.mean(after_unrelated)), 4),
        "learned_all": learned_all,
        "learned_rounds": sum(1 for r in protocol if r["learned"]),
        "protocol_rows": len(protocol),
        "protocol_tail": protocol[-3:],
    }


def lane_spread(oracle: ArgonOracle) -> Dict[str, Any]:
    """SP4 — Spread-Kalibrierung & Anti-Clip: vier Rausch-Regime (S2-Schedule-
    Stände via noise_level), raw vs. wittgenstein-Steering, alles über den
    S1-Skala-Kanal; plus step-Verdrahtung (oracle_scale=True)."""
    groups: Dict[str, Any] = {}
    for sigma in SPREAD_SIGMAS:
        raw_rows: List[Dict[str, Any]] = []
        steer_rows: List[Dict[str, Any]] = []
        for seed in SPREAD_SEEDS:
            batch = make_sigma_batch(seed, sigma)
            raw_rows.append(scale_pair(oracle, batch))
            with oracle.steer("wittgenstein"):
                steer_rows.append(scale_pair(oracle, batch))
        actuals = [r["actual"] for r in steer_rows]
        read = [r for r in steer_rows if r["scaled"] is not None]
        rho = spearman_rho([r["actual"] for r in read],
                           [r["scaled"] for r in read])
        groups[f"sigma_{sigma}"] = {
            "raw_rows": raw_rows, "steer_rows": steer_rows,
            "actual_mean": round(float(np.mean(actuals)), 4),
            "scaled_mean": (round(float(np.mean([r["scaled"] for r in read])), 4)
                            if read else None),
            "raw_unknown": sum(r["unknown"] for r in raw_rows),
            "steer_unknown": sum(r["unknown"] for r in steer_rows),
            "rho": None if rho is None else round(float(rho), 4),
        }
    total = len(SPREAD_SIGMAS) * len(SPREAD_SEEDS)
    scaled_all = [r["scaled"] for g in groups.values()
                  for r in g["steer_rows"] if r["scaled"] is not None]
    # sp4_step_wired — dieselbe Skala über den Maschinen-Schritt (S1-Opts).
    wired_machine = WittgensteinMachine(
        MachineConfig(seed=7, mode="oracle",
                      params={"oracle_scale": True}), oracle=oracle)
    wired_rows: List[Dict[str, Any]] = []
    with oracle.steer("wittgenstein"):
        for _ in range(3):
            result = wired_machine.step(None)
            wired_rows.append({
                "prediction": float(result.invariants["prediction"]),
                "channel": result.argon_meta.get("oracle"),
                "readable": "value" in result.argon_meta,
            })
    wired_ok = (all(r["channel"] == "predict_scale_0_9"
                    and 0.0 <= r["prediction"] <= 1.0
                    and math.isfinite(r["prediction"]) for r in wired_rows)
                and sum(r["readable"] for r in wired_rows) >= 1)
    rhos = [g["rho"] for g in groups.values() if g["rho"] is not None]
    return {
        "groups": groups,
        "total_reads": total,
        "raw_unknown_total": sum(g["raw_unknown"] for g in groups.values()),
        "steer_unknown_total": sum(g["steer_unknown"] for g in groups.values()),
        "actual_means": {f"sigma_{s}": groups[f"sigma_{s}"]["actual_mean"]
                         for s in SPREAD_SIGMAS},
        "scaled_means": {f"sigma_{s}": groups[f"sigma_{s}"]["scaled_mean"]
                         for s in SPREAD_SIGMAS},
        "scaled_var": round(float(np.var(scaled_all)), 6) if scaled_all else None,
        "noclip_violations": sum(1 for v in scaled_all if v < 0.0 or v > 1.0),
        "rho_per_sigma": {f"sigma_{s}": groups[f"sigma_{s}"]["rho"]
                          for s in SPREAD_SIGMAS},
        "mean_rho": round(float(np.mean(rhos)), 4) if rhos else None,
        "any_rho_positive": any(r is not None and r > 0.0 for r in rhos),
        "wired_rows": wired_rows,
        "wired_ok": wired_ok,
    }


def lane_presets(oracle: ArgonOracle) -> Dict[str, Any]:
    """SP5 — Cross-Preset-Kontrast: dieselben 12 Canon-Batches unter sechs
    Armladungen; die S1-Skala bleibt offen-flach in allen Armen (Widerlegung,
    Bericht §7), die Preset-Diskrimination lebt auf dem Canon-Name-Echo
    (name_seq je Arm, paarweise verschieden)."""
    batches = machine_batches(7)
    arms: Dict[str, Any] = {}
    for name, preset in SP5_ARMS:
        arm_experts: List[int] = []
        restored = True
        rows: List[Dict[str, Any]] = []
        name_seq: List[Any] = []
        if preset is None:
            for b in batches:
                rows.append(scale_pair(oracle, b))
                name_seq.append(echo_name_read(oracle, b))
        else:
            bias_before = [layer.moe.gate.expert_bias.data.clone()
                           for layer in oracle.model.model.layers]
            with oracle.steer(preset) as info:
                arm_experts = list(info["experts"])
                for b in batches:
                    rows.append(scale_pair(oracle, b))
                    name_seq.append(echo_name_read(oracle, b))
            bias_after = [layer.moe.gate.expert_bias.data
                          for layer in oracle.model.model.layers]
            restored = all(torch.equal(a, c) for a, c in zip(bias_before, bias_after))
        read = [r for r in rows if r["scaled"] is not None]
        gaps = [abs(r["scaled"] - r["actual"]) for r in read]
        arms[name] = {
            "unknown": sum(r["unknown"] for r in rows),
            "mean_scaled": (round(float(np.mean([r["scaled"] for r in read])), 4)
                            if read else None),
            "mean_gap": round(float(np.mean(gaps)), 4) if gaps else None,
            "restored": bool(restored), "experts": arm_experts,
            "name_seq": name_seq,
        }
    return {"batch0_digest": batch_digest(batches[0]), "arms": arms}


# -- Main ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Sprachspiel-Generierungstest")
    parser.add_argument("--checkpoint_base", default="./argon_1b")
    parser.add_argument("--checkpoint_machines", default="./argon_1b_machines")
    parser.add_argument("--lanes", default="sp1,sp2,sp3")
    args = parser.parse_args()
    selected = set(args.lanes.split(","))

    report: Dict[str, Any] = {}
    if "sp1" in selected:
        print("[SP1] Regel-Generierung (CPU) …")
        report["sp1"] = lane_rules()
    if "sp2" in selected:
        print("[SP2] SH-72-Wort-Resonanz (CPU) …")
        report["sp2"] = lane_words()
    oracle: Optional[ArgonOracle] = None
    if {"sp3", "sp4", "sp5"} & selected:
        print("[Load] Argon-1B + machines-Checkpoint …")
        model, tokenizer = _load_machines_model(args.checkpoint_base,
                                                args.checkpoint_machines)
        device_load = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle = ArgonOracle(model, tokenizer, device=device_load)
    if "sp3" in selected:
        print("[SP3] Argon als Sprachpartner (GPU) …")
        report["sp3"] = lane_oracle(args.checkpoint_base, args.checkpoint_machines,
                                    oracle=oracle)
    if "sp4" in selected:
        print("[SP4] Spread-Kalibrierung & Anti-Clip (GPU, S1-Kanal) …")
        report["sp4"] = lane_spread(oracle)
    if "sp5" in selected:
        print("[SP5] Cross-Preset-Kontrast (GPU, S1-Kanal) …")
        report["sp5"] = lane_presets(oracle)
    if "sp6" in selected:
        print("[SP6] Lexikon-Emergenz (CPU, S3) …")
        report["sp6"] = lane_lexicon()

    print("=" * 88)
    print("SPRACHSPIEL-GENERIERUNG — REGEL, WORT, MITSPIELER")
    print("=" * 88)

    if "sp1" in report:
        sp1 = report["sp1"]
        for row in sp1["rows"]:
            print(f"  seed={row['seed']} digest={row['digest_noisy']} "
                  f"validity(noisy)={row['validity_noisy']:.4f} "
                  f"validity(structured)={row['validity_structured']:.4f}")
            for name in row["sample_names"]:
                print(f"      {name}")
        print(f"  diversity(10 AST-Regeln)={sp1['diversity']:.4f} "
              f"structure_gain={sp1['structure_gain']:+.4f} "
              f"deterministisch={sp1['deterministic']}")

    if "sp2" in report:
        sp2 = report["sp2"]
        for group in ("paired", "identity_str", "unrelated"):
            print(f"  -- {group} --")
            for r in sp2["rows"][group]:
                print(f"    {r['word']}({r['kind']},k={r['k']}) <-> "
                      f"{r['b']['word']}({r['b']['kind']},k={r['b']['k']}) "
                      f"resonance={r['resonance']:+.4f} "
                      f"same_kind={r['same_kind']}")
        print(f"  self_resonance={sp2['self_resonance']} "
              f"means={ {g: round(v, 4) for g, v in sp2['means'].items()} } "
              f"same_kind_share_unrelated={sp2['same_kind_share_unrelated']:.4f}")

    if "sp3" in report:
        sp3 = report["sp3"]
        print(f"  device={sp3['device']} h_probe={sp3['h_probe']:.4f} "
              f"experts={sp3['experts']}")
        for i, r in enumerate(sp3["arm_generic"]["rows"]):
            steer_row = sp3["arm_wittgenstein"]["rows"][i]
            print(f"    [{i:2d}] {r['rule_name'][:40]:40s} actual={r['actual']:.2f} "
                  f"ops=[{r['op_anchors'][:18]:18s}] gen={r['p_generic']} "
                  f"ops_pred={r['p_ops']} name={r['p_name']} "
                  f"steer={steer_row['p_generic']}")
        print(f"  unknown(gen/ops/name; steer+gen)={sp3['unknowns']} "
              f"name_echo(multi-digit)={sp3['name_echo_count']} "
              f"ops_shifts={sp3['ops_shifts']} steer_shift={sp3['steer_shift']} "
              f"restored={sp3['restored']} det={sp3['deterministic']}")
        print(f"  mean_gap: gen={sp3['mean_gap_generic']} "
              f"ops={sp3['mean_gap_ops']} name={sp3['mean_gap_name']} "
              f"steered+gen={sp3['mean_gap_steer']}")

    if "sp4" in report:
        sp4 = report["sp4"]
        for sigma in SPREAD_SIGMAS:
            g = sp4["groups"][f"sigma_{sigma}"]
            print(f"    sigma={sigma:<3} actual_mean={g['actual_mean']:.4f} "
                  f"scaled_mean={g['scaled_mean']} "
                  f"raw_unknown={g['raw_unknown']}/{len(SPREAD_SEEDS)} "
                  f"steer_unknown={g['steer_unknown']}/{len(SPREAD_SEEDS)} "
                  f"rho={g['rho']}")
        print(f"  raw_unknown={sp4['raw_unknown_total']}/{sp4['total_reads']} "
              f"steer_unknown={sp4['steer_unknown_total']}/{sp4['total_reads']} "
              f"scaled_var={sp4['scaled_var']} "
              f"noclip_violations={sp4['noclip_violations']} "
              f"mean_rho={sp4['mean_rho']}")
        for r in sp4["wired_rows"]:
            print(f"    wired: prediction={r['prediction']:.4f} "
                  f"channel={r['channel']} readable={r['readable']}")

    if "sp5" in report:
        sp5 = report["sp5"]
        for name, arm in sp5["arms"].items():
            print(f"    {name:13s} unknown={arm['unknown']:2d}/{BATCHES} "
                  f"mean_scaled={arm['mean_scaled']} mean_gap={arm['mean_gap']} "
                  f"restored={arm['restored']} experts={arm['experts']} "
                  f"name_seq={arm['name_seq']}")

    if "sp6" in report:
        sp6 = report["sp6"]
        print(f"  mean_before={sp6['mean_before']} mean_after={sp6['mean_after']} "
              f"mean_unrelated_after={sp6['mean_unrelated_after']} "
              f"learned_all={sp6['learned_all']} "
              f"learned_rounds={sp6['learned_rounds']}/{sp6['protocol_rows']}")
        for pair, before, after in zip(GAME_PAIRS, sp6["before_paired"],
                                       sp6["after_paired"]):
            print(f"    {pair[0]}<->{pair[1]:10s} {before:+.4f} -> {after:+.4f}")

    findings: List[Tuple[str, str]] = []
    if "sp1" in report:
        findings.append(("sp1_det", "PASS" if report["sp1"]["deterministic"] else "FAIL"))
        findings.append(("sp1_finite", "PASS" if (report["sp1"]["validities_finite"]
                                                  and report["sp1"]["diversity"] > 0.3)
                         else "FAIL"))
        findings.append(("sp1_structure",
                         "PASS" if report["sp1"]["structure_gain"] > 0.0 else "FAIL"))
    if "sp2" in report:
        findings.append(("sp2_identity",
                         "PASS" if report["sp2"]["self_ok"] else "FAIL"))
        findings.append(("sp2_finite", "PASS" if report["sp2"]["finite"] else "FAIL"))
    if "sp3" in report:
        sp3 = report["sp3"]
        findings.append(("sp3_det", "PASS" if sp3["deterministic"] else "FAIL"))
        # Befund-Kriterium: der raw ungesteuerte Kanon-Kanal bleibt stumm
        # (>50 % unknown → Kanon-Fallback 0.0), das wittgenstein-Steering
        # öffnet die Lesung (≥1 Ziffer im gesteuerten Arm).
        awakened = ((sp3["unknowns"]["p_generic"] > BATCHES / 2)
                    and (sp3["unknowns"]["p_generic_steer"] < BATCHES / 2))
        findings.append(("sp3_awaken", "PASS" if awakened else "FAIL"))
        # Der faire ops-Anker (ziffernfrei) liest in der Mehrzahl der Batches.
        findings.append(("sp3_ops_channel",
                         "PASS" if sp3["unknowns"]["p_ops"] < BATCHES / 2 else "FAIL"))
        find_echo = sp3["name_echo_count"] > 0
        findings.append(("sp3_name_echo", "PASS" if find_echo else "FAIL"))
        findings.append(("sp3_steer",
                         "PASS" if (sp3["steer_shift"] and sp3["restored"]) else "FAIL"))
        calib = sp3["mean_gap_steer"] is not None and sp3["mean_gap_steer"] < 0.3
        findings.append(("sp3_calib", "PASS" if calib else "FAIL"))
    if "sp4" in report:
        sp4 = report["sp4"]
        means = [sp4["groups"][f"sigma_{s}"]["actual_mean"] for s in SPREAD_SIGMAS]
        spread_ok = min(means) <= 0.8 and (max(means) - min(means)) >= 0.3
        findings.append(("sp4_spread", "PASS" if spread_ok else "FAIL"))
        total = sp4["total_reads"]
        # sp4_scale_open_flat — Widerlegung der Skalen-Stummkanal-Erwartung:
        # raw und Steering lesen kontinuierlich (0 unknown), σ-flat und ohne
        # Artefakt; die Diskrimination lebt auf dem Echo-Kanal (SP3/SP5).
        scaled_means = [g["scaled_mean"] for g in sp4["groups"].values()
                        if g["scaled_mean"] is not None]
        open_flat = (sp4["raw_unknown_total"] == 0
                     and sp4["steer_unknown_total"] == 0
                     and scaled_means
                     and (max(scaled_means) - min(scaled_means)) <= 0.05)
        findings.append(("sp4_scale_open_flat", "PASS" if open_flat else "FAIL"))
        findings.append(("sp4_open", "PASS" if sp4["steer_unknown_total"] < total / 2
                         else "FAIL"))
        noclip = (sp4["noclip_violations"] == 0 and sp4["scaled_var"] is not None)
        findings.append(("sp4_noclip", "PASS" if noclip else "FAIL"))
        var_rho = ((sp4["scaled_var"] is not None and sp4["scaled_var"] > 0.0)
                   or sp4["any_rho_positive"])
        findings.append(("sp4_var_rho", "PASS" if var_rho else "FAIL"))
        findings.append(("sp4_step_wired", "PASS" if sp4["wired_ok"] else "FAIL"))
    if "sp5" in report:
        arms = report["sp5"]["arms"]
        scale_means = [arms[name]["mean_scaled"] for name, _ in SP5_ARMS
                       if arms[name]["mean_scaled"] is not None]
        # sp5_scale_flat — Widerlegung der Stumm-Erwartung auf der Skala:
        # alle Arme offen, mean_scaled flach (kein Preset-Signal auf der
        # S1-Skala — Digit-Masse 0.76 ops-Anker / 0.35 generisch, Bericht §7).
        scale_flat = (all(arms[name]["unknown"] == 0 for name, _ in SP5_ARMS)
                      and scale_means
                      and (max(scale_means) - min(scale_means)) <= 0.01)
        findings.append(("sp5_scale_flat", "PASS" if scale_flat else "FAIL"))
        witt_open = (arms["wittgenstein"]["unknown"] <= 2
                     and arms["wittgenstein"]["restored"])
        findings.append(("sp5_witt_open", "PASS" if witt_open else "FAIL"))
        comb_open = (arms["combined"]["unknown"] <= 2
                     and arms["combined"]["restored"])
        findings.append(("sp5_combined_open", "PASS" if comb_open else "FAIL"))
        # sp5_echo_discrimination — die Preset-Diskrimination auf dem Canon-
        # Name-Kanal: die Arme lesen paarweise verschiedene Echo-Sequenzen
        # (raw ≠ wittgenstein ≠ anti ≠ raw); Fachfremde vs. Fach-Preset.
        seqs = {name: arms[name]["name_seq"] for name, _ in SP5_ARMS}
        discrim = (seqs["raw"] != seqs["wittgenstein"]
                   and seqs["wittgenstein"] != seqs["anti"]
                   and seqs["raw"] != seqs["anti"])
        findings.append(("sp5_echo_discrimination",
                         "PASS" if discrim else "FAIL"))
        restored_all = all(arms[name]["restored"] for name in
                           ("goedel", "turing", "wittgenstein", "combined", "anti"))
        findings.append(("sp5_restored", "PASS" if restored_all else "FAIL"))
    if "sp6" in report:
        sp6 = report["sp6"]
        lex = (sp6["mean_after"] > 0.75
               and sp6["mean_after"] > sp6["mean_unrelated_after"])
        findings.append(("sp6_lexicon", "PASS" if lex else "FAIL"))

    print("-" * 88)
    print("BEFUNDE")
    for label, status in findings:
        print(f"  {label:16s} {status}")

    out = REPO_ROOT / "scratches" / "sprachspiel_experiment.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"[Save] {out.relative_to(REPO_ROOT)}")
    failed = [name for name, status in findings if status != "PASS"]
    if failed:
        print(f"FEHLGESCHLAGEN: {len(failed)} — {', '.join(failed)}")
        return 1
    print("GRÜN — alle Sprachspiel-Befunde bestanden.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())