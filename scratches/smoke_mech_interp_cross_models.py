"""smoke_mech_interp_cross_models.py — MI-5: Cross-Model-Mechanistik.

Die MI-1..MI-4-Canons und die SP3/Paris-Anker, gefahren auf der Lineage:
    gemma-3-1b-it (Ausgang, DENSE — Scout: Gemma3ForCausalLM, 26 Schichten,
                   vocab 262144, kein .moe, kein Softcapping)
    argon_1b        (topologische Initialisierung — Untrainiert-Router,
                     expert_bias ALL-ZERO std=0.0000 über 26 Schichten)
    argon_1b_specialized    (Router-Spezialisierung, learned bias)
    argon_1b_specialized_ml (dito, ML-Korpus)
Plan: ~/.claude/plans/MI-5-cross-model-mi.md (Denkmodus MechanisticCircuitMix-
Mind v1.0_20261026_CircMix, Pfad A; User-Laufzeichen „lgtm" 2026-10-26).

PREREG (P2 — registriert vor dem ersten GPU-Lauf, keine Band-Weitung):
  MI-5-Subset-Canon: 6 Texte, je Text[0] einer MI-1-Domäne (code=quicksort,
  math_logic=Summenformel, prose=Bergsee, goedel=Primfaktoren, turing=TM-
  Basisbewegung, wittgenstein=Bedeutung-im-Gebrauch). Alle Verdikt-Aggregate
  laufen nur auf diesem Subset; volle Felder persistent (T1).

  V1 mi5_lineage_routing_shift (argon_1b vs specialized/ml): je Schwester
     S ∈ {specialized, specialized_ml} und je Subset-Text t ist
       D_tv(S, t) = (1/26) · Σ_li canon_tv(pct_S[t][li], pct_1b[t][li])
     (8-dim Zeilen-Verteilungen der MI-1-Tomographie, canon_tv-Kanon) und
     PASS, wenn für BEIDE Schwestern mean_t D_tv > 0.01 (≥1% verschobene
     Routing-Masse je Schicht im Schnitt).
  V2 mi5_paris_recall_compare (alle 4): je Modell gilt
       argmax(last_probs("The capital of France is")) == die JEWEILS
       AUFGELOSTE Paris-ID (resolve_token_ids_exact, Oberflächen-Exakt,
       unique; die ID je Modell ist Datums-Feld gegen die argon-Konstante
       9079).
  V3 mi5_dense_parent_argmax (gemma): die SP3-Argmax-Maske ist kein MoE-
     Artefakt, sondern Erbe des Ausgangs-Substrats: (a) raw top1 am
     GENERIC_PROMPT ist ein FORMAT-TOKEN aus der registrierten Menge
     {"", "\n", "\n\n", " ", ".", ",", ":", ";", "!", "?"} AND
     (b) p('1') = digit_mass.digit_probs["1"] >= 0.05 AND
     (c) M_digit = digit_mass.M_digit >= 0.10. (Massen-Präsenz-Schwellen;
     die argon-M_ der Runde 0.354 bleibt Vergleichs-Datum, keine Basis-Rate
     eines anderen Modells.)
  V4 mi5_crystallization_compare (alle 4): je Modell gilt
       crystallization_start(layerwise_logits.rows) nicht None auf >= 4/6
       Subset-Texten (min_tail=2-Canon). Die star-Je-Text-Werte sowie
       p_final_top1/official_top1/anchor_tv je Text bleiben Datums-Felder
       (lens-versus-bf16-Abweichungen sind das SP3-Anker-Thema, keine
       Verdikt-Prädikate).
  V5 mi5_gate_entropy_gradient (alle 3 argon): je Variante gilt strikt
       mean-H(ℓ11)[Bergsee] + 0.01 < mean-H(ℓ11)[quicksort]
     (Prosa < Code am Capacity-Gate-Kanal, MI-3-P4-V9-Analog quer zur
     Lineage; H = assemble_pct.ent, tokenweise). Der Vergleich argon_1b
     [Bias-Zero] vs specialized/ml bleibt DATUM — gelernt oder
     Initialisierungs-Erbe entscheidet die Zahlenlage, nicht das Verdikt.
  V6 mi5_l17_attn_share_compare (alle 3 argon): je Variante ist
       dp_attn(ℓ17, Paris) / (dp_attn + dp_moe) >= 0.5
     (MI-2-Analog quer: die ℓ17-Attention-W Recall ist Linien-Erbe).

  DATUMS-ZUSATZ ([Datums-Zusatz], registriert, keine Verdikte):
   - bias_std_26 je argon-Variante (Bias-Zero-Diskriminante)
   - M_digit-Serie je argon-Variante: raw, ko_prose@core (knockout_experts
     {3,7}@BAND_CORE penalty -50), ko_code@core ({0,1,4}@BAND_CORE),
     steer_full_prose (steer_band {3,7} SET +50/-50 alle Schichten) — auf
     argon_1b wirkt -50 ab Null-Bias (reine Score-Minderung, Plan-Risiko 5)
   - Chimären-Verdikt-Felder je argon-Variante: CHIMERA_PROSE_CODE und
     CHIMERA_PROSE_MATH (User-Audit 1:1), t_switch = gemeinsame Token-Prefix-
     Länge gegen PREFIX_PROSE_CODE/MATH (IDs, MI-4-Kanon), pct/ent-Voll-Bahn
   - dense-arm: sublayer_logits_dense(PARIS_TEXT) auf gemma — telescope
     (Σ dp_attn + dp_mlp == endpoints) als Einheits-Prüfung, dp-Bahn je
     Schicht; scheitert die Konstruktion → OFFEN/strukturell, keine
     Simulation (Via-Negativa)
   - strukturell-nicht-existente Felder auf gemma buchen offiziell:
     layerwise_routing (keine Gates), tokenwise_routing, Steerings/KO
     (OFFEN/strukturell — kein Simulations-Ersatz)
   - resolve_token_ids_exact je Modell; Paris-ID 9079 als argon-Konstante
     nur Referenz-Datum (gemma-Vokabel per Scout gleich groß, ID kann
     abweichen — je Modell gebucht)

PILOT (R2 — promote keine Verdikte, S3-Muster der MI-Smokes):
  pilot_canon_tv_axioms     canon_tv: tv(a,a)=0, symmetrisch, perm > 0,
                            Bereich [0,1] auf 4-dim Wahrscheinlichkeitsfeld
  pilot_crystal_finder      Modul-Finder auf synthetischen rows (stabiler
                            TAIL, verspäteter TAIL, Oszillation → None)
  pilot_layer_bands_26      EDGE/Core decken 26 Schichten ohne Überlappung
  pilot_gate_clamp          capacity_gate = clamp(H/ln8, 0.25, 1.0)
                            (moe_layers.py:85-88)
  pilot_moe_guard           is_moe_model: moe-Stub True, dense-Stub False,
                            broken Stub False (AttributeError-Zweig)
  pilot_resolver_contract   resolve_token_ids_exact am Stub-Tokenizer:
                            unique → [id], Mehr-Token-Kandidat → [],
                            keine Treffer → []

PREREG-FREEZE 2026-10-26: die obigen Verdikte sind VOR den GPU-Runs
registriert; das KORREKTUR-LOG trägt nur nachweisliche Lauf-Bugs, keine
Band-Nachjustierung.

KORREKTUR-LOG:
  Lauf 1 (2026-10-26, GEBROCHEN, kein JSON — kein stiller Fallback):
    assemble_pct-Kanon liefert REINE LISTEN (pct[T][L][E], ent[T][L];
    R2-Modell-Entkopplung), der Smoke verwendete Tensor-Zugriffe
    (`ent[:, 11]` in ent_l11_series, `.tolist()` in chimera_tokenwise) —
    TypeError „list indices must be integers or slices" auf argon_1b,
    Lauf endete vor der Auswertung; keine Messung gebucht.
    Fix: Listen-Indexierung, keine Formel-Änderung.
    MI-5-Lauf 2 = voller Re-Lauf aller 4 Modelle.

Lauf:  python -m scratches.smoke_mech_interp_cross_models           (CPU-Pilot)
       python -m scratches.smoke_mech_interp_cross_models --gpu     (Hauptlauf)
       python -m scratches.smoke_mech_interp_cross_models --gpu --json \
           scratches/mi5_cross_models.json                          (JSON-Pfad)
Exit:  0 = alle Verdikte PASS — 1 = mindestens ein FAIL — 2 = Pilot/OFFEN/CPU.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Dict, List, Tuple

import torch

from machines.mech_interp import (
    MechInterpProbe,
    canon_tv,
    crystallization_start,
    is_moe_model,
    layer_bands,
    load_any_checkpoint,
)

GPU_DEFAULT_JSON = "scratches/mi5_cross_models.json"

MODEL_PATHS: Dict[str, str] = {
    "gemma3_1b_it": "HF:google/gemma-3-1b-it",
    "argon_1b": "./argon_1b",
    "argon_1b_specialized": "./argon_1b_specialized",
    "argon_1b_specialized_ml": "./argon_1b_specialized_ml",
}
ARGON_VARIANTS = ("argon_1b", "argon_1b_specialized", "argon_1b_specialized_ml")

# Canon-Anker der MI-Smokes (1:1):
GENERIC_PROMPT = ("Validity of a symbolic experiment batch, "
                  "we now test whether the generated sequence maintains "
                  "coherence under perturbation:")
PARIS_TEXT = "The capital of France is"

# Registrierte Chimären (User-Audit MI-4, verbatim) und t_switch-Praefixe:
CHIMERA_PROSE_CODE = ("Der warme Sommerwind wehte über den See, während "
                      "das Programm lief: def quicksort(arr): "
                      "return [x for x in arr]")
CHIMERA_PROSE_MATH = ("Oma erzählte im alten Garten ihre Geschichte, "
                      "doch plötzlich galt: Für alle n in N gilt "
                      "sum_{k=1}^n k = n(n+1)/2.")
PREFIX_PROSE_CODE = ("Der warme Sommerwind wehte über den See, "
                     "während das Programm lief: ")
PREFIX_PROSE_MATH = ("Oma erzählte im alten Garten ihre Geschichte, "
                     "doch plötzlich galt: ")

KO_SET_PROSE: Tuple[int, ...] = (3, 7)      # Prosa-Träger (MI-1/MI-3-Canon)
KO_SET_CODE: Tuple[int, ...] = (0, 1, 4)    # Code-Träger (MI-1/MI-3-Canon)
KO_PENALTY = -50.0           # MI-3-Canon (knockout_experts-ADDITIV-Arm)
STEER_AMPLITUDE = 50.0       # MI-3-Canon (steer_band-SET-Arm)

N_LAYERS = 26
N_EXPERTS = 8
_EDGE_LIST, _CORE_LIST = layer_bands(N_LAYERS)
BAND_CORE: Tuple[int, ...] = tuple(_CORE_LIST)          # 4..23 (Kanon)

# MI-1-Subset-Canon (je Text[0] der Domänen):
DOMAINS: Dict[str, str] = {
    "code":
        "def quicksort(items):\n    if len(items) <= 1:\n        return items\n"
        "    pivot = items[0]\n    return quicksort([x for x in items[1:] if x < pivot])"
        " + [pivot] + quicksort([x for x in items[1:] if x >= pivot])",
    "math_logic":
        "For all natural numbers n, the sum 1 + 2 + ... + n equals n(n+1)/2.",
    "prose":
        "Der warme Sommerwind trug in der Abenddämmerung den Duft von Kiefern "
        "über den stillen Bergsee.",
    "goedel":
        "Die Gödelnummer eines Formelausdrucks entsteht aus Primfaktoren: "
        "2^1 · 3^2 · 5^1 kodiert die Zeichenfolge (x).",
    "turing":
        "Eine Turingmaschine liest das Symbol, schreibt das neue Symbol, "
        "bewegt den Kopf nach links oder rechts und wechselt den Zustand.",
    "wittgenstein":
        "Ein Wort hat Bedeutung im Gebrauch, nicht im Wörterbuch — das "
        "Sprachspiel gibt der Zeichenfolge ihre Rolle.",
}
MI5_SUBSET: Tuple[str, ...] = tuple(DOMAINS.values())
V5_TEXT_PROSE = DOMAINS["prose"]   # Bergsee
V5_TEXT_CODE = DOMAINS["code"]     # quicksort

# Registrierte Verdikt-Schwellen (PREREG, siehe Docstring):
FORMAT_TOKENS = ("", "\n", "\n\n", " ", ".", ",", ":", ";", "!", "?")
TV_FLOOR = 0.01      # V1
V3_P1_FLOOR = 0.05   # V3b
V3_M_FLOOR = 0.10    # V3c
V4_MIN_DEFINED = 4   # V4 (>= 4/6)
V5_GAP_FLOOR = 0.01  # V5
V6_SHARE_FLOOR = 0.5 # V6


def common_prefix_len(a: List[int], b: List[int]) -> int:
    """Gemeiname Token-Praefixlaenge (t_switch-Marker, MI-4-Kanon)."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def resolve_token_ids_exact(tokenizer, target: str,
                            candidates: Tuple[str, ...]) -> List[int]:
    """MI-2-P4-Auflösung (Oberflächen-Exakt): len(encoded) == 1 UND
    decode([id]) == target EXAKT (kein lstrip); Mehr-Token-Kandidaten
    abgewiesen, bevor ein Vertreter entsteht."""
    ids: List[int] = []
    for cand in candidates:
        encoded = tokenizer.encode(cand, add_special_tokens=False)
        if len(encoded) != 1:
            continue
        decoded_first = tokenizer.decode([encoded[0]], skip_special_tokens=True)
        if decoded_first == target:
            ids.append(int(encoded[0]))
    return ids


def mean_value(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def layer_mean_tv(pct_s: torch.Tensor, pct_b: torch.Tensor) -> float:
    """V1-Distanz eines Texts: (1/26) Σ_li canon_tv(pct_S[li], pct_B[li])."""
    if pct_s.shape != pct_b.shape or pct_s.shape[-1] != N_EXPERTS:
        raise ValueError("layer_mean_tv: Form-Bruch — kein stiller Fallback.")
    return mean_value([canon_tv(pct_s[li], pct_b[li])
                       for li in range(pct_s.shape[0])])


def gate_value(ent: float) -> float:
    """Capacity-Gate canon (moe_layers.py:85-88)."""
    return min(max(ent / math.log(N_EXPERTS), 0.25), 1.0)


# -- CPU-Pilot (PILOT-Labels, promovieren keine Verdikte — R2) ---------------

def pilot_canon_tv_axioms() -> Tuple[str, bool, str]:
    a = torch.tensor([0.4, 0.3, 0.2, 0.1])
    b = torch.tensor([0.1, 0.2, 0.3, 0.4])
    tv_same = float(canon_tv(a, a))
    tv_sym = canon_tv(a, b) == canon_tv(b, a)
    tv_perm = canon_tv(a, torch.tensor([0.3, 0.4, 0.1, 0.2])) > 0.0
    tv_in_range = 0.0 <= canon_tv(a, b) <= 1.0
    ok = (tv_same < 1e-9) and bool(tv_sym) and bool(tv_perm) and bool(tv_in_range)
    return ("pilot_canon_tv_axioms", ok,
            f"tv_same={tv_same:.2e} sym={bool(tv_sym)} perm={bool(tv_perm)} "
            f"range={bool(tv_in_range)}")


def pilot_crystal_finder() -> Tuple[str, bool, str]:
    def rows_of(seq: List[int]) -> List[Dict[str, Any]]:
        return [{"kind": "layer", "layer": i, "top1_id": t}
                for i, t in enumerate(seq)]

    star_const = crystallization_start(rows_of([7, 7, 5, 9] + [42] * 20))
    star_late = crystallization_start(rows_of([7] * 10 + [9] * 24))
    star_osc = crystallization_start(rows_of([7, 9] * 13))
    ok = (star_const == 4 and star_late == 10 and star_osc is None)
    return ("pilot_crystal_finder", ok,
            f"const={star_const} late={star_late} osc={star_osc}")


def pilot_layer_bands_26() -> Tuple[str, bool, str]:
    edge, core = layer_bands(N_LAYERS)
    union = sorted(edge + core)
    ok = (union == list(range(N_LAYERS)) and len(union) == len(set(union)))
    return ("pilot_layer_bands_26", ok,
            f"edge={edge} core_first={core[0]} core_last={core[-1]}")


def pilot_gate_clamp() -> Tuple[str, bool, str]:
    ln8 = math.log(N_EXPERTS)
    ok = (abs(gate_value(0.15 * ln8) - 0.25) < 1e-9
          and abs(gate_value(0.5 * ln8) - 0.5) < 1e-9
          and abs(gate_value(9.0 * ln8) - 1.0) < 1e-9)
    return ("pilot_gate_clamp", ok,
            "floor 0.15·ln8, linear 0.5·ln8, cap 9.0·ln8 (moe_layers.py:85-88)")


def pilot_moe_guard() -> Tuple[str, bool, str]:
    class _LayerWithMoe:
        moe = object()

    class _MoeStub:
        model = type("_M", (), {"layers": [_LayerWithMoe()]})()

    class _DenseStub:
        model = type("_M", (), {"layers": [object()]})()

    class _BrokenStub:
        pass

    ok = (is_moe_model(_MoeStub) is True
          and is_moe_model(_DenseStub) is False
          and is_moe_model(_BrokenStub) is False)
    return ("pilot_moe_guard", ok, "moe->True dense->False broken->False")


def pilot_resolver_contract() -> Tuple[str, bool, str]:
    class _Tok:
        def encode(self, text: str, add_special_tokens: bool = False):
            return {" Paris": [9079], "Paris": [500, 9079]}[text]

        def decode(self, ids, skip_special_tokens: bool = True):
            return {9079: " Paris"}[ids[0]]

    got = resolve_token_ids_exact(_Tok(), " Paris", (" Paris", "Paris"))
    empty = resolve_token_ids_exact(_Tok(), " Span", (" Paris",))
    ok = (got == [9079] and empty == [])
    return ("pilot_resolver_contract", ok,
            f"unique={got} mehr_token_verworfen leere_treffer={empty}")


PILOTS: List = [
    pilot_canon_tv_axioms, pilot_crystal_finder, pilot_layer_bands_26,
    pilot_gate_clamp, pilot_moe_guard, pilot_resolver_contract,
]


# -- Batterie-Felder ---------------------------------------------------------

def bias_std_26(model) -> List[float]:
    """Datum je argon-Variante: std des learned expert_bias je Schicht."""
    return [float(layer.moe.gate.expert_bias.data.std().item())
            for layer in model.model.layers]


def sp3_fields(probe: MechInterpProbe) -> Dict[str, Any]:
    """SP3-Argmax am GENERIC_PROMPT (raw) + Paris-Recall-Felder."""
    p = probe.last_probs(GENERIC_PROMPT)
    top1 = int(p.argmax().item())
    m_gen = probe.digit_mass(GENERIC_PROMPT)
    p_paris = probe.last_probs(PARIS_TEXT)
    paris_top1 = int(p_paris.argmax().item())
    return {
        "generic_top1_id": top1,
        "generic_top1_str": probe.tokenizer.decode(
            [top1], skip_special_tokens=True),
        "p_digit1_generic": m_gen["digit_probs"]["1"],
        "M_digit_generic": m_gen["M_digit"],
        "M_unknown": m_gen["unknown"],
        "paris_top1_id": paris_top1,
        "paris_top1_str": probe.tokenizer.decode(
            [paris_top1], skip_special_tokens=True),
    }


def m_digit_series(probe: MechInterpProbe) -> Dict[str, Any]:
    """[Datums-Zusatz] M_digit-Serie unter MI-3-Canon-Bias-Bedingungen."""
    raw = float(probe.digit_mass(GENERIC_PROMPT)["M_digit"])
    series: Dict[str, Any] = {"raw": raw}
    with probe.knockout_experts(KO_SET_PROSE, layers=list(BAND_CORE),
                                penalty=KO_PENALTY):
        series["ko_prose_core"] = float(
            probe.digit_mass(GENERIC_PROMPT)["M_digit"])
    with probe.knockout_experts(KO_SET_CODE, layers=list(BAND_CORE),
                                penalty=KO_PENALTY):
        series["ko_code_core"] = float(
            probe.digit_mass(GENERIC_PROMPT)["M_digit"])
    with probe.steer_band(KO_SET_PROSE, None, steer=STEER_AMPLITUDE):
        series["steer_full_prose"] = float(
            probe.digit_mass(GENERIC_PROMPT)["M_digit"])
    return series


def chimera_tokenwise(probe: MechInterpProbe, chimera: str,
                      prefix: str) -> Dict[str, Any]:
    """[Datums-Zusatz] Chimären-Felder je argon-Variante (t_switch per IDs)."""
    ids_full = probe._encode(chimera)[0].tolist()
    ids_pre = probe._encode(prefix)[0].tolist()
    t_switch = common_prefix_len(ids_full, ids_pre)
    rt = probe.tokenwise_routing(chimera)
    T = rt["n_tokens"]
    if not (1 <= t_switch < T):
        raise RuntimeError(f"chimera_tokenwise: t_switch={t_switch}/{T}")
    return {"text": chimera,
            "prefix": prefix,
            "t_switch": t_switch,
            "n_tokens": T,
            "pct": rt["pct"],
            "ent": rt["ent"]}


def ent_l11_series(probe: MechInterpProbe) -> Dict[str, Any]:
    """V5-Feld: mean router entropy ℓ11 (tokenweise, Mi-4-Kanon) auf
    Bergsee (prose) und quicksort (code)."""
    out: Dict[str, Any] = {}
    for name, text in (("prose", V5_TEXT_PROSE), ("code", V5_TEXT_CODE)):
        rt = probe.tokenwise_routing(text)
        ent = rt["ent"]                       # Liste [T][26] (assemble_canon)
        H = [ent[t][11] for t in range(rt["n_tokens"])]
        h_mean = mean_value(H)
        out[name] = {"H_mean_l11": float(h_mean),
                     "H_min_l11": float(min(H)),
                     "H_max_l11": float(max(H)),
                     "gate_mean_l11": float(gate_value(h_mean))}
    return out


def subset_rows(probe: MechInterpProbe) -> Dict[str, Dict[str, Any]]:
    """Logit-Lens-Trajektorien je Subset-Text (modell-agnostisch)."""
    return {name: probe.layerwise_logits(text)
            for name, text in DOMAINS.items()}


# -- Läufe je Modell ---------------------------------------------------------

def gpu_determinism() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_name: str) -> MechInterpProbe:
    print(f"[MI-5] lade {model_name} …", flush=True)
    if model_name == "gemma3_1b_it":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it",
                                            local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-it", torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="eager",
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
    else:
        model, tok = load_any_checkpoint(MODEL_PATHS[model_name])
    probe = MechInterpProbe(model, tok)
    print(f"[MI-5] {model_name}: moe={is_moe_model(model)} "
          f"layers={probe.n_layers} experts={probe.n_experts}", flush=True)
    return probe


def run_model_battery(model_name: str) -> Dict[str, Any]:
    empty_cache()
    gpu_determinism()
    probe = load_model(model_name)
    moe = is_moe_model(probe.model)

    battery: Dict[str, Any] = {
        "model": model_name,
        "path": MODEL_PATHS[model_name],
        "is_moe": moe,
        "n_layers": probe.n_layers,
        "n_experts": probe.n_experts,
    }

    # Paris-Auflösung je Modell (Plan-Risiko 4):
    paris_ids = resolve_paris(probe)
    battery["paris_id"] = paris_ids[0]
    battery["paris_id_is_argon_constant"] = bool(paris_ids[0] == 9079)

    # SP3-Felder (modell-agnostisch):
    battery["sp3"] = sp3_fields(probe)
    battery["sp3"]["paris_recall"] = bool(
        battery["sp3"]["paris_top1_id"] == battery["paris_id"])

    # Kristallisations-Trajektorien (alle 4, Logit-Lens modell-agnostisch):
    rows = subset_rows(probe)
    battery["layerwise_logits"] = {
        name: {
            "star": crystallization_start(
                [r for r in info["rows"] if r["kind"] == "layer"]),
            "rows": [{"layer": r["layer"], "kind": r["kind"],
                      "top1_id": r["top1_id"], "top1_str": r["top1_str"],
                      "top1_prob": r["top1_prob"], "gap_tv": r["gap_tv"]}
                     for r in info["rows"]],
            "official_top1_id": info["official_top1_id"],
            "official_top1_str": info["official_top1_str"],
            "p_final_top1_id": info["p_final_top1_id"],
            "anchor_tv": info["anchor_tv"],
        }
        for name, info in rows.items()
    }

    if moe:
        # MI-1-Tomographie je Subset-Text:
        battery["routing_pct"] = {
            name: probe.layerwise_routing([text])["pct_26x8"].tolist()
            for name, text in DOMAINS.items()}
        battery["bias_std_26"] = bias_std_26(probe.model)

        # V5-Felder (tokenwise H(ℓ11) prose/code):
        battery["ent_l11"] = ent_l11_series(probe)

        # MI-2-arm (ℓ17-Paris-Share):
        sub = probe.sublayer_logits(
            PARIS_TEXT, {"id": battery["paris_id"], "str": " Paris"})
        row17 = sub["rows"][17]
        denom = row17["dp_attn"] + row17["dp_moe"]
        battery["l17_paris"] = {
            "dp_attn": row17["dp_attn"], "dp_moe": row17["dp_moe"],
            "p_mid": row17["p_mid"], "p_out": row17["p_out"],
            "share_attn_17": float(row17["dp_attn"] / denom) if denom != 0 else 0.0,
            "anchor_tv": sub["anchor_tv"],
            "telescope_broken": sub["telescope_broken"],
        }

        # [Datums-Zusatz] M_digit-Serie unter Bias-Bedingungen:
        battery["m_digit_series"] = m_digit_series(probe)

        # [Datums-Zusatz] Chimären je Variante (MI-4-Kanon):
        battery["chimera_tokenwise"] = {
            "prose_code": chimera_tokenwise(probe, CHIMERA_PROSE_CODE,
                                            PREFIX_PROSE_CODE),
            "prose_math": chimera_tokenwise(probe, CHIMERA_PROSE_MATH,
                                            PREFIX_PROSE_MATH),
        }
    else:
        # [Datums-Zusatz] dense-arm (MI-2-Analog auf gemma):
        try:
            sub = probe.sublayer_logits_dense(
                PARIS_TEXT, {"id": battery["paris_id"], "str": " Paris"})
            row17 = sub["rows"][17]
            denom = row17["dp_attn"] + row17["dp_mlp"]
            battery["dense_paris"] = {
                "status": "gemessen",
                "dp_attn_17": row17["dp_attn"], "dp_mlp_17": row17["dp_mlp"],
                "p_mid_17": row17["p_mid"], "p_out_17": row17["p_out"],
                "share_attn_17": float(row17["dp_attn"] / denom) if denom != 0 else 0.0,
                "telescope_target": sub["telescope_target"],
                "endpoints_target": sub["endpoints_target"],
                "telescope_broken": sub["telescope_broken"],
                "anchor_tv": sub["anchor_tv"],
                "rows_dp": [{"layer": r["layer"], "dp_attn": r["dp_attn"],
                             "dp_mlp": r["dp_mlp"],
                             "top1_out_str": r["top1_out_str"]}
                            for r in sub["rows"]],
            }
        except (RuntimeError, ValueError, TypeError, AttributeError) as err:
            battery["dense_paris"] = {"status": "OFFEN/strukturell",
                                      "error": str(err)}

    return battery


def resolve_paris(probe: MechInterpProbe) -> List[int]:
    ids = resolve_token_ids_exact(probe.tokenizer, " Paris", (" Paris",))
    if len(ids) != 1:
        raise RuntimeError(
            "MI-5: Paris-Auflösung nicht unique "
            f"({ids}) — kein stiller Fallback.")
    return ids


# -- Strikte Auswertung nach PREREG (kein stiller Fallback) ------------------

def classify_battery(batteries: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    verdicts: List[Dict[str, Any]] = []
    argon = {name: b for name, b in batteries.items()
             if name in ARGON_VARIANTS}
    base = argon["argon_1b"]

    # V1 — lineage routing shift:
    try:
        detail = {}
        for sibling in ("argon_1b_specialized", "argon_1b_specialized_ml"):
            d_s = [layer_mean_tv(
                torch.tensor(argon[sibling]["routing_pct"][dom]),
                torch.tensor(base["routing_pct"][dom]))
                for dom in DOMAINS]
            detail[sibling] = {"per_text": d_s, "mean": mean_value(d_s)}
        ok = all(d["mean"] > TV_FLOOR for d in detail.values())
        verdicts.append({"id": "mi5_lineage_routing_shift",
                         "status": "PASS" if ok else "FAIL",
                         "detail": detail, "floor": TV_FLOOR})
    except (KeyError, TypeError, ValueError) as err:
        verdicts.append({"id": "mi5_lineage_routing_shift", "status": "OFFEN",
                         "error": str(err)})

    # V2 — paris_recall_compare (alle 4):
    try:
        rows = {name: {"top1": b["sp3"]["paris_top1_id"],
                       "resolved": b["paris_id"],
                       "ok": b["sp3"]["paris_recall"]}
                for name, b in batteries.items()}
        ok = all(r["ok"] for r in rows.values())
        verdicts.append({"id": "mi5_paris_recall_compare",
                         "status": "PASS" if ok else "FAIL", "detail": rows})
    except (KeyError, TypeError) as err:
        verdicts.append({"id": "mi5_paris_recall_compare", "status": "OFFEN",
                         "error": str(err)})

    # V3 — dense_parent_argmax (gemma):
    try:
        g = batteries["gemma3_1b_it"]["sp3"]
        top1_str = g["generic_top1_str"]
        cond_a = top1_str in FORMAT_TOKENS
        cond_b = g["p_digit1_generic"] >= V3_P1_FLOOR
        cond_c = g["M_digit_generic"] >= V3_M_FLOOR
        ok = cond_a and cond_b and cond_c
        verdicts.append({
            "id": "mi5_dense_parent_argmax",
            "status": "PASS" if ok else "FAIL",
            "detail": {"top1_str": top1_str, "top1_id": g["generic_top1_id"],
                       "p_digit1": g["p_digit1_generic"],
                       "M_digit": g["M_digit_generic"],
                       "cond_a_format_token": cond_a,
                       "cond_b": cond_b, "cond_c": cond_c},
            "floors": {"p1": V3_P1_FLOOR, "M": V3_M_FLOOR},
        })
    except (KeyError, TypeError) as err:
        verdicts.append({"id": "mi5_dense_parent_argmax", "status": "OFFEN",
                         "error": str(err)})

    # V4 — crystallization_compare (alle 4):
    try:
        detail = {}
        for name, b in batteries.items():
            stars = {dom: info["star"]
                     for dom, info in b["layerwise_logits"].items()}
            defined = sum(1 for s in stars.values() if s is not None)
            detail[name] = {
                "star_by_domain": stars,
                "defined": defined,
                "of": len(b["layerwise_logits"]),
                "anchors": {dom: {"p_final_top1_id": info["p_final_top1_id"],
                                  "official_top1_id": info["official_top1_id"],
                                  "anchor_tv": info["anchor_tv"]}
                            for dom, info in b["layerwise_logits"].items()},
            }
        ok = all(d["defined"] >= V4_MIN_DEFINED for d in detail.values())
        verdicts.append({"id": "mi5_crystallization_compare",
                         "status": "PASS" if ok else "FAIL",
                         "detail": detail, "floor": V4_MIN_DEFINED})
    except (KeyError, TypeError) as err:
        verdicts.append({"id": "mi5_crystallization_compare", "status": "OFFEN",
                         "error": str(err)})

    # V5 — gate_entropy_gradient (alle 3 argon):
    try:
        detail = {}
        for name, b in argon.items():
            prose_h = b["ent_l11"]["prose"]["H_mean_l11"]
            code_h = b["ent_l11"]["code"]["H_mean_l11"]
            detail[name] = {"H_prose_l11": prose_h, "H_code_l11": code_h,
                            "gap_code_minus_prose": code_h - prose_h,
                            "gate_mean_l11_prose":
                                b["ent_l11"]["prose"]["gate_mean_l11"],
                            "gate_mean_l11_code":
                                b["ent_l11"]["code"]["gate_mean_l11"]}
        ok = all(d["gap_code_minus_prose"] > V5_GAP_FLOOR
                 for d in detail.values())
        verdicts.append({"id": "mi5_gate_entropy_gradient",
                         "status": "PASS" if ok else "FAIL",
                         "detail": detail, "floor": V5_GAP_FLOOR})
    except (KeyError, TypeError) as err:
        verdicts.append({"id": "mi5_gate_entropy_gradient", "status": "OFFEN",
                         "error": str(err)})

    # V6 — l17_attn_share_compare (alle 3 argon):
    try:
        detail = {name: {"share_attn_17": b["l17_paris"]["share_attn_17"],
                         "dp_attn": b["l17_paris"]["dp_attn"],
                         "dp_moe": b["l17_paris"]["dp_moe"]}
                  for name, b in argon.items()}
        ok = all(d["share_attn_17"] >= V6_SHARE_FLOOR
                 for d in detail.values())
        verdicts.append({"id": "mi5_l17_attn_share_compare",
                         "status": "PASS" if ok else "FAIL",
                         "detail": detail, "floor": V6_SHARE_FLOOR})
    except (KeyError, TypeError) as err:
        verdicts.append({"id": "mi5_l17_attn_share_compare", "status": "OFFEN",
                         "error": str(err)})

    return verdicts


def summarize(verdicts: List[Dict[str, Any]]) -> str:
    n_pass = sum(1 for v in verdicts if v["status"] == "PASS")
    n_fail = sum(1 for v in verdicts if v["status"] == "FAIL")
    n_offen = sum(1 for v in verdicts if v["status"] == "OFFEN")
    return f"{n_pass} PASS | {n_fail} FAIL | {n_offen} OFFEN"


def classify(verdicts: List[Dict[str, Any]], pilot_fails: int, gpu: bool) -> int:
    if any(v["status"] == "FAIL" for v in verdicts):
        return 1
    if pilot_fails:
        return 2
    if gpu and any(v["status"] == "OFFEN" for v in verdicts):
        return 2
    if gpu and verdicts:
        return 0
    return 2


def main() -> int:
    gpu = "--gpu" in sys.argv
    json_path = GPU_DEFAULT_JSON
    if "--json" in sys.argv:
        json_path = sys.argv[sys.argv.index("--json") + 1]

    pilot_rows = [{"id": rid, "status": "PILOT" if ok else "PILOT-FAIL",
                   "note": note}
                  for rid, ok, note in (p() for p in PILOTS)]
    pilot_fails = sum(1 for r in pilot_rows if r["status"] == "PILOT-FAIL")

    batteries: Dict[str, Dict[str, Any]] = {}
    verdicts: List[Dict[str, Any]] = []

    if gpu:
        for model_name in MODEL_PATHS:
            batteries[model_name] = run_model_battery(model_name)
            empty_cache()
        verdicts = classify_battery(batteries)
        payload = {"pilots": pilot_rows, "batteries": batteries,
                   "verdicts": verdicts}
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)

    for row in pilot_rows:
        print(f"{row['status']}: {row['id']} — {row['note']}", flush=True)
    if gpu:
        for v in verdicts:
            print(f"{v['status']}: {v['id']}", flush=True)
        print(summarize(verdicts), flush=True)
        print(f"JSON: {json_path}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    pil = f"{len(pilot_rows) - pilot_fails}/{len(pilot_rows)} PILOT"
    if gpu:
        print(f"PILOT {pil} | VERDIKTE {summarize(verdicts)}", flush=True)
    else:
        print(f"PILOT {pil} | BATTERIE OFFEN (CPU-Lane)", flush=True)

    return classify(verdicts, pilot_fails, gpu)


if __name__ == "__main__":
    raise SystemExit(main())