"""
smoke_regeln_woerter.py — SP10: Die Regeln und die Wörter (Composite × Lexikon).

Frage: macht das Lexikon das Programm, oder der Zufall? Der Composite-Zyklus
(Wittgenstein generiert Regel → Turing führt Programm → Gödel predet) hat
seine Bruchstelle (SP7: erzwungene Tautologie an Schritt 9, seed 7,
first_detect 11); das Lexikon hat seine Bewegung (SP6/SP9: Gebrauch dreht
Embeddings). SP10 koppelt beide, ohne eine Zeile Maschinen-Code zu ändern —
die Verschachtelung greift an `CompositeMachine.wittgenstein` (composite.py
:45-51); der Zwangs-Haken hängt am composite-step-Zähler (nicht am
wm._step_index), damit die Lexikon-Runden den Zwang nicht schlucken.

DREI HERMENEUTISCHE THESEN (als falsifizierbare Vorhersagen geprüft, 2026-10-25):
  T1 „Das Programm ist nicht aus den Wörtern gemacht" — der Composite-Verlauf
     (Spur: regel, verdikt, transitionen, gaps, Basis) ist invariant gegen
     jeden Lexikon-Zustand. Code-Fundament (verifiziert): `_generate_ast`
     (wittgenstein_machine.py:292-303) liest nur RNG aus fixen Pools,
     `_generate_name` (:107-117) nur AST-Struktur, `evaluate` (:119-137) nur
     den AST — die Regeln sind wort-blind.
  T2 „Das Programm ist aus dem Zufall gemacht" — wird das Lexikon-Spiel auf
     DEMSELBEN RNG-Strom gespielt (Lexikon-Runden zwischen Composite-
     Schritten, je Zugfolge create_rule → uniform ×10 → normal ×10), ver-
     schiebt der Zugverbrauch den Verlauf; die Bruchstelle (LOOP an 9, Ex-
     pansion, Basis > 3) bleibt trotzdem stichfest, weil sie am Zwang hängt,
     nicht am Zufall. Kripke/Quus-Kontrast auch unter Verschachtelung.
  T3 „Die Sprachdynamik ist inhalt-invariant" — gegeben des Lern-Schedules
     hängt die Lexikon-Dynamik an geschlossenen Formen:
       Halbwinkel β_j = atan(0.4^j · tan β0), β0 = θ0/2, θ0 = Winkel der
       Gematria-Vektoren (α = 0.3 → Shrink-Faktor 1−2α = 0.4 pro Episode);
       Paar-Resonanz c_j = cos(2β_j);
       Zuschauer-Resonanz (v = tisch lernt nie mit regel↔rule):
         R_j = cos β_j · (M·v) + sin β_j · (N·v),
         M = n(regel0+rule0), N = n(regel0−rule0) (Identität:
         regel0 = s/2 + d/2, |s| = 2cos β0, |d| = 2sin β0);
       Grenzwert R_∞ = M·v — der SP9-E2-Drift (+1.31e-4 über 45 Runden) ist
       das log-Asymptoten-Glied, kein Rauschen und kein Fixpunkt-Pin.

VERDICT-MAP (Prereg, fixiert VOR dem ersten Lauf; FAIL ist gleichwertiger
Befund, Errata nach der P4-Einstufung; Bands werden nach der Messung nie
geweitet):
  sp10_l0_anchor      — L0 (seed 7, Zwang {9}, 16 Schritte) == committetes
                        hermeneutik_experiment.json FELDWEISE: verdicts,
                        basis, final_basis, expansions, gaps, first_detect
                        == 11; Quus-Arm == committet. Falsifikator: jedes
                        abweichende Feld.
  sp10_det            — Doppel-Läufe (L0-Kripke/-Quus und L2-Kripke/Quus):
                        trace + gaps + Roh-Lexikonwerte bitidentisch.
  sp10_t1_spur        — L1: Composite-Spur bitidentisch zur L0-Spur bei
                        (a) Seitenband-Lexikon auf eigener Maschine und
                        (b) Lexikon auf derselben Maschine NACH dem Verlauf
                        (Spur-Snapshot vor/nach ==). Falsifikator: jede
                        Spur-Differenz.
  sp10_t1_lexikon     — L1a (seed 9, SP6-Spiel): after_paired ==
                        [0.999790, 0.999982, 1.000000], learned == 15;
                        L1b: je Paar End-Resonanz >= 0.99. Falsifikator:
                        Anker-Differenz / Konvergenz-Ausfall.
  sp10_t2_divergenz   — L2-Kripke: erste trace-Differenz zur L0-Natur-Lage
                        bei Schritt <= 3. Falsifikator: > 3 oder keine.
  sp10_t2_kripke      — L2-Kripke: verdicts[8] == "LOOP", "step9:LOOP" in
                        expansions, final_basis > 3, gaps endlich; L2-Quus
                        (derselbe Vorschub): verdicts[8] != "LOOP" und kein
                        "step9:LOOP". Die Bruchstelle ist wort-frei.
  sp10_t2_lexikon_lebt— L2-Lexikon (48 Runden, 3 Paare): je Paar Final-
                        Resonanz >= 0.99 und gelernte Episoden je Paar >= 5.
  sp10_t3_paar        — L2: geschlossene Form c_j je Spielrunde vs. Roh-
                        werte (48 Punkte über 3 Paare): max Differenz
                        <= 1e-9. Falsifikator: Überschreitung.
  sp10_t3_obs         — L2: tisch-Spur (49 Punkte, k = gelernte regel↔rule-
                        Episoden) vs. R_j: max Differenz <= 1e-9; kaffee/
                        baum-Zeilen bitkonstant über alle 48 Runden.
                        Falsifikator: Überschreitung.
  sp10_t3_rueck       — L3a SP9-Rückdiction (seed 9, Leg1 15 gated + Leg2
                        45 Einzel-Runden): rohe 46-Punkte-Spur vs. Formel
                        <= 1e-9; (regel,rule)/(spiel,game)/(gebrauch,use)-
                        Roh-Resonanzen über Leg2-Runden <= 1e-9; gerundete
                        Leg2-Spur == committet (Bitgleichheit nach round 6);
                        tisch15/tisch60/Drift/unrelated-Zeilen/delta-Zahl
                        == committet. Falsifikator: Überschreitung oder
                        Anker-Differenz.
  sp10_t3_asymptot    — |letzte tisch-Spur-Wert − M·v| <= 1e-4 in L2 UND
                        SP9-Replay (Erwartung aus der SP9-Spur: ~1e-10).
                        Falsifikator: Überschreitung.

P4-REGISTER (2026-10-25, nach dem Erstrun ergänzt; Bands UNVERÄNDERT, die
Ursprungs-Verdicte bleiben FAIL im Artefakt): drei Implementations-Errata.
(1) sp10_t1_spur — die Seitenband-Arme des Erstruns liefen ohne Forced-Haken
    (Zwang {9} fehlte); p4-Lesung fährt am SP7-Schema. (2) sp10_t1_lexikon —
    after_paired wurde gegen das Per-Runden-Protokoll geprüft statt gegen das
    Per-Paar-Spielende; die letzten drei Protokoll-Einträge des Erstruns
    waren bereits bitidentisch zum Anker. (3) sp10_t3_rueck — delta_count
    lief über eine Rohspur mit ungerundetem Punkt 0 (14) statt über die
    committete SP9-Konstruktion (Punkt 0 = unrelated15[0], gerundet; 15). Der
    15. Unterschied der committeten Spur ist das Rundungs-Anschluss-Residuum
    am Spurstoss, kein Sozialisierungs-Ereignis — dokumentiert, nicht getilgt;
    SP9-Verdicte nutzen dieses Feld nicht als Band. Korrigierte Lesungen
    kommen als _p4-Verdicte mit denselben Bands; die drei Thesen bleiben
    unverändert.

Run:  python3 -m scratches.smoke_regeln_woerter     # CPU, keine Abhängigkeit
Log:  ... | tee scratches/regeln_woerter_experiment.log
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from machines.composite import CompositeMachine  # noqa: E402
from machines.machine_base import MachineConfig  # noqa: E402
from machines.wittgenstein_machine import (  # noqa: E402
    GeneratedSymbolicRule, WittgensteinMachine)

TAUTAST = ("identity", "x")
QUUSAST = ("add", ("sin", "x"), 0.0)
TAUT_POOL6 = {"axiom_pool": ["ALPHA", "BETA", "GAMMA", "DELTA",
                             "EPSILON", "ZETA"]}
SEED_KP = 7          # SP7-Anker-Seed
SEED_SB = 9          # SP6/SP9-Lexikon-Anker-Seed
STEPS = 16           # Composite-Schritte (SP7-Länge)
FORCE_STEPS = {9}    # erzwungene Bruchstelle (SP7-Kripke)
STEP9 = 9
ALPHA = 0.3
SHRINK = 1.0 - 2.0 * ALPHA   # Halbwinkel-Faktor je Episode = 0.4
LEX_RPS = 3          # Lexikon-Runden je Composite-Schritt (48 gesamt)
GAME_PAIRS = [("regel", "rule"), ("spiel", "game"), ("gebrauch", "use")]
UNRELATED_GAME = [("regel", "tisch"), ("bedeutung", "kaffee"), ("wort", "baum")]
SP6_AFTER_PAIRED = [0.999790, 0.999982, 1.000000]
ANALYTIC_TOL = 1e-9
ASYM_TOL = 1e-4
MIN_PAIR_RES = 0.99
MIN_EPISODES = 5

# P4-Einstufungs-Register (2026-10-25, nach dem Erstrun ergänzt): Ursprungs-
# FAIL-Verdicte bleiben in den Artefakten unangetastet; korrigierte Lesungen
# kommen als NEUE Verdicte mit UNVERÄNDERTEN Bands hinzu (_p4-Suffix).
# Vor dem Erstrun war ERRATA leer (P2-Freeze); kein Band wurde geweitet.
ERRATA: Dict[str, str] = {
    "sp10_t1_spur":
        "Erstrun: die Seitenband-Arme liefen ohne Forced-Haken (Zwang {9} "
        "fehlte) — Implementations-Erratum, kein Wort-Befund. p4-Lesung fährt "
        "die Arme am SP7-Schema (run_forced).",
    "sp10_t1_lexikon":
        "Erstrun: after_paired wurde gegen das Per-Runden-Protokoll (15 "
        "Einträge) geprüft statt gegen das Per-Paar-Spielende (SP6-Anker mit "
        "3 Einträgen). Die letzten 3 Protokoll-Einträge waren schon "
        "bitidentisch zum Anker. p4-Lesung prüft per Paar (SP6-Schreibweise).",
    "sp10_t3_rueck":
        "Erstrun: delta_count wurde über eine Rohspur mit UNGERUNDEM Punkt 0 "
        "gezählt (14) statt über die committete SP9-Konstruktion (Punkt 0 = "
        "unrelated15[0], gerundet). Der 15. Unterschied ist das Rundungs-"
        "Anschluss-Residuum am Spurstoss (|Roh − gerundet| ~ 1e-7), kein "
        "Sozialisierungs-Ereignis. SP9-Verdicte nutzen dieses Feld nicht als "
        "Band; dokumentiert, nicht getilgt. p4-Lesung zählt SP9-exakt.",
}

FINDINGS: List[Tuple[str, str]] = []


def mark(name: str, ok: bool) -> bool:
    """Verdict zeichnen (Prereg-Namen; Reihenfolge = Findings-Reihenfolge)."""
    FINDINGS.append((name, "PASS" if ok else "FAIL"))
    return ok


def raw_res(tokenizer, w1: str, w2: str) -> float:
    return float(tokenizer.resonance(tokenizer.embed_word(w1),
                                     tokenizer.embed_word(w2)))


def dotv(u: List[float], v: List[float]) -> float:
    return float(sum(x * y for x, y in zip(u, v)))


def pair_geometry(tokenizer, w1: str, w2: str) -> Dict[str, Any]:
    """Gematria-Geometrie des ungelehrten Paars: c0, β0 = θ0/2, Mittelinie
    M = n(a+b), Achse N = n(a−b) (Seite von w1). Identität w1_0 = s/2 + d/2
    beweist die Zuschauer-Formel exakt."""
    a = tokenizer.embed_word(w1)
    b = tokenizer.embed_word(w2)
    c0 = float(tokenizer.resonance(a, b))
    sumv = [x + y for x, y in zip(a, b)]
    diff = [x - y for x, y in zip(a, b)]
    norm_s = math.sqrt(float(tokenizer.resonance(sumv, sumv)))
    norm_d = math.sqrt(float(tokenizer.resonance(diff, diff)))
    if norm_s < 1e-9 or norm_d < 1e-9:
        raise ValueError(f"pair_geometry: entartete Geometrie {w1}/{w2} "
                         f"(norms {norm_s:.3e}/{norm_d:.3e})")
    theta0 = math.acos(max(-1.0, min(1.0, c0)))
    return {"w1": w1, "w2": w2, "c0": c0, "beta0": theta0 / 2.0,
            "norm_sum": norm_s, "norm_diff": norm_d,
            "mid": [x / norm_s for x in sumv],
            "axis": [x / norm_d for x in diff]}


def closed_pair(geom: Dict[str, Any], episodes: int) -> float:
    """Paar-Resonanz nach `episodes` gelernten Episoden: c_j = cos(2β_j)."""
    beta = math.atan((SHRINK ** episodes) * math.tan(geom["beta0"]))
    return math.cos(2.0 * beta)


def closed_viewer(geom: Dict[str, Any], viewer: List[float],
                  episodes: int) -> float:
    """Zuschauer-Resonanz R_j = cos β_j (M·v) + sin β_j (N·v)."""
    beta = math.atan((SHRINK ** episodes) * math.tan(geom["beta0"]))
    return math.cos(beta) * dotv(geom["mid"], viewer) + \
        math.sin(beta) * dotv(geom["axis"], viewer)


def first_trace_diff(trace_a: List[str], trace_b: List[str]) -> Tuple[
        Optional[int], str]:
    """1-basiert der erste Differenzschritt; (None, …) wenn identisch."""
    for idx, (x, y) in enumerate(zip(trace_a, trace_b)):
        if x != y:
            return idx + 1, f"Schritt {idx + 1}: Spur-Differenz"
    if len(trace_a) != len(trace_b):
        return min(len(trace_a), len(trace_b)) + 1, "Laengen-Differenz"
    return None, "identisch"


def round6(values: List[float]) -> List[float]:
    return [round(v, 6) for v in values]


# ---------------------------------------------------------------------------
# Composite-Lauf-Harness (Zwangs-Haken am composite-step-Zähler)
# ---------------------------------------------------------------------------
def make_composite(force_ast: Tuple[Any, ...]) -> Tuple[CompositeMachine,
                                                        Dict[str, Any]]:
    """Composite (seed 7, SP7-Config) mit force-Haken: am Zwangsschritt wird
    die ERSTE Regel des Batch durch `force_ast` ersetzt — über einen
    pending-Flag, den die Lauf-Routine am composite-step-Zähler setzt, so
    dass Lexikon-Runden außerhalb der Composite-Schritte den Zwang nie fassen."""
    machine = CompositeMachine(MachineConfig(seed=SEED_KP, mode="in_loop",
                                             params=dict(TAUT_POOL6)))
    harness: Dict[str, Any] = {"pending": False}
    wm = machine.wittgenstein
    original = wm.create_rule

    def patched() -> GeneratedSymbolicRule:
        if harness["pending"]:
            harness["pending"] = False
            rule = GeneratedSymbolicRule(force_ast)
            wm.generated_rules.append(rule)
            return rule
        return original()

    wm.create_rule = patched
    return machine, harness


def run_composite_sp7(force_ast: Tuple[Any, ...],
                      force_steps: set) -> Dict[str, Any]:
    """Natur-Lage ohne Lexikon (L0/Kontroll-Form; SP7-Länge + volle Spur)."""
    machine, harness = make_composite(force_ast)
    verdicts: List[str] = []
    basis: List[int] = []
    gaps: List[float] = []
    paradigms: List[bool] = []
    for step in range(1, STEPS + 1):
        if step in force_steps:
            harness["pending"] = True
        result = machine.step(None)
        verdicts.append(str(result.invariants["verdict"]))
        basis.append(len(machine.turing.basis))
        gaps.append(float(result.gap))
        paradigms.append(bool(result.invariants.get(
            "paradigm_signal_composite", {}).get("detected")))
    first_detect = next((idx for idx, flag in enumerate(paradigms, start=1)
                         if flag), None)
    return {"verdicts": verdicts, "basis": basis, "final_basis": basis[-1],
            "expansions": list(machine.expansions), "gaps": gaps,
            "gaps_finite": all(g == g and abs(g) != float("inf") for g in gaps),
            "first_detect": first_detect,
            "trace": list(machine.trace)}


def run_forced(machine: CompositeMachine, harness: Dict[str, Any]
               ) -> Tuple[List[str], List[float]]:
    """SP7-Verlauf (Zwangs-Haken an FORCE_STEPS — identische Mechanik wie
    run_composite_sp7) für die Seitenband-Arme: gibt Spur + Gap-Rohwerte."""
    gaps: List[float] = []
    for step in range(1, STEPS + 1):
        if step in FORCE_STEPS:
            harness["pending"] = True
        result = machine.step(None)
        gaps.append(float(result.gap))
    return list(machine.trace), gaps


# ---------------------------------------------------------------------------
# L0: SP7-Anker-Feldvergleich
# ---------------------------------------------------------------------------
def lane_anchor() -> Dict[str, Any]:
    committed = json.loads((REPO_ROOT / "scratches" /
                            "hermeneutik_experiment.json").read_text(
                                encoding="utf-8"))
    k0, k1 = (run_composite_sp7(TAUTAST, FORCE_STEPS) for _ in range(2))
    q0, q1 = (run_composite_sp7(QUUSAST, FORCE_STEPS) for _ in range(2))
    ck, cq = committed["kripke"], committed["contrast_quus"]
    fields = {
        "verdicts": k0["verdicts"] == ck["verdicts"],
        "basis": k0["basis"] == ck["basis"],
        "final_basis": k0["final_basis"] == ck["final_basis"],
        "expansions": k0["expansions"] == ck["expansions"],
        "gaps": k0["gaps"] == ck["gaps"],
        "first_detect": k0["first_detect"] == committed["first_detect"],
        "quus_verdicts": q0["verdicts"] == cq["verdicts"],
        "quus_final_basis": q0["final_basis"] == cq["final_basis"],
        "quus_expansions": q0["expansions"] == cq["expansions"],
    }
    det0 = (k0["trace"] == k1["trace"] and k0["gaps"] == k1["gaps"]
            and q0["trace"] == q1["trace"] and q0["gaps"] == q1["gaps"])
    mark("sp10_l0_anchor", all(fields.values()))
    return {"fields": fields, "first_detect": k0["first_detect"],
            "final_basis": k0["final_basis"],
            "verdicts": k0["verdicts"], "expansions": k0["expansions"],
            "quus_final_basis": q0["final_basis"],
            "det0": det0,
            "trace": k0["trace"], "gaps": k0["gaps"]}


# ---------------------------------------------------------------------------
# L1: Seitenband (These 1)
# ---------------------------------------------------------------------------
def lane_seitenband(anchor: Dict[str, Any]) -> Dict[str, Any]:
    """(a) Seitenband-Lexikon auf eigener Maschine (seed 9, SP6-Spiel) neben
    einem frischen Composite (seed 7, SP7-Zwang): Composite-Spur == L0-Spur,
    Lexikon == SP6-Anker (per Paar). (b) Lexikon auf DERSELBEN Composite-
    Maschine NACH dem 16-Schritt-Verlauf: Spur + Gap-Historie (Snapshot
    vor/nach) bitidentisch, Lexikon konvergiert."""
    side = WittgensteinMachine(MachineConfig(seed=SEED_SB, mode="in_loop"))
    protocol_side = side.lexicon_game(GAME_PAIRS, rounds=15, alpha=ALPHA)
    # SP6-Schreibweise: End-Resonanz je PAAR (nicht das Per-Runden-Protokoll).
    after_paired = [round(raw_res(side.tokenizer, *p), 6)
                    for p in GAME_PAIRS]
    protocol_res = [float(p["resonance_after"]) for p in protocol_side]
    learned_side = sum(1 for p in protocol_side if p["learned"])
    unrelated_side = [round(raw_res(side.tokenizer, *p), 6)
                      for p in UNRELATED_GAME]

    comp_a, harness_a = make_composite(TAUTAST)
    trace_a, gaps_a = run_forced(comp_a, harness_a)

    comp_b, harness_b = make_composite(TAUTAST)
    trace_b_before, gaps_b_before = run_forced(comp_b, harness_b)
    snapshot_before = json.dumps({"trace": trace_b_before,
                                  "gaps": list(comp_b._gap_history)})
    protocol_b = comp_b.wittgenstein.lexicon_game(GAME_PAIRS, rounds=15,
                                                  alpha=ALPHA)
    snapshot_after = json.dumps({"trace": list(comp_b.trace),
                                 "gaps": list(comp_b._gap_history)})
    final_b = [raw_res(comp_b.wittgenstein.tokenizer, *p) for p in GAME_PAIRS]

    # Erstrun-Konstruktionen (P3, s. ERRATA): Seitenband-Arm OHNE Zwang +
    # after_paired als Per-Runden-Protokoll — reproduzieren die FAIL-Lesung.
    comp_c, _ = make_composite(TAUTAST)
    for _ in range(STEPS):
        float(comp_c.step(None).gap)
    spur_erstrun = list(comp_c.trace) == anchor["trace"]
    lexikon_erstrun = protocol_res == SP6_AFTER_PAIRED

    anchor_trace, anchor_gaps = anchor["trace"], anchor["gaps"]
    spur_ok = (trace_a == anchor_trace and gaps_a == anchor_gaps
               and trace_b_before == trace_a and gaps_b_before == gaps_a
               and snapshot_before == snapshot_after)
    lexikon_ok = (after_paired == SP6_AFTER_PAIRED
                  and learned_side == 15
                  and all(r >= MIN_PAIR_RES for r in final_b))
    mark("sp10_t1_spur", spur_erstrun)       # Erstrun-Konstruktion (FAIL)
    mark("sp10_t1_lexikon", lexikon_erstrun)  # Erstrun-Konstruktion (FAIL)
    mark("sp10_t1_spur_p4", spur_ok)
    mark("sp10_t1_lexikon_p4", lexikon_ok)
    return {"after_paired": after_paired, "protocol_res": protocol_res,
            "learned_side": learned_side, "unrelated_side": unrelated_side,
            "trace_invariant": spur_ok, "trace_len": len(trace_a),
            "unforced_trace_matches": spur_erstrun,
            "final_b": round6(final_b),
            "learned_b": sum(1 for p in protocol_b if p["learned"])}


# ---------------------------------------------------------------------------
# L2: gezählter Zufalls-Vorschub (These 2)
# ---------------------------------------------------------------------------
def lane_counted(force_ast: Tuple[Any, ...],
                 anchor_trace: List[str]) -> Dict[str, Any]:
    """Verschachtelung: je Composite-Schritt 3 Lexikon-Runden (round-robin
    über GAME_PAIRS) auf DEMSELBEN Wittgenstein-RNG-Strom — Composite-Schritt
    zuerst, Lexikon-Runden danach (der erste Composite-Schritt zieht noch aus
    der unverschobenen Position, die Divergenz erscheint also naturwüchsig
    ab dem Schritt nach dem ersten Vorschub)."""
    machine, harness = make_composite(force_ast)
    wm = machine.wittgenstein
    tok = wm.tokenizer
    state_fresh = json.dumps(wm._rng.bit_generator.state, sort_keys=True)
    geoms = {pair: pair_geometry(tok, *pair) for pair in GAME_PAIRS}
    viewer = tok.embed_word("tisch")
    pre_rows = [raw_res(tok, *pair) for pair in UNRELATED_GAME]
    episodes = [0, 0, 0]
    lex_rows: List[Dict[str, Any]] = []
    rows_series: List[List[float]] = []   # je Spielrunde die 3 Zeilen (roh)
    verdicts: List[str] = []
    basis: List[int] = []
    gaps: List[float] = []
    lex_i = 0
    for step in range(1, STEPS + 1):
        if step in FORCE_STEPS:
            harness["pending"] = True
        result = machine.step(None)
        verdicts.append(str(result.invariants["verdict"]))
        basis.append(len(machine.turing.basis))
        gaps.append(float(result.gap))
        for _ in range(LEX_RPS):
            pair = GAME_PAIRS[lex_i % len(GAME_PAIRS)]
            line = wm.lexicon_game([pair], rounds=1, alpha=ALPHA)[0]
            if line["learned"]:
                episodes[lex_i % len(GAME_PAIRS)] += 1
            k = episodes[lex_i % len(GAME_PAIRS)]
            rows_now = [raw_res(tok, *p) for p in UNRELATED_GAME]
            lex_rows.append({
                "round": lex_i, "pair": list(pair),
                "learned": bool(line["learned"]), "episodes": k,
                "res": raw_res(tok, *pair),
                "closed": closed_pair(geoms[pair], k),
                "success": line["success_rate"], "k0": episodes[0]})
            rows_series.append(rows_now)
            lex_i += 1
    rows_post = [raw_res(tok, *p) for p in UNRELATED_GAME]
    pair_final = [raw_res(tok, *p) for p in GAME_PAIRS]
    trace = list(machine.trace)
    first_diff, diff_note = first_trace_diff(trace, anchor_trace)
    return {"force_ast": str(force_ast), "verdicts": verdicts,
            "expansions": list(machine.expansions),
            "final_basis": basis[-1], "gaps": gaps,
            "gap_finite": all(g == g and abs(g) != float("inf") for g in gaps),
            "trace": trace,
            "first_diff": first_diff, "diff_note": diff_note,
            "rng_shifted_after_lexikon": json.dumps(
                wm._rng.bit_generator.state, sort_keys=True) != state_fresh,
            "episodes": list(episodes),
            "pair_final": round6(pair_final),
            "rows_pre": round6(pre_rows), "rows_post": round6(rows_post),
            "_rows_pre": list(pre_rows), "_rows_post": list(rows_post),
            "lex_rounds": len(lex_rows),
            "_lex_rows": lex_rows, "_rows_series": rows_series,
            "_geoms": geoms, "_viewer": viewer}


def lane_counted_pair(anchor_trace: List[str]) -> Tuple[Dict[str, Any],
                                                        Dict[str, Any]]:
    """L2: kripke-Arm (Doppel-Lauf) + quus-Arm (Doppel-Lauf); Marks:
    t2_divergenz, t2_kripke, t2_lexikon_lebt; Determinismus-Fracht det2."""
    k1 = lane_counted(TAUTAST, anchor_trace)
    k2 = lane_counted(TAUTAST, anchor_trace)
    q1 = lane_counted(QUUSAST, anchor_trace)
    q2 = lane_counted(QUUSAST, anchor_trace)
    det2 = (k1["trace"] == k2["trace"] and k1["gaps"] == k2["gaps"]
            and q1["trace"] == q2["trace"] and q1["gaps"] == q2["gaps"]
            and [r["res"] for r in k1["_lex_rows"]]
            == [r["res"] for r in k2["_lex_rows"]]
            and k1["_rows_series"] == k2["_rows_series"])
    step9 = STEP9
    kripke_ok = (k1["verdicts"][step9 - 1] == "LOOP"
                 and "step9:LOOP" in k1["expansions"]
                 and k1["final_basis"] > 3 and k1["gap_finite"])
    quus_ok = (q1["verdicts"][step9 - 1] != "LOOP"
               and "step9:LOOP" not in q1["expansions"])
    divergenz_ok = k1["first_diff"] is not None and k1["first_diff"] <= 3
    lexikon_lebt = (all(r >= MIN_PAIR_RES for r in k1["pair_final"])
                    and all(n >= MIN_EPISODES for n in k1["episodes"]))
    mark("sp10_t2_divergenz", divergenz_ok)
    mark("sp10_t2_kripke", kripke_ok and quus_ok)
    mark("sp10_t2_lexikon_lebt", lexikon_lebt)
    summary = {
        "kripke": {key: k1[key] for key in
                   ("verdicts", "expansions", "final_basis", "first_diff",
                    "diff_note", "rng_shifted_after_lexikon", "episodes",
                    "pair_final", "rows_pre", "rows_post", "lex_rounds",
                    "gap_finite")},
        "quus": {key: q1[key] for key in
                 ("verdicts", "expansions", "final_basis", "episodes",
                  "pair_final")},
        "det2": det2, "kripke_ok": kripke_ok, "quus_ok": quus_ok,
    }
    return summary, {"_k1": k1, "_q1": q1}


# ---------------------------------------------------------------------------
# L3b: Analytik der L2-Lage (These 3, LIVE)
# ---------------------------------------------------------------------------
def lane_analytik(k1: Dict[str, Any]) -> Dict[str, Any]:
    """Geschlossene Formen gegen die L2-Rohwerte: Paar-Resonanz über alle
    48 Spielrunden (c_j je Runde gegen den Rohwert NACH der Runde), tisch-
    Zuschauer-Spur (49 Punkte: Punkt 0 = Pre-Lage mit k = 0 vor dem ersten
    Vorschub, Punkte 1..48 = die Zeilenlage je Runde mit der bis dahin
    gelernten regel↔rule-Episodenzahl), Zeilen-Konstanz für kaffee/baum,
    Grenzwert R_∞ = M·v."""
    geoms, viewer = k1["_geoms"], k1["_viewer"]
    lex_rows = k1["_lex_rows"]
    rows_series = k1["_rows_series"]
    rows_pre = k1["_rows_pre"]
    rows_post = k1["_rows_post"]

    # 1) Paar-Formel je Spielrunde (der Rohwert ist immer NACH der Runde
    #    genommen, `episodes` zählt ein ggf. Lernen derselben Runde).
    pair_devs = [abs(r["res"] - r["closed"]) for r in lex_rows]
    max_pair = max(pair_devs)

    # 2) tisch-Spur: Punkt 0 (k = 0) + 48 Runden; k = gelernte regel↔rule-
    #    Episoden (= lex_rows[i]["k0"], Buchhaltung des ersten Paares).
    regel_geom = geoms[GAME_PAIRS[0]]
    k0_series = [0] + [row["k0"] for row in lex_rows]
    tisch_series = [rows_pre[0]] + [rs[0] for rs in rows_series]
    obs_devs = [abs(R - closed_viewer(regel_geom, viewer, k))
                for R, k in zip(tisch_series, k0_series)]
    max_obs = max(obs_devs)

    # 3) kaffee/baum-Zeilen bitkonstant über alle Runden und in der Endlage.
    frozen_ok = (all(rs[1] == rows_pre[1] and rs[2] == rows_pre[2]
                     for rs in rows_series)
                 and rows_post[1] == rows_pre[1]
                 and rows_post[2] == rows_pre[2])

    # 4) Grenzwert: die letzte tisch-Lage liegt im 1e-4-Band um M·v.
    asym = abs(tisch_series[-1] - dotv(regel_geom["mid"], viewer))

    t3_paar = mark("sp10_t3_paar", max_pair <= ANALYTIC_TOL)
    t3_obs = mark("sp10_t3_obs", max_obs <= ANALYTIC_TOL and frozen_ok)
    return {"max_pair_dev": max_pair, "max_obs_dev": max_obs,
            "frozen_ok": frozen_ok, "t3_paar": t3_paar, "t3_obs": t3_obs,
            "asym": asym, "asym_reference": dotv(regel_geom["mid"], viewer),
            "tisch_series": round6(tisch_series), "k0_series": k0_series}

# ---------------------------------------------------------------------------
# L3a: SP9-Rückdiction
# ---------------------------------------------------------------------------
def lane_rueckdiction() -> Dict[str, Any]:
    committed = json.loads((REPO_ROOT / "scratches" /
                            "lexikon_dynamik_experiment.json").read_text(
                                encoding="utf-8"))
    soc = committed["social"]
    machine = WittgensteinMachine(MachineConfig(seed=SEED_SB, mode="in_loop"))
    tok = machine.tokenizer
    geoms = {pair: pair_geometry(tok, *pair) for pair in GAME_PAIRS}
    viewer = tok.embed_word("tisch")
    episodes = [0, 0, 0]
    pair_rows: List[Dict[str, Any]] = []

    # Leg 1: 15 gated Runden (einzelner lexicon_game-Aufruf, wie SP9).
    protocol = machine.lexicon_game(GAME_PAIRS, rounds=15, alpha=ALPHA)
    for line in protocol:
        pair = tuple(line["pair"])
        pi = GAME_PAIRS.index(pair)
        if line["learned"]:
            episodes[pi] += 1
        pair_rows.append({"round": int(line["round"]), "leg": 1,
                          "pair": list(pair), "learned": bool(line["learned"]),
                          "episodes": episodes[pi],
                          "res": None, "closed": None})
    unrelated15 = [round(raw_res(tok, *p), 6) for p in UNRELATED_GAME]
    # SP9-Schreibweise: der Spur-Anchluss ist der gerundete SP6-Anker
    # (unrelated15[0]); die Rohspur kommt daneben für die geschlossene Form.
    traj: List[float] = [unrelated15[0]]
    traj_raw: List[float] = [raw_res(tok, *UNRELATED_GAME[0])]
    k_list: List[int] = [episodes[0]]

    # Leg 2: 45 Einzel-Runden (wie SP9, je Runde beobachtbar).
    for r in range(45):
        pair = GAME_PAIRS[r % len(GAME_PAIRS)]
        line = machine.lexicon_game([pair], rounds=1, alpha=ALPHA)[0]
        if line["learned"]:
            episodes[r % len(GAME_PAIRS)] += 1
        k = episodes[r % len(GAME_PAIRS)]
        pair_rows.append({"round": r, "leg": 2, "pair": list(pair),
                          "learned": bool(line["learned"]),
                          "episodes": k,
                          "res": raw_res(tok, *pair),
                          "closed": closed_pair(geoms[pair], k)})
        row = raw_res(tok, *UNRELATED_GAME[0])
        traj.append(row)
        traj_raw.append(row)
        k_list.append(episodes[0])
    unrelated60 = [round(raw_res(tok, *p), 6) for p in UNRELATED_GAME]

    # 1) geschlossene Form gegen die Rohspur
    regel_geom = geoms[GAME_PAIRS[0]]
    devs = [abs(R - closed_viewer(regel_geom, viewer, k))
            for R, k in zip(traj_raw, k_list)]
    max_traj = max(devs)
    pair_res_rows = [row for row in pair_rows if row["res"] is not None]
    max_pair = max(abs(row["res"] - row["closed"]) for row in pair_res_rows)
    # 2) Bit-Anker gegen die committete SP9-Spur (round6 über der SP9-Konstr.)
    rounded = round6(traj_raw)
    anchor_traj = soc["tisch_traj_leg2"] == rounded
    delta_count = sum(1 for a, b in zip(traj, traj[1:]) if b - a != 0.0)
    delta_01 = traj[1] - traj[0]      # Rundungs-Anschluss-Residuum (s. ERRATA)
    anchors = {
        "tisch_traj": anchor_traj,
        "tisch15": soc["tisch15"] == unrelated15[0],
        "tisch60": soc["tisch60"] == unrelated60[0],
        "drift": soc["tisch_drift_p4"] == round(abs(unrelated60[0]
                                                    - unrelated15[0]), 9),
        "delta_count": soc["tisch_deltas_nonzero"] == delta_count,
        "unrelated15": soc["unrelated15"] == unrelated15,
        "unrelated60": soc["unrelated60"] == unrelated60,
    }
    # Erstrun-Lesung (P3): delta über die Rohspur mit ungerundetem Punkt 0.
    delta_count_raw = sum(1 for a, b in zip(traj_raw, traj_raw[1:])
                          if b - a != 0.0)
    anchors_raw = dict(anchors)
    anchors_raw["delta_count"] = soc["tisch_deltas_nonzero"] == delta_count_raw
    analytic_ok = max_traj <= ANALYTIC_TOL and max_pair <= ANALYTIC_TOL
    anchor_ok = all(anchors.values())
    anchor_ok_raw = all(anchors_raw.values())
    mark("sp10_t3_rueck", analytic_ok and anchor_ok_raw)  # Erstrun FAIL (Erratum)
    mark("sp10_t3_rueck_p4", analytic_ok and anchor_ok)
    mid = regel_geom["mid"]
    asym = abs(traj_raw[-1] - dotv(mid, viewer))
    return {"committed_size": (len(soc["tisch_traj_leg2"]),),
            "unrelated15": unrelated15, "unrelated60": unrelated60,
            "tisch15": round(unrelated15[0], 6),
            "tisch60": round(unrelated60[0], 6),
            "delta_count": delta_count, "delta_count_raw": delta_count_raw,
            "delta_01": delta_01, "anchors": anchors,
            "max_traj_dev": max_traj, "max_pair_dev": max_pair,
            "asym": asym,
            "traj": rounded, "k_list": k_list,
            "learned_leg1": sum(1 for row in pair_rows
                                if row["leg"] == 1 and row["learned"]),
            "learned_leg2": sum(1 for row in pair_rows
                                if row["leg"] == 2 and row["learned"]),
            "analytic": analytic_ok, "anchor_ok": anchor_ok}


def main() -> int:
    print("=" * 78)
    print("SP10 — DIE REGELN UND DIE WOERTER (Composite × Lexikon)")
    print("=" * 78)

    print("[SP10.L0] SP7-Anker-Feldvergleich (seed 7, Zwang {9}) …")
    anchor = lane_anchor()

    print("[SP10.L1] Seitenband (These 1) …")
    t1 = lane_seitenband(anchor)

    print("[SP10.L2] Gezaehlter Zufalls-Vorschub (These 2) …")
    summary, artifacts = lane_counted_pair(anchor["trace"])

    print("[SP10.L3] Analytik (These 3: L2 live) …")
    analytik = lane_analytik(artifacts["_k1"])
    print("[SP10.L3a] SP9-Rueckdiction + geschlossene Form …")
    rueck = lane_rueckdiction()
    mark("sp10_det", anchor["det0"] and summary["det2"])
    mark("sp10_t3_asymptot", analytik["asym"] <= ASYM_TOL
         and rueck["asym"] <= ASYM_TOL)

    print("-" * 78)
    named = FINDINGS
    n_pass = sum(1 for _, v in named if v == "PASS")
    n_fail = sum(1 for _, v in named if v == "FAIL")
    print("BEFUNDE")
    for name, ok in named:
        print(f"  {name:22s} {ok}")
    if ERRATA:
        print("-" * 78)
        print("ERRATA (P4-Einstufung; Ursprungs-Verdicte unveraendert)")
        for name, note in ERRATA.items():
            print(f"  {name:22s} {note}")
    failed = [n for n, v in named if v == "FAIL" and n not in ERRATA]
    all_pass = not failed
    bilanzen = f"{n_pass} PASS / {n_fail} dokumentierte Errata" \
        if ERRATA else f"{n_pass} PASS"
    print("BILANZ:", bilanzen)

    out = REPO_ROOT / "scratches" / "regeln_woerter_experiment.json"
    out.write_text(json.dumps({
        "anchor": anchor, "seitenband": t1, "gezaehlt": summary,
        "analytik": analytik,
        "rueckdiction": {k: v for k, v in rueck.items()
                         if k != "committed"},
        "findings": [{"name": n, "status": v} for n, v in named],
        "errata": dict(ERRATA),
        "all_pass": all_pass, "summary": bilanzen,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[Save] {out.relative_to(REPO_ROOT)}")
    if not all_pass:
        print(f"FEHLGESCHLAGEN: {len(failed)} — {', '.join(failed)}")
        return 1
    print("GRUEN — Loesungskette geschlossen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())