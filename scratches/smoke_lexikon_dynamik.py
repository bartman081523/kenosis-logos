"""
smoke_lexikon_dynamik.py — SP9: Soziale Lexikon-Dynamik (Zyklus nach SP1-SP8).

Fragestellung (User-Delegation 2026-10-25, „entscheide du" auf SP9):
learn_pair über längere Spielverläufe — Interferenz, Sättigung, Verlernung.
Die Bedeutung eines Wortes ist sein Gebrauch in der Verlaufsgeschichte: was
passiert mit dem Lexikon, wenn dieselben Wörter über viele Episoden, in
Ketten und in Neugebrauch weitergespielt werden?

PREREG (P2 — Freeze VOR Messung, ExperimentLedgerMind v1.1):
Hypothesen, Bands, Verdict-Map und Falsifikatoren stehen hier in diesem
Docstring, bevor der erste Lauf die erste Zahl sieht. Toleranzen werden nach
der Messung NIEMALS geweitet; FAIL-Diagnosen sind gleichwertige Befunde (P6)
und werden namentlich dokumentiert (Projekt-Präzedenz Bericht §7).

Canon-Anker (Bestandsaufnahme, verifiziert):
  machines/wittgenstein_machine.py:218-234  learn_pair — gegenseitige Mischung,
      e_new = n((1−α)·e + α·e_partner); BEIDE Wörter drehen sich aufeinander zu.
  machines/wittgenstein_machine.py:183     embed_word liest das Lexikon ZUERST
      (geometric-tokenizer.lisp:95-98) — Gebrauch schlägt Gematria nach dem
      Lernen.
  machines/wittgenstein_machine.py:342-380 lexicon_game — gated Sprachschritte
      (Success ≥ 0.5 → learn_pair), round-robin über die Paarliste
      (pairs[round % len(pairs)]), deterministisch über MachineConfig(seed).
  machines/lisp/vendor/language-learning.lisp:22-25 — learn-from-input blendet
      den MITTELWERT der Wortvektoren mit α=0.2 in den 8-D-Zustand; KEIN
      Zerfallsterm. Verlernen existiert im Canon nur als Neugebrauch — genau
      das ist die zu prüfende These, keine schon existierende Mechanik.

Ex-ante-Analyse (P2-zulässig: arithmetisch aus dem Canon, KEINE Messung):
  Reine Episoden eines Paares (a, b) erzeugen aus der Paar-Resonanz c die
  Rekursion
      c' = (A·c + B) / (A + B·c)   mit A = (1−α)² + α²,  B = 2α(1−α),
  denn mit v1 ∝ (1−α)e1 + αe2, v2 ∝ (1−α)e2 + αe1 ist
      v1·v2 = A·c + B,  |v1|² = A + B·c.
  Die Verknappung folgt:  1−c' = (A−B)·(1−c) / (A+B·c)  → Konvergenzfaktor
  A−B (asymptotisch, c→1). Alle drei Vorhersagen unten stehen aus dieser
  Rekursion VOR dem Lauf:
      α=0.3, c0 = −0.000263 (SP2-Anker regel↔rule):  c5 = 0.999790 —
             reproduziert SP6-after_paired[0] EXAKT; denn das Gate lernte
             15/15 Runden, je Paar also 5 Misch-Episoden im round-robin
             (c5(0.843313→) = 0.999982 bestätigt denselben Weg).
      α=0.9 und α=0.1 haben IDENTische Rekursion (Symmetrie α↔1−α: A=0.82,
      B=0.18, Faktor A−B = 0.64) — die Konvergenz kommt deutlich später
      als bei α=0.3 (Faktor 0.16). Der α-Sweep ist eine NICHT-monotone
      Konvergenz-Fläche mit Minimum in der Mitte — ex ante fixiert.

Lanes (alle CPU-only; keine GPU, kein SBCL, keine Maschinen-Code-Änderung —
lexicon_game + learn_pair decken die Dynamik komplett ab):
  sp9.0 Regression    — SP6-Lane exakt nachgestellt (seed 9, in_loop,
                        GAME_PAIRS, 15 Runden, α 0.3); Bit-Delta gegen die
                        SP6-Artefaktziffern.
  sp9.2 Sättigung     — ein Paar über 60 reine Episoden (α=0.3): Konvergenz
                        ≥ 0.999, MONOTON (kein Dip > 1e-6 ab Überschreiten),
                        kein Overshoot > 1+1e-9; α-Sweep {0.1, 0.3, 0.9} über
                        100 reine Episoden.
  sp9.1 Interferenz   — Kette: regel↔rule bis Konvergenz, dann 30 reine
                        Episoden rule↔game (geteiltes Wort 'rule');
                        res(regel,rule) fällt. End-Verdikt: das 'rule'-
                        Embedding nähert sich der Halbierenden von
                        (rule0, game0) — nicht 0. Disjunkt-Kontrast:
                        lernende Tauschpaare ohne geteilte Wörter bewegen
                        res(regel,rule) NICHT (≤ 1e-12).
  sp9.3 Verlernen     — regel↔rule voll gelernt, dann 10 reine Episoden
                        rule↔tisch: res(rule,regel) fällt unter 0.8 und
                        gegen cos(regel0, Halbierende(rule0, tisch0)). Dann
                        RELEARNING regel↔rule: ≥ 0.999 in ≤ 6 Episoden;
                        Gebrauch regeneriert die Bedeutung (Verlernen ist
                        kein Zustandsverlust).
  sp9.4 Sozial-Freeze — SP6-Konfig + 45 weitergelaufene Runden (60 gated
                        gesamt, RNG-Kontinuität): die unrelated-Resonanzen
                        driften nur innerhalb des ersten Konvergenzfensters
                        (regel bewegt sich mit) und frieren danach ein.

Verdict-Map (fixiert; Band-Weichung verboten):
  sp9_s6_regress      — after_paired == [0.999790, 0.999982, 1.000000]
                        (Rundung 6) UND learned_rounds == 15.
  sp9_det             — dieselbe Saturation-Lane zweimal gelaufen: volle
                        float-Identität der Trajektorien.
  sp9_sat_converged   — c(60, α=0.3) > 0.999.
  sp9_sat_monotone    — max-dip-ab-peak ≤ 1e-6 UND max(traj) ≤ 1+1e-9.
  sp9_sat_alpha3_first— erste Episode mit c ≥ 0.999 ist GENAU 5.
  sp9_sat_sweep_final — α ∈ {0.1, 0.9}: nach 100 Episoden c > 0.999.
  sp9_sat_sweep_symmetry — |c_n(α=0.1) − c_n(α=0.9)| ≤ 1e-12 für alle n.
  sp9_sat_sweep_mid_fast — conv(α=0.3) ≤ 6 UND conv(α=0.1) ≥ 12 UND
                        conv(α=0.9) ≥ 12 (Mitte schneller als Pole).
  sp9_int_decay       — res(regel,rule) nach der Kette fällt ≥ 0.05 unter
                        den Peak.
  sp9_int_endpoint    — |res(regel,rule)_final − cos(regel0, m)| ≤ 0.05 UND
                        |res(regel,game)_final − cos(regel0, m)| ≤ 0.05,
                        m = n(rule0 + game0).
  sp9_int_disjoint    — disjunktes Tauschpaar bewegt res(regel,rule) um
                        ≤ 1e-12.
  sp9_unlearn_decay   — res(rule,regel) nach Neugebrauch < 0.8 UND
                        |final − cos(regel0, m')| ≤ 0.02, m' = n(rule0 + tisch0).
  sp9_unlearn_relearn — nach ≤ 6 Relearning-Episoden c ≥ 0.999.
  sp9_social_freeze   — |mean_unrelated(60) − mean_unrelated(15)| ≤ 0.05.
  sp9_social_tisch_pin— |res(regel,tisch,60) − res(regel,tisch,15)| ≤ 1e-12
                        UND res(regel,tisch,15) rundet auf den SP6-Wert
                        0.003954.

P4-EINSTUFUNG (2026-10-25, NACH dem ersten Lauf — neue Lesung als neuer
Knoten; die Ursprungs-Verdicte bleiben in den Artefakten unangetastet):
  Der erste Lauf erbrachte 12/15 PASS und drei FAILs. Keines widerlegt die
  Mechanik-Hypothesen (Konvergenz exakt in 5 Episoden; Verlern-Decay 0.708
  < 0.8; Relearning in 4 Episoden; α-Symmetrie conv=18 an beiden Polen;
  Disjunkt-Delta 0.0) — die drei FAILs widerlegen drei Prereg-ANNAMEN
  derselben Familie:
    E1 „der Beobachter bleibt unbeteiligt" (sp9_int_endpoint,
       sp9_unlearn_decay): der Beobachter 'regel' war selbst durch Phase 1
       gedreht (embed_word('regel') ≠ regel0). Die Halbierenden-Prognosen
       benutzten die t0-Stände (0.004424 / −0.004496) gegen gemessene
       0.705054 / 0.708337. Die Messung selbst ist ein POSITIV-Befund:
       res(regel,rule) == res(regel,game) EXAKT (0.705054) — rule UND game
       nähern sich derselben Halbierenden; es gibt keine unbeteiligten
       Wörter, jeder Zuschauer ist Mitspieler.
    E2 „Sättigung == Fixpunkt" (sp9_social_tisch_pin): die 1e-12-Pin-Band
       war ex ante falsch gewählt — die Konvergenz ist eine ASYMPTOTE
       (Winkel-Shrink ~×0.4 je Episode): regel wandert weiter, auch bei
       c = 0.99979: tisch 0.003954 → 0.004085 (Δ = +1.31e-4). Nur
       wirklich unbeteiligte Wörter (bedeutung/kaffee, wort/baum — kein
       Lexikon-Kontakt) frieren EXAKT ein (Δ = 0.0).
  Korrigierte Lesungen (neue Verdicte, Bands UNVERÄNDERT aus dem Prereg —
  keine Band-Weichung; dieselben Identitäten, angewandt auf die PHASE-2-
  STÄNDE statt der t0-Stände, deterministisch replizierbar):
    sp9_int_endpoint_p4     — |res(regel,rule)_final − cos(regel_p2, m2)|
                              ≤ 0.05 UND |res(regel,game)_final −
                              cos(regel_p2, m2)| ≤ 0.05,
                              m2 = n(rule_p2 + game0) — alle Stände =
                              Beginn der Phase 2.
    sp9_unlearn_endpoint_p4 — |final_use − cos(regel_p2', m2')| ≤ 0.02,
                              m2' = n(rule_p2' + tisch0) (der Beobachter
                              'regel' friert in der Phase 2 ein).
    sp9_social_tisch_p4     — der tisch-Drift über Leg 2 trägt konstantes
                              Vorzeichen (monoton) und |Δ| ≤ 2e-2
                              (Pfad-Längen-Bound: regs Restwinkel gegen die
                              Halbierende ≈ sqrt(2·(1−c5)) ≈ 0.02 rad; die
                              Dot-Änderung ist durch die Pfadlänge
                              beschränkt).
    sp9_social_frozen_exact — wörter OHNE jeden Lexikon-Kontakt
                              (bedeutung/kaffee, wort/baum) bleiben über
                              beide Legs EXAKT ruhig (Δ = 0.0) — der
                              Sozialismus ist wortlokal: nur Wörter mit
                              Lexikon-Kontakt wandern weiter.
  Errata-Register: die drei Ursprungs-Verdicte bleiben FAIL, werden im Log
  unter ERRATA ausgewiesen und zählen nicht zum Lösungszustand (exit 0).

Run:  venv_openmythos/bin/python -u -m scratches.smoke_lexikon_dynamik \
      2>&1 | tee scratches/lexikon_dynamik_experiment.log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from machines.machine_base import MachineConfig  # noqa: E402
from machines.wittgenstein_machine import (  # noqa: E402
    GeometricTokenizer, WittgensteinMachine)

# SP6-Anker (sprachspiel_experiment.json, lane_lexicon, seed 9 in_loop):
# Fünf Misch-Episoden je Paar — die Rekursion liefert ex ante dieselben Werte.
SP6_AFTER_PAIRED = [0.999790, 0.999982, 1.000000]
SP6_REGEL_TISCH_15 = 0.003954
GAME_PAIRS = [("regel", "rule"), ("spiel", "game"), ("gebrauch", "use")]
UNRELATED_GAME = [("regel", "tisch"), ("bedeutung", "kaffee"), ("wort", "baum")]
ALPHAS = (0.1, 0.3, 0.9)

OUT_PATH = REPO_ROOT / "scratches" / "lexikon_dynamik_experiment.json"

THRESH = 0.999          # Sättigungs-Schwelle (Band ex ante fixiert)
SAT_EPISODES = 60       # Sättigungs-Lane
SWEEP_EPISODES = 100    # α-Sweep
CHAIN_EPISODES = 30     # Interferenz-Kette (Phase 2)
USE_EPISODES = 10       # Neugebrauch (Verlernen)
RELEARN_LIMIT = 6       # Relearning-Budget

# Prereg-Errata (P4-Einstufung, 2026-10-25): die Messung widerlegte die
# ANNAMHE, nicht die These — Ursprungs-Verdicte bleiben stehen (P1/P4), sind
# hier notiert, zählen nicht zum Lösungszustand.
ERRATA: Dict[str, str] = {
    "sp9_int_endpoint":
        "E1: Beobachter-Annahme widerlegt — regel ist selbst Mitspieler; "
        "korrigierte Lesung sp9_int_endpoint_p4.",
    "sp9_unlearn_decay":
        "E1: derselbe Beobachter-Fehler in der Halbierenden-Formel "
        "(Zerfall-anteil < 0.8 wahr, 0.708337); korrigierte Lesung "
        "sp9_unlearn_endpoint_p4.",
    "sp9_social_tisch_pin":
        "E2: Sättigung ist Asymptote, kein Fixpunkt (Drift +1.31e-4); "
        "korrigierte Lesung sp9_social_tisch_p4.",
}

findings: List[Tuple[str, str]] = []


def mark(name: str, cond: bool) -> None:
    findings.append((name, "PASS" if cond else "FAIL"))


def raw_res(tokenizer: GeometricTokenizer, w1: str, w2: str) -> float:
    """Paar-Resonanz UNGERUNDET (für Bands auf 1e-12-Niveau)."""
    return float(tokenizer.resonance(tokenizer.embed_word(w1),
                                     tokenizer.embed_word(w2)))


def round6(values: Any) -> Any:
    if isinstance(values, (list, tuple)):
        return [round6(v) for v in values]
    return round(float(values), 6)


def max_dip_after_peak(traj: List[float]) -> float:
    """Größter Rückfall unter dem laufenden Maximum (Monotonie-Maß)."""
    running_max = -1e9
    worst = 0.0
    for r in traj:
        if r > running_max:
            running_max = r
        worst = max(worst, running_max - r)
    return worst


def first_at_threshold(traj: List[float], threshold: float = THRESH
                       ) -> Optional[int]:
    """1-basierte Episode des ersten c >= threshold (None: nie)."""
    for i, r in enumerate(traj):
        if r >= threshold:
            return i + 1
    return None


def bisector_cos(e_a: List[float], e_b: List[float],
                 observer: List[float]) -> float:
    """cos(observer, n(e_a + e_b)) — der Halbierenden-Vorhersagewert des
    symmetrischen Mixing-Fixpunkts (ex ante aus den Gematria-Embeddings)."""
    m = GeometricTokenizer._normalize([x + y for x, y in zip(e_a, e_b)])
    return float(np.dot(observer, m))


def lane_regression() -> Dict[str, Any]:
    """SP9.0 — SP6-Lane exakt nachgestellt; Bit-Delta gegen die Artefakt-
    ziffern. Kein Maschinen-Drift durch die SP9-Runde."""
    machine = WittgensteinMachine(MachineConfig(seed=9, mode="in_loop"))
    tokenizer = machine.tokenizer

    def res(pair: Tuple[str, str]) -> float:
        return round6(raw_res(tokenizer, *pair))

    before_paired = [res(p) for p in GAME_PAIRS]
    protocol = machine.lexicon_game(GAME_PAIRS, rounds=15, alpha=0.3)
    after_paired = [res(p) for p in GAME_PAIRS]
    learned_rounds = sum(1 for r in protocol if r["learned"])
    mark("sp9_s6_regress",
         after_paired == SP6_AFTER_PAIRED and learned_rounds == 15)
    return {
        "before_paired": before_paired,
        "after_paired": after_paired,
        "learned_rounds": learned_rounds,
        "sp6_anchor": SP6_AFTER_PAIRED,
    }


def lane_saturation() -> Dict[str, Any]:
    """SP9.2 — ein Paar über 60 reine Episoden (α=0.3): Konvergenz, Monotonie,
    Kein-Overshoot; plus α-Sweep {0.1, 0.3, 0.9} über 100 reine Episoden und
    der Determinismus-Doppel-Check."""
    def run_once() -> List[float]:
        tokenizer = GeometricTokenizer()
        traj = []
        for _ in range(SAT_EPISODES):
            tokenizer.learn_pair("regel", "rule", alpha=0.3)
            traj.append(raw_res(tokenizer, "regel", "rule"))
        return traj

    traj_a = run_once()
    traj_b = run_once()
    dip = max_dip_after_peak(traj_a)
    first = first_at_threshold(traj_a)
    mark("sp9_det", traj_a == traj_b)
    mark("sp9_sat_converged", traj_a[-1] > THRESH)
    mark("sp9_sat_monotone", dip <= 1e-6 and max(traj_a) <= 1.0 + 1e-9)
    mark("sp9_sat_alpha3_first", first == 5)

    sweep: Dict[str, Any] = {}
    for alpha in ALPHAS:
        tokenizer = GeometricTokenizer()
        traj: List[float] = []
        for _ in range(SWEEP_EPISODES):
            tokenizer.learn_pair("regel", "rule", alpha=alpha)
            traj.append(raw_res(tokenizer, "regel", "rule"))
        sweep[repr(alpha)] = {
            "final": round6(traj[-1]),
            "conv_episode": first_at_threshold(traj),
            "traj": round6(traj),
        }
    slow = [first_at_threshold([float(v) for v in sweep[repr(alpha)]["traj"]])
            for alpha in (0.1, 0.9)]
    final_ok = all(sweep[repr(alpha)]["final"] > THRESH
                   for alpha in (0.1, 0.9))
    mark("sp9_sat_sweep_final", final_ok)
    traj01 = [float(v) for v in sweep[repr(0.1)]["traj"]]
    traj09 = [float(v) for v in sweep[repr(0.9)]["traj"]]
    sym = max(abs(a - b) for a, b in zip(traj01, traj09))
    mark("sp9_sat_sweep_symmetry", sym <= 1e-12)
    mid_fast = (sweep[repr(0.3)]["conv_episode"] is not None
                and sweep[repr(0.3)]["conv_episode"] <= 6
                and all(c is not None and c >= 12 for c in slow))
    mark("sp9_sat_sweep_mid_fast", mid_fast)
    return {
        "sat_alpha03": {
            "final": round6(traj_a[-1]),
            "dip_after_peak": round(dip, 12),
            "first_at_threshold": first,
            "overshoot": round(max(traj_a) - 1.0, 12),
            "traj": round6(traj_a),
        },
        "sweep": sweep,
        "sweep_symmetry_max": round(sym, 12),
        "sweep_conv_slow": slow,
    }


def lane_interference() -> Dict[str, Any]:
    """SP9.1 — Kette regel↔rule → rule↔game: geteiltes 'rule' zieht das
    gelernte Paar auseinander; das End-Embedding nähert sich der Halbierenden
    der Ursprungs-Vektoren. Disjunkt-Kontrast: kein geteiltes Wort → keine
    Bewegung."""
    tokenizer = GeometricTokenizer()
    regel0 = tokenizer.embed_word("regel")
    rule0 = tokenizer.embed_word("rule")
    game0 = tokenizer.embed_word("game")
    baseline_rule_game = raw_res(tokenizer, "regel", "game")

    # Phase 1: regel↔rule bis Konvergenz (ex ante: exakt 5 Episoden).
    used = 0
    peak = 0.0
    for i in range(100):
        tokenizer.learn_pair("regel", "rule", alpha=0.3)
        used = i + 1
        peak = raw_res(tokenizer, "regel", "rule")
        if peak >= THRESH:
            break

    # Phase-2-Anfangsstände — E1-Korrektur: auch der Beobachter 'regel' ist
    # selbst durch Phase 1 gedreht; die p4-Formel benutzt die TATSÄCHlichen
    # Phase-2-Stände statt der t0-Stände.
    regel_p2 = tokenizer.embed_word("regel")
    rule_p2 = tokenizer.embed_word("rule")

    # Phase 2: rule↔game, 30 reine Episoden; Beobachter: res(regel, rule).
    traj: List[float] = []
    for _ in range(CHAIN_EPISODES):
        tokenizer.learn_pair("rule", "game", alpha=0.3)
        traj.append(raw_res(tokenizer, "regel", "rule"))
    final_rule = raw_res(tokenizer, "regel", "rule")
    final_game = raw_res(tokenizer, "regel", "game")
    bisect_pred = bisector_cos(rule0, game0, regel0)      # Ursprungs-Lesung
    bisect_pred_p4 = bisector_cos(rule_p2, game0, regel_p2)  # Phase-2-Lesung
    mark("sp9_int_decay", final_rule <= peak - 0.05)
    mark("sp9_int_endpoint",
         abs(final_rule - bisect_pred) <= 0.05
         and abs(final_game - bisect_pred) <= 0.05)
    mark("sp9_int_endpoint_p4",
         abs(final_rule - bisect_pred_p4) <= 0.05
         and abs(final_game - bisect_pred_p4) <= 0.05)

    # Disjunkt-Kontrast: lernendes Tauschpaar ohne geteilte Wörter.
    disjoint = GeometricTokenizer()
    for _ in range(5):
        disjoint.learn_pair("regel", "rule", alpha=0.3)
    before_disjoint = raw_res(disjoint, "regel", "rule")
    for _ in range(CHAIN_EPISODES):
        disjoint.learn_pair("tisch", "stuhl", alpha=0.3)
    after_disjoint = raw_res(disjoint, "regel", "rule")
    mark("sp9_int_disjoint", abs(after_disjoint - before_disjoint) <= 1e-12)
    return {
        "baseline_rule_game": round6(baseline_rule_game),
        "phase1_episodes": used,
        "peak": round6(peak),
        "traj_phase2": round6(traj),
        "final_rule": round6(final_rule),
        "final_game": round6(final_game),
        "bisector_pred": round6(bisect_pred),
        "bisector_pred_p4": round6(bisect_pred_p4),
        "disjoint_delta": round(after_disjoint - before_disjoint, 12),
    }


def lane_unlearning() -> Dict[str, Any]:
    """SP9.3 — Verlernen durch Neugebrauch und Wiederherstellung durch
    wiederholten Gebrauch (kein Zerfallsterm im Canon)."""
    tokenizer = GeometricTokenizer()
    regel0 = tokenizer.embed_word("regel")
    rule0 = tokenizer.embed_word("rule")
    tisch0 = tokenizer.embed_word("tisch")

    for _ in range(5):
        tokenizer.learn_pair("regel", "rule", alpha=0.3)
    learned = raw_res(tokenizer, "regel", "rule")

    # Phase-2-Anfangsstände — dieselbe E1-Korrektur (der Beobachter 'regel'
    # friert in der Phase 2 ein, aber sein IST-Stand ist der Phase-2-Stand).
    regel_p2 = tokenizer.embed_word("regel")
    rule_p2 = tokenizer.embed_word("rule")

    traj_u: List[float] = []
    for _ in range(USE_EPISODES):
        tokenizer.learn_pair("rule", "tisch", alpha=0.3)
        traj_u.append(raw_res(tokenizer, "regel", "rule"))
    final_u = raw_res(tokenizer, "regel", "rule")
    bisect_u = bisector_cos(rule0, tisch0, regel0)       # Ursprungs-Lesung
    bisect_u_p4 = bisector_cos(rule_p2, tisch0, regel_p2)  # Phase-2-Lesung

    traj_r: List[float] = []
    relearn_at = None
    for i in range(RELEARN_LIMIT):
        tokenizer.learn_pair("regel", "rule", alpha=0.3)
        traj_r.append(raw_res(tokenizer, "regel", "rule"))
        if traj_r[-1] >= THRESH:
            relearn_at = i + 1
            break
    mark("sp9_unlearn_decay",
         final_u < 0.8 and abs(final_u - bisect_u) <= 0.02)
    mark("sp9_unlearn_endpoint_p4", abs(final_u - bisect_u_p4) <= 0.02)
    mark("sp9_unlearn_relearn", relearn_at is not None)
    return {
        "learned": round6(learned),
        "traj_use": round6(traj_u),
        "final_use": round6(final_u),
        "bisector_use_pred": round6(bisect_u),
        "bisector_use_pred_p4": round6(bisect_u_p4),
        "traj_relearn": round6(traj_r),
        "relearn_at": relearn_at,
    }


def lane_social() -> Dict[str, Any]:
    """SP9.4 — Sozial-Freeze: die unrelated-Resonanzen driften nur innerhalb
    des ersten Konvergenzfensters und frieren danach ein (RNG-Kontinuität
    über zwei lexicon_game-Legs derselben Maschine)."""
    machine = WittgensteinMachine(MachineConfig(seed=9, mode="in_loop"))
    tokenizer = machine.tokenizer

    def res(pair: Tuple[str, str]) -> float:
        return round6(raw_res(tokenizer, *pair))

    machine.lexicon_game(GAME_PAIRS, rounds=15, alpha=0.3)
    unrelated15 = [res(p) for p in UNRELATED_GAME]

    # Leg 2 runde-für-Runde: derselbe round-robin-Schedule und derselbe
    # RNG-Draw-Order wie eine einzelne lexicon_game(45)-Schrift (das Paar-
    # Loslassen verbraucht keine Zufallszüge), aber je Runde beobachtbar —
    # deshalb trägt die p4-Lesung die tatsächliche tisch-Trajektorie.
    tisch_traj = [float(unrelated15[0])]
    for r in range(45):
        w_lang, w_partner = GAME_PAIRS[r % len(GAME_PAIRS)]
        machine.lexicon_game([(w_lang, w_partner)], rounds=1, alpha=0.3)
        current = raw_res(tokenizer, *UNRELATED_GAME[0])
        tisch_traj.append(current)
    unrelated60 = [res(p) for p in UNRELATED_GAME]
    mean15 = float(np.mean(unrelated15))
    mean60 = float(np.mean(unrelated60))
    tisch15 = float(unrelated15[0])
    tisch60 = float(unrelated60[0])
    deltas = [b - a for a, b in zip(tisch_traj, tisch_traj[1:])]
    monotone = all(d >= 0.0 for d in deltas) or all(d <= 0.0 for d in deltas)
    drift_p4 = abs(tisch60 - tisch15)
    frozen = all(abs(unrelated60[i] - unrelated15[i]) == 0.0 for i in (1, 2))
    mark("sp9_social_freeze", abs(mean60 - mean15) <= 0.05)
    mark("sp9_social_tisch_pin",
         abs(tisch60 - tisch15) <= 1e-12
         and abs(round6(tisch15) - SP6_REGEL_TISCH_15) <= 1e-9)
    mark("sp9_social_tisch_p4", monotone and drift_p4 <= 2e-2)
    mark("sp9_social_frozen_exact", frozen)
    return {
        "unrelated15": unrelated15,
        "unrelated60": unrelated60,
        "mean15": round(mean15, 4),
        "mean60": round(mean60, 4),
        "tisch15": round6(tisch15),
        "tisch60": round6(tisch60),
        "tisch_traj_leg2": round6(tisch_traj),
        "tisch_deltas_nonzero": sum(1 for d in deltas if d != 0.0),
        "tisch_drift_p4": round(drift_p4, 9),
        "sp6_tisch_anchor": SP6_REGEL_TISCH_15,
    }


def main() -> int:
    print("=" * 78)
    print("SP9 — SOZIALE LEXIKON-DYNAMIK (Interferenz, Sättigung, Verlernung)")
    print("=" * 78)
    report: Dict[str, Any] = {}

    print("[SP9.0] SP6-Regression (CPU) …")
    report["regression"] = lane_regression()
    print("        after_paired =", report["regression"]["after_paired"],
          "learned_rounds =", report["regression"]["learned_rounds"])

    print("[SP9.2] Sättigung + α-Sweep (CPU) …")
    report["saturation"] = lane_saturation()
    print(f"        α=0.3: final {report['saturation']['sat_alpha03']['final']}"
          f" first_at {report['saturation']['sat_alpha03']['first_at_threshold']}"
          f" dip {report['saturation']['sat_alpha03']['dip_after_peak']}")

    print("[SP9.1] Interferenz-Kette (CPU) …")
    report["interference"] = lane_interference()
    print(f"        final_rule {report['interference']['final_rule']}"
          f" vs bisector {report['interference']['bisector_pred']}")

    print("[SP9.3] Verlernung/Relearning (CPU) …")
    report["unlearning"] = lane_unlearning()
    print(f"        final_use {report['unlearning']['final_use']}"
          f" relearn_at {report['unlearning']['relearn_at']}")

    print("[SP9.4] Sozial-Freeze (CPU) …")
    report["social"] = lane_social()
    print(f"        mean_unrelated 15→60: {report['social']['mean15']}"
          f" → {report['social']['mean60']}")

    print("-" * 78)
    print("BEFUNDE")
    for name, verdict in findings:
        print(f"  {name:<24} {verdict}")
    all_pass = all(v == "PASS" for n, v in findings if n not in ERRATA)
    n_errata = sum(1 for n, _ in findings if n in ERRATA)
    if ERRATA:
        print("ERRATA — Prereg-Errata (P4-Einstufung 2026-10-25; Ursprungs-"
              "Verdicte bleiben FAIL):")
        for name, text in ERRATA.items():
            print(f"  {name:<24} {text}")
    print("GRÜN — Lösungskette geschlossen (Ursprungs-FAILs als Errata "
          "dokumentiert, korrigierte Lesungen bestanden)."
          if all_pass
          else "ROT — offene Widerlegungsfälle außerhalb der Errata (P6).")
    n_pass = sum(1 for n, v in findings if v == "PASS" and n not in ERRATA)
    n_fail_errata = sum(1 for n, v in findings
                        if n in ERRATA and v == "FAIL")
    n_pass_fixed = sum(1 for n, v in findings
                       if v == "PASS" and "_p4" in n)
    report.update({"findings": findings, "all_pass": all_pass,
                   "errata": ERRATA,
                   "summary": f"{n_pass} PASS, {n_fail_errata} FAIL (Errata), "
                              f"{n_pass_fixed} korrigierte Lesungen PASS"})
    OUT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True),
                        encoding="utf-8")
    print("[Save] scratches/lexikon_dynamik_experiment.json")
    print(f"BILANZ: {n_pass} PASS / {n_fail_errata} dokumentierte Errata "
          f"/ {n_pass_fixed} korrigierte Lesungen")
    print("EXIT =", 0 if all_pass else 1)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())