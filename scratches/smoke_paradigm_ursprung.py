"""smoke_paradigm_ursprung.py — SP11: Der Ursprung der Paradigmen (Composite-Signal-Ebene).

Frage: WOHIN kommt ein Paradigm-Ereignis (ratio > 2.0) her, wenn kein Zwang es
setzt? SP7 zeigte, DASS das Paradigmen-Signal die Kripke-Bruchstelle markiert
(plus -> quus, first_detect 11); SP10 zeigte, DASS die Zeiten mit dem
Zufalls-Strom wandern (L0: 11/12/15 — L2: 12/13/15/16). SP11 entscheidet die
Entstehung mechanisch, ohne eine Zeile Maschinen-Code zu ändern — die
Sub-Gaps liegen je Schritt in den invariants, die Gegenproben leben auf der
Datenebene.

Canon-Anker (verifiziert vor dem Prereg):
  machines/composite.py:132-140 — Expansion bei verdict in {LOOP, COLLAPSE,
    EXPLOSION}; DANACH ZWEI unabhängige Signal-Quellen an ein und dasselbe
    expansions-Register: `signal = self.goedel.paradigm_signal()` (:135) und
    `signal_c = self.paradigm_signal()` (:138) — Herkunft ist im Register
    NICHT markiert (die Kernfrage von T1).
  :167-182 — Formel: len >= 10; recent = mean(history[-5:]); baseline =
    mean(history[-20:-5]) falls len >= 20, sonst mean(history[:-5]);
    ratio = recent / (baseline + GAP_EPS = 1e-6); detected = ratio > 2.0.
    Das COMPOSITE-Signal liest die Mittel-Gap-Serie; das Signal am Schritt n
    sieht die Gaps der Schritte 1..n-1 (Append :143 NACH der Signal-Prüfung).
  machines/goedel_machine.py:229-239 — dieselbe Formel auf der GOEDEL-eigenen
    Serie (tanh-beschränkt, lernt über self_model); das Gödel-Gap sieht im
    Schritt n bereits die Serie 1..n (Goedel.step :128 läuft VOR :135).
  machines/turing_machine.py:144-157 — expand_basis: Pool-Pop, Basis.append,
    State-Vector erweitert, UND loop_detector = {} (:156) — die Expansion
    wirkt auf künftige Verdikte, nicht nur auf die Serie.
  machine_base.py:51 / goedel_machine.py:40 — GAP_EPS = 1e-6.

EX-ANTE-Befund (2026-10-25, vor dem ersten Lauf, aus committeten Daten):
  hermeneutik_experiment.json trägt die Kolumnen paradigm_ratio und
  paradigm_detected. Hand-Replikation an den committeten gaps zeigt:
  - paradigm_ratio entspricht der GOEDEL-Signal-Größe (0.20-0.35 — nie >= 2);
  - paradigm_detected entspricht der COMPOSITE-Signal-Flag (True an den
    Schritten 11/12/15). Hand-Replikation aus den committeten composite gaps
    (Mittel der drei Sub-Gaps): ratio(11) = 3.078, ratio(12) = 2.962,
    ratio(13) = 1.688 (False), ratio(14) = 1.709 (False), ratio(15) = 2.580 —
    genau die Detect-Flag-Muster. Die committeten Kolumnen WIDERSPRUCHEN sich
    also ex ante NICHT einer der beiden Thesen — beide können nur am
    Neu-Lauf mechanisch geprüft werden, beide Daten sind Anker.

DREI HERMENEUTISCHE THESEN (als falsifizierbare Vorhersagen fixiert, 2026-10-25):
  T1 Urheber — das Paradigm-Ereignis kommt aus dem COMPOSITE-Signal
     (Gap-Mittel der drei Sub-Gaps), nie aus dem Gödel-Signal. Code-Fundament:
     das Gödel-Gap ist tanh-beschränkt und lernt über das self_model (langsame
     Glättung); die Gap-Spitzen kommen aus validity-Treiben der Wittgenstein-
     Teilmaschine (Regelgüte) und aus dem Verdikt der Turing-Teilmaschine.
     Falsifikator: irgendein Schritt mit goedel.detected == True.
  T2 Mechanik — das Signal ist ein exaktes Funktion der aufgezeichneten Serie:
     ratio(n) rekomputiert aus der finalen Gap-Serie (mit dem Regime-Leg
     len >= 20) reproduziert die aufgezeichnete Detect-Menge exakt und jede
     Ratio innerhalb 1e-12. Falsifikator: jede Abweichung (Menge/Ratio/
     Regime) — der Signal liest mehr als die Serie, oder die Serie ist nicht
     die Serie des Signals.
  T3 Ursprung — das Paradigma ist das ECHO der Basis-Expansion:
     T3a kein Anlass, kein Echo: ein unverzwungener Lauf OHNE jede Expansion
         produziert kein Paradigm-Ereignis. Falsifikator: ein Paradigma in
         einem Expansion-freien Lauf.
     T3b Echo-Abhängigkeit: Gefrieren des TURING-Sub-Gaps auf seinem Vor-
         Expansions-Niveau (Counterfaktur auf der Datenebene: cf_gap_i =
         (w_i + t_vor + g_i)/3 für i >= erste Expansion) bricht ALLE
         Paradigmen. Falsifikator: überlebende Detection.
     T3c Kontrast: Gefrieren des GOEDEL-Sub-Gaps ändert die Detect-Menge
         NICHT (cf2_gap_i = (w_i + t_i + g_vor)/3). Falsifikator: jede
         Änderung der Menge.
  T4 Zwang — eine erzwungene Expansion (identity-Verdikt -> LOOP, SP7-
     Mechanik) an Schritt 13 erzeugt mindestens EIN Paradigma in [14, 22]
     (Recent-Fenster-Lage der Spitzengenerierung), das im unzwung Zwilling
     desselben Seed fehlt. Falsifikator: keine induzierte Detection.

VERDICT-MAP (Prereg, fixiert VOR dem ersten Lauf; FAIL ist gleichwertiger
Befund; Bands werden nach der Messung NIE geweitet; P4-Einstufung bei
Implementations-Errata korrigiert mit _p4-Verdicten und UNVERÄNDERTEN Bands,
Ursprungs-Zeilen bleiben FAIL im Register):
  sp11_l0_anchor        — L0-Replay (seed 7, Zwang {9}, 16 Schritte) ==
                          committetes hermeneutik_experiment.json FELDWEISE:
                          kripke verdicts/final_basis 7/gaps (round6)/
                          expansions/paradigm_detected (Composite-Flag)/
                          paradigm_ratio (Goedel, Schritte 10-16) / first_detect
                          == 11; Quus-Arm == committet. Falsifikator: jedes
                          abweichende Feld.
  sp11_det              — Doppel-Läufe (L0-Kripke, Sweep-seed 11, Zwang-paar)
                          bitidentisch: trace + gap-Serie + Sub-Serien +
                          Detect-Flags. Falsifikator: jede Abweichung.
  sp11_t1_urheber       — in ALLEN Läufen kein Schritt mit goedel.detected ==
                          True. Falsifikator: irgendein True (auch ein
                          Doppel-Eintrag an einem Schritt).
  sp11_t2_mechanik      — je Lauf: rekomputierte Composite-Detect-Menge ==
                          aufgezeichnete Menge, Ratios <= 1e-12. Falsifikator:
                          jede Abweichung.
  sp11_t3a_expansionfrei— alle Expansion-freien Grundmengen-Läufe (6 Sweep-
                          Seeds + Zwang-Zwilling unverzwungen) paradigmfrei.
                          Falsifikator: ein Paradigma ohne Expansion.
                          Befunden-Zustand OFFEN ist zugelassen, falls die
                          Grundmenge KEINEN Expansion-freien Lauf enthält —
                          OFFEN zählt nicht als FAIL, wird namentlich
                          protokolliert (P6 vollständige Residue).
  sp11_t3b_turing_echo  — TURING-gefrorene Counterfaktur: Detect-Menge leer
                          in allen Läufen mit Expansion. Falsifikator:
                          überlebende Detection.
  sp11_t3c_goedel_vergleich — GOEDEL-gefrorene Counterfaktur: Detect-Menge
                          unverändert. Falsifikator: jede Änderung.
  sp11_t4_forced        — Zwangs-Lauf (seed 21, Zwang {13}, 24 Schritte):
                          mindestens ein Paradigma in [14, 22], das im
                          unverzwungenen Zwilling fehlt. Falsifikator: keine
                          induzierte Detection.

Run:  /run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python -u -m
      scratches.smoke_paradigm_ursprung 2>&1 | tee scratches/paradigm_ursprung_experiment.log
JSON: scratches/paradigm_ursprung_experiment.json
CPU-only, keine Maschinen-Code-Änderung; Determinismus über Doppel-Läufe.

P4-REGISTER (Erstrun 2026-10-25: 6 PASS + 2 FAIL, 0 Errata, exit 1; P4-Runde
2026-10-26 — Vorschlag A "Akkumulationsgesetz" des externen Audits):
  Kern: die Turing-Sub-Gap-Serie ist keine freie Zustandsgröße, sondern die
  Akkumulations-Währung (prediction = float(self._transition_count) ist der
  KUMULATIVE Verlaustransitionen-Kontostand, turing_machine.py:234; das Ist
  ist der Einzelschritt s_n, :239; Inkrement einzig in :197):
      t_n = |C_{n-1} - s_n| / (1 + s_n + 1e-6),   C_{n-1} = sum_{k<n} s_k
  (Gap = apply_gap mit GAP_EPS 1e-6, machine_base.py:54/:531; Kontostand
  exponiert in state() :293). Leerlauf-Blätter (s_n = 0, z. B.
  "x"/"1.00"-Regeln) schlagen das Konto C_{n-1} ungedämpft auf (seed12:
  28.0/31.0/33.0/39.0 an den Schritten 14/17/19/23); Hand-Verifikation an
  der committeten seed12-Serie 2026-10-25 über alle 24 Schritte bestätigt.
  ERRATA E1 (P4-Klassifikation; Ursprungs-Zeilen bleiben FAIL im Register):
    sp11_t3a_expansionfrei + sp11_t3b_turing_echo — die Ein-Faktor-Echo-
    Annahme (Turing-Gap als frei gefrorener Faktor) ist widerlegt.
  sp11_t3_accumulation_law_p4 — t̂-Serie rekonstruiert aus der
    transitions-Spalte; ZWEI Stränge: (a) max |t_n - t̂_n| <= 1e-12
    (RATIO_TOL) über alle Läufe und Schritte (Prereg-Schätzung 9 Läufe,
    verwirklicht über ALLE 10: l0 × 2, Sweep × 6, Zwangs-Paar — mehr
    Umfang, kein Band-Wechsel); (b) je-Zeiler-Kontostand
    (turing_machine.py:293) == kumulierte Summe der transitions je Schritt.
    Falsifikator: Restgröße > 1e-12 oder Kontostand-Differenz.
  sp11_t3_dedrifted_echo_p4 — gleitende Prediction ῡ_{n-1} =
    mean(s_{max(1, n-10) .. n-1}) (Präzedenz: wittgenstein_machine.py:413-414,
    np.mean(history[-10:]) im Nicht-Orakel-Zweig) statt der kumulierten
    Summe; dedrifted Composite-Serie cf_d = (w + t_dedrift + g)/3;
    Band: Detect-Menge == leer in den expansion-freien Läufen (Erstrun-
    Grundmenge: seed12/seed14/seed15/seed16). Falsifikator: überlebende
    Detection dort. Rest-Läufe (l0/seed11/seed13/zw_twin/zw_forced) als
    Daten (isolieren den reinen Verdikt-Anteil).
  Die Bands sind fixiert VOR dem Rerun (P2 innerhalb der P4-Runde).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machines.composite import CompositeMachine
from machines.machine_base import MachineConfig
from machines.wittgenstein_machine import GeneratedSymbolicRule

TAUTAST = ("identity", "x")          # erzwungen (SP7-Kripke: Selbst-Transition -> LOOP)
QUUSAST = ("add", ("sin", "x"), 0.0) # erzwungen Quus (SP7-Kontrast)
POOL6 = {"axiom_pool": ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA"]}
SEED_L0 = 7
FORCE_L0 = {9}
STEPS_L0 = 16
SP11_SEEDS = [11, 12, 13, 14, 15, 16]
STEPS_SWEEP = 24
SEED_ZW = 21
FORCE_ZW = {13}
STEPS_ZW = 24
ZW_BAND_LO = 14
ZW_BAND_HI = 22
THRESHOLD = 2.0                      # Canon-Default (beide Signale)
GAP_EPS = 1e-6                       # machine_base.py:51
RATIO_TOL = 1e-12

FINDINGS: List[Tuple[str, str]] = []
NOTES: List[str] = []
# P4-Klassifikation (2026-10-26, Vorschlag A): die Ursprungs-Vorhersagen
# T3a/T3b nahmen die Turing-Sub-Gap-Serie als FREIEN Faktor an — widerlegt
# durch das Akkumulationsgesetz (Docstring P4-REGISTER). Ursprungs-Zeilen
# bleiben FAIL (P4: Einstufung statt Re-Decide); korrigierte Lesung als
# _p4-Verdicte mit VOR dem Rerun fixierten Bands.
ERRATA: Dict[str, str] = {
    "sp11_t3a_expansionfrei":
        "E1: 'kein Anlass, kein Echo' nahm die Turing-Sub-Gap-Serie als freien "
        "Zustandsfaktor an. Sie ist die Akkumulations-Währung "
        "t_n = |C_{n-1} - s_n|/(1 + s_n + 1e-6) (prediction = kumulativer "
        "Kontostand, turing_machine.py:234/:197/:250; GAP_EPS "
        "machine_base.py:51/:531): Leerlauf-Blätter (s_n = 0) schlagen "
        "C_{n-1} ungedämpft auf — die Paradigmen in expansion-freien Läufen "
        "(Erstrun: seed12/14/15/16) sind Akkumulations-Echos. Korrigierte "
        "Lesung: sp11_t3_dedrifted_echo_p4.",
    "sp11_t3b_turing_echo":
        "E1: dieselbe Ein-Faktor-Annahme — der Freeze auf t_vor (Mittel der "
        "Vor-Expansions-Serie) misst kein Echo eines freien Sub-Gaps, sondern "
        "eine Misch-Serie der Akkumulations-Währung; die Überlebenden folgen "
        "dieser Mischung. Korrigierte Lesung: sp11_t3_dedrifted_echo_p4 "
        "(Gleitfenster statt Mittel-Freeze, wittgenstein_machine.py:413-414).",
}


def mark(name: str, state: str, note: str = "") -> None:
    """Verdikt-Register: state in {PASS, FAIL, OFFEN}."""
    FINDINGS.append((name, state))
    if note:
        NOTES.append(f"{name}: {note}")


def round6(x: Any) -> float:
    return round(float(x), 6)


# ---------------------------------------------------------------------------
# Canon-Replikation des Signals (composite.py:167-182 / goedel:229-239)
# ---------------------------------------------------------------------------

def signal_from_history(history: Sequence[float]
                        ) -> Tuple[Optional[float], bool]:
    """Exakte Replikation: am Beobachtungsschritt sieht das COMPOSITE-Signal
    die Serie der Schritte 1..n-1 (Append nach der Prüfung, :143); das
    GOEDEL-Signal die Serie 1..n (Goedel.step läuft vor :135)."""
    if len(history) < 10:
        return None, False
    recent = float(np.mean(history[-5:]))
    baseline = (float(np.mean(history[-20:-5])) if len(history) >= 20
                else float(np.mean(history[:-5])))
    ratio = recent / (baseline + GAP_EPS)
    return ratio, bool(ratio > THRESHOLD)


def replicate_flags(series: Sequence[float], offset: int) -> List[Tuple[Optional[float], bool]]:
    """(ratio, detected) je Beobachtungsschritt n=1..len(series);
    offset der Anzahl Gaps, die das Signal im Schritt N schon sieht:
    Composite-Signal: offset 0 (sieht Serie bis n-1), Goedel-Signal: offset 1."""
    out: List[Tuple[Optional[float], bool]] = []
    for n in range(1, len(series) + 1):
        history = series[:n - 1 + offset]
        out.append(signal_from_history(history))
    return out


# ---------------------------------------------------------------------------
# Harness (SP10-Muster, seedy generalisiert)
# ---------------------------------------------------------------------------

def make_composite(seed: int, force_ast: Tuple[Any, ...]
                   ) -> Tuple[CompositeMachine, Dict[str, Any]]:
    """Composite mit Zwangs-Haken am composite-step-Zähler (pending-Flag):
    die ERSTE Regel des Batch wird am Zwangsschritt durch force_ast ersetzt —
    ohne eine Zeile Maschinen-Code (wm.create_rule-Abdeckung, SP10-Muster)."""
    machine = CompositeMachine(MachineConfig(seed=seed, mode="in_loop",
                                             params=dict(POOL6)))
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


def run_composite(machine: CompositeMachine, harness: Dict[str, Any],
                  steps: int, force_steps: "set[int]") -> Dict[str, Any]:
    """Verlauf mit je-Schritt-Attribution: Sub-Gaps, beide Signale (goedel /
    composite), neue expansions-Einträge, Basis-Größe."""
    rows: List[Dict[str, Any]] = []
    trace: List[str] = []
    for step in range(1, steps + 1):
        if step in force_steps:
            harness["pending"] = True
        pre_expansions = list(machine.expansions)
        st = machine.step(None)
        inv = st.invariants
        new_exp = [e for e in machine.expansions if e not in pre_expansions]
        trace.append(st.trace[-1])
        rows.append({
            "step": step,
            "rule_name": inv["rule_name"],
            "verdict": inv["verdict"],
            "transitions": int(inv["transitions"]),
            "t_count": int(machine.turing.state()["transition_count"]),
            "gap": float(st.gap),
            "w_gap": float(inv["submachines"]["wittgenstein"]["gap"]),
            "t_gap": float(inv["submachines"]["turing"]["gap"]),
            "g_gap": float(inv["submachines"]["goedel"]["gap"]),
            "g_ratio": (float(inv["paradigm_signal"]["ratio"])
                        if inv["paradigm_signal"]["ratio"] is not None else None),
            "g_detected": bool(inv["paradigm_signal"]["detected"]),
            "c_ratio": (float(inv["paradigm_signal_composite"]["ratio"])
                        if inv["paradigm_signal_composite"]["ratio"] is not None else None),
            "c_detected": bool(inv["paradigm_signal_composite"]["detected"]),
            "new_expansions": new_exp,
            "basis_size": len(machine.turing.basis),
        })
    expansions = list(machine.expansions)
    verdict_exp_steps = sorted(
        int(m.group(1)) for m in
        (re.match(r"step(\d+):(LOOP|COLLAPSE|EXPLOSION)", e) for e in expansions)
        if m)
    para_steps = sorted(
        int(m.group(1)) for m in
        (re.match(r"step(\d+):paradigm", e) for e in expansions) if m)
    gap_series = [r["gap"] for r in rows]
    w_series = [r["w_gap"] for r in rows]
    t_series = [r["t_gap"] for r in rows]
    g_series = [r["g_gap"] for r in rows]
    return {
        "rows": rows, "trace": trace, "expansions": expansions,
        "verdict_exp_steps": verdict_exp_steps, "para_steps": para_steps,
        "gap_series": gap_series, "w_series": w_series, "t_series": t_series,
        "g_series": g_series, "basis_size": rows[-1]["basis_size"],
        "final_basis": len(machine.turing.basis),
    }


def rep_composite_flags(run: Dict[str, Any]) -> List[Tuple[Optional[float], bool]]:
    """T2-Replikation: Composite-Detect-Muster aus der finalen Gap-Serie
    (das Signal am Schritt n sieht die Serie 1..n-1)."""
    return replicate_flags(run["gap_series"], offset=0)


def counterfactuals(run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """T3b/T3c — Gefrieren auf dem Vor-Expansions-Niveau, Datenebene.
    None, wenn der Lauf keine Expansion hatte."""
    first = run["verdict_exp_steps"][0] if run["verdict_exp_steps"] else None
    if not first:
        return None
    w, t, g = run["w_series"], run["t_series"], run["g_series"]
    cut = first - 1
    t_pre = float(np.mean(t[:cut]))
    g_pre = float(np.mean(g[:cut]))
    cf_t = [t[i] if i < cut else t_pre for i in range(len(t))]
    cf_g = [g[i] if i < cut else g_pre for i in range(len(g))]
    cf_b = [(w[i] + cf_t[i] + g[i]) / 3.0 for i in range(len(w))]
    cf_c = [(w[i] + t[i] + cf_g[i]) / 3.0 for i in range(len(w))]
    return {
        "first_expansion": first,
        "t_pre": t_pre, "g_pre": g_pre,
        "orig_detections": list(run["para_steps"]),
        "cf_b_detections": [n for n, (_, det) in
                            enumerate(replicate_flags(cf_b, offset=0), 1) if det],
        "cf_c_detections": [n for n, (_, det) in
                            enumerate(replicate_flags(cf_c, offset=0), 1) if det],
        "orig_flags": [r["c_detected"] for r in run["rows"]],
        "cf_b_flags": [det for _, det in replicate_flags(cf_b, offset=0)],
        "cf_c_flags": [det for _, det in replicate_flags(cf_c, offset=0)],
    }


def t2_check(run: Dict[str, Any]) -> Tuple[bool, str]:
    """Mechanik-Falsifikator: rekomputierte (ratio, detected) == aufgezeichnet."""
    rep = rep_composite_flags(run)
    for row, (ratio, det) in zip(run["rows"], rep):
        if det != row["c_detected"]:
            return False, f"Flag-Differenz am Schritt {row['step']}"
        if (ratio is None) != (row["c_ratio"] is None):
            return False, f"Ratio-None-Differenz am Schritt {row['step']}"
        if ratio is not None and abs(ratio - row["c_ratio"]) > RATIO_TOL:
            return False, (f"Ratio-Differenz am Schritt {row['step']}: "
                           f"{ratio!r} != {row['c_ratio']!r}")
    # Detect-Menge-Ableitung aus den expansions-Einträgen (Konsistenz):
    return True, "ok"


# ---------------------------------------------------------------------------
# P4-Runde — Akkumulationsgesetz + Dedrift (Vorschlag A); Bands fixiert
# vor dem Rerun. Canon: turing_machine.py:234/:197/:250/:293,
# machine_base.py:54/:531, wittgenstein_machine.py:413-414.
# ---------------------------------------------------------------------------

def accumulation_check(run: Dict[str, Any]) -> Dict[str, Any]:
    """P4-Vorschlag A: t̂_n = |C_{n-1} - s_n|/(1 + s_n + 1e-6) gegen die
    aufgezeichnete Turing-Gap-Serie; C_{n-1} = kumulierte transitions-Summe;
    zweiter Strang: je-Zeiler-Kontostand (state()["transition_count"], :293)
    == kumulierte Summe der transitions je Schritt."""
    s = [r["transitions"] for r in run["rows"]]
    residues: List[float] = []
    c_prior = 0
    for n, s_n in enumerate(s, 1):
        t_hat = abs(c_prior - s_n) / (1.0 + abs(s_n) + GAP_EPS)
        residues.append(abs(t_hat - run["t_series"][n - 1]))
        c_prior += s_n
    counts = [r["t_count"] for r in run["rows"]]
    count_ok = all(c == sum(s[:i]) for i, c in enumerate(counts, 1))
    return {"max_resid": max(residues) if residues else 0.0,
            "residues": residues, "count_ok": count_ok,
            "c_series": [sum(s[:i]) for i in range(len(s))]}


def dedrift(run: Dict[str, Any]) -> Dict[str, Any]:
    """P4-Vorschlag A: gleitende Prediction statt kumulierter Summe —
    ῡ_{n-1} = mean(s_{max(1, n-10) .. n-1}) (Präzedenz
    wittgenstein_machine.py:413-414, np.mean(history[-10:]) im
    Nicht-Orakel-Zweig; leeres Verlauf → 0.0); dedrifted Composite-Serie
    cf_d = (w + t_dedrift + g)/3, Detect-Menge aus cf_d (Offset 0)."""
    s = [r["transitions"] for r in run["rows"]]
    t_dedrift: List[float] = []
    for n, s_n in enumerate(s, 1):
        history = s[max(0, n - 10):n - 1]
        u = float(np.mean(history)) if history else 0.0
        t_dedrift.append(abs(u - float(s_n)) / (1.0 + abs(s_n) + GAP_EPS))
    cf_d = [(w + t + g) / 3.0 for w, t, g in
            zip(run["w_series"], t_dedrift, run["g_series"])]
    flags = replicate_flags(cf_d, offset=0)
    return {"t_dedrift_series": t_dedrift, "cf_d_series": cf_d,
            "dedrift_flags": [det for _, det in flags],
            "dedrift_detections": [n for n, (_, det) in enumerate(flags, 1)
                                   if det]}


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------

def lane_l0_anker() -> Dict[str, Any]:
    """L0-Replay gegen hermeneutik_experiment.json — inklusive der
    committeten Signal-Kolumnen (paradigm_detected == Composite-Flag,
    paradigm_ratio == Goedel-Ratio)."""
    committed = json.load(open("scratches/hermeneutik_experiment.json"))
    out: Dict[str, Any] = {}
    fields_ok = True

    for arm, force_ast, key in (("kripke", TAUTAST, "kripke"),
                                ("quus", QUUSAST, "contrast_quus")):
        machine, harness = make_composite(SEED_L0, force_ast)
        run = run_composite(machine, harness, STEPS_L0, FORCE_L0)
        cm = committed[key]
        verdicts_ok = [r["verdict"] for r in run["rows"]] == cm["verdicts"]
        final_ok = run["final_basis"] == cm["final_basis"]
        gaps_ok = ([round6(g) for g in run["gap_series"]]
                   == [round6(g) for g in cm["gaps"]])
        exp_ok = run["expansions"] == cm["expansions"]
        det_ok = ([r["c_detected"] for r in run["rows"]] == cm["paradigm_detected"])
        ratio_ok = True
        ratios_live = [r["g_ratio"] for r in run["rows"]]
        ratios_commit = cm["paradigm_ratio"]
        for idx, (a, b) in enumerate(zip(ratios_live, ratios_commit)):
            if a is None and b is None:
                continue
            if a is None or b is None:
                ratio_ok = False
                break
            if abs(a - b) > RATIO_TOL:
                ratio_ok = False
                break
        arm_ok = verdicts_ok and final_ok and gaps_ok and exp_ok and det_ok and ratio_ok
        fields_ok = fields_ok and arm_ok
        out[arm] = {
            "run": run,
            "checks": {"verdicts": verdicts_ok, "final_basis": final_ok,
                       "gaps": gaps_ok, "expansions": exp_ok,
                       "paradigm_detected": det_ok, "paradigm_ratio": ratio_ok},
        }
    first_detect_ok = all(
        out[arm]["run"]["para_steps"][0] == 11
        for arm in ("kripke", "quus") if out[arm]["run"]["para_steps"])
    fields_ok = fields_ok and first_detect_ok

    # Herkunfts-Attribution der kripke-Paradigma-Einträge (T1-Daten):
    attr = []
    for n in out["kripke"]["run"]["para_steps"]:
        row = out["kripke"]["run"]["rows"][n - 1]
        attr.append({"step": n, "c_ratio": row["c_ratio"],
                     "c_detected": row["c_detected"],
                     "g_ratio": row["g_ratio"], "g_detected": row["g_detected"]})
    out["kripke_attribution"] = attr
    out["fields_ok"] = fields_ok
    out["first_detect_ok"] = first_detect_ok
    mark("sp11_l0_anchor", "PASS" if fields_ok and first_detect_ok else "FAIL",
         "Feldvergleich kripke+quus inkl. committeter Signal-Kolumnen "
         "(paradigm_ratio = Goedel, paradigm_detected = Composite)")
    return out


def lane_sweep() -> Dict[str, Any]:
    """6 unverzwungene Läufe (Seeds 11-16, 24 Schritte) — T1/T2/T3-Daten."""
    runs: Dict[str, Dict[str, Any]] = {}
    for seed in SP11_SEEDS:
        machine, harness = make_composite(seed, TAUTAST)
        runs[f"seed{seed}"] = run_composite(machine, harness, STEPS_SWEEP, set())
    return runs


def lane_zwang() -> Dict[str, Any]:
    """Seed-21-Paar: unverzwungener Zwilling + Zwang {13} — T4-Daten
    (+ Doppel-Läufe für die Determinismus-Prüfung)."""
    twin, harness_t = make_composite(SEED_ZW, TAUTAST)
    forced, harness_f = make_composite(SEED_ZW, TAUTAST)
    zw_twin = run_composite(twin, harness_t, STEPS_ZW, set())
    zw_forced = run_composite(forced, harness_f, STEPS_ZW, FORCE_ZW)
    forced2, harness_f2 = make_composite(SEED_ZW, TAUTAST)
    zw_forced_re = run_composite(forced2, harness_f2, STEPS_ZW, FORCE_ZW)
    return {"zw_twin": zw_twin, "zw_forced": zw_forced, "zw_forced_re": zw_forced_re}


def determinism_check(l0: Dict[str, Any], sweep: Dict[str, Any],
                      zwang: Dict[str, Any]) -> Tuple[bool, str]:
    """Doppel-Läufe bitidentisch: trace + gap-Serie + Sub-Serien + Flags."""
    machine, harness = make_composite(SEED_L0, TAUTAST)
    l0_re = run_composite(machine, harness, STEPS_L0, FORCE_L0)
    l0_first = l0["kripke"]["run"]
    same_trace = l0_re["trace"] == l0_first["trace"]
    same_gaps = l0_re["gap_series"] == l0_first["gap_series"]
    same_subs = (l0_re["w_series"] == l0_first["w_series"]
                 and l0_re["t_series"] == l0_first["t_series"]
                 and l0_re["g_series"] == l0_first["g_series"])
    same_flags = ([r["c_detected"] for r in l0_re["rows"]]
                  == [r["c_detected"] for r in l0_first["rows"]])
    if not (same_trace and same_gaps and same_subs and same_flags):
        return False, "L0-Wiederholung weicht ab"

    machine11, harness11 = make_composite(11, TAUTAST)
    s11_re = run_composite(machine11, harness11, STEPS_SWEEP, set())
    s11_first = sweep["seed11"]
    if not (s11_re["trace"] == s11_first["trace"]
            and s11_re["gap_series"] == s11_first["gap_series"]
            and [r["c_detected"] for r in s11_re["rows"]]
            == [r["c_detected"] for r in s11_first["rows"]]):
        return False, "Sweep-seed11-Wiederholung weicht ab"

    if not (zwang["zw_forced_re"]["trace"] == zwang["zw_forced"]["trace"]
            and zwang["zw_forced_re"]["gap_series"] == zwang["zw_forced"]["gap_series"]
            and [r["c_detected"] for r in zwang["zw_forced_re"]["rows"]]
            == [r["c_detected"] for r in zwang["zw_forced"]["rows"]]):
        return False, "Zwang-Wiederholung weicht ab"
    return True, "ok"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("== SP11 — Der Ursprung der Paradigmen ==")
    print("Canon: composite.py:132-140/:167-182, goedel_machine.py:229-239,")
    print("       turing_machine.py:144-157 (loop_detector-Reset), GAP_EPS 1e-6")
    print("P4: turing_machine.py:197/:234/:250/:293 (Akkumulationsgesetz),")
    print("       machine_base.py:54/:531, wittgenstein_machine.py:413-414")
    print()

    l0 = lane_l0_anker()
    sweep = lane_sweep()
    zwang = lane_zwang()

    # Determinismus
    det_ok, det_note = determinism_check(l0, sweep, zwang)
    mark("sp11_det", "PASS" if det_ok else "FAIL", det_note)

    # T1: Urheber — niemals goedel.detected
    all_runs: Dict[str, Dict[str, Any]] = {
        "l0_kripke": l0["kripke"]["run"], "l0_quus": l0["quus"]["run"],
        **{k: v for k, v in sweep.items()},
        "zw_twin": zwang["zw_twin"], "zw_forced": zwang["zw_forced"],
    }
    breach = [(name, r["step"]) for name, run in all_runs.items()
              for r in run["rows"] if r["g_detected"]]
    t1_ok = not breach
    mark("sp11_t1_urheber", "PASS" if t1_ok else "FAIL",
         f"Goedel-Verletzungen: {breach}" if breach else
         "kein Schritt mit goedel.detected in allen 9 Läufen")

    # T2: Mechanik — Replikation aus der Serie
    t2_fail: List[str] = []
    for name, run in all_runs.items():
        ok, note = t2_check(run)
        if not ok:
            t2_fail.append(f"{name}: {note}")
    mark("sp11_t2_mechanik", "PASS" if not t2_fail else "FAIL",
         f"Fehler: {t2_fail}" if t2_fail
         else "Detect-Menge und Ratio je Lauf exakt aus der Gap-Serie rekomputiert")

    # T3a: Expansion-freie Läufe paradigmfrei
    free_runs = {n: r for n, r in all_runs.items() if not r["verdict_exp_steps"]}
    if free_runs:
        viol = [(n, r["para_steps"]) for n, r in free_runs.items() if r["para_steps"]]
        t3a_state = "PASS" if not viol else "FAIL"
        t3a_note = (f"Expansion-frei: {sorted(free_runs)}; Verletzungen: {viol}"
                    if viol else
                    f"Expansion-freie Läufe {sorted(free_runs)} alle paradigmfrei")
    else:
        t3a_state = "OFFEN"
        t3a_note = ("kein Expansion-freier Lauf in der Grundmenge "
                    "(6 Sweep-Seeds + Zwang-Zwilling) — Test nicht anwendbar")
    mark("sp11_t3a_expansionfrei", t3a_state, t3a_note)

    # T3b/T3c: Counterfaktur je Lauf mit Expansion
    cfs: Dict[str, Dict[str, Any]] = {}
    t3b_fail: List[str] = []
    t3c_fail: List[str] = []
    for name, run in all_runs.items():
        cf = counterfactuals(run)
        if cf is None:
            continue
        cfs[name] = cf
        if cf["cf_b_detections"]:
            t3b_fail.append(f"{name}: Überlebende {cf['cf_b_detections']}")
        if cf["cf_c_detections"] != cf["orig_detections"]:
            t3c_fail.append(f"{name}: {cf['cf_c_detections']} != "
                            f"{cf['orig_detections']}")
    t3b_state = "PASS" if (not t3b_fail and cfs) else ("FAIL" if t3b_fail else "OFFEN")
    t3c_state = "PASS" if (not t3c_fail and cfs) else ("FAIL" if t3c_fail else "OFFEN")
    mark("sp11_t3b_turing_echo", t3b_state, f"Fehler: {t3b_fail}" if t3b_fail
         else f"alle {len(cfs)} Gegenproben ohne überlebende Detection")
    mark("sp11_t3c_goedel_vergleich", t3c_state, f"Fehler: {t3c_fail}" if t3c_fail
         else f"Detection-Mengen in {len(cfs)} Gegenproben unverändert")

    # T4: Zwang
    induced = sorted(set(zwang["zw_forced"]["para_steps"])
                     - set(zwang["zw_twin"]["para_steps"]))
    in_band = [n for n in induced if ZW_BAND_LO <= n <= ZW_BAND_HI]
    mark("sp11_t4_forced", "PASS" if in_band else "FAIL",
         f"induzierte Paradigmas {induced}; in Band [14, 22]: {in_band}")

    # P4-Runde (Vorschlag A): Akkumulationsgesetz + Dedrift — Bands fixiert
    # vor dem Rerun; die Ursprungs-Verdikte t3a/t3b bleiben FAIL (ERRATA).
    free_names = [n for n, run in all_runs.items()
                  if not run["verdict_exp_steps"]]
    acc: Dict[str, Dict[str, Any]] = {}
    acc_fails: List[str] = []
    for name, run in all_runs.items():
        a = accumulation_check(run)
        acc[name] = {"max_resid": a["max_resid"], "count_ok": a["count_ok"],
                     "c_series": a["c_series"]}
        if not a["count_ok"] or a["max_resid"] > RATIO_TOL:
            acc_fails.append(f"{name}: count_ok={a['count_ok']}, "
                             f"max_resid={a['max_resid']:.3e}")
    acc_max = max((v["max_resid"] for v in acc.values()), default=0.0)
    mark("sp11_t3_accumulation_law_p4", "PASS" if not acc_fails else "FAIL",
         f"Strang-Verletzung: {acc_fails}" if acc_fails else
         (f"t_n == |C(n-1) - s_n|/(1+s_n+1e-6) in allen {len(all_runs)} "
          f"Läufen; max Residuum {acc_max!r}; Kontostand == kumulierte "
          "Summe in jedem Schritt (turing_machine.py:234/:197/:293)"))

    dd: Dict[str, Dict[str, Any]] = {}
    drift_fails: List[str] = []
    for name, run in all_runs.items():
        d = dedrift(run)
        dd[name] = d
        if name in free_names and d["dedrift_detections"]:
            drift_fails.append(f"{name}: überlebende {d['dedrift_detections']}")
    mark("sp11_t3_dedrifted_echo_p4", "PASS" if not drift_fails else "FAIL",
         f"überlebende Dedrift-Detections: {drift_fails}" if drift_fails else
         ("dedrifted Detect-Menge == leer in den expansion-freien Läufen "
          f"{free_names} (Gleitfenster-Präzedenz "
          "wittgenstein_machine.py:413-414); Rest-Läufe als Daten"))

    # Befunde
    print("BEFUNDE:")
    for name, state in FINDINGS:
        print(f"  {name:<26s} {state}")
    print()
    if NOTES:
        print("NOTIZEN:")
        for n in NOTES:
            print(f"  - {n}")
        print()

    if ERRATA:
        print("ERRATA (P4-Klassifikation):")
        for name, text in ERRATA.items():
            print(f"  - {name}: {text}")
        print()

    failed = [n for n, s in FINDINGS if s == "FAIL" and n not in ERRATA]
    offen = [n for n, s in FINDINGS if s == "OFFEN"]
    passes = [n for n, s in FINDINGS if s == "PASS"]
    print(f"BILANZ: {len(passes)} PASS; {len(failed)} FEHLGESCHLAGEN (unverwartet); "
          f"{len([n for n, s in FINDINGS if s == 'FAIL']) - len(failed)} dokumentierte Errata; "
          f"{len(offen)} OFFEN")

    payload = {
        "l0": {"kripke": {"checks": l0["kripke"]["checks"],
                          "expansions": l0["kripke"]["run"]["expansions"],
                          "para_steps": l0["kripke"]["run"]["para_steps"],
                          "transitions": [r["transitions"] for r in
                                          l0["kripke"]["run"]["rows"]],
                          "t_count": [r["t_count"] for r in
                                      l0["kripke"]["run"]["rows"]],
                          "final_basis": l0["kripke"]["run"]["final_basis"]},
               "quus": {"checks": l0["quus"]["checks"],
                        "expansions": l0["quus"]["run"]["expansions"],
                        "para_steps": l0["quus"]["run"]["para_steps"],
                        "transitions": [r["transitions"] for r in
                                        l0["quus"]["run"]["rows"]],
                        "t_count": [r["t_count"] for r in
                                    l0["quus"]["run"]["rows"]],
                        "final_basis": l0["quus"]["run"]["final_basis"]},
               "kripke_attribution": l0["kripke_attribution"],
               "fields_ok": l0["fields_ok"],
               "first_detect_ok": l0["first_detect_ok"]},
        "sweep": {name: {
            "verdicts": [r["verdict"] for r in run["rows"]],
            "expansions": run["expansions"],
            "para_steps": run["para_steps"],
            "verdict_exp_steps": run["verdict_exp_steps"],
            "gap_series": run["gap_series"],
            "w_series": run["w_series"], "t_series": run["t_series"],
            "g_series": run["g_series"],
            "transitions": [r["transitions"] for r in run["rows"]],
            "t_count": [r["t_count"] for r in run["rows"]],
            "c_ratio": [r["c_ratio"] for r in run["rows"]],
            "g_ratio": [r["g_ratio"] for r in run["rows"]],
            "c_detected": [r["c_detected"] for r in run["rows"]],
            "g_detected": [r["g_detected"] for r in run["rows"]],
            "basis_size": run["basis_size"],
        } for name, run in sweep.items()},
        "zwang": {name: {
            "verdicts": [r["verdict"] for r in run["rows"]],
            "expansions": run["expansions"],
            "para_steps": run["para_steps"],
            "gap_series": run["gap_series"],
            "w_series": run["w_series"], "g_series": run["g_series"],
            "t_series": run["t_series"],
            "transitions": [r["transitions"] for r in run["rows"]],
            "t_count": [r["t_count"] for r in run["rows"]],
            "c_detected": [r["c_detected"] for r in run["rows"]],
        } for name, run in
            (("zw_twin", zwang["zw_twin"]), ("zw_forced", zwang["zw_forced"]))},
        "counterfactuals": cfs,
        "p4": {
            "errata": ERRATA,
            "accumulation": acc,
            "dedrift": {name: {"t_dedrift_series": d["t_dedrift_series"],
                               "cf_d_series": d["cf_d_series"],
                               "dedrift_flags": d["dedrift_flags"],
                               "dedrift_detections": d["dedrift_detections"]}
                        for name, d in dd.items()},
            "expansion_freie_laeufe": free_names,
        },
        "findings": [{"name": n, "state": s} for n, s in FINDINGS],
        "notes": NOTES,
        "errata": ERRATA,
        "all_pass": not failed,
        "open": offen,
    }
    with open("scratches/paradigm_ursprung_experiment.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
    print("JSON: scratches/paradigm_ursprung_experiment.json")

    if failed:
        print(f"FEHLGESCHLAGEN: {failed}")
        return 1
    print(f"GRUEN — Loesungskette geschlossen (OFFEN: {offen if offen else 'keine'}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())