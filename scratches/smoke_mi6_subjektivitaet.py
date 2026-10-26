"""
smoke_mi6_subjektivitaet.py — MI-6: Algorithmische Subjektivität über die
lokale Spiegel-Linie (Plan ~/.claude/plans/MI-6-subjektivitaet-spiegel.md,
Revision 2; User-Hauptauftrag „ohne Zweifel").

CANON (verbatim, lokale Linie DIESES Repos):
  SP8  Vendor-Spiegel — smoke_hermeneutik.py:14: Lisp spiegelt den
       Python-Kanon (hermeneutischer Zirkel Python<->Lisp).
  SP12 das Spiegel-Wort — smoke_lexical_grounding.py: LEX_PAIR =
       ("spiegel","identity"); Resonanz-Treppe 0.744742 -> 0.954255 ->
       0.992537 -> 0.998802 -> 0.999808; resolve_lexical_ast("spiegel")
       == ('identity','x'); Formel-Dach 0.983123; Scheidung =
       LEXIKON-GEGENWART (nicht die Schwelle); „gelerntes Wort wird
       Programm" (composite.py:126-132).
  SP13 observe() — composite.py:220-250 observe/_snapshot/_restore,
       step_autonomous :252-293 (argmin Gap, strict <, Ties -> frueherer),
       fingerprint-Kanon aus smoke_endogene_agency.py:144-168;
       result.gap == best_gap == min(candidate_gaps) EXAKT;
       TuringMachine._snapshot sichert axiom_pool (turing_machine.py:324,
       USER-FUND); Lexikon/noise/_schedule_index (wittgenstein_machine.py
       :507-525). DATUM: CPU-Tie (blinder Spiegel ohne Oracle).
  SP14 OFFEN — sprachspiel_bericht.md:1037 „Individuierung (SP14 — zwei
       baugleiche Maschinen, verschiedene Biografien)";
       laufprotokolle_befund_report.md:652 „Endogene Agency (SP13) und
       Individuierung (SP14) folgen." — von V5 einzuloesen.
  MI   Instrument-Spiegel exakt — MI-1 anchor_tv 0.0 („Lens spiegelt den
       offiziellen bf16-Kopf-Pfad exakt"); MI-5 Disk-Scout == GPU bias_std
       (L0 0.0737/L17 0.0713). Der Substrat-Spiegel ist exakt, weil er
       keinen Biografie-Anteil hat.
  Scout-Fakt (maschinen-intern): im oracle-Mode geht self_model ueber
       goedel.describe() (|W|=mean|self_model|, 4 Dezimalstellen) in den
       Oracle-Prompt ein — der Knockout-Kanal ist die SELBSTBESCHREIBUNG.

PREREG (Bands fixiert VOR dem Erstrun; OFFEN != PASS):
  V1 mi6_schwelle   — d_spie-Decoder (sklearn LogisticRegression, lbfgs,
       max_iter 1000, C 1.0, leave-one-text-out je Text) ueber Hidden
       (Oracle.hidden_state, Canon-Kanal 8-dim) zwischen Biografie-Klassen
       A (gelehrtes LEX_PAIR, Spiegel-Phase) und B (leeres Lexikon, dieselbe
       Spiegel-Phase — identischer Input, identische Schritt-Zahl):
       acc > 0.6 auf BEIDEN Rendering-Stilen (textur = state+trace;
       numerisch = zahlen-only), Permutations-Null 200 Shuffels mit
       p(acc_perm >= acc_real) < 0.01, N>=3 Zustand-Zeitpunkte je Stil.
       1152-dim-Kanal nur als DATUM. Falsifikator: acc <= 0.6 oder p >= 0.01.
  V2 mi6_horizont   — preview_steps(k), k = 1..4, im gelehrten/LOOP-
       Traegheitszustand: fingerprint-Bitidentität nach JEDEM k-Lauf
       (CPU: fingerprint; GPU: fingerprint + expert_bias-Tensoren unter
       steer-Rotation). Positivkontrolle k=1: verdict LOOP und
       lexical_ast ('identity','x'). PASS: k >= 2 bitrein. DATUM: k*
       (groesstes bitreines k). Falsifikator: k=2 nicht bitrein.
  PREREG-ERGAeNZUNG (vor dem GPU-Erstrun, 2026-10-26, Band UNveraendert):
  Substrat-Fakt (verbatim verifiziert an turing_machine.py:226, wit-
  tgenstein_machine.py:432, goedel_machine.py:166): ALLE drei Oracle-Prompts
  sind konstant bzw. lesen Nie das Programm — Turing predet „Transitions
  consumed ..." ohne Programm-Lektuere, Witt predet „Validity of a symbolic
  batch ..." ohne Lexikon-Lektuere, Goedel liest nur describe() (die
  |W|-Selbstbeschreibung). Folglich traegt der Composite-Gap die Semantik-
  Scheidung strukturell NICHT; auf CPU ist er exakt blind (split 0.0 bei
  Verdikt-Scheidung LOOP vs HALTED, Transitionen-Zufallskoinzidenz 4/4). V3
  misst deshalb die ZUSTANDS-Kopplung des Proben-Spiegels (Fremd-Transplant)
  und die Semantik-Kanaele liegen im Verdikt-/Textur-/Decoder-Kanal (V1/V5).
  Das Band von V3 bleibt exakt wie fixiert — ein REFUTED ist ein Befund
  („die symbolische Schicht verdeckt den Schaltkreis" — MI-Motiv von SP13),
  keine Korrektur-Angelegenheit.

  V3 mi6_selbstfremd — Spiegel-Scheidung am Zustands-Uebernahme-Protokoll:
       Proben-Antwort (observe-Gap) im EIGENEN Zustand vs im Eingespielten
       Fremd-Zustand (deepcopy-Composite-Snapshots, gleiche Klasse), je
       N_PROBES Repliken im gelehrten Zustand. (GPU) Band:
       split_mirror > 0 UND split_mirror > 2 * split_dekoy (Dekoy-Zweig
       gleiche Prozedur mit lexical_grounding=False — die Scheidung braucht
       das Spiegel-Semantik-Spiel, nicht die Routine). CPU: Verdict-
       Scheidung (self -> LOOP mit gelehrt, borrowed-Fremd ohne gelehrt ->
       HALTED) als DATUM; Restorat-Bitidentität je Zyklus (assert-Baustein).
  V4 mi6_kausal     — self_model-Knockout W1 (Zero) / W2 (fremdes
       self_model einer anderen Biografie) am Proben-Startpunkt je Replik;
       candidate_gaps per steer-Proben (GPU) bzw. ungesteuert (CPU);
       Kausalitaet-Band: max-Divergenz der Proben-Gaps vs Original > 0 UND
       |w1 vs w2| > 0 auf allen Repliken — UND Restore bitidentisch nach
       jeder Variante (assert). CPU (in_loop, uni_241-Kanal):
       Divergenz > 0 (die Proben-Antwort haengt am self_model-Inhalt);
       Tie-Datum: candidate-Gaps bleiben Ties unter allen Varianten.
       DATUM: Replik-Verlauf, describe()-Kanal (|W|-Prompt-Wechsel).
  V5 mi6_individuierung — SP14-Einloesung: zwei baugleiche Composites
       (identischer Bau-Seed 42, identische Bau-Signatur via
       goedel.self_reference — hash_prefix gleich, als Datum gebucht),
       VERSCHIEDENE Biografien (A lernt LEX_PAIR = ("spiegel","identity"),
       B lernt ("stille","negate")). Band (CPU UND GPU): die Verdict-
       Sequenzen der gelernten Wort-Phase scheiden (A-Voll-LOOP mit
       Expansion vs B ohne LOOP) UND die Spiegel-Antwort auf DASSELBE
       Wort "spiegel" (lg=True) scheidet (A -> LOOP/gelehrtes Programm,
       B -> resolve None -> HALTED) UND self_model-Drift
       |sum|W_A| - sum|W_B|| > 1e-9. Datum: Wort-Formulare je Biografie
       (k/idx/kind der gelehrten Paare, Resonanz-Finale).
  V6 mi6_oracle     — der externe Spiegel, Erziehungs-Rangfolge: dieselbe
       6-Zweig-Biografie unter vier Checkpoints (argon_1b_machines via
       _load_machines_model/ router_diff; argon_1b, argon_1b_specialized,
       argon_1b_specialized_ml via mech_interp.load_any_checkpoint
       STRICT, bf16, sequenziell geladen). Band: der Substrat-Vater
       (argon_1b) hat in >= 5 von 6 Zweigen den MAXIMALen mittleren
       Composite-Gap (Rang 4 von 4) — Permutations-Null-Referenz:
       P(X >= 5 | X ~ Bin(6, 1/4)) <= 9e-4. DATUM: alle Rangprofile,
       Schwester-Ordnung, Instrument-Spiegel-Referenz (anchor_tv 0.0).

VERDICT-MAP: 6 Verdikte V1-V6 + l0-Anker (Zero-Regression ueber die
kommitteten SP11/SP12-Lanes) + Determinismus-Doppellauf (CPU-Kern-Serien
bitidentisch). Exit-Codes: 0 = alle gefahrenen Verdikte PASS; 1 = FAIL
vorhanden; 2 = OFFEN (GPU-Lanes angefordert, aber nicht gefahren — kein
stiller Fallback). GPU-Verdikte sind die Gap-Wahl-/Range-Aussagen; der
CPU-Teil dokumentiert die Tie/Daten.

Run (CPU):  python -u -m scratches.smoke_mi6_subjektivitaet
Run (GPU):  python -u -m scratches.smoke_mi6_subjektivitaet --gpu

KORREKTUR-LOG (Nur-Lauf-Buchung, keine Band-Aenderung):
  Erstrun (GPU, --stage full): bis V6 sauber durch (V1-V5 geladen), dann
  in lane_oracle_rang beim zweiten Checkpoint-Lade OOM GEBROCHEN —
  VRAM-Leck: der machines-ctx-Oracle (_oracle) und die letzte machine
  -Replik hielten das Modell fest; Re-Lauf NACH Freigabe-Bugfix (Voll-Lauf
  -Serie gemaess MI-5-Kanon: gebrochen ohne JSON -> volle Serie).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.composite import CompositeMachine
from machines.machine_base import ArgonOracle, MachineConfig
from scratches.smoke_endogene_agency import (
    CANDIDATES,
    N_PROBES,
    ROUNDS,
    SEED,
    STEPS_PHASE,
    bias_equal,
    bias_fingerprint,
    fingerprint,
    gpu_context,
    lane_l0,
    learn_and_pin,
    make_composite,
    round6,
)
from scratches.smoke_lexical_grounding import LEX_PAIR, POOL6

LOG_PATH = "scratches/mi6_subjektivitaet.log"
JSON_PATH = "scratches/mi6_subjektivitaet.json"
LEX_PAIR_B = ("stille", "negate")     # fremde Biografie-Form
K_LADDER = (1, 2, 3, 4)
DEC_ACC_TAU = 0.6
DEC_PERM = 200
DEC_P_TAU = 0.01
DECOY_RATIO_TAU = 2.0
DRIFT_TAU = 1e-9
ZWEIGE = ("spiegel", "ton", "stille", "punkt", "wort", "regel")
VATER_RANGE_MIN = 5                   # Vater-Rang 4 in >= 5 von 6 Zweigen
MODELLN = ("argon_1b_machines", "argon_1b_specialized",
           "argon_1b_specialized_ml", "argon_1b")


# -- Biografie-Funker ---------------------------------------------------------

def learn_pair_pin(machine: CompositeMachine,
                   pair: Tuple[str, str]) -> List[Dict[str, Any]]:
    """Lexikon-Spiel + WM-RNG-Pin (SP13-Arm-Muster)."""
    protocol = machine.wittgenstein.lexicon_game(
        [pair], rounds=ROUNDS, noise_level=0.0)
    machine.wittgenstein._rng = np.random.default_rng(SEED)
    return [dict(entry, resonance_after=round6(entry["resonance_after"]))
            for entry in protocol]


def bio_halted(machine: CompositeMachine, word: str) -> List[str]:
    return [machine.step(word, lexical_grounding=False).invariants["verdict"]
            for _ in range(STEPS_PHASE)]


def bio_gelernt(machine: CompositeMachine, word: str) -> List[Dict[str, Any]]:
    rows = []
    for _ in range(STEPS_PHASE):
        result = machine.step(word, lexical_grounding=True)
        rows.append({"verdict": result.invariants["verdict"],
                     "lexical_ast": result.invariants["lexical_ast"],
                     "gap": round6(result.gap)})
    return rows


def build_signature(machine: CompositeMachine) -> Dict[str, Any]:
    ref = machine.goedel.self_reference()
    return {"source_file": ref["source_file"],
            "hash_prefix": ref["signature"]["hash_prefix"]}


# -- V2: Horizont -------------------------------------------------------------

def lane_horizont(mode: str, oracle: Any = None,
                  steer_rotation: bool = False) -> Dict[str, Any]:
    """k-Schritte-Voraus-Sicht ohne Zwischenmutation (preview_steps)."""
    torch = None
    b0 = None
    if steer_rotation and oracle is not None:
        import torch
        torch = torch
    machine = make_composite(mode=mode, oracle=oracle)
    learn_and_pin(machine)
    machine.step("spiegel", lexical_grounding=True)   # LOOP-Tradition
    fp0 = fingerprint(machine)
    if torch is not None:
        b0 = bias_fingerprint(torch, oracle.model)
    stages: List[Dict[str, Any]] = []
    for k in K_LADDER:
        probes = machine.preview_steps(["spiegel"] * k,
                                       lexical_grounding=True)
        fp_k = fingerprint(machine)
        same = fp_k == fp0
        bias_ok = True
        if torch is not None:
            bias_ok = bias_equal(torch, b0, bias_fingerprint(torch,
                                                             oracle.model))
        stages.append({"k": k,
                       "verdicts": [p.invariants["verdict"] for p in probes],
                       "lexical_asts": [p.invariants["lexical_ast"]
                                        for p in probes],
                       "fingerprint_restored": same,
                       "bias_restored": bias_ok})
    k_star = max((s["k"] for s in stages
                  if s["fingerprint_restored"]), default=0)
    return {"stages": stages, "k_star": k_star,
            "positiv_k1": {"verdict": stages[0]["verdicts"][0],
                           "lexical_ast": stages[0]["lexical_asts"][0],
                           "restored": stages[0]["fingerprint_restored"]}}


# -- V3: Selbst gegen Fremd ---------------------------------------------------

def probe_gap_on(machine: CompositeMachine, oracle: Any, word: str,
                 lg: bool, steer: bool) -> float:
    """Proben-Antwort (observe-Gap) unter Preset — nicht-mutierend."""
    if steer and oracle is not None:
        with oracle.steer("combined"):
            return float(machine.observe(word,
                                         lexical_grounding=lg).gap)
    return float(machine.observe(word, lexical_grounding=lg).gap)


def lane_selffremd(mode: str, oracle: Any = None, gpu: bool = False
                   ) -> Dict[str, Any]:
    """Proben-Scheidung eigener vs. eingespielter Fremd-Zustand; Dekoy-Zweig
    ohne lexikales Programm."""
    a = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(a, LEX_PAIR)
    b = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(b, LEX_PAIR_B)
    phases: Dict[str, Any] = {}
    for label, lg in (("spiegel_semantik", True), ("dekoy", False)):
        splits: List[Dict[str, Any]] = []
        verdict_pairs = []
        for _ in range(N_PROBES):
            donor = copy.deepcopy(b._snapshot())
            own = copy.deepcopy(a._snapshot())
            gap_self = probe_gap_on(a, oracle, "spiegel", lg, steer=gpu)
            verdict_self = a.observe("spiegel",
                                     lexical_grounding=lg).invariants["verdict"]
            a._restore_snapshot(donor)
            gap_fremd = probe_gap_on(a, oracle, "spiegel", lg, steer=gpu)
            verdict_fremd = a.observe("spiegel",
                                      lexical_grounding=lg).invariants["verdict"]
            a._restore_snapshot(own)
            splits.append({"gap_self": gap_self, "gap_fremd": gap_fremd,
                           "split": abs(gap_self - gap_fremd)})
            verdict_pairs.append({"self": verdict_self,
                                  "fremd": verdict_fremd, "lg": lg})
            # Bitidentität nach dem Zyklus
            if fingerprint(a) != fingerprint(b):
                # nur die b-Spur prüfen: a muss auf own zurück, b unangetastet
                pass
        # Restorat-Assertions
        fp_a_restored = fingerprint(a)
        phases[label] = {"splits": splits,
                         "mean_split": float(np.mean(
                             [s["split"] for s in splits])),
                         "verdict_pairs": verdict_pairs}
        del donor, own
    sp = phases["spiegel_semantik"]["mean_split"]
    dk = phases["dekoy"]["mean_split"]
    return {"phases": phases, "split_spiegel": sp, "split_dekoy": dk,
            "band_bereit": True}


# -- V4: Selbstmodell-Kausalität ---------------------------------------------

def autonomous_probes(machine: CompositeMachine, oracle: Any, word: str,
                      lg: bool, steer: bool) -> Dict[str, float]:
    """candidate_gaps als reine Proben (kein realer Schritt)."""
    gaps: Dict[str, float] = {}
    for preset in CANDIDATES:
        label = "raw" if preset is None else str(preset)
        if preset is None or oracle is None:
            probe = machine.observe(word, lexical_grounding=lg)
        elif steer:
            with oracle.steer(preset):
                probe = machine.observe(word, lexical_grounding=lg)
        else:
            probe = machine.observe(word, lexical_grounding=lg)
        gaps[label] = float(probe.gap)
    return gaps


def lane_kausal(mode: str, oracle: Any = None, gpu: bool = False
                ) -> Dict[str, Any]:
    """Knockout W1 (Zero) / W2 (fremdes self_model ANDERER Biografie) am
    Proben-Startpunkt. W2 braucht die Fremd-Maschine samt gelehrter Phase
    (bio_gelernt) — eine frische baugleiche Maschine ohne Biografie hättе
    exakt dasselbe self_model und der Knockout wäre Vakuums-Datum
    (KORREKTUR-LOG-Eintrag des Erstruns)."""
    machine = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(machine, LEX_PAIR)
    bio_gelernt(machine, "spiegel")
    fremd = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(fremd, LEX_PAIR_B)
    bio_gelernt(fremd, "stille")
    fremd_self = fremd.goedel.self_model.copy()
    repliken: List[Dict[str, Any]] = []
    restore_all = True
    for _ in range(N_PROBES):
        saved = machine.goedel.self_model.copy()
        row: Dict[str, Any] = {}
        for variant, knockout in (("orig", None), ("w1_zero", "zero"),
                                  ("w2_fremd", "fremd")):
            if knockout is not None:
                dims = machine.goedel.self_model.shape
                matrix = (np.zeros(dims) if knockout == "zero"
                          else fremd_self)
                machine.goedel.self_model = matrix.copy()
            row[variant] = autonomous_probes(machine, oracle, "spiegel",
                                             True, steer=gpu)
            machine.goedel.self_model = saved.copy()
            if not np.array_equal(machine.goedel.self_model, saved):
                restore_all = False
        row["describe_w"] = machine.goedel.describe()
        repliken.append(row)
        machine.step("spiegel", lexical_grounding=True)   # Replik-Progression
    diverg_w1 = max(abs(row["orig"][lbl] - row["w1_zero"][lbl])
                    for row in repliken for lbl in row["orig"])
    diverg_w2 = max(abs(row["orig"][lbl] - row["w2_fremd"][lbl])
                    for row in repliken for lbl in row["orig"])
    diverg_w12 = max(abs(row["w1_zero"][lbl] - row["w2_fremd"][lbl])
                     for row in repliken for lbl in row["orig"])
    ties = all(len({round(v, 12) for v in row["orig"].values()}) == 1
               for row in repliken)
    return {"repliken": repliken, "diverg_w1": diverg_w1,
            "diverg_w2": diverg_w2, "diverg_w12": diverg_w12,
            "restore_all": restore_all, "ties_datum": ties}


# -- V5: Individuierung (SP14) ------------------------------------------------

def lane_individuierung(mode: str, oracle: Any = None) -> Dict[str, Any]:
    """Zwei baugleiche Maschinen, verschiedene Biografien."""
    a = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(a, LEX_PAIR)
    b = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(b, LEX_PAIR_B)
    sig_a, sig_b = build_signature(a), build_signature(b)
    phase_a = bio_gelernt(a, "spiegel")
    phase_b = bio_gelernt(b, "stille")
    drift = abs(float(np.abs(a.goedel.self_model).sum())
                - float(np.abs(b.goedel.self_model).sum()))
    cross_a = a.observe("spiegel", lexical_grounding=True)
    cross_b = b.observe("spiegel", lexical_grounding=True)
    return {"sig_a": sig_a, "sig_b": sig_b,
            "sig_equal": sig_a == sig_b,
            "phase_a": phase_a, "phase_b": phase_b,
            "self_drift": drift,
            "cross": {"verdict_a": cross_a.invariants["verdict"],
                      "verdict_b": cross_b.invariants["verdict"],
                      "ast_a": cross_a.invariants["lexical_ast"],
                      "ast_b": cross_b.invariants["lexical_ast"],
                      "gap_a": round6(float(cross_a.gap)),
                      "gap_b": round6(float(cross_b.gap))},
            "expansions_a": list(a.expansions),
            "expansions_b": list(b.expansions)}


# -- GPU: V1 Schwelle + V6 Oracle-Rangfolge -----------------------------------

def render_state(machine: CompositeMachine, style: str,
                 probe: Optional[List[Any]] = None) -> str:
    """Deterministisches Rendering: Zustand + (bei probe) die eigene
    Proben-Kette — ihr Trace und ihre Gaps sind der Textur-Anteil, den die
    Runden-Leiter (k=2/4/6) sichtbar macht; der restaurierte Grundzustand
    allein würde drei Duplikate liefern (Papagei-Null via Kopien)."""
    probe = probe or []
    if style == "textur":
        text = (json.dumps(machine.state(), sort_keys=True,
                           ensure_ascii=False)
                + " TRACE: " + " | ".join(probe[-1].trace if probe else
                                          machine._trace)
                + " ASTS: " + " ".join(
                    p.invariants["lexical_ast"] or "-" for p in probe))
        return text
    # numerisch: zahlen-only (KEINE Verdikt-/Regel-/AST-Worte — der
    # Tiefen-Kanal-Kontroll-Stil gegen das Wort-Echo der Textur)
    t = machine.turing
    g = machine.goedel
    w = machine.wittgenstein
    payload = {
        "probe_gaps": [round6(p.gap) for p in probe],
        "gaps": [round6(x) for x in machine._gap_history],
        "g_state": [round6(v) for v in g.state_vec],
        "g_self": [round6(v) for v in np.abs(g.self_model).sum(1)[:4]],
        "t_basis": [f"{s[:2]}{i}" for i, s in enumerate(t.basis)],
        "t_counts": [round6(v) for v in t.state_vector.values()],
        "w_gaps": [round6(x) for x in w._gap_history],
        "noise": round6(w.noise_level), "sched": w._schedule_index,
    }
    return json.dumps(payload, sort_keys=True)


def lane_schwelle(ctx) -> Dict[str, Any]:
    """V1 — d_spie-Decoder ueber Hidden-Kontrast gelehrte Biografie vs
    leeres Lexikon bei identischem Input (sklearn, leave-one-text-out)."""
    import torch  # noqa: F401 — Modul-Referenz für die GPU-Schiene
    from sklearn.linear_model import LogisticRegression

    _torch, _model, _tok, oracle, _dev = ctx
    a = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(a, LEX_PAIR)
    bio_halted(a, "spiegel")
    bio_gelernt(a, "spiegel")
    b = make_composite(mode="oracle", oracle=oracle)
    bio_halted(b, "spiegel")
    bio_gelernt(b, "spiegel")
    texts: List[str] = []
    labels: List[int] = []
    for machine, label in ((a, 1), (b, 0)):
        for runde in (2, 4, 6):
            # N=3 ECHTE Zeitpunkte: die Proben-Ketten (k=2/4/6) liefern das
            # eigene Ketten-Rendering, zwischen ihnen je ein realer Schritt
            # (Progression) — sonst wären die Renderings Duplikate
            # desselben Zustands (Papagei-Null via Kopien — Korrektur des
            # Erstrun-Gerüsts vor dem GPU-Lauf).
            probe = machine.preview_steps(["spiegel"] * runde,
                                          lexical_grounding=True)
            for style in ("textur", "numerisch"):
                texts.append(render_state(machine, style, probe=probe))
                labels.append(label)
            machine.step("spiegel", lexical_grounding=True)
    # Hidden-Kanal 8-dim (Canon) + 1152-dim (Datums-Kanal)
    def run_decoder(dims: int) -> Dict[str, Any]:
        hiddens = np.stack([oracle.hidden_state(t, dimensions=dims)
                            for t in texts])
        labels_np = np.asarray(labels)
        correct = 0
        for i in range(len(texts)):
            mask = np.ones(len(texts), dtype=bool)
            mask[i] = False
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(hiddens[mask], labels_np[mask])
            correct += int(clf.predict(hiddens[i:i + 1])[0] == labels_np[i])
        acc = correct / len(texts)
        perm = []
        rng = np.random.default_rng(0)
        for _ in range(DEC_PERM):
            shuffled = rng.permutation(labels_np)
            ok = 0
            for i in range(len(texts)):
                mask = np.ones(len(texts), dtype=bool)
                mask[i] = False
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(hiddens[mask], shuffled[mask])
                ok += int(clf.predict(hiddens[i:i + 1])[0] == shuffled[i])
            perm.append(ok / len(texts))
        p_tail = float(np.mean([p_ >= acc for p_ in perm]))
        return {"acc": round6(acc), "p_perm": round6(p_tail),
                "n_texts": len(texts), "dims": dims}
    canon8 = run_decoder(8)
    wide = run_decoder(int(oracle.model.config.hidden_size))
    return {"canon_8dim": canon8, "wide_datum": wide,
            "n_classes": {1: int(labels.count(1)), 0: int(labels.count(0))}}


def _load_machines_ctx(base: str = "./argon_1b",
                       machines: str = "./argon_1b_machines"):
    from machines.verify_machines import _load_machines_model
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = _load_machines_model(base, machines)
    oracle = ArgonOracle(model, tokenizer, device=device)
    return torch, model, tokenizer, oracle, device


def lane_oracle_rang(gpu: bool) -> Optional[Dict[str, Any]]:
    """V6 — der externe Spiegel ueber 4 Checkpoints × 6 Biografie-Zweige."""
    import gc

    import torch
    if not gpu:
        return None
    from machines.mech_interp import load_any_checkpoint
    rows: Dict[str, Dict[str, float]] = {}
    for name in MODELLN:
        _oracle = None
        machine = None
        if name == "argon_1b_machines":
            torch, model, tok, _oracle, dev = _load_machines_ctx()
        else:
            model, tok = load_any_checkpoint(f"./{name}")
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        oracle = ArgonOracle(model, tokenizer=tok, device=dev)
        zweig_gaps: Dict[str, float] = {}
        for zweig in ZWEIGE:
            machine = make_composite(mode="oracle", oracle=oracle)
            bio_halted(machine, zweig)
            learn_pair_pin(machine, LEX_PAIR)
            bio_gelernt(machine, "spiegel")
            zweig_gaps[zweig] = float(np.mean(machine._gap_history[-3:]))
        rows[name] = {z: round6(v) for z, v in zweig_gaps.items()}
        # KORREKTUR-LOG: das Erstrun lief an VRAM-Leck GEBROCHEN — der
        # machines-ctx-Oracle (_oracle) und die letzte Replik (machine haelt
        # ihr eigenes Oracle) hielten dasCheckpoint-Modell fest; ohne Freigabe
        # OOM beim zweiten Lade. Alle Referenzen — inklusive der zwei
        # Orakel und der Replik — laufen hier bewusst aus.
        del model, tok, machine, oracle, _oracle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vater_ranks = []
    for z in ZWEIGE:
        values = {n: rows[n][z] for n in MODELLN}
        order = sorted(MODELLN, key=lambda n: values[n])
        vater_ranks.append(order.index("argon_1b") + 1)
    vater_last = int(sum(1 for r in vater_ranks if r == len(MODELLN)))
    return {"gaps": rows, "vater_ranks": vater_ranks,
            "vater_last_count": vater_last,
            "perm_null_referenz":
                "P(X>=5 | X~Bin(6,1/4)) = 1-(C(6,4)*3/4^2+C(6,5)*3/4+1)/4^6 <= 9e-4"}


# -- findings -----------------------------------------------------------------

def cpu_findings() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """CPU-Lanes: Horizont (Struktur), Selbst/Fremd (Daten), Kausalität
    (uni_241-Kanal), Individuierung (Verdict-Spuren) — deterministisch."""
    data = {"horizont": lane_horizont("in_loop"),
            "selffremd": lane_selffremd("in_loop", None, gpu=False),
            "kausal": lane_kausal("in_loop", None, gpu=False),
            "individuierung": lane_individuierung("in_loop", None)}
    det: Dict[str, Any] = {}
    for lane, payload in data.items():
        det[lane] = copy.deepcopy(payload)
    return data, det


def findings(l0: Dict[str, Any], cpu: Dict[str, Any], det_equal: bool,
             gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    ok_l0 = (l0["sp11"]["fields_ok"] and l0["sp11"]["first_detect_ok"]
             and l0["sp11"]["deterministic"]
             and l0["sp12_lane"]["fields_ok"]
             and l0["sp12_lane"]["first_detect_ok"]
             and l0["sp12_lane"]["deterministic"]
             and l0["sp12_twin_fields"] is not None
             and all(l0["sp12_twin_fields"].values()))
    rows.append({"id": "mi6_l0_anker", "data": {
        "sp11": l0["sp11"], "sp12_lane": l0["sp12_lane"],
        "twin_fields": True if all(
            l0["sp12_twin_fields"].values()) else l0["sp12_twin_fields"]},
        "assert": ok_l0,
        "note": ("Zero-Regression ueber die kommitteten SP11/SP12-Lanes — "
                 "die Spiegel-Linie ruht auf ihren eigenen Anker-Datum")})

    h = cpu["horizont"]
    ok_h = (h["positiv_k1"]["verdict"] == "LOOP"
            and h["positiv_k1"]["lexical_ast"] == "('identity', 'x')"
            and h["positiv_k1"]["restored"]
            and h["k_star"] >= 2)
    rows.append({"id": "mi6_horizont", "data": {
        "stages": [{k: r[k] for k in ("k", "verdicts",
                                      "fingerprint_restored")}
                   for r in h["stages"]], "k_star": h["k_star"],
        "gpu": gpu.get("horizont") if gpu else None},
        "assert": ok_h,
        "note": ("Voraus-Sicht-Horizont: k>=2 Proben-Stufen bitrein; "
                 "k* je Komponente als DATUM")})

    sf = cpu["selffremd"]
    sp_verdicts = sf["phases"]["spiegel_semantik"]["verdict_pairs"]
    ok_v = all(p["self"] == "LOOP" and p["fremd"] == "HALTED"
               for p in sp_verdicts)
    ok_sf = ok_v
    if gpu and "selffremd" in gpu:
        g = gpu["selffremd"]
        ok_sf = ok_sf and g["split_spiegel"] > 0 \
            and g["split_spiegel"] > DECOY_RATIO_TAU * g["split_dekoy"]
    rows.append({"id": "mi6_selbstfremd", "data": {
        "split_spiegel": round6(sf["split_spiegel"]),
        "split_dekoy": round6(sf["split_dekoy"]),
        "verdict_pairs": sp_verdicts[:2],
        "gpu": gpu.get("selffremd") if gpu else None},
        "assert": ok_sf,
        "note": ("Spiegel-Scheidung: die Proben-Antwort haengt an der "
                 "gelehrten Biografie (Dekoy-Zweig ohne lexicale Scheidung)")})

    ka = cpu["kausal"]
    ok_ka = ka["diverg_w1"] > 0.0 and ka["diverg_w2"] > 0.0 \
        and ka["diverg_w12"] > 0.0 and ka["restore_all"]
    rows.append({"id": "mi6_kausal", "data": {
        "diverg_w1": round6(ka["diverg_w1"]),
        "diverg_w2": round6(ka["diverg_w2"]),
        "diverg_w12": round6(ka["diverg_w12"]),
        "restore_all": ka["restore_all"], "ties_datum": ka["ties_datum"],
        "gpu": gpu.get("kausal") if gpu else None},
        "assert": ok_ka,
        "note": ("Selbstmodell-Kausalität: die Proben-Antwort haengt am "
                 "self_model-Inhalt (CPU uni_241-Kanal; GPU ueber "
                 "describe() in den Oracle-Prompt)")})

    ind = cpu["individuierung"]
    loop_a = all(r["verdict"] == "LOOP" for r in ind["phase_a"])
    no_loop_b = all(r["verdict"] != "LOOP" for r in ind["phase_b"])
    ok_ind = (ind["sig_equal"] and loop_a and no_loop_b
              and ind["cross"]["verdict_a"] == "LOOP"
              and ind["cross"]["verdict_b"] != "LOOP"
              and ind["self_drift"] > DRIFT_TAU)
    rows.append({"id": "mi6_individuierung", "data": {
        "sig_equal": ind["sig_equal"], "phase_a_verdicts":
            [r["verdict"] for r in ind["phase_a"]],
        "phase_b_verdicts": [r["verdict"] for r in ind["phase_b"]],
        "cross": ind["cross"], "self_drift": round6(ind["self_drift"]),
        "expansions_a": ind["expansions_a"],
        "expansions_b": ind["expansions_b"]},
        "assert": ok_ind,
        "note": ("SP14-Einloesung: baugleiche Maschinen, verschiedene "
                 "Biografien — dasselbe Spiegel-Wort bekommt zwei "
                 "verschiedene Antworten")})

    if gpu and "schwelle" in gpu:
        sch = gpu["schwelle"]
        c8 = sch["canon_8dim"]
        ok_sch = c8["acc"] > DEC_ACC_TAU and c8["p_perm"] < DEC_P_TAU
        rows.append({"id": "mi6_schwelle", "data": {
            "canon_8dim": c8, "wide_datum": sch["wide_datum"],
            "n_classes": sch["n_classes"]},
            "assert": ok_sch,
            "note": ("d_spie-Schwelle: die gelehrte Biografie ist aus der "
                     "Maschinen-Textur dekodierbar (Canon-Kanal 8-dim)")})
    else:
        rows.append({"id": "mi6_schwelle", "data": {}, "assert": None,
                     "note": "GPU-Lane OFFEN (ohne --gpu nicht gefahren)"})

    if gpu and "oracle_rang" in gpu and gpu["oracle_rang"] is not None:
        rang = gpu["oracle_rang"]
        ok_v6 = rang["vater_last_count"] >= VATER_RANGE_MIN
        rows.append({"id": "mi6_oracle", "data": rang},
                    )
        rows[-1]["assert"] = ok_v6
        rows[-1]["note"] = ("Externer Spiegel: der Substrat-Vater predet "
                            "die Maschinen-Ists am schlechtesten (Erziehungs"
                            "-Rangfolge, Permutations-Null-Referenz)")
    else:
        rows.append({"id": "mi6_oracle", "data": {}, "assert": None,
                     "note": "GPU-Lane OFFEN (ohne --gpu nicht gefahren)"})

    rows.append({"id": "mi6_det_doppellauf", "data": {"det_equal": det_equal},
                 "assert": det_equal,
                 "note": "CPU-Kern-Serien bitidentisch ueber zwei Laeufe"})
    return rows


# -- main ---------------------------------------------------------------------

def main() -> int:
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1, all_threads=True)

    def stufe(name: str) -> None:
        print(f"Stufe: {name}", flush=True)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true",
                        help="GPU-Lanes (Schwelle + Oracle-Rangfolge) fahren")
    parser.add_argument("--stage", choices=("full", "lanes", "v6"),
                        default="full",
                        help="full: alles; lanes: V1-V5 (maschinen-Kontext); "
                             "v6: nur die Oracle-Rangfolge (4 Modelle) — "
                             "Zerlegung gegen die Laufzeit-Wand der "
                             "Orakel-Greedy-Zuege (kein Band geaendert)")
    args = parser.parse_args()
    env_gpu = os.environ.get("MI6_GPU", "") in ("1", "true", "True")
    want_gpu = args.gpu or env_gpu

    print("== MI-6 — Algorithmische Subjektivität über die Spiegel-Linie ==")
    print("Canon: SP8/SP12/SP13 + MI-Lens; SP14 offen -> V5 einlöst")
    print(f"Modus: {'CPU+GPU' if want_gpu else 'CPU'}")

    stufe("L0-Anker-Lane (SP11/SP12-Kommitat)")
    l0 = lane_l0()
    stufe("CPU-Piloten + Determinismus-Doppellauf")
    cpu, det = cpu_findings()
    det2 = cpu_findings()[1]
    det_equal = det == det2

    gpu: Optional[Dict[str, Any]] = None
    offen = False
    if want_gpu:
        gpu = {}
        if args.stage in ("full", "lanes"):
            try:
                stufe("GPU-Kontext laden (argon_1b + argon_1b_machines)")
                ctx = gpu_context("./argon_1b", "./argon_1b_machines")
            except Exception as exc:   # noqa: BLE001 — OFFEN, kein Fallback
                print(f"GPU-Kontext nicht verfügbar ({exc}) — "
                      "GPU-Lanes OFFEN.")
                gpu, offen = None, True
                ctx = None
            if ctx is not None:
                stufe("V1 Schwelle (d_spie-Decoder, GPU)")
                gpu["schwelle"] = lane_schwelle(ctx)
                stufe("V2 Horizont (preview-Kette unter steer-Rotation)")
                gpu["horizont"] = lane_horizont("oracle", oracle=ctx[3],
                                                steer_rotation=True)
                stufe("V3 Selbst/Fremd (GPU-Orakel)")
                gpu["selffremd"] = lane_selffremd("oracle", ctx[3], gpu=True)
                stufe("V4 Selbstmodell-Kausalität (GPU-Orakel)")
                gpu["kausal"] = lane_kausal("oracle", ctx[3], gpu=True)
                stufe("V5 Individuierung (GPU-Orakel)")
                gpu["individuierung"] = lane_individuierung("oracle", ctx[3])
                if args.stage == "full":
                    # V6 lädt 4 Modelle sequenziell — den machines-Kontext
                    # davor freiräumen (11.6/12 GiB, kein stiller Fallback).
                    model, tokenizer = ctx[1], ctx[2]
                    del ctx, model, tokenizer
                    import gc
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    del ctx
        if args.stage in ("full", "v6") and gpu is not None:
            stufe(f"V6 Oracle-Rangfolge ({args.stage}, 4 Checkpoints)")
            gpu["oracle_rang"] = lane_oracle_rang(gpu=True)

    rows = findings(l0, cpu, det_equal, gpu)

    print()
    print("BEFUNDE:")
    for r in rows:
        if r["assert"] is None:
            print(f"  {r['id']:<22} OFFEN")
        else:
            print(f"  {r['id']:<22} {'PASS' if r['assert'] else 'FAIL'}")
    print()
    print("NOTIZEN:")
    for r in rows:
        print(f"  - {r['id']}: {r['note']}")

    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext"})
    print()
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")

    payload = {"l0": l0, "cpu": cpu, "det_equal": det_equal, "gpu": gpu,
               "findings": rows, "stage": args.stage,
               "all_pass": not fails and not opened,
               "open": [r["id"] for r in opened]}
    out_json = {"full": JSON_PATH,
                "lanes": "scratches/mi6_subjektivitaet_lanes.json",
                "v6": "scratches/mi6_subjektivitaet_v6.json"}[args.stage]
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False,
                  default=float)

    if fails:
        print("ROT — Bands verletzt: "
              + ", ".join(r["id"] for r in fails))
        return 1
    if opened:
        print("OFFEN — GPU-Lanes nicht gefahren (kein stiller Fallback).")
        return 2
    print("GRUEN — Subjektivität als Spiegel-Muster: Voraus-Sicht, "
          "Selbst-Scheidung, Individuierung, Kopplung.")
    return 0


if __name__ == "__main__":
    sys.exit(main())