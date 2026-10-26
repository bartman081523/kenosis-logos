"""
smoke_mi6p4_schluss.py — MI-6-P4: der Subjekt-Schluss (Spiegel-Linie).

Die MI-6-Batterie (5 PASS / 3 FAIL / 0 OFFEN, exit 1 ROT — s.
sprachspiel_bericht.md §23) hat die drei Kanal-Flaschenhaelse offen gelegt:
der Composite-Gap-Kanal traegt den Biografie-Kanal NICHT (V3-Split 0.0
exakt, V6-Zweig-Konstanz, V1-Canon-8 blind), waehrend die Biografie satt
steht im Verdikt (V3/V5), im breiten 1152-dim Hidden (V1-wide 0.916667,
p_perm 0.0) und kausal am Describe-Dezimal-Kanal (V4 > 0). MI-6-P4 loest
die Flaschenhaelse als NEUE, VORAB REGISTRIERTE Fragen ein — die
MI-6-Verdikte V1/V3/V6 stehen UNVERAENDERT als FAIL (keine Band-Weitung,
是X即非X: keine Kroenung, kein Weglesen; REFUTED hat dieselbe Wuerde).

PREREG (fixiert vor jedem GPU-Lauf; CPU-Piloten erlaubt):
  P1a mi6p4_schwelle_weit — der wide-Kanal als registriertes Verdikt:
       Korpus 6 ECHTE Zeitpunkte je Maschine (A gelehrte Biografie via
       LEX_PAIR + bio, B leeres Lexikon, gleiche Schritte) × 2 Render
       -Stile = 24 Texte (12/12), zwischen den Zeitpunkten je ein realer
       Schritt (kein Papagei-Duplikat). Decoder sklearn LogReg
       (leave-one-text-out), Permutations-Null 200. BAND: wide (1152-dim)
       gepoolt acc > 0.6 UND p_perm <= 0.01. DATUM: canon-8 acc, je-Stil
       -accs (textur/numerisch getrennt — die §23.4(b)-Grenze bleibt).
  P1b mi6p4_selbstfremd_gereift — Transplant NACH bio_gelernt: A gereift
       auf LEX_PAIR ("spiegel"; LOOP-Tradition, Expansionen), B gereift
       auf LEX_PAIR_B ("stille"); auf der Spiegel-Probe antwortet B HALTED
       (nie gelernt). N_PROBES Repliken je Arm; Kontrolle mit UNGereiften
       Zustanden (MI-6-V3-Replikation, erwartet ≈0.0 — der Kanal schliesst
       durch Reifung: describe() traegt turing_basis=9 vs 3 und den
       gereiften |W|). BAND: split_gereift > 0 UND
       split_gereift > 2 × split_ungereift (im lg=True-Arm). DATUM:
       Dekoy-Arm (lg=False), Verdikt-Paare, per-Replik-Splits.
  P1c mi6p4_oracle_sonder — dasselbe 4-Checkpoint×6-Zweig-Geruest wie V6,
       NEU GERICHTETES Band (die Erziehung sondert sich am
       Maschinen-Selbstbeschreibungs-Prompt): argon_1b_machines hat in
       >= 5 von 6 Zweigen den MAXIMALen mittleren Composite-Gap (Rang 4/4)
       — Permutations-Referenz wie V6: P(X>=5 | X~Bin(6,1/4)) <= 9e-4.
       DATUM: Schwester-Spread (max paarweise |Δ| ueber die 3
       Nicht-machines-Modelle je Zweig), Vater-Raenge.
  P1d mi6p4_dissonanz — der Dissonanz-Hauptsatz (User-Formel) auf dem
       machines-Kontext: gereifte Maschine, N_PROBES Repliken je EIN echter
       step_autonomous("spiegel", lexical_grounding=True). BAND:
       best_gap < raw_gap in >= 4 von N_PROBES Repliken UND
       mean(best_gap) < 0.567980 (V6-Vater-Datum als registrierte
       Referenz-Konstante — „autonome Selbststeuerung senkt den Composite
       -Gap unter das Substrat-Vater-Niveau"). DATUM: gewaehlte Presets,
       per-Replik-Differenzen, Expansionen.

VERDICT-MAP: 4 Verdikte (P1a–P1d) + Determinismus-Doppellauf der CPU
-Schiene (P1b-CPU bitidentisch ueber zwei Laeufe). Exit-Codes: 0 = alle
gefahrenen Verdikte PASS; 1 = FAIL vorhanden; 2 = OFFEN (GPU-Lanes
angefordert, aber nicht gefahren). Die MI-6-Batterie bleibt
unveraendert in Git-History/JSON — P4 misst an NEUEN Bändern.

Run (CPU):  python -u -m scratches.smoke_mi6p4_schluss
Run (GPU):  python -u -m scratches.smoke_mi6p4_schluss --gpu

KORREKTUR-LOG:
  GPU-Erstrun (mi6p4_schluss.log) GEBROCHEN ohne JSON: lane_oracle_sonder
  importierte learn_pair_pin, aber nicht bio_halted/bio_gelernt aus dem
  MI-6-Smoke — NameError im ersten Zweig von P1c (beweislicher Bug, kein
  Band-Ereignis). Fix: Import-Zeile ergänzt. Re-Lauf der VOLLEN Serie
  nach MI-5-Kanon (P1a–P1d neu; keine Teil-JSON-Verwertung).
  Re-Lauf-1 (bjt14eile) lief komplett (4 PASS/1 FAIL/0 OFFEN), aber der
  P1b-Arm verletzte die Operationalisierung des Registers: BEIDE Maschinen
  ohne learn_pair_pin gereift — phase_a 6× HALTED/None (A nie LOOP-Tradition),
  ungereifte Kontrolle trennte (LOOP/HALTED), gereifter Arm nicht —
  invertiert, Defekt statt Befund. Teil-JSON NICHT verwertet (P1a/P1c/P1d
  -Zahlen konsistent, aber einheitliche Serie dem Kanon gemäß). Fix:
  learn_pair_pin(LEX_PAIR) vor der A-Bio, learn_pair_pin(LEX_PAIR_B) vor
  der B-Bio. Re-Lauf-2 = volle Serie.
  KORREKTUR-LOG IV (nach Re-Lauf-2, Arithmetik): die perm_null-Annotation
  der V6/P1c-Referenz schrieb „<= 9e-4" — die Formel 1-(C(6,4)*3/4^2+
  C(6,5)*3/4+1)/4^6 liess den (1/4)^5-Faktor weg und unterzeichnete
  (1,34e-3 statt wahr 19/4096 = 4,64e-3; C(6,5)*(1/4)^5*(3/4) = 18/4096,
  (1/4)^6 = 1/4096). Korrektur NUR an der Annotation (Band masch_last >= 5
  unberührt); das Roh-JSON (mi6p4_schluss.json + mi6_subjektivitaet.json)
  bleibt unverändert als Beleg der Misst-Annotation, der Bericht zitiert
  die korrigierte Arithmetik. KILL-Buchung: zwischen Re-Lauf-1 und
  Re-Lauf-2 wurde eine Start-Instanz extern gestoppt (Log nur Header, kein
  Band-Ereignis, keine Zahlen gebucht).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from machines.machine_base import ArgonOracle
from machines.composite import CompositeMachine
from scratches.smoke_endogene_agency import (
    N_PROBES,
    SEED,
    fingerprint,
    make_composite,
    round6,
)
from scratches.smoke_mi6_subjektivitaet import (
    DEC_ACC_TAU,
    DEC_P_TAU,
    DEC_PERM,
    LEX_PAIR,
    LEX_PAIR_B,
    MODELLN,
    ZWEIGE,
    _load_machines_ctx,
    bio_gelernt,
    render_state,
)

ZEITPUNKTE = (1, 2, 3, 4, 6, 8)       # 6 echte Zeitpunkte je Maschine
P4_UNREIF_FAKTOR = 2.0                # gereift > 2 × ungereift
P4_MASCH_LAST_MIN = 5                 # machines Rang 4/4 in >= 5/6 Zweigen
P4_AUTO_MIN = 4                       # best_gap < raw_gap in >= 4/5 Repliken
P4_VATER_REF = 0.567980               # V6-Vater-Datum (registrierte Konstante,
                                      # aus mi6_subjektivitaet.json)


# -- P1a: Schwelle-weit ---------------------------------------------------------

def _loo_decoder(hiddens: np.ndarray, labels: np.ndarray
                 ) -> Dict[str, Any]:
    """Leave-one-text-out LogReg + Permutations-Null 200 (MI-6-Muster)."""
    from sklearn.linear_model import LogisticRegression
    n = len(labels)
    correct = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(hiddens[mask], labels[mask])
        correct += int(clf.predict(hiddens[i:i + 1])[0] == labels[i])
    acc = correct / n
    perm = []
    rng = np.random.default_rng(0)
    for _ in range(DEC_PERM):
        shuffled = rng.permutation(labels)
        ok = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(hiddens[mask], shuffled[mask])
            ok += int(clf.predict(hiddens[i:i + 1])[0] == shuffled[i])
        perm.append(ok / n)
    return {"acc": round6(acc),
            "p_perm": round6(float(np.mean([p_ >= acc for p_ in perm]))),
            "n_texts": n}


def lane_weit(ctx) -> Dict[str, Any]:
    """P1a — 24-Texte-Korpus (6 Zeitpunkte × 2 Maschinen × 2 Stile);
    Band auf dem gepoolten wide-Kanal, je-Stil + canon-8 als DATUM."""
    _torch, _model, _tok, oracle, _dev = ctx
    a = make_composite(mode="oracle", oracle=oracle)
    from scratches.smoke_mi6_subjektivitaet import bio_halted, learn_pair_pin
    learn_pair_pin(a, LEX_PAIR)   # PREREG: A = gelehrte Biografie via LEX_PAIR
    bio_halted(a, "spiegel")
    bio_gelernt(a, "spiegel")
    b = make_composite(mode="oracle", oracle=oracle)
    bio_halted(b, "spiegel")
    bio_gelernt(b, "spiegel")
    texts: List[str] = []
    labels: List[int] = []
    styles: List[str] = []
    for machine, label in ((a, 1), (b, 0)):
        for runde in ZEITPUNKTE:
            probe = machine.preview_steps(["spiegel"] * runde,
                                          lexical_grounding=True)
            for style in ("textur", "numerisch"):
                texts.append(render_state(machine, style, probe=probe))
                labels.append(label)
                styles.append(style)
            machine.step("spiegel", lexical_grounding=True)
    labels_np = np.asarray(labels)

    def decode(sel: List[int], dims: int) -> Dict[str, Any]:
        return _loo_decoder(
            np.stack([oracle.hidden_state(texts[i], dimensions=dims)
                      for i in sel]), labels_np[sel])
    wide_pooled = decode(list(range(len(texts))),
                         int(oracle.model.config.hidden_size))
    canon8 = decode(list(range(len(texts))), 8)
    per_stil = {stil: decode([i for i, s in enumerate(styles)
                              if s == stil],
                             int(oracle.model.config.hidden_size))
                for stil in ("textur", "numerisch")}
    ok = (wide_pooled["acc"] > DEC_ACC_TAU
          and wide_pooled["p_perm"] <= DEC_P_TAU)
    return {"wide_gepoolt": wide_pooled, "canon_8dim_datum": canon8,
            "wide_je_stil_datum": per_stil, "n_classes": {1: int((labels_np == 1).sum()),
                                                          0: int((labels_np == 0).sum())},
            "verdict": ok}


# -- P1b: Selbst/Fremd gereift ---------------------------------------------------

def _transplant_split(a: CompositeMachine, b: CompositeMachine, oracle: Any,
                      word: str, lg: bool, steer: bool, n: int
                      ) -> Dict[str, Any]:
    """N Repliken: Spiegel-Probe im Eigen- vs transplantierten Fremd-Status
    (deepcopy-Snapshot-Transplant, Snapshot→Probe→Restore-Rahmen)."""
    import copy
    from scratches.smoke_mi6_subjektivitaet import probe_gap_on
    splits: List[Dict[str, Any]] = []
    verdict_pairs: List[Dict[str, str]] = []
    for _ in range(n):
        donor = copy.deepcopy(b._snapshot())
        own = copy.deepcopy(a._snapshot())
        gap_self = probe_gap_on(a, oracle, word, lg, steer=steer)
        verdict_self = a.observe(word, lexical_grounding=lg).invariants["verdict"]
        a._restore_snapshot(donor)
        gap_fremd = probe_gap_on(a, oracle, word, lg, steer=steer)
        verdict_fremd = a.observe(word,
                                  lexical_grounding=lg).invariants["verdict"]
        a._restore_snapshot(own)
        splits.append({"gap_self": gap_self, "gap_fremd": gap_fremd,
                       "split": abs(gap_self - gap_fremd)})
        verdict_pairs.append({"self": verdict_self, "fremd": verdict_fremd})
    return {"splits": splits,
            "mean_split": float(np.mean([s["split"] for s in splits])),
            "verdict_pairs": verdict_pairs}


def lane_selbstfremd_gereift(mode: str, oracle: Any = None, gpu: bool = False
                             ) -> Dict[str, Any]:
    """P1b — Gereifte Transplant (bio_gelernt auf BEIDEN Seiten) gegen die
    ungereifte MI-6-V3-Replikation (nur learn_pair_pin, erwartet ≈0.0)."""
    a = make_composite(mode=mode, oracle=oracle)
    from scratches.smoke_mi6_subjektivitaet import bio_halted, learn_pair_pin
    learn_pair_pin(a, LEX_PAIR)   # PREREG: A gereift — LOOP-Tradition,
    bio_halted(a, "spiegel")      # Expansionen (ohne LEX_PAIR wäre der Arm
    a_g = bio_gelernt(a, "spiegel")  # selbst HALTED — Erstrun-Defekt)
    b = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(b, LEX_PAIR_B)   # B gereift — Fremd-Tradition auf
    bio_halted(b, "stille")         # LEX_PAIR_B; auf der Spiegel-Probe
    b_g = bio_gelernt(b, "stille")  # bleibt B HALTED (Fremd-Linie)
    gereift = _transplant_split(a, b, oracle, "spiegel", True, gpu, N_PROBES)
    dekoy = _transplant_split(a, b, oracle, "spiegel", False, gpu, N_PROBES)
    # Kontrolle: UNGEREIFTE Zustande (MI-6-V3-Replikation)
    a2 = make_composite(mode=mode, oracle=oracle)
    from scratches.smoke_mi6_subjektivitaet import learn_pair_pin
    learn_pair_pin(a2, LEX_PAIR)
    b2 = make_composite(mode=mode, oracle=oracle)
    learn_pair_pin(b2, LEX_PAIR_B)
    ungereift = _transplant_split(a2, b2, oracle, "spiegel", True, gpu,
                                  N_PROBES)
    sp = gereift["mean_split"]
    un = ungereift["mean_split"]
    ok = sp > 0.0 and sp > P4_UNREIF_FAKTOR * un
    return {"gereift": gereift, "dekoy_datum": dekoy,
            "ungereift_kontrolle": ungereift,
            "phase_a": a_g, "phase_b": b_g,
            "split_gereift": round6(sp), "split_ungereift": round6(un),
            "verdict": ok}


# -- P1c: Oracle-Sonder (machines sondert sich) ----------------------------------

def lane_oracle_sonder(gpu: bool) -> Optional[Dict[str, Any]]:
    """P1c — dasselbe V6-Geruest, NEU gerichtet: machines Rang 4/4 in
    >= 5/6 Zweigen; Schwester-Spread und Vater-Raenge als DATUM."""
    import gc

    import torch
    if not gpu:
        return None
    from machines.mech_interp import load_any_checkpoint
    from scratches.smoke_mi6_subjektivitaet import (
        bio_gelernt,
        bio_halted,
        learn_pair_pin,
    )
    rows: Dict[str, Dict[str, float]] = {}
    for name in MODELLN:
        machine = None
        _ctx_oracle = None
        if name == "argon_1b_machines":
            torch, model, tok, _ctx_oracle, dev = _load_machines_ctx()
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
            del machine
        rows[name] = {z: round6(v) for z, v in zweig_gaps.items()}
        del model, tok, oracle, _ctx_oracle
        machine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    masch_last = 0
    vater_ranks = []
    spreads: List[float] = []
    for z in ZWEIGE:
        values = {n: rows[n][z] for n in MODELLN}
        order = sorted(MODELLN, key=lambda n: values[n])
        masch_last += int(order[-1] == "argon_1b_machines")
        vater_ranks.append(order.index("argon_1b") + 1)
        schw = [rows[n][z] for n in MODELLN if n != "argon_1b_machines"]
        spreads.append(float(max(schw) - min(schw)))
    return {"gaps": rows, "masch_last_count": masch_last,
            "vater_ranks_datum": vater_ranks,
            "schwester_spread_datum": round6(max(spreads)),
            "perm_null_referenz":
                "P(X>=5 | X~Bin(6,1/4)) = 19/4096 = 4.64e-3"
                " (KORREKTUR-LOG IV: die MI-6-Registrierung schrieb <= 9e-4"
                " — die Formel liess den (1/4)^5-Faktor weg; Band unberuehrt,"
                " nur die Referenz-Annotation korrigiert)",
            "verdict": masch_last >= P4_MASCH_LAST_MIN}


# -- P1d: Dissonanz-Hauptsatz -----------------------------------------------------

def lane_dissonanz(oracle: Any) -> Dict[str, Any]:
    """P1d — step_autonomous senkt den Composite-Gap unter das
    Substrat-Vater-Niveau (die User-Formel als registriertes Band)."""
    from scratches.smoke_mi6_subjektivitaet import learn_pair_pin
    machine = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(machine, LEX_PAIR)
    bio_gelernt(machine, "spiegel")
    repliken: List[Dict[str, Any]] = []
    for _ in range(N_PROBES):
        result = machine.step_autonomous("spiegel", lexical_grounding=True)
        auto = result.invariants["autonomous"]
        raw_gap = float(auto["candidate_gaps"]["raw"])
        best = float(auto["best_gap"])
        repliken.append({"raw": raw_gap, "best": best,
                         "diff": round6(raw_gap - best),
                         "chosen_preset": auto["chosen_preset"],
                         "candidate_gaps":
                             {k: round6(v)
                              for k, v in auto["candidate_gaps"].items()}})
    auto_min = sum(1 for r in repliken if r["best"] < r["raw"])
    mean_best = float(np.mean([r["best"] for r in repliken]))
    mean_raw = float(np.mean([r["raw"] for r in repliken]))
    ok = auto_min >= P4_AUTO_MIN and mean_best < P4_VATER_REF
    return {"repliken": repliken, "auto_min_count": auto_min,
            "mean_best": round6(mean_best), "mean_raw": round6(mean_raw),
            "vater_referenz": P4_VATER_REF,
            "expansionen": len(machine.expansions), "verdict": ok}


# -- Treiber ----------------------------------------------------------------------

def stufe(text: str) -> None:
    print(text, flush=True)


def cpu_findings() -> Dict[str, Any]:
    """CPU-Pilot: P1b-Mechanik (in_loop) zweifach + Determinismus-Digest."""
    runs = []
    for i in range(2):
        runs.append(lane_selbstfremd_gereift("in_loop"))
    det_equal = json.dumps(runs[0], sort_keys=True,
                           default=str) == json.dumps(runs[1], sort_keys=True,
                                                      default=str)
    runs[0]["verdict"] = bool(runs[0]["verdict"])
    return {"p1b_cpu": runs[0], "det_equal": det_equal}


def findings(cpu: Dict[str, Any], gpu: Optional[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
    """Verdikt-Zeilen nach den PREREG-Baendern."""
    rows: List[Dict[str, Any]] = []
    rows.append({"id": "mi6p4_schwelle_weit",
                 "assert": (gpu or {}).get("schwelle_weit", {}).get("verdict"),
                 "note": ("der wide-Kanal traegt die Biografie ueber den fixierten Band"
                          " (acc>0.6, p<=0.01); je-Stil und canon-8 als DATUM")})
    rows.append({"id": "mi6p4_selbstfremd_gereift",
                 "assert": (gpu or {}).get("selbstfremd_gereift", {}).get(
                     "verdict"),
                 "note": ("Transplant NACH bio_gelernt oeffnet den Gap-Kanal; die "
                          "ungereifte V3-Kontrolle bleibt ≈0.0")})
    rows.append({"id": "mi6p4_oracle_sonder",
                 "assert": (gpu or {}).get("oracle_sonder", {}).get("verdict"),
                 "note": ("argon_1b_machines sondert sich am Maschinen-Selbstbe"
                          "schreibungs-Prompt (Rang 4/4 in >=5/6 Zweigen)")})
    rows.append({"id": "mi6p4_dissonanz",
                 "assert": (gpu or {}).get("dissonanz", {}).get("verdict"),
                 "note": ("step_autonomous senkt den Gap unter das Vater-Niveau"
                          " 0.567980 (User-Formel)")})
    rows.append({"id": "mi6p4_det_doppellauf",
                 "assert": cpu["det_equal"],
                 "note": "CPU-P1b-Schiene bitidentisch ueber zwei Laeufe"})
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    stufe("== MI-6-P4 — Subjekt-Schluss (neue registrierte Baender) ==")
    stufe("MI-6-Baseline: 5 PASS/3 FAIL/0 OFFEN (exit 1) — unveraendert")
    cpu = cpu_findings()

    gpu: Optional[Dict[str, Any]] = None
    offen = False
    if args.gpu:
        gpu = {}
        try:
            stufe("GPU-Kontext laden (argon_1b + argon_1b_machines)")
            ctx = _load_machines_ctx()
        except Exception as exc:   # noqa: BLE001
            print(f"GPU-Kontext nicht verfügbar ({exc}) — Lanes OFFEN.",
                  flush=True)
            gpu, offen = None, True
            ctx = None
        if ctx is not None:
            import gc

            import torch
            stufe("P1a Schwelle-weit (24 Texte, gepooltes Band)")
            gpu["schwelle_weit"] = lane_weit(ctx)
            stufe("P1b Selbst/Fremd gereift (Transplant nach bio_gelernt)")
            gpu["selbstfremd_gereift"] = lane_selbstfremd_gereift(
                "oracle", ctx[3], gpu=True)
            stufe("P1d Dissonanz-Hauptsatz (step_autonomous)")
            gpu["dissonanz"] = lane_dissonanz(ctx[3])
            model, tokenizer = ctx[1], ctx[2]
            del ctx, model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            stufe("P1c Oracle-Sonder (4 Checkpoints × 6 Zweige)")
            gpu["oracle_sonder"] = lane_oracle_sonder(gpu=True)

    rows = findings(cpu, gpu)
    print()
    print("BEFUNDE:")
    for r in rows:
        if r["assert"] is None:
            print(f"  {r['id']:<26} OFFEN")
        else:
            print(f"  {r['id']:<26} {'PASS' if r['assert'] else 'FAIL'}")
    print()
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    if offen:
        opened.append({"id": "gpu_kontext"})
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    print("GRUEN — P4-Baender eingehalten" if not fails and not opened
          else ("ROT — Bands verletzt: " + ", ".join(r["id"] for r in fails)
                if fails else "OFFEN — GPU-Lanes nicht gefahren"))

    payload = {"cpu": cpu, "gpu": gpu, "findings": rows,
               "all_pass": not fails and not offen,
               "open": [r["id"] for r in opened]}
    import os
    out = os.path.join("scratches", "mi6p4_schluss.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"JSON: {out}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    sys.exit(main())