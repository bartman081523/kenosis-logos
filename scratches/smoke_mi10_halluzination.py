"""
smoke_mi10_halluzination.py — MI-10 (REV-4): Provenienz-Audit der
Halluzination.

DEFINITION (Nutzer-Vorgabe, wörtlich operationalisiert):
  Halluzination ist, was durchs TRAINING nicht gerechtfertigt ist UND was
  der PROMPT (und die Projektion des Prompts auf das Training) nicht
  hergibt.

  Das ist eine PROVENIENZ-Aussage, keine Wahrheits-Aussage: ein memorierter
  Irrtum ist „durchs Training gerechtfertigt" und fällt NICHT unter diese
  Definition. Die Metrik behauptet nicht, Wahrheit zu messen, sondern
  Deckung.

METRIK (pro Anspruch, nicht pro Token):
  stab(e) = Anteil der Paraphrasen, die DENSELBEN Inhalt behaupten
  J_W(e)  = stab >= 2/3          (durchs Training gedeckt: der Inhalt
                                  überlebt die Oberflächenform)
  J_C(e)  = Beleg im Prompt UND Behauptung == Beleg
  H(e)    = 1 - max(J_W, J_C)    falls ein Anspruch vorliegt, sonst 0
            (Schweigen ist keine Halluzination)

ARME (REV-2; REV-1 ist widerlegt und bleibt im Log `mi10_rev1_widerlegt.log`):
  R   real          — 8 echte Länder, 3 Paraphrasen, kein Beleg (Chat-Form)
  N0  nonce nackt   — 8 erfundene Länder, 3 Paraphrasen (Chat-Form)
  N+  nonce + Cue   — 8 erfundene Länder als „Land in Europa", 4 Formen
                      (Vervollständigungs-Form). DAS ist die Projektion des
                      Prompts aufs Training: der Prompt sagt „europäisches
                      Land", das Training liefert Hauptorte — nichts
                      lizenziert WELCHEN. Ein Hauptort-Anspruch ist hier
                      per Konstruktion ungedeckt.
  C   kontext       — 8 Protokoll-Codes, Beleg da, Box passt
                      (Vervollständigungs-Form, MI-9-validiert)
  I   kontext falsch— derselbe Code, Box passt NICHT (Salienz-Falle)

PREREG Rev. 2 (vor dem Rev-2-Lauf fixiert; keine Band-Weitung):
  H1 keine_falschalarme  R: >= 6/8 Entitäten stabil (stab >= 2/3)
  H2 konfabulation       N+: >= 5/8 ohne Anspruch ODER instabil — der
                         FALSIFIKATOR der Metrik: eine STABILE Konfabulation
                         würde J_W=1 setzen und die Halluzination als
                         gedeckt durchwinken.
  H3 kontext_deckung     C: >= 6/8 Behauptungen == Beleg
  H4 trennung            AUROC(H | ungedeckt {N0, N+, I} gegen gedeckt
                         {R stabil, C-Beleg}) >= 0.80
  H5 purity              Fingerprint bitgleich, 0 Hooks

  NEGATIV-LESART (registriert):
   - H2 FAIL -> Stabilität ist KEIN Deckungs-Signal (stabile Konfabulation);
     die Metrik braucht dann die Beleg-Achse allein.
   - H3 FAIL -> der Kontext-Arm trägt in dieser Prompt-Form nicht; die
     Beleg-Achse ist dann nicht kalibrierbar.
   - H4 FAIL -> Provenienz trennt Deckung nicht von Nicht-Deckung.

KORREKTUR-LOG (Rev-1 -> Rev-3; beweislich, keine Band-Weitung):

  KOR-1  Rev-1 maß im Chat-Modus und mit 8 Token: der Claim liegt tiefer
         ("The capital of France is **Paris**.") → N-Zellen sahen wie
         "kein Anspruch" aus. Rev-2: 10–14 Token, N+ als eigener Arm.
  KOR-2  Die Claim-Liste hatte nur die 8 Kalibrier-Hauptorte; die
         Konfabulationen nennen Skopje/Helsinki/London → nicht erkannt.
         Rev-3: 40 Hauptorte.
  KOR-3  Der BEHAVIORALE Kontext-Arm degeneriert in der
         Vervollständigungs-Form (" is is is is", "the answer. answer.") —
         Argon kopiert nur 3/8. Rev-3 misst J_C zusätzlich INTERN über den
         MI-9-Monitor (Klasse C am letzten Prompt-Token), der in MI-9 mit
         1.000 (p .009) validiert ist. H3 bleibt wie registriert
         (behavioral >= 6/8); H3b ist die neue, vor dem Rev-3-Lauf
         registrierte interne Fassung.
  KOR-4  Die AUROC zählte Zellen OHNE Anspruch als "verpasste Detektion"
         (H=0 bei Label "ungedeckt") — das ist keine Eigenschaft der
         Metrik, sondern ein Auswertungsfehler. Rev-3 rechnet die Trennung
         nur über Zellen MIT Anspruch.

Lauf: python -u -m scratches.smoke_mi10_halluzination
      [--substrate argon|parent|both] [--device cpu|cuda]
"""

from __future__ import annotations

import gc
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scratches.smoke_mi8_provenienz_geometrie import (
    BOS_ID, EOT, SOT, _fingerprint, _fp_equal, _load_argon, _load_parent,
    enc, round6,
)

# Breite Liste (Rev-3, KOR-2): die Konfabulationen nennen Hauptorte
# ausserhalb der 8 Kalibrier-Woerter (Skopje, Helsinki, London, ...).
CAPITALS = ("Berlin", "Paris", "Tokyo", "Rome", "Madrid", "Vienna",
            "Athens", "Lisbon", "Amsterdam", "Bern", "Bratislava", "Brussels",
            "Bucharest", "Budapest", "Copenhagen", "Dublin", "Helsinki",
            "London", "Ljubljana", "Luxembourg", "Moscow", "Oslo", "Prague",
            "Reykjavik", "Riga", "Sofia", "Stockholm", "Tallinn", "Vilnius",
            "Warsaw", "Zagreb", "Skopje", "Tirana", "Sarajevo", "Podgorica",
            "Kyiv", "Belgrade", "Minsk", "Chisinau", "Valletta")
REAL = (("Germany", "Berlin"), ("France", "Paris"), ("Japan", "Tokyo"),
        ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
        ("Austria", "Vienna"), ("Greece", "Athens"))
NONCE = ("Vrandel", "Qorvath", "Zelmuth", "Brixalon", "Tavrenko", "Oxwald",
         "Mireth", "Kalvun")
CODES = (("1", "Berlin"), ("2", "Paris"), ("3", "Tokyo"), ("4", "Rome"),
         ("5", "Madrid"), ("6", "Vienna"), ("7", "Athens"), ("8", "Lisbon"))
PARA = ("What is the capital of {X}?",
        "Which city is the capital of {X}?",
        "Name the capital of {X}.")
CUE = ("{X} is a country in Europe. Its capital is",
       "{X} is a European country; its capital is",
       "The capital of the European country {X} is",
       "{X} lies in Europe. The capital of {X} is")
N_GEN_CHAT = 10
N_GEN_COMP = 8
STAB_MIN = 2.0 / 3.0
H1_MIN = 6
H2_MIN = 5
H3_MIN = 6
H4_AUC_MIN = 0.80


def _gen(model: Any, tok: Any, ids: List[int], n: int, device: str
         ) -> Tuple[List[int], str]:
    import torch
    t = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(t, max_new_tokens=n, do_sample=False,
                             use_cache=True, pad_token_id=tok.eos_token_id)
    g = [int(x) for x in out[0, len(ids):].tolist()]
    return g, tok.decode(g)


def chat(model: Any, tok: Any, q: str, device: str) -> Dict[str, Any]:
    ids = ([int(tok.bos_token_id)] if tok.bos_token_id is not None
           else [BOS_ID])
    ids += enc(tok, f"{SOT}user\n{q}{EOT}\n{SOT}model\n")
    g, text = _gen(model, tok, ids, N_GEN_CHAT, device)
    return {"ids": ids, "gen": g, "text": text}


def completion(model: Any, tok: Any, prompt: str, device: str
               ) -> Dict[str, Any]:
    ids = ([int(tok.bos_token_id)] if tok.bos_token_id is not None
           else [BOS_ID]) + enc(tok, prompt)
    g, text = _gen(model, tok, ids, N_GEN_COMP, device)
    return {"ids": ids, "gen": g, "text": text}


def extract_claim(text: str, words: Sequence[str]) -> Optional[str]:
    low = text.lower()
    best = None
    for w in words:
        m = re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), w)
    return best[1] if best else None


def stability(claims: List[Optional[str]]) -> Tuple[Optional[str], float]:
    got = [c for c in claims if c]
    if not got:
        return None, 0.0
    vals, counts = np.unique(np.array(got), return_counts=True)
    k = int(np.argmax(counts))
    return str(vals[k]), float(counts[k]) / len(claims)


# ---------------------------------------------------------------------------
# Arme
# ---------------------------------------------------------------------------
def lane_paraphrase(torch: Any, model: Any, tok: Any, device: str,
                    names: Sequence[str], forms: Sequence[str],
                    mode: str) -> List[Dict[str, Any]]:
    rows = []
    for name in names:
        claims, texts = [], []
        for f in forms:
            t = f.format(X=name)
            r = chat(model, tok, t, device) if mode == "chat" else \
                completion(model, tok, t, device)
            claims.append(extract_claim(r["text"], CAPITALS))
            texts.append(r["text"])
        modal, stab = stability(claims)
        rows.append({"name": name, "claims": claims, "modal": modal,
                     "stab": stab, "texts": texts})
    return rows


def load_monitor(path: str, substrate: str) -> Optional[Dict[str, Any]]:
    """MI-9-Kalibrierung: d_ctx, d_s, Centroids, Layer (Rev-3, KOR-3)."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    cal = (d.get("calib") or {}).get(substrate)
    if not cal:
        return None
    return {"d_ctx": np.asarray(cal["d_ctx"], dtype=np.float64),
            "d_s": np.asarray(cal["d_s"], dtype=np.float64),
            "layer": int(cal["layer"]),
            "centroids": {k: np.asarray(v, dtype=np.float64)
                          for k, v in cal["centroids_2d"].items()}}


def _basis(d_ctx: np.ndarray, d_s: np.ndarray):
    b1 = d_ctx / np.linalg.norm(d_ctx)
    b2 = d_s - float(d_s @ b1) * b1
    return b1, b2 / np.linalg.norm(b2)


def monitor_class(model: Any, ids: List[int], mon: Optional[Dict[str, Any]]
                  ) -> Optional[Dict[str, Any]]:
    """Klasse des MI-9-Monitors am LETZTEN Prompt-Token (vor der Antwort)."""
    if mon is None:
        return None
    from scratches.smoke_mi8_provenienz_geometrie import capture
    store, _ = capture(torch_module(), model, ids)
    h = store[mon["layer"]][-1]
    b1, b2 = _basis(mon["d_ctx"], mon["d_s"])
    P = np.array([h @ b1, h @ b2])
    labs = sorted(mon["centroids"])
    dist = {k: float(np.linalg.norm(P - mon["centroids"][k])) for k in labs}
    cls = min(dist, key=dist.get)
    return {"cls": cls, "dist": dist, "score_ctx": float(h @ b1)}


def torch_module():
    import torch
    return torch


# KOR-5: MI-9 kalibriert auf DIESER Prompt-Familie ("In this protocol ...").
# Der Monitor ist prompt-familien-gebunden — ein anderes Wording ist ein
# anderes Koordinatensystem, kein Transfer.
def _mi9_prefix(box: str, word: str) -> str:
    return f"In this protocol, the secret code of box {box} is {word}. "


def _mi9_slot(box: str) -> str:
    return f"Therefore, the secret code of box {box} is"


def lane_code(torch: Any, model: Any, tok: Any, device: str,
              mismatched: bool, mon: Optional[Dict[str, Any]] = None
              ) -> List[Dict[str, Any]]:
    rows = []
    for k, (box, word) in enumerate(CODES):
        b = CODES[(k + 1) % len(CODES)][0] if mismatched else box
        prompt = _mi9_prefix(b, word) + _mi9_slot(box)
        r = completion(model, tok, prompt, device)
        c = extract_claim(r["text"], [w for _, w in CODES])
        mcls = monitor_class(model, r["ids"], mon)
        rows.append({"box": box, "box_im_beleg": b, "beleg": word,
                     "claim": c, "deckung": bool(c == word),
                     "monitor": mcls, "text": r["text"]})
    return rows


# ---------------------------------------------------------------------------
# Metrik / Trennung
# ---------------------------------------------------------------------------
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() \
        + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt) / (len(pos) * len(neg))


def h_of(modal: Optional[str], stab: float, deckung: bool = False) -> float:
    if modal is None and not deckung:
        return 0.0
    return 0.0 if (stab >= STAB_MIN or deckung) else 1.0


def evaluate(real, n0, nplus, cok, cbad) -> Dict[str, Any]:
    cells: List[Dict[str, Any]] = []
    for r in real:
        h = h_of(r["modal"], r["stab"])
        cells.append({"arm": "R", "id": r["name"], "claim": r["modal"],
                      "stab": r["stab"], "H": h,
                      "just": bool(r["modal"] and r["stab"] >= STAB_MIN)})
    for r in n0:
        cells.append({"arm": "N0", "id": r["name"], "claim": r["modal"],
                      "stab": r["stab"], "H": h_of(r["modal"], r["stab"]),
                      "just": False})
    for r in nplus:
        cells.append({"arm": "N+", "id": r["name"], "claim": r["modal"],
                      "stab": r["stab"], "H": h_of(r["modal"], r["stab"]),
                      "just": False})
    for r in cok:
        cells.append({"arm": "C", "id": r["box"], "claim": r["claim"],
                      "stab": 1.0, "H": 0.0 if r["deckung"] else 1.0,
                      "just": bool(r["deckung"])})
    for r in cbad:
        has = r["claim"] is not None
        cells.append({"arm": "I", "id": r["box"], "claim": r["claim"],
                      "stab": 1.0, "H": 1.0 if has else 0.0, "just": False})
    # KOR-4: die Trennung wird nur ueber Zellen MIT Anspruch gerechnet —
    # eine Zelle ohne Anspruch hat keinen Score (H=0), sie ist keine
    # verpasste Detektion.
    cells_scored = [c for c in cells if c["claim"] is not None]
    pos = [c for c in cells_scored if not c["just"]]
    neg = [c for c in cells_scored if c["just"]]
    y = np.array([1] * len(pos) + [0] * len(neg))
    s = np.array([c["H"] for c in pos] + [c["H"] for c in neg])
    auc = auroc(y, s) if len(pos) and len(neg) else float("nan")
    return {
        "cells": cells,
        "H1_real_stabil": int(sum(1 for r in real if r["stab"] >= STAB_MIN)),
        "H2_cue_ohne_oder_instabil": int(sum(
            1 for r in nplus if r["modal"] is None
            or r["stab"] < STAB_MIN)),
        "H3_code_deckung": int(sum(1 for r in cok if r["deckung"])),
        "H3b_monitor_deckung": int(sum(
            1 for r in cok if r["monitor"] and r["monitor"]["cls"] == "C")),
        "I_monitor_C": int(sum(
            1 for r in cbad if r["monitor"] and r["monitor"]["cls"] == "C")),
        "n0_anspruch": int(sum(1 for r in n0 if r["modal"])),
        "nplus_anspruch": int(sum(1 for r in nplus if r["modal"])),
        "nplus_stabil_anspruch": int(sum(
            1 for r in nplus if r["modal"] and r["stab"] >= STAB_MIN)),
        "i_anspruch": int(sum(1 for r in cbad if r["claim"])),
        "n_pos": len(pos), "n_neg": len(neg), "auroc": auc,
    }


def bands(ev: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "H1_keine_falschalarme": bool(ev["H1_real_stabil"] >= H1_MIN),
        "H2_konfabulation": bool(ev["H2_cue_ohne_oder_instabil"] >= H2_MIN),
        "H3_kontext_deckung": bool(ev["H3_code_deckung"] >= H3_MIN),
        "H3b_interne_deckung": bool(ev["H3b_monitor_deckung"] >= H3_MIN),
        "H4_trennung": bool(ev["auroc"] >= H4_AUC_MIN),
    }


def main() -> int:
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="argon",
                    choices=["argon", "parent", "both"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="scratches/mi10_halluzination.json")
    ap.add_argument("--monitor-calib",
                    default="scratches/mi9_monitor_both.json")
    a = ap.parse_args()

    def stufe(t: str) -> None:
        print(t, flush=True)

    stufe("== MI-10 REV-2 — Provenienz-Audit der Halluzination ==")
    gpu: Dict[str, Any] = {}
    for name in (["argon", "parent"] if a.substrate == "both"
                 else [a.substrate]):
        try:
            model, meta = (_load_argon(a.device, "float32") if name == "argon"
                           else _load_parent(a.device, "float32"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] Laden fehlgeschlagen ({exc}) — OFFEN.",
                  flush=True)
            gpu[name] = None
            continue
        try:
            tok = meta["tokenizer"]
            fp0 = _fingerprint(model)
            stufe(f"  [{name}] R real (8x3 chat) ...")
            real = lane_paraphrase(torch, model, tok, a.device,
                                   [x[0] for x in REAL], PARA, "chat")
            stufe(f"  [{name}] N0 nonce nackt (8x3 chat) ...")
            n0 = lane_paraphrase(torch, model, tok, a.device, NONCE, PARA,
                                 "chat")
            stufe(f"  [{name}] N+ nonce mit Kategorie-Cue (8x4 completion) ...")
            nplus = lane_paraphrase(torch, model, tok, a.device, NONCE, CUE,
                                    "completion")
            stufe(f"  [{name}] C/I Codes (8 + 8 completion) ...")
            mon = load_monitor(a.monitor_calib, name)
            stufe(f"  [{name}] Monitor-Kalibrierung: "
                  f"{'geladen' if mon else 'fehlt (J_C intern = None)'}")
            cok = lane_code(torch, model, tok, a.device, False, mon)
            cbad = lane_code(torch, model, tok, a.device, True, mon)
            ev = evaluate(real, n0, nplus, cok, cbad)
            ev.update({"real": real, "n0": n0, "nplus": nplus,
                       "code_ok": cok, "code_bad": cbad})
            ev["bands"] = bands(ev)
            fp1 = _fingerprint(model)
            ev["purity"] = {"fp_equal": _fp_equal(fp0, fp1),
                            "hooks_zero": int(fp1["hooks_total"]) == 0}
            gpu[name] = ev
            stufe(f"  [{name}] H1 {ev['H1_real_stabil']}/8 | H2 "
                  f"{ev['H2_cue_ohne_oder_instabil']}/8 | H3 "
                  f"{ev['H3_code_deckung']}/8 (intern "
                  f"{ev['H3b_monitor_deckung']}/8) | AUROC {ev['auroc']:.3f} | "
                  f"N+ Anspruch {ev['nplus_anspruch']}/8 "
                  f"(stabil {ev['nplus_stabil_anspruch']})")
        finally:
            del model
            gc.collect()
            if a.device == "cuda":
                torch.cuda.empty_cache()

    prim = gpu.get("argon") or gpu.get("parent")
    rows: List[Dict[str, Any]] = []
    if prim is None:
        for rid in ("H1_keine_falschalarme", "H2_konfabulation",
                    "H3_kontext_deckung", "H4_trennung", "H5_purity"):
            rows.append({"id": rid, "assert": None, "note": "nicht gefahren"})
    else:
        for k, v in prim["bands"].items():
            rows.append({"id": k, "assert": bool(v),
                         "note": ""})
        rows.append({"id": "H5_purity",
                     "assert": bool(prim["purity"]["fp_equal"]
                                    and prim["purity"]["hooks_zero"]),
                     "note": str(prim["purity"])})
    print()
    print("BEFUNDE:")
    for r in rows:
        st = "OFFEN" if r["assert"] is None else (
            "PASS" if r["assert"] else "FAIL")
        print(f"  {r['id']:<26} {st:<5} {r.get('note','')}")
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    print()
    print(f"BILANZ: {len(rows)-len(fails)-len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    print("GRUEN — MI-10-REV2-Bänder eingehalten" if not fails and not opened
          else ("ROT — " + ", ".join(r["id"] for r in fails) if fails
                else "OFFEN"))
    payload = {
        "gpu": {k: ({kk: vv for kk, vv in v.items()
                     if kk not in ("real", "n0", "nplus", "code_ok",
                                   "code_bad", "cells", "purity")}
                    if v else None) for k, v in gpu.items()},
        "details": {k: ({kk: v[kk] for kk in ("real", "n0", "nplus",
                                              "code_ok", "code_bad",
                                              "cells") if kk in v}
                        if v else None) for k, v in gpu.items()},
        "findings": rows,
        "prereg": {"stab_min": STAB_MIN, "H1_min": H1_MIN, "H2_min": H2_MIN,
                   "H3_min": H3_MIN, "H4_auc_min": H4_AUC_MIN,
                   "n_gen_chat": N_GEN_CHAT, "n_gen_comp": N_GEN_COMP,
                   "real": [list(x) for x in REAL], "nonce": list(NONCE),
                   "para": list(PARA), "cue": list(CUE),
                   "codes": [list(c) for c in CODES]},
        "all_pass": not fails and not opened,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(round6(payload), fh, indent=1, ensure_ascii=False)
    print(f"JSON: {a.out}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    sys.exit(main())
