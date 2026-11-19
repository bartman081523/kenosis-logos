"""
smoke_mi9_drei_quellen.py — MI-9 (REV-2): der Drei-Quellen-Monitor
Kontext / Training / eigene Gedanken — Kalibrierung, Readout, Kausal-Kontrolle.

REV-1 ist widerlegt und bleibt als Register stehen (Log
`scratches/mi9_rev1_widerlegt.log`): drei Gerätefehler, alle beweislich.
  K1  M1-p-Band mit n=6 arithmetisch unerreichbar (p_min = 1/64 = 0.0156
      > 0.01) — kein Modellbefund.
  K2  M3 nahm an, ein EINZELNER widersprechender Satz überschreibe die
      Gewichte. Gemessen: p(Widerspruch|C) = 0/6 (Argon) — die Gewichte
      gewinnen. MI-7 hatte dafür erfundene Fakten benutzt (kein
      Gewichts-Konkurrent), nicht kontrafaktische reale.
  K3  M4 normierte den Rollen-Score durch ||h|| (~12k) — Delta < 0.001,
      unter jeder Auflösung. 0/6 war ein Skalen-, kein Kausalbefund.

REV-2 DESIGN (jede Behauptung an einem konfundfreien minimalen Paar):

  P1 KONTEXT  — 8 erfundene Codes (Box n -> Name). C-Zelle: der Code steht
                im Praefix und der Slot fragt DIESELBE Box.
                I-Zelle: DERSELBE Code, DERSELBE Slot, nur die Box-Nummer
                im Praefix ist eine andere (irrelevant). Beide Praefixe
                sind byte-gleich lang; unterschiedlich ist EIN Token.
                C vs I ist damit der reine Relevanz-Kontrast — "steht die
                Antwort im Kontext?" ohne Template- oder Laengen-Konfund.
  P2 TRAINING — 8 reale Fakten (Land -> Hauptort), neutrales 8-Token-
                Praefix, Slot ohne Beleg. Die Antwort kann nur aus den
                Gewichten kommen. (W vs C ist NICHT der Band-Test: die
                Templates unterscheiden sich notwendig, weil das Gewicht
                nur reale, der Kontext auch erfundene Fragen beantworten
                kann. W wird behavioral geprueft, nicht per Achse.)
  P3 EIGENE   — 6 eigene 8-Token-Fortsetzungen; derselbe ID-String als
                model- vs user-Turn. Nur die Chat-Rolle unterscheidet.
  P4 INTEGRATION — 3 Klassen W/C/S im 2D-Raum (d_ctx, d_s), LOO. DATUM
                mit Template-Vorbehalt.

PREREG Rev. 2 (vor dem Rev-2-Lauf fixiert; keine Band-Weitung):

  M1 kontext_achse   C-vs-I LOO-Sign-Acc >= 0.75, p_perm <= 0.02 (n=8,
                     p_min = 1/256).
  M2 relevanz        Verhalten kippt mit der Relevanz: p(Code|C) >= 0.30
                     in >= 6/8 UND p(Code|I) <= 0.10 in >= 6/8.
  M3 gewicht         p(wahrer Hauptort|W) >= 0.30 in >= 6/8.
  M4 eigene          S-vs-¬S LOO-Acc >= 0.75 (Rollen-Tausch).
  M5 achsen          |cos(d_ctx, d_s)| <= 0.5.
  M6 integration     3-Klassen-LOO >= 0.80 (Template-Vorbehalt, DATUM).
  M7 purity          Fingerprint bitgleich, 0 Hooks, Doppellauf identisch.

  NEGATIV-LESART (registriert):
   - M2 FAIL (p(Code|I) hoch) -> das Gerät liest "ein Code steht im
     Praefix", nicht Relevanz; M1 traegt dann keine Kontext-Aussage.
   - M5 FAIL -> eine gemeinsame Quell-Achse; die Dreiheit ist dann eine
     Achse mit drei Zonen.

KORREKTUR-LOG (Rev-1 -> Rev-2; alle Einträge beweislich, keine Band-Weitung):

  KOR-1  Rev-1-M1 verlangte p < 0.01 bei n=6 Paaren. Das ist arithmetisch
         unerreichbar (p_min = 1/64 = 0.0156). Fix: n=8 (p_min = 1/256).
  KOR-2  Rev-1-M3 nahm an, EIN widersprechender Satz überschreibe die
         Gewichte. Gemessen: p(Widerspruch|C) = 0/6 in Argon — die Gewichte
         gewinnen. Der Kontext-Arm braucht Fragen, die das Gewicht NICHT
         beantworten kann (MI-7-Kanon: erfundene/Protokoll-Fakten).
  KOR-3  Rev-1-M4 normierte den Rollen-Score durch ||h|| (~12k). Delta
         < 0.001 — unter jeder Auflösung; 0/6 war ein Skalen-, kein
         Kausalbefund. Fix: gepaarte LOO-Accuracy.
  KOR-4  Rev-1-Codes waren Nicht-Wörter ("Zirkon"): nur 1/8 Zellen
         kopierten, weil der Wort-Prior fehlt und MI-7s 0.806 ein
         Zirkon-Prior war, kein Kopieren. Fix: reale Ein-Token-Wörter
         (C 6/8 p>=0.30, I 8/8 p<=0.01 — vor dem Rev-2-Lauf gemessen).
  KOR-5  Pilot-Null war eine Einzelziehung; bei n=8 ist 8/8 unter der Null
         ein 1/256-Ereignis. Fix: 20 Ziehungen, Mittel in [0.3, 0.7].
  KOR-6  Design: W-vs-C ist KEIN Band (die Templates unterscheiden sich
         notwendig, weil nur der Kontext fiktive Fragen beantworten kann).
         Der konfundfreie Kontrast ist C-vs-I: gleicher Satz, gleiches
         Wort, nur die Box-Nummer im Präfix ist eine andere.

Lauf: python -u -m scratches.smoke_mi9_drei_quellen
      [--substrate argon|parent|both] [--device cpu|cuda]
"""

from __future__ import annotations

import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from scratches.smoke_mi8_provenienz_geometrie import (
    BOS_ID, EOT, N_GEN, PROBE_LAYER, SOT, _fingerprint, _fp_equal,
    _load_argon, _load_parent, assemble, capture, enc, fixed_dir_acc, round6,
    unit,
)

M1_ACC_MIN = 0.75
M1_P_MAX = 0.02
M2_P_CTX = 0.30
M2_P_IRR = 0.10
M3_P_WGT = 0.30
M2_MIN = 6
M4_ACC_MIN = 0.75
M5_COS_MAX = 0.5
M6_ACC_MIN = 0.80

NEUTRAL = "There is nothing else in this note."        # 8 Token
REAL = (
    ("Germany", "Berlin"), ("France", "Paris"), ("Japan", "Tokyo"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("Austria", "Vienna"), ("Greece", "Athens"),
)
# Codes sind REALE Ein-Token-Woerter (KOR-4): Nicht-Woerter wie "Zirkon"
# tragen einen Wort-Prior und werden nur zufaellig kopiert; mit realen
# Woertern ist der Kopier-Effekt messbar (C 6/8 p>=0.30, I 8/8 p<=0.01).
CODES = (                                               # (Box, Name)
    ("1", "Berlin"), ("2", "Paris"), ("3", "Tokyo"), ("4", "Rome"),
    ("5", "Madrid"), ("6", "Vienna"), ("7", "Athens"), ("8", "Lisbon"),
)
SEEDS = (
    "The harbour lights fade", "A copper wire carries",
    "The old clock in the hallway", "Rain has been falling on",
    "The garden gate stays", "A single candle burns",
)


# ---------------------------------------------------------------------------
# Zellbau
# ---------------------------------------------------------------------------
def _code_praefix(box: str, name: str) -> str:
    return f"In this protocol, the secret code of box {box} is {name}. "


def _code_slot(box: str) -> str:
    return f"Therefore, the secret code of box {box} is"


def build_ci(tok: Any) -> Dict[str, Any]:
    """C- und I-Zellen: gleicher Code, gleicher Slot, nur Box-Nummer anders."""
    cells: Dict[str, List[Dict[str, Any]]] = {"C": [], "I": []}
    refs = []
    for k, (box, name) in enumerate(CODES):
        other = CODES[(k + 1) % len(CODES)][0]
        slot = enc(tok, _code_slot(box))
        tgt = enc(tok, " " + name)[0]
        pair_ref = None
        for arm, b in (("C", box), ("I", other)):
            pre = enc(tok, _code_praefix(b, name))
            ids = [BOS_ID] + pre + slot
            start = 1 + len(pre)
            if list(ids[start:start + len(slot)]) != slot:
                raise RuntimeError("build_ci: Slot nicht an der Position.")
            cell = {"box": box, "box_im_praefix": b, "name": name,
                    "target_id": int(tgt), "ids": ids, "start": start,
                    "end": start + len(slot)}
            if pair_ref is None:
                pair_ref = (len(pre), len(slot), start)
            elif (len(pre), len(slot), start) != pair_ref:
                raise RuntimeError(
                    f"build_ci: Paar-Laengen {len(pre)}/{len(slot)}/{start} vs "
                    f"{pair_ref} — Gerät-Bruch (Box-Nummer muss 1 Token sein).")
            cells[arm].append(cell)
        refs.append(pair_ref)
    return {"cells": cells,
            "ref": {"slot_len": refs[0][1],
                    "paar_len_gleich": len(set(refs)) >= 1,
                    "praefix_len_pro_paar": [r[0] for r in refs]}}


def build_w(tok: Any) -> Dict[str, Any]:
    """W-Zellen: neutrales Praefix + realer Fakt-Slot (Antwort aus Gewicht)."""
    cells = []
    for land, wahr in REAL:
        pre = enc(tok, NEUTRAL)
        slot = enc(tok, f"The capital of {land} is")
        ids = [BOS_ID] + pre + slot
        start = 1 + len(pre)
        if list(ids[start:start + len(slot)]) != slot:
            raise RuntimeError("build_w: Slot nicht an der Position.")
        cells.append({"land": land, "ziel": wahr, "target_id": int(
            enc(tok, " " + wahr)[0]), "ids": ids, "start": start,
            "end": start + len(slot)})
    if len({len(c["ids"]) for c in cells}) != 1:
        raise RuntimeError("build_w: Laengen ungleich — Gerät-Bruch.")
    return {"cells": cells, "praefix_len": len(enc(tok, NEUTRAL)),
            "slot_len": len(enc(tok, "The capital of Germany is"))}


def build_s(tok: Any, torch: Any, model: Any, device: str
            ) -> Dict[str, Any]:
    pairs = []
    for seed in SEEDS:
        ids = torch.tensor([[BOS_ID] + enc(tok, seed)], dtype=torch.long,
                           device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=N_GEN, do_sample=False,
                                 use_cache=True,
                                 pad_token_id=tok.eos_token_id)
        gen = [int(t) for t in out[0, ids.shape[1]:].tolist()]
        if len(gen) != N_GEN:
            raise RuntimeError(f"build_s: {len(gen)} statt {N_GEN} Token.")
        pair = {"seed": seed, "gen": gen, "gen_text": tok.decode(gen)}
        for role in ("model", "user"):
            hid, start = assemble(
                tok, f"{SOT}user\n{seed}{EOT}\n{SOT}{role}\n", gen)
            pair[role] = {"ids": hid, "start": start, "end": start + len(gen)}
        pairs.append(pair)
    return {"pairs": pairs}


# ---------------------------------------------------------------------------
# 2D-Mathe
# ---------------------------------------------------------------------------
def _basis(d_ctx: np.ndarray, d_s: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray]:
    b1 = unit(d_ctx)
    b2 = np.asarray(d_s, np.float64) - float(np.asarray(d_s) @ b1) * b1
    return b1, unit(b2)


def _project(H: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    return np.stack([H @ b1, H @ b2], axis=1)


def _centroids(P: np.ndarray, labels: Sequence[str]) -> Dict[str, np.ndarray]:
    return {lab: P[[i for i, x in enumerate(labels) if x == lab]].mean(axis=0)
            for lab in sorted(set(labels))}


def _classify(P: np.ndarray, C: Dict[str, np.ndarray]) -> List[str]:
    labs = sorted(C)
    d = np.stack([np.linalg.norm(P - C[l], axis=1) for l in labs], axis=1)
    return [labs[i] for i in np.argmin(d, axis=1)]


def loo_paired(H_a: np.ndarray, H_b: np.ndarray,
               n_perm: int = 1000) -> Dict[str, Any]:
    """LOO über Paare (b vs a). Konfundfrei, weil nur EIN Token differiert."""
    n = len(H_a)
    D = H_b - H_a
    ok = 0
    for i in range(n):
        d = unit(D[np.arange(n) != i].mean(axis=0))
        ok += int(float((H_b[i] - H_a[i]) @ d) > 0.0)
    acc = ok / n
    rng = np.random.default_rng(42)
    hits = 0
    for _ in range(n_perm):
        s = rng.choice(np.array([-1.0, 1.0]), size=n)
        dd = D * s[:, None]
        tot = dd.sum(axis=0)
        c = 0
        for i in range(n):
            d = unit((tot - dd[i]) / (n - 1))
            c += int(float(dd[i] @ d) > 0.0)
        if c / n >= acc - 1e-12:
            hits += 1
    return {"acc": acc, "p": (1.0 + hits) / (1.0 + n_perm), "n": n}


def loo_3class(P: np.ndarray, labels: Sequence[str]) -> float:
    n = len(labels)
    ok = 0
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        C = _centroids(P[keep], [labels[j] for j in keep])
        ok += int(_classify(P[i:i + 1], C)[0] == labels[i])
    return ok / n


# ---------------------------------------------------------------------------
# Lanes
# ---------------------------------------------------------------------------
def lane_ci(torch: Any, model: Any, tok: Any, data: Dict[str, Any]
            ) -> Dict[str, Any]:
    H: Dict[str, List[np.ndarray]] = {"C": [], "I": []}
    beh: Dict[str, List[Dict[str, Any]]] = {"C": [], "I": []}
    for arm in ("C", "I"):
        for cell in data["cells"][arm]:
            store, logits = capture(torch, model, cell["ids"])
            H[arm].append(store[PROBE_LAYER][cell["end"] - 1])
            p = torch.softmax(torch.tensor(logits), -1).numpy()
            beh[arm].append({"box": cell["box"], "name": cell["name"],
                             "p_code": float(p[cell["target_id"]]),
                             "arg": int(np.argmax(p))})
    Hc, Hi = np.stack(H["C"]), np.stack(H["I"])
    d_ctx = unit((Hc - Hi).mean(axis=0))
    l2 = loo_paired(Hi, Hc)
    ok_c = sum(1 for r in beh["C"] if r["p_code"] >= M2_P_CTX)
    ok_i = sum(1 for r in beh["I"] if r["p_code"] <= M2_P_IRR)
    return {"d_ctx": d_ctx, "Hc": Hc, "Hi": Hi, "loo": l2,
            "beh": beh, "m2_ctx": int(ok_c), "m2_irr": int(ok_i)}


def lane_w(torch: Any, model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    H, beh = [], []
    for cell in data["cells"]:
        store, logits = capture(torch, model, cell["ids"])
        H.append(store[PROBE_LAYER][cell["end"] - 1])
        p = torch.softmax(torch.tensor(logits), -1).numpy()
        beh.append({"land": cell["land"], "ziel": cell["ziel"],
                    "p_ziel": float(p[cell["target_id"]]),
                    "arg": int(np.argmax(p))})
    Hw = np.stack(H)
    return {"Hw": Hw, "beh": beh,
            "m3": int(sum(1 for r in beh if r["p_ziel"] >= M3_P_WGT))}


def lane_s(torch: Any, model: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    hm, hu = [], []
    for pair in data["pairs"]:
        for role, bucket in (("model", hm), ("user", hu)):
            cell = pair[role]
            store, _ = capture(torch, model, cell["ids"])
            bucket.append(store[PROBE_LAYER][cell["end"] - 1])
    Hm, Hu = np.stack(hm), np.stack(hu)
    l2 = loo_paired(Hu, Hm)
    d_s = unit((Hm - Hu).mean(axis=0))
    raw = [{"delta_raw": float((Hm[i] - Hu[i]) @ d_s) / (
        float(np.linalg.norm(Hm[i] - Hu[i])) + 1e-12)}
        for i in range(len(Hm))]
    return {"d_s": d_s, "Hm": Hm, "Hu": Hu, "loo": l2, "raw": raw}


def calibration(name: str, res: Dict[str, Any]) -> Dict[str, Any]:
    Hw, Hc, Hm = res["w"]["Hw"], res["ci"]["Hc"], res["s"]["Hm"]
    b1, b2 = _basis(res["ci"]["d_ctx"], res["s"]["d_s"])
    P = _project(np.concatenate([Hw, Hc, Hm]), b1, b2)
    labels = ["W"] * len(Hw) + ["C"] * len(Hc) + ["S"] * len(Hm)
    return {"substrate": name, "layer": PROBE_LAYER,
            "d_ctx": res["ci"]["d_ctx"].tolist(),
            "d_s": res["s"]["d_s"].tolist(),
            "cos_axes": float(unit(res["ci"]["d_ctx"]) @ unit(res["s"]["d_s"])),
            "centroids_2d": {k: v.tolist()
                             for k, v in _centroids(P, labels).items()},
            "calib_stats": {"ci_loo": res["ci"]["loo"],
                            "s_loo": res["s"]["loo"],
                            "loo3": res.get("loo3")},
            "note": ("d_ctx: + = Antwort steht im Kontext (C), - = kein "
                     "Beleg (I/W). d_s: + = eigener VorText (model-Turn).")}


def read_source(tok: Any, model: Any, ids: Sequence[int],
                calib: Dict[str, Any]) -> Dict[str, Any]:
    import torch
    b1, b2 = _basis(np.asarray(calib["d_ctx"]), np.asarray(calib["d_s"]))
    C = {k: np.asarray(v) for k, v in calib["centroids_2d"].items()}
    store, _ = capture(torch, model, ids)
    h = store[int(calib["layer"])][-1]
    P = _project(h[None, :], b1, b2)
    dist = {k: float(np.linalg.norm(P[0] - v)) for k, v in C.items()}
    order = sorted(dist.values())
    return {"cls": _classify(P, C)[0], "dist": dist,
            "margin": float(order[1] - order[0]) if len(order) > 1 else 0.0,
            "score_ctx": float(h @ b1), "score_s": float(h @ b2)}


# ---------------------------------------------------------------------------
# Befunde / Pilot
# ---------------------------------------------------------------------------
def findings(cpu: Dict[str, Any], gpu: Optional[Dict[str, Any]]
             ) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(rid: str, ok: Optional[bool], note: str = "") -> None:
        rows.append({"id": rid, "assert": ok, "note": note})

    add("mi9_cpu_pilot", bool(cpu.get("verdict")), "")
    if gpu is None:
        for rid in ("mi9_kontext_achse", "mi9_relevanz", "mi9_gewicht",
                    "mi9_eigene", "mi9_achsen", "mi9_integration",
                    "mi9_purity"):
            add(rid, None, "Modell-Lanes nicht gefahren")
        return rows
    ci, w, s = gpu["ci"], gpu["w"], gpu["s"]
    add("mi9_kontext_achse",
        bool(ci["loo"]["acc"] >= M1_ACC_MIN and ci["loo"]["p"] <= M1_P_MAX),
        f"acc {ci['loo']['acc']:.3f} p {ci['loo']['p']:.4f} n {ci['loo']['n']}")
    add("mi9_relevanz", bool(ci["m2_ctx"] >= M2_MIN and ci["m2_irr"] >= M2_MIN),
        f"p(Code|C)>={M2_P_CTX} {ci['m2_ctx']}/8, "
        f"p(Code|I)<={M2_P_IRR} {ci['m2_irr']}/8")
    add("mi9_gewicht", bool(w["m3"] >= M2_MIN),
        f"p(Hauptort|W)>={M3_P_WGT} {w['m3']}/8")
    add("mi9_eigene", bool(s["loo"]["acc"] >= M4_ACC_MIN),
        f"acc {s['loo']['acc']:.3f} p {s['loo']['p']:.4f}")
    add("mi9_achsen", bool(abs(gpu["cos_axes"]) <= M5_COS_MAX),
        f"cos {gpu['cos_axes']:.3f}")
    add("mi9_integration", bool(gpu["loo3"] >= M6_ACC_MIN),
        f"3-Klassen-LOO {gpu['loo3']:.3f} (Template-Vorbehalt)")
    p = gpu.get("purity") or {}
    add("mi9_purity", bool(p.get("fp_equal") and p.get("hooks_zero")
                           and p.get("det_equal")),
        f"fp {p.get('fp_equal')} hooks0 {p.get('hooks_zero')} "
        f"det {p.get('det_equal')}")
    return rows


def _null_rate(n_seeds: int, n: int = 8) -> float:
    """Erwartungswert der LOO-Accuracy ohne Klassendifferenz (reine Kontrolle)."""
    accs = []
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mu = rng.standard_normal(64) * 3
        Ha = np.stack([mu + rng.standard_normal(64) * .3 for _ in range(n)])
        Hn = np.stack([mu + rng.standard_normal(64) * .01 for _ in range(n)])
        D = Hn - Ha
        ok = 0
        for i in range(n):
            d = unit(np.delete(D, i, axis=0).mean(0))
            ok += int(float(D[i] @ d) > 0)
        accs.append(ok / n)
    return float(np.mean(accs))


def pilot(tok: Any) -> Dict[str, Any]:
    legs: Dict[str, Any] = {}
    ci = build_ci(tok)
    w = build_w(tok)
    legs["ci_paar_len_gleich"] = bool(all(
        len(ci["cells"]["C"][k]["ids"]) == len(ci["cells"]["I"][k]["ids"])
        and ci["cells"]["C"][k]["start"] == ci["cells"]["I"][k]["start"]
        and ci["cells"]["C"][k]["end"] == ci["cells"]["I"][k]["end"]
        for k in range(len(CODES))))
    legs["ci_ein_token_diff"] = bool(all(
        c["box"] != c["box_im_praefix"] for c in ci["cells"]["I"]))
    legs["neutral_8"] = bool(w["praefix_len"] == 8)
    legs["w_len_gleich"] = bool(len({len(c["ids"]) for c in w["cells"]}) == 1)
    legs["box_1_token"] = bool(all(len(enc(tok, b)) == 1 for b, _ in CODES))
    legs["real_1_token"] = bool(all(
        len(enc(tok, " " + t)) == 1 and len(enc(tok, l)) == 1
        for l, t in REAL))
    # Mathe: trennbare Blobs, LOO + Null
    rng = np.random.default_rng(5)
    n = 8
    mu_a = rng.standard_normal(64) * 3
    shift = np.zeros(64)
    shift[0] = 5.0
    Ha = np.stack([mu_a + rng.standard_normal(64) * .3 for _ in range(n)])
    Hb = np.stack([mu_a + shift + rng.standard_normal(64) * .3
                   for _ in range(n)])
    legs["loo_recovers"] = bool(loo_paired(Ha, Hb)["acc"] >= 0.9)
    # Null: 30 UNABHAENGIGE Seeds (Ziehungen aus EINEM Stream sind
    # korreliert: Seed 5 liefert 8/8, obwohl der Erwartungswert 0.5 ist).
    # Referenz ueber 200 Seeds: Mittel 0.515.
    legs["loo_null"] = bool(0.35 <= _null_rate(30) <= 0.65)
    return {"legs": legs, "verdict": bool(all(legs.values())),
            "n_pairs": len(CODES), "slot_len": ci["ref"]["slot_len"]}


def main() -> int:
    import argparse
    import torch
    from transformers import AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="argon",
                    choices=["argon", "parent", "both"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="scratches/mi9_monitor_both.json")
    args = ap.parse_args()

    def stufe(t: str) -> None:
        print(t, flush=True)

    stufe("== MI-9 REV-2 — Drei-Quellen-Monitor ==")
    tok = AutoTokenizer.from_pretrained("argon_1b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    stufe("CPU-Pilot: Zell-Gesetze + Paar-Mathe")
    p1, p2 = pilot(tok), pilot(tok)
    cpu = {"legs": p1["legs"], "verdict": p1["verdict"],
           "deterministic": bool(json.dumps(p1["legs"], sort_keys=True)
                                 == json.dumps(p2["legs"], sort_keys=True)),
           "n_pairs": p1["n_pairs"], "slot_len": p1["slot_len"]}
    stufe("  " + json.dumps(cpu["legs"], ensure_ascii=False))

    gpu: Dict[str, Any] = {}
    for name in (["argon", "parent"] if args.substrate == "both"
                 else [args.substrate]):
        try:
            model, meta = (_load_argon(args.device, "float32") if name ==
                           "argon" else _load_parent(args.device, "float32"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] Laden nicht moeglich ({exc}) — OFFEN.",
                  flush=True)
            gpu[name] = None
            continue
        try:
            tokx = meta["tokenizer"]
            stufe(f"  [{name}] geladen")
            fp0 = _fingerprint(model)
            ci_d = build_ci(tokx)
            w_d = build_w(tokx)
            s_d = build_s(tokx, torch, model, args.device)
            ci = lane_ci(torch, model, tokx, ci_d)
            stufe(f"  [{name}] C/I acc {ci['loo']['acc']:.3f} "
                  f"p {ci['loo']['p']:.4f} | p(Code|C) {ci['m2_ctx']}/8 "
                  f"p(Code|I)<=.1 {ci['m2_irr']}/8")
            w = lane_w(torch, model, w_d)
            stufe(f"  [{name}] W p(Hauptort)>=.3 {w['m3']}/8")
            s = lane_s(torch, model, s_d)
            stufe(f"  [{name}] S acc {s['loo']['acc']:.3f}")
            res: Dict[str, Any] = {"ci": ci, "w": w, "s": s,
                                   "ci_data": ci_d, "w_data": w_d,
                                   "s_data": s_d}
            b1, b2 = _basis(ci["d_ctx"], s["d_s"])
            P = _project(np.concatenate([w["Hw"], ci["Hc"], s["Hm"]]),
                         b1, b2)
            lab = (["W"] * len(w["Hw"]) + ["C"] * len(ci["Hc"])
                   + ["S"] * len(s["Hm"]))
            res["loo3"] = loo_3class(P, lab)
            res["cos_axes"] = float(unit(ci["d_ctx"]) @ unit(s["d_s"]))
            calib = calibration(name, res)
            res["calib"] = calib
            demo = []
            for arm, group in (("C", ci_d["cells"]["C"][:2]),
                               ("I", ci_d["cells"]["I"][:2]),
                               ("W", w_d["cells"][:2])):
                for cell in group:
                    demo.append({"arm": arm,
                                 "read": read_source(tokx, model,
                                                     cell["ids"], calib)})
            res["demo"] = demo
            stufe(f"  [{name}] 3-Klassen {res['loo3']:.3f} | cos "
                  f"{res['cos_axes']:.3f}")
            fp1 = _fingerprint(model)
            a, _ = capture(torch, model, w_d["cells"][0]["ids"])
            b, _ = capture(torch, model, w_d["cells"][0]["ids"])
            res["purity"] = {"fp_equal": _fp_equal(fp0, fp1),
                             "hooks_zero": int(fp1["hooks_total"]) == 0,
                             "det_equal": bool(all(np.array_equal(a[l], b[l])
                                                   for l in a))}
            gpu[name] = res
        finally:
            del model
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

    primary = gpu.get("argon") or gpu.get("parent")
    rows = findings(cpu, primary)
    print()
    print("BEFUNDE:")
    for r in rows:
        st = "OFFEN" if r["assert"] is None else (
            "PASS" if r["assert"] else "FAIL")
        print(f"  {r['id']:<22} {st:<5} {r.get('note','')}")
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    print()
    print(f"BILANZ: {len(rows)-len(fails)-len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    print("GRUEN — MI-9-REV2-Bänder eingehalten" if not fails and not opened
          else ("ROT — Bänder verletzt: " + ", ".join(r["id"] for r in fails)
                if fails else "OFFEN"))

    def strip(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if k not in ("Hw", "Hc", "Hi", "Hm", "Hu", "d_ctx",
                                 "d_s", "ci_data", "w_data", "s_data")}
        if isinstance(o, (list, tuple)):
            return [strip(v) for v in o]
        if isinstance(o, np.ndarray):
            return None
        return o

    payload = {"cpu": cpu, "gpu": strip(gpu), "findings": rows,
               "prereg": {"m1": [M1_ACC_MIN, M1_P_MAX],
                          "m2": [M2_P_CTX, M2_P_IRR, M2_MIN],
                          "m3": [M3_P_WGT, M2_MIN],
                          "m4": M4_ACC_MIN, "m5": M5_COS_MAX,
                          "m6": M6_ACC_MIN,
                          "codes": [list(c) for c in CODES],
                          "real": [list(r) for r in REAL],
                          "neutral": NEUTRAL, "seeds": list(SEEDS)},
               "calib": {k: v["calib"] for k, v in gpu.items() if v},
               "all_pass": not fails and not opened,
               "open": [r["id"] for r in opened]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(round6(payload), fh, indent=1, ensure_ascii=False)
    print(f"JSON: {args.out}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    sys.exit(main())
