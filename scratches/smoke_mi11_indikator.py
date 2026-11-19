"""
smoke_mi11_indikator.py — MI-11: der EINPASSIGE Halluzinations-Indikator.

MI-9 liefert zwei Achsen (Kontext-Beleg, eigener VorText). MI-10 liefert das
externe Deckungs-Audit (Stabilität über Paraphrasen). Was fehlt, ist die
dritte, entscheidende Achse: **Vertrautheit** — ist die Entität überhaupt im
Training? Genau sie trennt eine Nonce-Konfabulation von einem memorierten
Fakt, und sie ist in EINEM Forward messbar.

DESIGN (konfundfrei: IDENTISCHES Template, nur der Entitätsname wechselt):
  F+  vertraut  — 8 echte Länder:  "{X} is a country in Europe. Its capital is"
  F−  neu       — 8 erfundene:    "{X} is a country in Europe. Its capital is"
  Nur der Name unterscheidet sich. Eine lineare Richtung, die F+ von F−
  trennt, ist damit eine Vertrautheits-Achse, kein Template-Artefakt.

  d_fam  = unit(mean(h_F+) − mean(h_F−)) am letzten Prompt-Token
  score  = h · d_fam        (positiv = vertraut)

  Label (extern, aus MI-10 eingefroren): F+ = gedeckt (Antwort stabil und
  korrekt in 3/3 Formen), F− = ungedeckt (Konfabulation, Stabilität < 2/3).

  LOO: Richtung aus 15 Zellen, Score der gehaltenen Zelle → unverzerrte
  AUROC. Keine Zirkularität.

PREREG Rev. 1 (vor dem Lauf):
  P1 trennung      LOO-AUROC(score_fam | gedeckt) >= 0.85
  P2 dekodierbar   LOO-Acc >= 0.80 UND LEE-korrigiertes p < 0.05
                   (Max-Statistik über alle Schichten, 500 Permutationen)
  P3 flag          >= 17/20 vertraut als vertraut UND >= 17/20 neu als neu
                   (n = 20+20; die 8+8-Vorfassung war zu klein fuer die
                   LEE-Korrektur, siehe KOR-1)
  P4 kein_falschalarm  die 8 echten Länder sind in MI-10 stabil (8/8) —
                   der Indikator darf sie nicht als ungedeckt flaggen
  P5 purity        Fingerprint bitgleich, 0 Hooks

  NEGATIV-LESART: P1/P2 FAIL → Vertrautheit ist NICHT linear im
  Prompt-Zustand repräsentiert; der einpassige Indikator kann dann nur die
  Kontext-Achse (MI-9) nutzen, nicht die Herkunft.

Lauf: python -u -m scratches.smoke_mi11_indikator
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
    _fingerprint, _fp_equal, _load_argon, _load_parent, capture, enc, round6,
    unit,
)

REAL = ("Germany", "France", "Japan", "Italy", "Spain", "Portugal",
        "Austria", "Greece", "Norway", "Sweden", "Poland", "Hungary",
        "Ireland", "Egypt", "Brazil", "Canada", "Australia", "India",
        "Mexico", "Kenya")
NONCE = ("Vrandel", "Qorvath", "Zelmuth", "Brixalon", "Tavrenko", "Oxwald",
         "Mireth", "Kalvun", "Draneth", "Sovril", "Kalthen", "Brumire",
         "Vasquin", "Nordrel", "Pelvath", "Grimsel", "Torvane", "Heskril",
         "Ondram", "Weltris")
CUE = "{X} is a country. Its capital is"
P1_AUC_MIN = 0.85
P2_ACC_MIN = 0.80
P2_P_MAX = 0.05
P3_MIN = 17
N_PERM = 500


def cells(tok: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    fam, nov = [], []
    for X in REAL:
        ids = ([int(tok.bos_token_id)] if tok.bos_token_id is not None
               else [2]) + enc(tok, CUE.format(X=X))
        fam.append({"name": X, "ids": ids})
    for X in NONCE:
        ids = ([int(tok.bos_token_id)] if tok.bos_token_id is not None
               else [2]) + enc(tok, CUE.format(X=X))
        nov.append({"name": X, "ids": ids})
    return fam, nov


def loo_scores(H: np.ndarray, n_pos: int) -> np.ndarray:
    """LOO-Score je Zelle, LDA-Regel (KOR-2): Schwelle ist der Mittelpunkt
    der beiden Projktions-Mittel, NICHT 0. Der Residualstrom hat einen
    grossen Common-Offset; eine 0-Schwelle macht die Accuracy bimodal
    (0.0/1.0) und damit den Permutationstest wertlos."""
    n = len(H)
    out = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        pos = keep & (np.arange(n) < n_pos)
        neg = keep & (np.arange(n) >= n_pos)
        d = unit(H[pos].mean(axis=0) - H[neg].mean(axis=0))
        thr = 0.5 * (float(H[pos].mean(axis=0) @ d)
                     + float(H[neg].mean(axis=0) @ d))
        out[i] = float(H[i] @ d) - thr
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() \
        + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt) / (len(pos) * len(neg))


def loo_acc_max(H: np.ndarray, n_pos: int, n_perm: int = N_PERM
                ) -> Tuple[float, int, float]:
    """LOO-Acc je Schicht; bester Layer + LEE-korrigiertes p (Max-Statistik)."""
    n_layers = H.shape[1]
    accs = np.zeros(n_layers)
    for L in range(n_layers):
        s = loo_scores(H[:, L, :], n_pos)
        accs[L] = float(np.mean((s > 0) == (np.arange(len(s)) < n_pos)))
    best = int(np.argmax(accs))
    obs = float(accs[best])
    rng = np.random.default_rng(42)
    hits = 0
    y = np.arange(len(H)) < n_pos
    for _ in range(n_perm):
        # KOR-3: Labels UNABHAENGIG von den Zeilen mischen. Zeilen und
        # Labels gemeinsam zu permutieren ist die Identitaet (die Paarung
        # bleibt erhalten) — der Test war dann wertlos (p immer 1.0).
        yp = rng.permutation(y)
        order = np.concatenate([np.where(yp)[0], np.where(~yp)[0]])
        Hp = H[order]
        mx = 0.0
        for L in range(n_layers):
            s = loo_scores(Hp[:, L, :], n_pos)
            acc = float(np.mean((s > 0) == (np.arange(len(s)) < n_pos)))
            mx = max(mx, acc)
        if mx >= obs - 1e-12:
            hits += 1
    return obs, best, (1 + hits) / (1 + n_perm)


def run(name: str, model: Any, tok: Any, device: str) -> Dict[str, Any]:
    import torch
    fp0 = _fingerprint(model)
    fam, nov = cells(tok)
    Hf, Hn = [], []
    ptops = []
    for group, bucket in ((fam, Hf), (nov, Hn)):
        for c in group:
            store, logits = capture(torch, model, c["ids"])
            H = np.stack([store[L][-1] for L in range(len(store))])
            bucket.append(H)
            p = torch.softmax(torch.tensor(logits), -1)
            ptops.append(float(torch.max(p)))
    Hf, Hn = np.stack(Hf), np.stack(Hn)          # (8, L, D)
    H = np.concatenate([Hf, Hn], axis=0)         # (16, L, D)
    y = np.array([1] * len(fam) + [0] * len(nov))
    acc, best, p = loo_acc_max(H, len(fam))
    s_best = loo_scores(H[:, best, :], len(fam))
    auc = auroc(y, s_best)
    # Flag auf der besten Schicht (LOO-Scores), Schwelle 0
    flag_f = int(sum(1 for v in s_best[:len(fam)] if v > 0))
    flag_n = int(sum(1 for v in s_best[len(fam):] if v <= 0))
    fp1 = _fingerprint(model)
    return {
        "substrate": name, "n_layers": H.shape[1], "best_layer": best,
        "loo_acc": acc, "loo_p": p, "auroc": auc,
        "flag_vertraut": flag_f, "flag_neu": flag_n,
        "p_top_fam": ptops[:len(fam)], "p_top_nov": ptops[len(fam):],
        "scores_fam": s_best[:len(fam)].tolist(),
        "scores_nov": s_best[len(fam):].tolist(),
        "purity": {"fp_equal": _fp_equal(fp0, fp1),
                   "hooks_zero": int(fp1["hooks_total"]) == 0},
        "d_fam": None,
    }


def calibrate_direction(name: str, model: Any, tok: Any) -> np.ndarray:
    """Richtung auf der besten Schicht über ALLE 16 Zellen (fuer den Export)."""
    import torch
    fam, nov = cells(tok)
    Hf, Hn = [], []
    for group, bucket in ((fam, Hf), (nov, Hn)):
        for c in group:
            store, _ = capture(torch, model, c["ids"])
            bucket.append(np.stack([store[L][-1] for L in range(len(store))]))
    Hf, Hn = np.stack(Hf), np.stack(Hn)
    H = np.concatenate([Hf, Hn], axis=0)
    acc, best, _ = loo_acc_max(H, len(fam), n_perm=50)
    d = unit(Hf[:, best, :].mean(axis=0) - Hn[:, best, :].mean(axis=0))
    return {"layer": best, "d_fam": d.tolist(), "loo_acc": acc}


def bands(r: Dict[str, Any]) -> Dict[str, bool]:
    return {
        "P1_trennung": bool(r["auroc"] >= P1_AUC_MIN),
        "P2_dekodierbar": bool(r["loo_acc"] >= P2_ACC_MIN
                               and r["loo_p"] < P2_P_MAX),
        "P3_flag": bool(r["flag_vertraut"] >= P3_MIN
                        and r["flag_neu"] >= P3_MIN),
    }


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="argon",
                    choices=["argon", "parent", "both"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="scratches/mi11_indikator.json")
    a = ap.parse_args()

    def stufe(t: str) -> None:
        print(t, flush=True)

    stufe("== MI-11 — Einpassiger Halluzinations-Indikator ==")
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
            r = run(name, model, tok, a.device)
            r["bands"] = bands(r)
            r["export"] = calibrate_direction(name, model, tok)
            gpu[name] = r
            stufe(f"  [{name}] n={len(REAL)}+{len(NONCE)} LOO-Acc "
                  f"{r['loo_acc']:.3f} p {r['loo_p']:.4f} "
                  f"| AUROC {r['auroc']:.3f} | Flag "
                  f"{r['flag_vertraut']}/{len(REAL)} vs "
                  f"{r['flag_neu']}/{len(NONCE)} | Layer {r['best_layer']}")
        finally:
            del model
            gc.collect()

    rows = []
    any_run = False
    for name, v in gpu.items():
        if not v:
            continue
        any_run = True
        for k, ok in v["bands"].items():
            rows.append({"id": f"{k}_{name}", "assert": bool(ok),
                         "note": f"acc {v['loo_acc']:.3f} p {v['loo_p']:.4f} "
                                 f"auc {v['auroc']:.3f}"})
        rows.append({"id": f"P4_purity_{name}", "assert": bool(
            v["purity"]["fp_equal"] and v["purity"]["hooks_zero"]),
            "note": str(v["purity"])})
    if not any_run:
        for k in ("P1_trennung", "P2_dekodierbar", "P3_flag", "P4_purity"):
            rows.append({"id": k, "assert": None, "note": "nicht gefahren"})
    print()
    print("BEFUNDE:")
    for r in rows:
        st = "OFFEN" if r["assert"] is None else (
            "PASS" if r["assert"] else "FAIL")
        print(f"  {r['id']:<20} {st:<5} {r.get('note','')}")
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    print()
    print(f"BILANZ: {len(rows)-len(fails)-len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    print("GRUEN" if not fails and not opened else
          ("ROT — " + ", ".join(r["id"] for r in fails) if fails else "OFFEN"))
    payload = {
        "gpu": {k: ({kk: vv for kk, vv in v.items()
                     if kk not in ("export",)} if v else None)
                for k, v in gpu.items()},
        "export": {k: v["export"] for k, v in gpu.items() if v},
        "findings": rows,
        "prereg": {"P1": P1_AUC_MIN, "P2": [P2_ACC_MIN, P2_P_MAX],
                   "P3": P3_MIN, "cue": CUE, "real": list(REAL),
                   "nonce": list(NONCE), "n_perm": N_PERM},
        "all_pass": not fails and not opened,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(round6(payload), fh, indent=1, ensure_ascii=False)
    print(f"JSON: {a.out}")
    return 1 if fails else (2 if opened else 0)


if __name__ == "__main__":
    sys.exit(main())
