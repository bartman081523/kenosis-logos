"""
provenienz_monitor.py — CLI für den MI-9-Drei-Quellen-Monitor.

Liest eine Kalibrierung (aus `smoke_mi9_drei_quellen.py`) und annotiert eine
Textstelle tokenweise mit den Quell-Scores W (Training), C (Kontext),
S (eigene Gedanken).

    python -u -m scratches.provenienz_monitor \
        --calib scratches/mi9_monitor_both.json --substrate argon \
        --text "The capital of France is"

  --file PATH       statt --text (UTF-8, eine Zeile oder ganzer Text)
  --json            Rohausgabe (JSON) statt Tabelle

WICHTIG (Auflösungs-Grenze, im Gerät registriert): die Achsen sind am
LETZTEN Token der MI-9-Kalibrier-Zellen geschätzt. Die tokenweise
Anwendung auf beliebige Positionen ist eine Extrapolation; belastbar ist
der Readout am Ende eines Slots/Turns, nicht in der Mitte eines Satzes.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from scratches.smoke_mi8_provenienz_geometrie import (
    BOS_ID, _load_argon, _load_parent, capture, enc, round6,
)
from scratches.smoke_mi9_drei_quellen import _basis, _classify, _project


def load_calib(path: str, substrate: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    calib = payload.get("calib") or {}
    if substrate not in calib:
        raise SystemExit(
            f"Kalibrierung für {substrate!r} fehlt (vorhanden: "
            f"{sorted(calib)}). Erst `smoke_mi9_drei_quellen.py "
            f"--substrate both` laufen lassen.")
    return calib[substrate]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", default="scratches/mi9_monitor_both.json")
    ap.add_argument("--substrate", default="argon",
                    choices=["argon", "parent"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--file")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    calib = load_calib(args.calib, args.substrate)
    d_ctx = np.asarray(calib["d_ctx"], dtype=np.float64)
    d_s = np.asarray(calib["d_s"], dtype=np.float64)
    b1, b2 = _basis(d_ctx, d_s)
    C = {k: np.asarray(v, dtype=np.float64)
         for k, v in calib["centroids_2d"].items()}
    layer = int(calib["layer"])

    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            text = fh.read()
    else:
        text = args.text

    import torch
    if args.substrate == "argon":
        model, meta = _load_argon(args.device, "float32")
    else:
        model, meta = _load_parent(args.device, "float32")
    tok = meta["tokenizer"]
    ids = [BOS_ID] + enc(tok, text)
    store, _ = capture(torch, model, ids)
    H = store[layer]

    rows = []
    for t in range(len(ids)):
        P = _project(H[t][None, :], b1, b2)
        cls = _classify(P, C)[0]
        dist = {k: float(np.linalg.norm(P[0] - v)) for k, v in C.items()}
        rows.append({
            "pos": t,
            "token": tok.decode([ids[t]]),
            "cls": cls,
            "score_ctx": float(H[t] @ b1),
            "score_s": float(H[t] @ b2),
            "dist": dist,
        })

    # Lesart: am ENDE der Sequenz ist der Readout belastbar
    last = rows[-1]
    if args.json:
        print(json.dumps(round6({"substrate": args.substrate,
                                 "layer": layer, "last": last,
                                 "rows": rows}), indent=1,
                         ensure_ascii=False))
        return 0
    print(f"Kalibrierung: {args.calib} [{args.substrate}], Schicht {layer}")
    print(f"Text: {text!r} ({len(ids)} Token inkl. BOS)")
    print(f"{'pos':>4} {'token':<14} {'Klasse':<7} {'Kontext':>9} {'S':>9}")
    for r in rows:
        print(f"{r['pos']:>4} {r['token']!r:<14} {r['cls']:<7} "
              f"{r['score_ctx']:>9.2f} {r['score_s']:>9.2f}")
    print()
    print(f"READOUT (letztes Token): {last['cls']}  "
          f"(Abstände: " + ", ".join(f"{k} {v:.1f}"
                                     for k, v in sorted(last['dist'].items()))
          + ")")
    print("Hinweis: belastbar am Slot-/Turn-Ende; tokenweise Anwendung ist "
          "Extrapolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
