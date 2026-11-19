"""
mi8_embedding_nachweis.py — entscheidender Zusatztest zu MI-8 (DATUM, kein
eigenes Verdikt-Band).

Frage: ist die in MI-8 gemessene "Rollen-Achse" eine EIGENSTÄNDIGE
Provenienz-Repräsentation — oder ist sie das Rollen-Token-Embedding selbst,
per Attention in die Inhaltspositionen kopiert?

Test (modellfrei entscheidbar, kein neues Band):
  e_diff   = embed["model"] - embed["user"]            (1152,)
  d_layer  = mean_Inhalt(model) - mean_Inhalt(user)    (MI-8 P1, je Schicht)
  cos(d_layer, e_diff)                                  je Schicht
  acc(e_diff als FESTE Richtung) auf P1 (Chat-Rolle) und P2 (Q:/A:)

Lesart (vor dem Lauf registriert):
  cos ~ 1 auf fruehen Schichten und acc(e_diff)=1.0 auf P1  -> die Achse IST
  das Embedding (Leak), keine semantische Herkunfts-Repraesentation.
  acc(e_diff) ~ 0.5 auf P2                                   -> kein Transfer
  ueber die Oberflaeche, konsistent mit MI-8 B2 FAIL.
  cos deutlich < 1                                           -> es gibt einen
  vom Embedding verschiedenen Anteil; dann ist die Achse mehr als Leak.

Lauf: python -u -m scratches.mi8_embedding_nachweis
"""
from __future__ import annotations

import json
import sys

import numpy as np

from scratches.smoke_mi8_provenienz_geometrie import (
    BOS_ID, PROBE_LAYER, _load_argon, build_transfer_cells, capture,
    collect_geometry, enc, fixed_dir_acc, loo_sign_acc, round6, unit,
)


def main() -> int:
    import torch
    print("== MI-8 Zusatz: Embedding-Nachweis ==", flush=True)
    model, meta = _load_argon("cpu", "float32")
    tok = meta["tokenizer"]
    geo = collect_geometry(tok)
    tr = build_transfer_cells(tok, geo)

    emb = model.model.embed_tokens.text_embed.weight.detach().float().cpu().numpy()
    uid = int(enc(tok, "user")[0])
    mid = int(enc(tok, "model")[0])
    e_diff = emb[mid] - emb[uid]
    e_unit = unit(e_diff)
    print(f"  e_diff Norm {np.linalg.norm(e_diff):.4f} "
          f"(user {uid}, model {mid})", flush=True)

    caps = {}
    for cell in geo["cells"]:
        store, _ = capture(torch, model, cell["ids"])
        caps[(cell["content"], cell["hist_name"], cell["role"])] = store

    n_layers = len(model.model.layers)
    rows = []
    D_all = np.zeros((n_layers, len(geo["content_ids"]), emb.shape[1]))
    for ci, cids in enumerate(geo["content_ids"]):
        sl = slice(geo["start"], geo["start"] + len(cids))
        for hi, hn in enumerate(("q", "s")):
            a = caps[(ci, hn, "model")][0]
            b = caps[(ci, hn, "user")][0]
            for li in range(n_layers):
                D_all[li, ci] += 0.5 * (caps[(ci, hn, "model")][li][sl].mean(0)
                                        - caps[(ci, hn, "user")][li][sl].mean(0))
    for li in range(n_layers):
        d = D_all[li].mean(axis=0)
        rows.append({
            "layer": li,
            "cos_ediff": float(unit(d) @ e_unit),
            "acc_ediff": float(np.mean(D_all[li] @ e_unit > 0)),
            "loo_acc": float(loo_sign_acc(D_all[li][None, :, :])[0]),
        })

    # P2 (Q:/A:) mit der Embedding-Richtung als fester Sonde
    qa = []
    for ci, cids in enumerate(geo["content_ids"]):
        for hn in ("q", "s"):
            cm = next(c for c in tr["marker"]
                      if c["content"] == ci and c["hist_name"] == hn
                      and c["role"] == "model")
            cu = next(c for c in tr["marker"]
                      if c["content"] == ci and c["hist_name"] == hn
                      and c["role"] == "user")
            sm, _ = capture(torch, model, cm["ids"])
            su, _ = capture(torch, model, cu["ids"])
            qa.append(sm[PROBE_LAYER][cm["start"]:cm["end"]].mean(0)
                      - su[PROBE_LAYER][cu["start"]:cu["end"]].mean(0))
    Dqa = np.stack(qa)
    acc_qa, p_qa = fixed_dir_acc(Dqa, e_unit)
    cos_qa = float(unit(Dqa.mean(0)) @ e_unit)

    # Outlier-Diagnose: tragen wenige Dimensionen die Differenz? (DATUM)
    diag = {}
    for li in (0, 4, 8, 12, 16, 21, 25):
        d = D_all[li].mean(axis=0)
        nrm = float(np.linalg.norm(d))
        order = np.argsort(-np.abs(d))[:8]
        top2 = float(np.abs(d[order[:2]]).sum())
        diag[f"L{li}"] = {
            "norm": nrm,
            "top8_dims": [int(i) for i in order],
            "top8_vals": [float(d[i]) for i in order],
            "top2_share": top2 / nrm if nrm > 0 else None,
            "n_dims_90pct": int(np.searchsorted(
                np.cumsum(np.sort(np.abs(d))[::-1]), 0.9 * np.abs(d).sum())
                + 1),
        }
    # typische Zustandsnorm am Probe-Layer (Inhaltspositionen)
    hnorm = []
    for cell in geo["cells"][:8]:
        st, _ = capture(torch, model, cell["ids"])
        hnorm.append(float(np.linalg.norm(
            st[PROBE_LAYER][cell["start"]:cell["end"]].mean(axis=0))))
    out = {"e_diff_norm": float(np.linalg.norm(e_diff)),
           "outlier_diag": diag,
           "h_norm_l21_mean": float(np.mean(hnorm)),
           "user_id": uid, "model_id": mid,
           "per_layer": rows,
           "qa_acc_ediff": acc_qa, "qa_p_ediff": p_qa,
           "qa_cos_ediff": cos_qa,
           "p1_acc_ediff_at_probe_layer": rows[PROBE_LAYER]["acc_ediff"],
           "verdict_reading": {
               "achse_ist_embedding": bool(rows[0]["acc_ediff"] >= 0.95
                                           and rows[0]["cos_ediff"] >= 0.9),
               "kein_marker_transfer": bool(acc_qa <= 0.70),
           }}
    print(json.dumps(round6(out), indent=1, ensure_ascii=False))
    with open("scratches/mi8_embedding_nachweis.json", "w",
              encoding="utf-8") as fh:
        json.dump(round6(out), fh, indent=1, ensure_ascii=False)
    print("JSON: scratches/mi8_embedding_nachweis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
