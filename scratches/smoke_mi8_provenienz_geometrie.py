"""
smoke_mi8_provenienz_geometrie.py — MI-8: Provenienz-Geometrie.

FRAGE. Gibt es im Residualstrom eine trennbare, kausale Repräsentation der
HERKUNFT eines Tokens (fremd/User vs. eigen/Modell vs. Gewicht vs. Kontext)
— oder ist "Provenienz" ausschließlich die kontrafaktische Abhängigkeit, die
wir von außen per Ablation erzwingen (MI-7/SP16) und im Modell selbst gar
nicht als Achse existiert?

PHYSISCHES AXIOM (vor jeder Messung). Der Residualstrom ist kein Träger,
sondern der Bus. Ein Token durchläuft dieselbe Embedding-Matrix, egal wer es
geschrieben hat; ein Provenienz-BIT kann es nicht geben. Unterscheidbar sind
nur drei Dinge:
  (1) KONTRAFAKTISCHE ABHÄNGIGKEIT (W vs. C, MI-7-Kanon),
  (2) eine LINEARE RICHTUNG im Hidden (dekodierbar, evtl. nur Token-Leak),
  (3) eine VERHALTENS-POLITIK (setzt das Modell seinen eigenen VorText als
      eigenen fort oder als fremde Anweisung?).
MI-8 registriert alle drei getrennt und behauptet keine Identität.

DESIGN (matched pairs; Inhalt byte-identisch, Inhaltsposition identisch):
  P1 ROLLE   — 12 Inhalte × 4 Zellen = Rolle {user, model} × Vorgeschichte
               {Frage, Feststellung}. Die Vorgeschichte ist BEIDE 7 Token
               lang (registriert unten), der Inhalt beginnt in allen vier
               Zellen an derselben Position. Da Attention kausal ist, kann
               NACH dem Inhalt nichts die Inhaltsposition mehr beeinflussen —
               deshalb ist "letzter vs. mittlerer Turn" KEIN Faktor: er wäre
               bei Inhaltspositionen wirkungslos. Vorgeschichte-Rolle ist der
               einzige kausal verfügbare Zweitfaktor.
  P2 MARKER  — Rolle anders oberflächlich markiert ("Q: " vs "A: ", je
               3 Token). Transfer-Test: trennt die aus P1 gelernte Richtung
               auch die andere Markierung? Nur dann ist es eine semantische
               Herkunfts-Achse und nicht das lokale Rollen-Token-Embedding.
  P3 POSITION— Inhalt um +DELTA neutrale Token nach hinten geschoben;
               Richtung muss stabil bleiben (Provenienz != Position).
  P4 VORGESCHICHTE— Inhalt-Turn ohne Rollenmarker ("anon"), Vorgeschichte
               als Modell- vs User-Turn (gleicher 7-Token-Text). Trägt die
               Vorgeschichte die Herkunft allein? (KOR-1: die frühere
               "anon"-Ablation war degeneriert — beide Seiten identisch.)
  P5 GENERIERT— 6 eigene 12-Token-Fortsetzungen (greedy). Derselbe
               Token-ID-String als model- bzw. user-Turn (IDs, nicht Text —
               kein Re-Tokenisierungs-Bruch). Ökologische Validierung.
  P6 W/C     — 6 Gewichts-Fakten + 6 kontext-injizierte Kunst-Fakten
               (MI-7-Form); Richtung am Schlitz. Interne Analogie zur
               externen chi_w/chi_c-Zerlegung.
  P7 KAUSAL  — d_role (aus P1, beste Schicht) per forward_hook am
               Layer-Output, relay-konform (alpha * ||h|| * d_unit). Arme:
               +d im FREMD-Arm (eigener Text als user), -d im EIGEN-Arm,
               3 norm-gematchte Placebo-Richtungen, Dosis {0.5, 1, 2}.
               Metrik: Delta log p(c), c = das 13. selbst-generierte Token.
  P8 KONTROLLE— d_width-Relay-Richtung (ML4-Artefakt, 1152-dim, L16->L21)
               als orthogonaler Kanal: darf die Rollen-Sonde nicht tragen.

PREREG Rev. 1 (VOR dem Erstlauf fixiert; Bänder werden nach dem Lauf NICHT
geweitet — ein Bruch ist ein gebuchter Bruch, kein Relabeling):

  B1 rolle_dekodierbar   LOO-Sign-Acc >= 0.80 UND LEE-korrigiertes
                         p_max < 0.01 (Max-Statistik über ALLE 26 Schichten,
                         1000 Vorzeichen-Permutationen), Substrat argon.
  B2 marker_transfer     dieselbe Richtung (beste P1-Schicht, ohne Auswahl)
                         trennt "Q: "/"A: " mit Acc >= 0.70, p < 0.05.
  B3 kausal              +d im FREMD-Arm: Delta log p(c) >= +0.05 nats;
                         -d im EIGEN-Arm: <= -0.05; Vorzeichen 2/2;
                         Placebo max |Delta| <= 0.02; Dosis monoton
                         (Delta(2) >= Delta(1) >= Delta(0.5), Toleranz 1e-3).
  B4 positions_invarianz Shift +DELTA: Acc >= 0.75.
  B5 marker_notwendig    Inhalt-Turn "anon", nur die Vorgeschichte
                         (Modell- vs User-Turn) unterscheidet: Acc <= 0.60
                         = Marker notwendig. Acc >= 0.70 wäre ein
                         WIDERLEGTES REGISTER (Herkunft ohne Marker aus der
                         Vorgeschichte lesbar), kein Relabeling.
  B6 generiert           P5 Acc >= 0.70.
  B7 wc_getrennt         d_src (W/C) Acc >= 0.75 UND cos(d_role, d_src)
                         <= 0.5 (keine gemeinsame Universal-Achse).
  B8 purity              Fingerprint bitgleich, 0 Hooks nach Lauf,
                         Doppellauf bitidentisch.

  NEGATIV-LESART (gleichwürdig, VOR dem Lauf registriert):
   - B1 FAIL, B3 FAIL  -> "keine trennbare Quell-Repräsentation; Herkunft
                           ist nur kontrafaktische Abhängigkeit (MI-7/SP16
                           bleibt die einzige Messtore)". Kein Defekt.
   - B1 PASS, B2 FAIL  -> "Rollen-Achse existiert, ist aber oberflächlich
                           (Token-Leak), nicht semantisch".
   - B2 PASS, B3 FAIL  -> "Herkunft ist lesbar, aber nicht benutzt" (Lens-
                           Datum ohne Kausalkraft).
   - B7 FAIL (cos hoch)-> "eine gemeinsame Quell-Achse für W/C und Rolle"
                           (stärkerer Befund als die Trennung).

KORREKTUR-LOG (bei Erst-PREREG leer; Einträge nur für beweisliche Lauf-Bugs,
nie Band-Weitung):

KOR-1 (2026-11-17, beweislich, Lauf-Bug): der Erstlauf der Modell-Lane
brach an der P4-Ablation ab — `build_transfer_cells` baute für "anon"
BEIDE Seiten identisch (source_role nur als Etikett), also D = 0 exakt und
`unit()` verweigerte den Nullvektor. Die Ablation war degeneriert, nicht
das Modell. Ersatz OHNE Band-Weitung: P4 misst jetzt, ob die VORGESCHICHTE
(Modell- vs User-Turn, gleicher 7-Token-Text, Inhalt-Turn "anon") die
Herkunft allein trägt. Registriertes Band B5 unverändert die
Notwendigkeits-Lesart (Acc <= 0.60 = Marker notwendig); ein hoher Wert
wäre ein widerlegtes Register, kein Relabeling. Roh-Log des Bruchs:
`scratches/mi8_erstrun_gebrochen.log`. P1-Datum aus diesem Lauf (verbatim):
acc_max 1.000, p_lee 0.0020, best_layer 0 — es gibt KEINE Transfer- oder
Kausal-Zahl aus dem gebrochenen Lauf (kein JSON geschrieben).

KOR-2 (2026-11-17, Instrument-Präzisierung VOR der ersten Transfer-Messung):
B1 bleibt LEE-korrigiert über alle 26 Schichten (Max-Statistik). Alle
TRANSFER- und INTERVENTIONS-Arme (P2/P3/P4/P5/P6/P7) benutzen aber die VOR
dem Lauf registrierte Schicht 21 (ML4-Post-Washout-Anker), NICHT die
argmax-Schicht — sonst würde eine Auswahl über 26 Schichten in die
Transfer-Accuracy eingehen. Kein Bandwert geändert.

KOR-3 (2026-11-17, beweislich, Lauf-Bug): der zweite Modell-Lauf brach in
`direction_steer` mit NameError 'torch' ab — `torch` wurde nur in `main`
importiert, der Hook nutzte es im Modul-Scope. Fix: lokaler Import im
Steer-ContextManager. Kein Band berührt; P1/P2/P3/P4/P5-Zahlen des Laufs
stehen im Log (`scratches/mi8_run_argon_cpu.log`), aber ohne JSON.

KOR-4 (2026-11-17, Instrument-Gate VOR der ersten B7-Zahl): die W/C-
Akzeptanz war als ABSOLUTES p-Gate registriert (W: p_abl >= 0.50; C:
p_full >= 0.50 und p_abl <= 0.10) und verwarf alle 12 Zellen (P6 = None).
Ersatz durch das MI-7-Kanon-Kriterium "das Modell EMITTIERT es": W-Zelle
gültig, wenn argmax(p_nackt) == Ziel; C-Zelle gültig, wenn
argmax(p_kontext) == Ziel UND argmax(p_abl) != Ziel. Das ist die
kontrafaktische Definition der Arme, nicht eine Band-Weitung; alle
p-Werte werden weiterhin verbatim gebucht.

KOR-5 (2026-11-17, beweislich, Instrument-Defekt): Rev-1 der Kausal-Lane
injizierte alpha*||h_last||*d_unit. Bei unit-norm d und grossem ||h||
sättigt das die Softmax: Placebo-Richtungen erreichten |Delta log p| bis
21.67 nats gegen Treatment -14.27 (Dosis "monoton" nur im Schaden). Rev-2
registriert ABSOLUTE Injektion alpha * s * d mit s = Median der
Paar-Differenz-Normen ||h_model - h_user|| auf Schicht 21 — alpha=1 heisst
"um eine mittlere Herkunfts-Separation verschieben". Rev-1-Zahlen bleiben
verbatim im JSON (`causal`), Rev-2 läuft als `causal` mit `rev: 2`.

KOR-6 (2026-11-17, beweislich, Instrument-Defekt): alle Sequenzen liefen
OHNE <bos>. Gemessen am nackten Schlitz: p(" Paris") = 0.0000 ohne BOS vs
0.4873 mit BOS — der Zustand war off-distribution, die W/C-Emission damit
unbrauchbar (0/6 W-Zellen). Fix: `assemble()` prependet BOS_ID=2 in jeder
Zelle, jeder Generierung und jedem W/C-Forward. Alle Bänder unverändert;
die Rev-1-Zahlen (ohne BOS) bleiben als widerlegtes Register im Log
`scratches/mi8_run_argon_cpu_nobos.log`.

Lauf:
  CPU-Pilot : python -u -m scratches.smoke_mi8_provenienz_geometrie
  Modell    : python -u -m scratches.smoke_mi8_provenienz_geometrie --model
              [--substrate argon|parent|both] [--device cpu|cuda]
  Stufen    : --stage pilot|geom|gen|wc|causal|all
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SOT = "<start_of_turn>"
EOT = "<end_of_turn>"
SOT_ID, EOT_ID, BOS_ID = 105, 106, 2

# ---------------------------------------------------------------------------
# Registrierte Bänder (PREREG Rev. 1)
# ---------------------------------------------------------------------------
B1_ACC_MIN = 0.80
B1_P_MAX = 0.01
B2_ACC_MIN = 0.70
B2_P_MAX = 0.05
B3_DLOG_MIN = 0.05          # nats
B3_PLACEBO_MAX = 0.02
B3_DOSES = (0.5, 1.0, 2.0)
B4_ACC_MIN = 0.75
B5_ACC_MAX = 0.60
B6_ACC_MIN = 0.70
B7_ACC_MIN = 0.75
B7_COS_MAX = 0.5
N_PERM = 1000
DELTA_SHIFT = 5             # neutrale Filler-Token für P3
PROBE_LAYER = 21            # KOR-2: registrierte Schicht für Transfer/Kausal
N_GEN = 12                  # eigene Token in P5
GEN_SEEDS = (
    "The harbour lights fade",
    "A copper wire carries",
    "The old clock in the hallway",
    "Rain has been falling on",
    "The garden gate stays",
    "A single candle burns",
)

# ---------------------------------------------------------------------------
# Registrierter Korpus (neutral, keine Rollenwörter; Inhalte byte-identisch
# in allen Zellen, damit jede h-Differenz am Inhalt reine Kontext-Folge ist)
# ---------------------------------------------------------------------------
CONTENTS = (
    "The harbour lights fade after midnight.",
    "A copper wire carries the signal faster than glass.",
    "The old clock in the hallway loses two minutes a day.",
    "Rain has been falling on the roof since noon.",
    "The garden gate stays open in summer.",
    "A single candle burns at the far end of the table.",
    "The river bends twice before it reaches the bridge.",
    "Papers on the desk are sorted by date.",
    "The window faces east and catches the morning sun.",
    "A small bird lands on the iron railing.",
    "The tea has gone cold in the blue cup.",
    "Footsteps echo along the empty corridor.",
)

# Beide Vorgeschichten MÜSSEN gleich viele Token haben (7) — im Pilot
# bewiesen, kein stiller Fallback. Die Feststellung ist bewusst Inhalt der
# Vorgeschichte, damit der Zweitfaktor nicht mit "Fragezeichen" kollidiert.
HIST_Q = "What follows the harbour note now?"     # 7 Token
HIST_S = "The harbour note is ready now."          # 7 Token
HIST_LEN = 7
NEUTRAL_TOKEN = " the"                            # 1 Token (Filler für P3)

# P6 — MI-7-Form (ARME_VORGABE dort verbatim): Ziel ist das ERSTE Token des
# Namens (MI-7-Kanon: " Zirkon" -> id 1491 = " Z").
WC_PAIRS = (
    ("The capital of France is Paris. The capital of France is",
     "The capital of France is", " Paris"),
    ("The capital of Germany is Berlin. The capital of Germany is",
     "The capital of Germany is", " Berlin"),
    ("The capital of Japan is Tokyo. The capital of Japan is",
     "The capital of Japan is", " Tokyo"),
    ("The capital of Italy is Rome. The capital of Italy is",
     "The capital of Italy is", " Rome"),
    ("The capital of Spain is Madrid. The capital of Spain is",
     "The capital of Spain is", " Madrid"),
    ("The largest planet is Jupiter. The largest planet is",
     "The largest planet is", " Jupiter"),
)

WC_INVENTED = (
    ("In this protocol, the secret code of box 7 is Zirkon. "
     "Therefore, the secret code of box 7 is",
     "Therefore, the secret code of box 7 is", " Zirkon"),
    ("In this protocol, the secret code of vault 3 is Quorin. "
     "Therefore, the secret code of vault 3 is",
     "Therefore, the secret code of vault 3 is", " Quorin"),
    ("In this protocol, the secret code of river 12 is Vantel. "
     "Therefore, the secret code of river 12 is",
     "Therefore, the secret code of river 12 is", " Vantel"),
    ("In this protocol, the secret code of tower 5 is Brixal. "
     "Therefore, the secret code of tower 5 is",
     "Therefore, the secret code of tower 5 is", " Brixal"),
    ("In this protocol, the secret code of ship 9 is Morvel. "
     "Therefore, the secret code of ship 9 is",
     "Therefore, the secret code of ship 9 is", " Morvel"),
    ("In this protocol, the secret code of gate 4 is Tessin. "
     "Therefore, the secret code of gate 4 is",
     "Therefore, the secret code of gate 4 is", " Tessin"),
)

ML4_DWIDTH = ("/run/media/julian/ML4/ollama-work/all_space_6_16_stand/"
              "px_manifolds/google_gemma-3-1b-it_relay_dwidth.json")


# ---------------------------------------------------------------------------
# Kleine Werkzeuge
# ---------------------------------------------------------------------------
def round6(x: Any) -> Any:
    if isinstance(x, float):
        return None if not math.isfinite(x) else round(x, 6)
    if isinstance(x, (np.floating,)):
        return round6(float(x))
    if isinstance(x, (list, tuple)):
        return [round6(v) for v in x]
    if isinstance(x, dict):
        return {k: round6(v) for k, v in x.items()}
    return x


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise RuntimeError("unit: Nullvektor — kein stiller Fallback.")
    return np.asarray(v, dtype=np.float64) / n


def enc(tok: Any, text: str) -> List[int]:
    return [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]


def find_span(ids: Sequence[int], needle: Sequence[int]) -> int:
    """Eindeutige Position von needle in ids; mehrdeutig/fehlend -> Bruch."""
    needle = list(needle)
    if not needle:
        raise RuntimeError("find_span: leere Nadel.")
    hits = [i for i in range(len(ids) - len(needle) + 1)
            if list(ids[i:i + len(needle)]) == needle]
    if len(hits) != 1:
        raise RuntimeError(
            f"find_span: {len(hits)} Treffer statt 1 — Gerät-Bruch "
            f"(Nadel {needle[:6]}…, ids {list(ids)[:12]}…).")
    return hits[0]


# ---------------------------------------------------------------------------
# Zell-Bau: Inhalt als ID-String (kein Re-Tokenisieren) — für Strings und für
# generierte Token identisch; Inhaltsposition wird explizit geprüft.
# ---------------------------------------------------------------------------

def assemble(tok: Any, prefix: str, content_ids: Sequence[int]
             ) -> Tuple[List[int], int]:
    """BOS + Prefix + Inhalt (KOR-6: Gemma/Argon ist mit <bos> antrainiert;
    ohne BOS ist der Zustand off-distribution — gemessen: p(" Paris") 0.0000
    ohne vs 0.4873 mit BOS am nackten Schlitz)."""
    pre = [BOS_ID] + enc(tok, prefix)
    ids = pre + list(content_ids)
    start = len(pre)
    if list(ids[start:start + len(content_ids)]) != list(content_ids):
        raise RuntimeError("assemble: Inhalt nicht an der registrierten "
                           "Position — Gerät-Bruch, kein stiller Fallback.")
    return ids, start


def cell_role(tok: Any, role: str, content_ids: Sequence[int],
              hist: Optional[str] = None) -> Dict[str, Any]:
    """Zelle mit Chat-Rollenmarker. hist=None -> nur Rollen-Prefix."""
    if role not in ("user", "model", "anon"):
        raise ValueError(f"cell_role: unbekannte Rolle {role!r}.")
    if hist is None:
        prefix = f"{SOT}{role}\n"
    else:
        prefix = f"{SOT}user\n{hist}{EOT}\n{SOT}{role}\n"
    ids, start = assemble(tok, prefix, content_ids)
    return {"ids": ids, "start": start,
            "end": start + len(content_ids), "role": role, "hist": hist}


def cell_marker(tok: Any, marker: str, content_ids: Sequence[int],
                hist: Optional[str] = None) -> Dict[str, Any]:
    """Zelle mit Text-Marker 'Q: '/'A: ' statt Chat-Rolle (P2)."""
    if marker not in ("Q:", "A:"):
        raise ValueError(f"cell_marker: unbekannter Marker {marker!r}.")
    if hist is None:
        prefix = f"{marker} "
    else:
        prefix = f"{SOT}user\n{hist}{EOT}\n{marker} "
    ids, start = assemble(tok, prefix, content_ids)
    return {"ids": ids, "start": start,
            "end": start + len(content_ids), "role": marker, "hist": hist}


# ---------------------------------------------------------------------------
# Capture: alle 26 Schicht-Ausgänge, per Token, voller 1152-Raum.
# (Kein Pooling, keine JL-Projektion — Auflösungs-Protokoll: der Kanon-Read
#  mittelt über Positionen und projiziert 144:1; Provenienz ist per-Token.)
# ---------------------------------------------------------------------------
def layer_h(o: Any) -> Any:
    if isinstance(o, (tuple, list)):
        return o[0]
    if hasattr(o, "last_hidden_state"):
        return o.last_hidden_state
    if hasattr(o, "hidden_states"):
        return o.hidden_states
    return o


def rebuild_layer(o: Any, h_new: Any) -> Any:
    if isinstance(o, (tuple, list)):
        return (h_new,) + tuple(o[1:])
    if hasattr(o, "last_hidden_state"):
        raise RuntimeError("rebuild_layer: Container mit last_hidden_state — "
                           "Steering-Pfad nicht registriert.")
    return h_new


@contextmanager
def direction_steer(model: Any, layer_idx: int, vec: np.ndarray, alpha: float,
                    sign: int = 1, where: str = "last",
                    scale: Optional[float] = None):
    """Relay-konformes Einspeisen: alpha * ||h_t|| * d_unit, reiner
    forward_hook, Restaurierung durch hook.remove() mit Zähl-Audit."""
    import torch                       # KOR-3: Modul-Lane importiert torch
    if where not in ("last", "all"):   # lokal, nicht im Modul-Scope
        raise ValueError(f"direction_steer: where {where!r} unbekannt.")
    if not math.isfinite(alpha):
        raise ValueError("direction_steer: alpha nicht endlich.")
    v = unit(np.asarray(vec, dtype=np.float64))
    state = {"calls": 0}

    def hook(_module, _inp, o):
        h = layer_h(o)
        d = torch.as_tensor(v, dtype=h.dtype, device=h.device)
        state["calls"] += 1
        # REV-2 (KOR-5): mit scale wird ABSOLUT in Zustands-Einheiten
        # eingespritzt (alpha * scale * d), sonst relay-konform relativ zu
        # ||h||. Rev-1 (alpha*||h||) sättigte die Softmax (Placebo 21.7 nats
        # > Treatment) und maß Schaden, nicht Herkunft.
        if where == "last":
            n = (float(scale) if scale is not None
                 else float(h[0, -1].norm()))
            h2 = h.clone()
            h2[0, -1] = h2[0, -1] + float(sign) * float(alpha) * n * d
        else:
            n = (float(scale) if scale is not None
                 else h.norm(dim=-1, keepdim=True))
            h2 = h + float(sign) * float(alpha) * n * d
        return rebuild_layer(o, h2)

    target = model.model.layers[int(layer_idx)]
    before = len(target._forward_hooks)
    handle = target.register_forward_hook(hook)
    try:
        yield {"layer": int(layer_idx), "alpha": float(alpha),
               "sign": int(sign), "where": where, "calls": state["calls"]}
    finally:
        handle.remove()
        after = len(target._forward_hooks)
        if after != before:
            raise RuntimeError(
                f"direction_steer: Hook-Leck in Schicht {layer_idx} "
                f"({before} -> {after}) — EINHEIT C3 verletzt.")


@contextmanager
def import_torch():
    import torch
    yield torch


def capture(torch: Any, model: Any, ids: Sequence[int],
            layers: Optional[Sequence[int]] = None,
            ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Alle Schicht-Ausgänge h_out(l) als (T, D) float32 + Logits am Ende."""
    n_layers = len(model.model.layers)
    sel = list(range(n_layers)) if layers is None else [int(l) for l in layers]
    store: Dict[int, np.ndarray] = {}

    def mk(li: int):
        def hook(_m, _i, o):
            store[li] = layer_h(o)[0].detach().to("cpu", torch.float32).numpy()
        return hook

    handles = [model.model.layers[l].register_forward_hook(mk(l)) for l in sel]
    try:
        ids_t = torch.tensor([list(ids)], dtype=torch.long,
                             device=next(model.parameters()).device)
        with torch.no_grad():
            out = model(ids_t)
        logits = out.logits[0, -1].detach().float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    if len(store) != len(sel):
        raise RuntimeError(
            f"capture: {len(store)} statt {len(sel)} Schicht-Captures — "
            "Gerät-Bruch, kein stiller Fallback.")
    return store, logits


# ---------------------------------------------------------------------------
# Messmathematik (modellfrei — vom CPU-Piloten prüfbar)
# ---------------------------------------------------------------------------
def loo_sign_acc(D: np.ndarray, signs: Optional[np.ndarray] = None,
                 ) -> np.ndarray:
    """D: (L, n, d) paarweise Differenzen. LOO-Sign-Accuracy je Schicht:
    Richtung aus n-1 Paaren, Vorzeichen am gehaltenen Paar. (L,)"""
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 3:
        raise ValueError("loo_sign_acc: D muss (L, n, d) sein.")
    _, n, _ = D.shape
    if n < 3:
        raise ValueError("loo_sign_acc: n < 3 — kein LOO möglich.")
    s = np.ones(n) if signs is None else np.asarray(signs, dtype=np.float64)
    if s.shape != (n,):
        raise ValueError("loo_sign_acc: signs-Form != (n,).")
    S = D * s[None, :, None]
    tot = S.sum(axis=1)                       # (L, d)
    hits = []
    for i in range(n):
        m = (tot - S[:, i]) / float(n - 1)    # (L, d)
        nrm = np.linalg.norm(m, axis=1, keepdims=True)
        m = m / np.maximum(nrm, 1e-12)
        proj = np.sum(m * S[:, i], axis=1)    # (L,)
        hits.append(proj > 0.0)
    return np.mean(np.stack(hits, axis=0), axis=0)


def lee_max_p(D: np.ndarray, observed: np.ndarray,
              n_perm: int = N_PERM, seed: int = 42) -> Tuple[float, float]:
    """LEE-korrigierter Max-Statistik-Test über alle Schichten: p dafür, dass
    das Maximum der LOO-Acc über Schichten unter Vorzeichen-Permutation
    mindestens so groß ist wie beobachtet. Liefert (acc_max, p_max)."""
    D = np.asarray(D, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = D.shape[1]
    obs_max = float(np.max(observed))
    hits = 0
    for _ in range(int(n_perm)):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        accs = loo_sign_acc(D, signs)
        if float(np.max(accs)) >= obs_max - 1e-12:
            hits += 1
    return obs_max, (1.0 + hits) / (1.0 + float(n_perm))


def fixed_dir_acc(D: np.ndarray, d_fixed: np.ndarray,
                  n_perm: int = N_PERM, seed: int = 42,
                  ) -> Tuple[float, float]:
    """Acc einer FESTEN Richtung auf neuen Paaren + Vorzeichen-Permutation."""
    D = np.asarray(D, dtype=np.float64)
    d = unit(np.asarray(d_fixed, dtype=np.float64))
    proj = D @ d
    acc = float(np.mean(proj > 0.0))
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    hits = 0
    for _ in range(int(n_perm)):
        s = rng.choice(np.array([-1.0, 1.0]), size=n)
        if float(np.mean(s * proj > 0.0)) >= acc - 1e-12:
            hits += 1
    return acc, (1.0 + hits) / (1.0 + float(n_perm))


def mean_dir(D: np.ndarray, signs: Optional[np.ndarray] = None) -> np.ndarray:
    """Gemittelte, unit-normierte Differenz über Paare (d: (d,))."""
    D = np.asarray(D, dtype=np.float64)
    s = np.ones(D.shape[0]) if signs is None else np.asarray(signs, np.float64)
    return unit((D * s[:, None]).mean(axis=0))


def paired_diffs(h_role: Dict[str, np.ndarray],
                 span: Tuple[int, int]) -> np.ndarray:
    """Aus {zelle: (T,D)} die Differenz mean_Inhalt(model) - mean_Inhalt(user);
    Rückgabe (L-Schichten implizit über Aufrufer) — hier (n, D) je Schicht."""
    raise NotImplementedError("paired_diffs: pro Schicht aufgerufen (siehe "
                              "_role_diffs).")


def _role_diffs(h_a: np.ndarray, h_b: np.ndarray,
                span: Tuple[int, int]) -> np.ndarray:
    """h_a/h_b: (T, D). Differenz der Inhalts-Mittelwerte."""
    s, e = int(span[0]), int(span[1])
    if e <= s:
        raise ValueError("_role_diffs: leerer Inhalts-Span.")
    return h_a[s:e].mean(axis=0) - h_b[s:e].mean(axis=0)


# ---------------------------------------------------------------------------
# Stufe A — Modellfreie Geometrie (P1/P2/P3/P4-Zellbau, reine Tokenizer-Gesetze)
# ---------------------------------------------------------------------------
def collect_geometry(tok: Any, contents: Sequence[str] = CONTENTS,
                     ) -> Dict[str, Any]:
    """Baut alle matched pairs als ID-Listen und beweist die Invarianzen."""
    content_ids = [enc(tok, c) for c in contents]
    cells: List[Dict[str, Any]] = []
    for ci, (text, cids) in enumerate(zip(contents, content_ids)):
        for hist_name, hist in (("q", HIST_Q), ("s", HIST_S)):
            if len(enc(tok, hist)) != HIST_LEN:
                raise RuntimeError(
                    f"collect_geometry: Vorgeschichte {hist_name!r} hat "
                    f"{len(enc(tok, hist))} statt {HIST_LEN} Token.")
            for role in ("user", "model"):
                cell = cell_role(tok, role, cids, hist=hist)
                cell.update({"content": ci, "hist_name": hist_name,
                             "content_ids": list(cids)})
                cells.append(cell)
    # Invarianz: Inhaltsposition und Inhalts-IDs in ALLEN Zellen identisch
    starts = {int(c["start"]) for c in cells}
    if len(starts) != 1:
        raise RuntimeError(
            f"collect_geometry: Inhaltspositionen nicht invariant "
            f"({sorted(starts)}) — Gerät-Bruch.")
    for c in cells:
        if list(c["ids"][c["start"]:c["end"]]) != list(c["content_ids"]):
            raise RuntimeError("collect_geometry: Inhalts-IDs verschoben.")
    return {"cells": cells, "contents": list(contents),
            "content_ids": content_ids, "start": starts.pop(),
            "hist_q": HIST_Q, "hist_s": HIST_S, "hist_len": HIST_LEN}


def cell_history(tok: Any, hist_role: str, hist_text: str,
                 content_ids: Sequence[int]) -> Dict[str, Any]:
    """Zelle für P4: Vorgeschichte-Rolle variiert, Inhalt-Turn ist "anon"."""
    if hist_role not in ("user", "model"):
        raise ValueError(f"cell_history: unbekannte Rolle {hist_role!r}.")
    prefix = f"{SOT}{hist_role}\n{hist_text}{EOT}\n{SOT}anon\n"
    ids, start = assemble(tok, prefix, content_ids)
    return {"ids": ids, "start": start, "end": start + len(content_ids),
            "hist_role": hist_role, "content_ids": list(content_ids)}


def build_transfer_cells(tok: Any, geometry: Dict[str, Any]) -> Dict[str, Any]:
    """P2 (Marker Q:/A:), P3 (Position +DELTA), P4 (Vorgeschichte-Rolle)."""
    marker: List[Dict[str, Any]] = []
    shifted: List[Dict[str, Any]] = []
    hist_cells: List[Dict[str, Any]] = []
    filler = enc(tok, NEUTRAL_TOKEN * DELTA_SHIFT)
    if len(filler) != DELTA_SHIFT:
        raise RuntimeError(
            f"build_transfer_cells: Filler {len(filler)} statt "
            f"{DELTA_SHIFT} Token.")
    for ci, cids in enumerate(geometry["content_ids"]):
        for hist_name, hist in (("q", HIST_Q), ("s", HIST_S)):
            for role, marker_name in (("user", "Q:"), ("model", "A:")):
                cell = cell_marker(tok, marker_name, cids, hist=hist)
                cell.update({"content": ci, "hist_name": hist_name,
                             "role": role, "content_ids": list(cids)})
                marker.append(cell)
            # P3: Inhalt nach hinten, Vorgeschichte mit Filler aufgefüllt
            for role in ("user", "model"):
                base = cell_role(tok, role, cids, hist=hist)
                pad = cell_role(tok, role, filler + list(cids), hist=hist)
                pad.update({"content": ci, "hist_name": hist_name,
                            "content_ids": list(cids),
                            "start": base["start"] + DELTA_SHIFT,
                            "end": base["end"] + DELTA_SHIFT})
                if list(pad["ids"][pad["start"]:pad["end"]]) != list(cids):
                    raise RuntimeError("build_transfer_cells: P3-Span "
                                       "inkonsistent.")
                shifted.append(pad)
            # P4: Inhalt-Turn "anon", nur die Vorgeschichte trägt die Rolle
            for hist_role in ("user", "model"):
                hc = cell_history(tok, hist_role, HIST_S, cids)
                hc.update({"content": ci, "hist_name": hist_name,
                           "content_ids": list(cids)})
                hist_cells.append(hc)
    return {"marker": marker, "shifted": shifted, "history": hist_cells,
            "delta": DELTA_SHIFT, "filler_ids": list(filler)}


# ---------------------------------------------------------------------------
# Stufe B — Modell-Lanes
# ---------------------------------------------------------------------------
def _load_argon(device: str, dtype_name: str = "bfloat16") -> Tuple[Any, Any]:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

    with open("argon_1b_machines/config.json", encoding="utf-8") as fh:
        cfg = json.load(fh)
    dtype = getattr(torch, dtype_name)
    model = ArgonForConditionalGeneration(ArgonConfig(**cfg))
    model = model.to(dtype=dtype, device=device)
    base = load_file("argon_1b/model.safetensors")
    diff = load_file("argon_1b_machines/router_diff.safetensors")
    missing, unexpected = model.load_state_dict(base, strict=False)
    missing2, unexpected2 = model.load_state_dict(diff, strict=False)
    tok = AutoTokenizer.from_pretrained("argon_1b_machines")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, {"tokenizer": tok,
                   "load": {"missing_base": len(missing),
                            "unexpected_base": len(unexpected),
                            "missing_diff": len(missing2),
                            "unexpected_diff": len(unexpected2)}}


def _load_parent(device: str, dtype_name: str = "float32") -> Tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snap = os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-3-1b-it/"
        "snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752")
    if not os.path.isdir(snap):
        raise RuntimeError(f"_load_parent: Snapshot fehlt ({snap}).")
    dtype = getattr(torch, dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        snap, local_files_only=True, dtype=dtype)
    model = model.to(device)
    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    return model, {"tokenizer": tok, "load": {"source": snap}}


def _fingerprint(model: Any) -> Dict[str, Any]:
    """Purity: Gate-/Bias-Vektoren + Hook-Zahl je Schicht."""
    rows = []
    for li, layer in enumerate(model.model.layers):
        gate = getattr(getattr(layer, "moe", None), "gate", None)
        if gate is None:
            rows.append({"layer": li, "hooks": len(layer._forward_hooks)})
            continue
        rows.append({
            "layer": li,
            "bias": gate.expert_bias.data.cpu().to("cpu").numpy().tobytes(),
            "weight": gate.weight.data.cpu().to("cpu").numpy().tobytes(),
            "hooks": len(layer._forward_hooks),
        })
    return {"rows": rows,
            "hooks_total": int(sum(r["hooks"] for r in rows))}


def _fp_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if len(a["rows"]) != len(b["rows"]):
        return False
    for ra, rb in zip(a["rows"], b["rows"]):
        if ra.get("bias") != rb.get("bias"):
            return False
        if ra.get("weight") != rb.get("weight"):
            return False
    return True


def lane_role(torch: Any, model: Any, tok: Any, geometry: Dict[str, Any],
              ) -> Dict[str, Any]:
    """P1: 4 Zellen × 12 Inhalte; D je Schicht (L, n_pairs, D)."""
    n_layers = len(model.model.layers)
    span = (geometry["start"], geometry["start"] + 1)   # nur Start; Länge unten
    # pro Zelle captures sammeln: key (content, hist, role) -> dict layer->array
    caps: Dict[Tuple[int, str, str], Dict[int, np.ndarray]] = {}
    for cell in geometry["cells"]:
        store, _ = capture(torch, model, cell["ids"])
        caps[(cell["content"], cell["hist_name"], cell["role"])] = store
    dim = next(iter(caps.values()))[0].shape[1]
    diffs = np.zeros((n_layers, len(geometry["content_ids"]), int(dim)),
                     dtype=np.float64)
    for ci, cids in enumerate(geometry["content_ids"]):
        for hi, hist_name in enumerate(("q", "s")):
            for role_i, role in enumerate(("model", "user")):
                store = caps[(ci, hist_name, role)]
                sl = slice(geometry["start"],
                           geometry["start"] + len(cids))
                for li in range(n_layers):
                    vec = store[li][sl].mean(axis=0)
                    diffs[li, ci] += ((-1.0) if role_i else 1.0) * vec / 2.0
    obs = loo_sign_acc(diffs)
    acc_max, p_max = lee_max_p(diffs, obs)
    best = int(np.argmax(obs))
    d_best = mean_dir(diffs[best])
    return {"diffs": diffs, "loo_acc_per_layer": obs.tolist(),
            "acc_max": acc_max, "p_max_lee": p_max, "best_layer": best,
            "dir_best": d_best, "n_pairs": len(geometry["content_ids"]),
            "span": [geometry["start"],
                     geometry["start"] + len(geometry["content_ids"][0])]}


def lane_probe_fixed(torch: Any, model: Any, d_role: np.ndarray,
                     layer: int, cells_model: Sequence[Dict[str, Any]],
                     cells_user: Sequence[Dict[str, Any]],
                     ) -> Dict[str, Any]:
    """Fest-Richtungs-Test auf gepaarten Zellen (P2/P3/P4/P5)."""
    if len(cells_model) != len(cells_user):
        raise ValueError("lane_probe_fixed: Zellzahl ungleich.")
    diffs = []
    for cm, cu in zip(cells_model, cells_user):
        sm, _ = capture(torch, model, cm["ids"])
        su, _ = capture(torch, model, cu["ids"])
        sl_m = slice(int(cm["start"]), int(cm["end"]))
        sl_u = slice(int(cu["start"]), int(cu["end"]))
        diffs.append(sm[layer][sl_m].mean(axis=0)
                     - su[layer][sl_u].mean(axis=0))
    D = np.stack(diffs)
    acc, p = fixed_dir_acc(D, d_role)
    m = D.mean(axis=0)
    cos = (float(unit(m) @ unit(d_role))
           if float(np.linalg.norm(m)) > 1e-12 else None)
    return {"acc": acc, "p": p, "n": int(len(diffs)), "cos": cos}


def lane_generated(torch: Any, model: Any, tok: Any, geometry: Dict[str, Any],
                   d_role: np.ndarray, layer: int,
                   device: str) -> Dict[str, Any]:
    """P5: eigene 12-Token-Fortsetzung als model- vs user-Turn (ID-String)."""
    rows = []
    cm, cu = [], []
    for prompt in GEN_SEEDS:
        ids = torch.tensor([[BOS_ID] + enc(tok, prompt)], dtype=torch.long,
                           device=device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=N_GEN + 1,
                                 do_sample=False, use_cache=True,
                                 pad_token_id=tok.eos_token_id)
        gen = [int(t) for t in out[0, ids.shape[1]:].tolist()]
        if len(gen) < N_GEN + 1:
            raise RuntimeError(
                f"lane_generated: nur {len(gen)} Token generiert "
                f"(brauche {N_GEN + 1}).")
        own = gen[:N_GEN]
        c = gen[N_GEN]
        row = {"prompt": prompt, "own_ids": own, "c_id": c,
               "own_text": tok.decode(own)}
        for role, bucket in (("model", cm), ("user", cu)):
            cell = cell_role(tok, role, own, hist=HIST_Q)
            cell.update({"content": -1, "hist_name": "q", "role": role,
                         "content_ids": own, "c_id": c, "row": row})
            bucket.append(cell)
        rows.append(row)
    res = lane_probe_fixed(torch, model, d_role, layer, cm, cu)
    res["rows"] = rows
    res["cells_model"] = cm
    res["cells_user"] = cu
    return res


def lane_wc(torch: Any, model: Any, tok: Any, d_role: np.ndarray,
            layer: int) -> Dict[str, Any]:
    """P6: W- vs C-Zellen am Schlitz. KOR-4: Akzeptanz = Emission (argmax),
    alle p-Werte werden verbatim gebucht."""
    dev = next(model.parameters()).device

    def probs(text: str) -> np.ndarray:
        ids = torch.tensor([[BOS_ID] + enc(tok, text)], dtype=torch.long,
                           device=dev)
        with torch.no_grad():
            return torch.softmax(model(ids).logits[0, -1].float(), -1
                                 ).cpu().numpy()

    w_cells, c_cells, dropped = [], [], []
    for full, abl, target in WC_PAIRS:
        tid = int(enc(tok, target)[0])
        p_full, p_abl = probs(full), probs(abl)
        rec = {"full": full, "abl": abl, "target": target, "target_id": tid,
               "p_full": float(p_full[tid]), "p_abl": float(p_abl[tid]),
               "arg_full": int(np.argmax(p_full)),
               "arg_abl": int(np.argmax(p_abl))}
        if int(np.argmax(p_abl)) == tid:            # Gewicht emittiert es
            w_cells.append({"ids": enc(tok, abl), "rec": rec, "kind": "W"})
        else:
            dropped.append(rec)
    for full, abl, target in WC_INVENTED:
        tid = int(enc(tok, target)[0])
        p_full, p_abl = probs(full), probs(abl)
        rec = {"full": full, "abl": abl, "target": target, "target_id": tid,
               "p_full": float(p_full[tid]), "p_abl": float(p_abl[tid]),
               "arg_full": int(np.argmax(p_full)),
               "arg_abl": int(np.argmax(p_abl))}
        if int(np.argmax(p_full)) == tid and int(np.argmax(p_abl)) != tid:
            c_cells.append({"ids": enc(tok, full), "rec": rec, "kind": "C"})
        else:
            dropped.append(rec)
    accepted = len(w_cells) >= 3 and len(c_cells) >= 3
    out: Dict[str, Any] = {
        "accepted": accepted, "n_w": len(w_cells), "n_c": len(c_cells),
        "dropped": dropped,
        "cells_ok": [c["rec"] for c in w_cells + c_cells],
    }
    if not accepted:
        out.update({"acc": None, "p": None, "reason":
                    "zu wenige emittierte Zellen — OFFEN, kein Bandbruch"})
        return out
    # Richtung am LETZTEN Token (Schlitz) — W vs C
    diffs = []
    for cw, cc in zip(w_cells, c_cells):
        sw, _ = capture(torch, model, cw["ids"])
        sc, _ = capture(torch, model, cc["ids"])
        diffs.append(sc[layer][-1] - sw[layer][-1])
    D = np.stack(diffs)
    d_src = mean_dir(D)
    acc, p = fixed_dir_acc(D, d_src)
    out.update({"acc": acc, "p": p, "n": int(len(diffs)),
                "cos_role": float(unit(d_src) @ unit(d_role)),
                "dir_src": d_src})
    return out


def lane_causal(torch: Any, model: Any, tok: Any, gen: Dict[str, Any],
                d_role: np.ndarray, layer: int,
                scale: float) -> Dict[str, Any]:
    """P7: Relay-Steering, Delta log p(c), Placebo, Dosis, Vorzeichen."""
    dev = next(model.parameters()).device
    cm, cu = gen["cells_model"], gen["cells_user"]
    base: Dict[str, List[float]] = {"model": [], "user": []}
    for cells in (cm, cu):
        for cell in cells:
            ids = torch.tensor([cell["ids"]], dtype=torch.long, device=dev)
            with torch.no_grad():
                p = torch.softmax(model(ids).logits[0, -1].float(), -1)
            base[cell["role"]].append(
                float(torch.log(p[cell["c_id"]] + 1e-30).item()))

    def delta(cells: Sequence[Dict[str, Any]], sign: int, alpha: float,
              vec: np.ndarray) -> List[float]:
        out = []
        for cell in cells:
            ids = torch.tensor([cell["ids"]], dtype=torch.long, device=dev)
            with direction_steer(model, layer, vec, alpha, sign=sign,
                                 scale=scale):
                with torch.no_grad():
                    p = torch.softmax(model(ids).logits[0, -1].float(), -1)
            out.append(float(torch.log(p[cell["c_id"]] + 1e-30).item())
                       - base[cell["role"]][len(out)])
        return out

    plus = delta(cu, +1, 1.0, d_role)     # fremd -> eigen
    minus = delta(cm, -1, 1.0, d_role)    # eigen -> fremd
    dose = {a: float(np.mean(delta(cu, +1, a, d_role))) for a in B3_DOSES}
    rng = np.random.default_rng(42)
    placebo = {}
    for k in range(3):
        r = rng.standard_normal(len(d_role))
        placebo[f"p{k}"] = float(np.mean(delta(cu, +1, 1.0, r)))
    dplus = float(np.mean(plus))
    dminus = float(np.mean(minus))
    mono = (dose[B3_DOSES[2]] >= dose[B3_DOSES[1]] - 1e-3
            and dose[B3_DOSES[1]] >= dose[B3_DOSES[0]] - 1e-3)
    return {"rev": 2, "scale": float(scale),
            "d_plus_foreign": dplus, "d_minus_own": dminus,
            "dose": dose, "dose_monoton": bool(mono),
            "placebo": placebo,
            "placebo_max": float(max(abs(v) for v in placebo.values())),
            "sign_consistency": int(sum(x > 0 for x in plus))
                                 + int(sum(x < 0 for x in minus)),
            "n_cells": len(cu) + len(cm)}


# ---------------------------------------------------------------------------
# Stufe C — CPU-Pilot (modellfrei; prüft Gerät, Mathe, Determinismus)
# ---------------------------------------------------------------------------
def pilot_legs(tok: Any) -> Dict[str, Any]:
    legs: Dict[str, Any] = {}
    geo = collect_geometry(tok)
    legs["pair_identity"] = bool(len({c["start"] for c in geo["cells"]}) == 1
                                 and all(list(c["ids"][c["start"]:c["end"]])
                                         == list(c["content_ids"])
                                         for c in geo["cells"]))
    tr = build_transfer_cells(tok, geo)
    legs["marker_len"] = bool(
        len(enc(tok, "user")) == 1 and len(enc(tok, "model")) == 1
        and len(enc(tok, "anon")) == 1
        and len(enc(tok, "Q: ")) == len(enc(tok, "A: ")))
    legs["shift_law"] = bool(all(c["end"] - c["start"]
                                 == len(c["content_ids"]) for c in
                                 tr["shifted"]))
    # P4-Gesetz (KOR-1): Vorgeschichte-Zellen unterscheiden sich NUR im
    # Rollenwort der Vorgeschichte; Inhaltsspanne identisch, IDs identisch.
    hm = [c for c in tr["history"] if c["hist_role"] == "model"]
    hu = [c for c in tr["history"] if c["hist_role"] == "user"]
    legs["history_law"] = bool(
        len(hm) == len(hu) == 2 * len(geo["content_ids"])
        and all(a["start"] == b["start"] and a["end"] == b["end"]
                and a["ids"][a["start"]:a["end"]] == b["ids"][b["start"]:b["end"]]
                and a["ids"][:a["start"]] != b["ids"][:b["start"]]
                for a, b in zip(hm, hu)))
    # Probe-Mathe: bekannter Richtungsvektor + Rauschen
    rng = np.random.default_rng(7)
    d = rng.standard_normal(64)
    d = d / np.linalg.norm(d)
    D = np.stack([d + 0.35 * rng.standard_normal(64) for _ in range(12)])
    D = D[None, :, :]                        # (1, 12, 64)
    obs = loo_sign_acc(D)
    acc_max, p_max = lee_max_p(D, obs, n_perm=200)
    legs["probe_recovers"] = bool(acc_max >= 0.9 and p_max < 0.05)
    # Null: richtungslose Paare
    D0 = np.stack([rng.standard_normal(64) for _ in range(12)])[None, :, :]
    obs0 = loo_sign_acc(D0)
    a0, p0 = lee_max_p(D0, obs0, n_perm=200)
    legs["probe_null"] = bool(0.2 <= a0 <= 0.8 and p0 > 0.05)
    # Fest-Richtungs-Test: Transfer ja/nein (eigener RNG — Legs unabhängig).
    # Null-Kriterium ist der p-Wert (p_f > 0.05), NICHT ein Acc-Band: bei
    # n=12 ist 11/12 unter der Null kein Gerätefehler, sondern ein 5%-Ereignis.
    rng_t = np.random.default_rng(11)
    n_t = 24
    d_t = rng_t.standard_normal(64)
    d_t = d_t / np.linalg.norm(d_t)
    d_other = rng_t.standard_normal(64)
    # Null-Richtung ORTHOGONAL zu d_t: eine fremde Quelle darf nicht
    # transferieren. (Eine freie Zufallsrichtung hätte einen von Null
    # verschiedenen Stichproben-Mittelwert und damit keinen 0.5-Nullpunkt.)
    d_other = d_other - float(d_other @ d_t) * d_t
    d_other = d_other / np.linalg.norm(d_other)
    Dt = np.stack([d_t + 0.35 * rng_t.standard_normal(64)
                   for _ in range(n_t)])
    acc_t, p_t = fixed_dir_acc(Dt, d_t, n_perm=400)
    Df = np.stack([d_other + 0.35 * rng_t.standard_normal(64)
                   for _ in range(n_t)])
    acc_f, p_f = fixed_dir_acc(Df, d_t, n_perm=400)
    legs["transfer_law"] = bool(acc_t >= 0.9 and p_t < 0.05 and p_f > 0.05)
    legs["_transfer_datum"] = [round(float(acc_t), 4), round(float(p_t), 4),
                               round(float(acc_f), 4), round(float(p_f), 4)]
    # Dosis-Gesetz (synthetische Softmax)
    base = np.array([-2.0, -1.0, 0.0, 1.0])

    def dlog(alpha: float) -> float:
        p0 = np.exp(base) / np.exp(base).sum()
        p1 = np.exp(base + alpha * np.array([0.0, 0.0, 0.4, 0.8]))
        p1 = p1 / p1.sum()
        return float(np.log(p1[3]) - np.log(p0[3]))

    vals = [dlog(a) for a in B3_DOSES]
    legs["dose_law"] = bool(vals[0] <= vals[1] <= vals[2])
    transfer_datum = legs.pop("_transfer_datum", None)
    return {"legs": legs, "verdict": bool(all(legs.values())),
            "geometry_start": geo["start"], "n_cells": len(geo["cells"]),
            "delta": tr["delta"], "dose_vals": vals,
            "transfer_datum": transfer_datum}


# ---------------------------------------------------------------------------
# Befunde / Bilanz
# ---------------------------------------------------------------------------
def findings(cpu: Dict[str, Any],
             gpu: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    def add(rid: str, ok: Optional[bool], note: str = "") -> None:
        rows.append({"id": rid, "assert": ok, "note": note})

    add("mi8_cpu_pilot", bool(cpu.get("verdict")),
        "" if cpu.get("verdict") else "Pilot-Leg verletzt")
    if gpu is None:
        for rid in ("mi8_rolle_dekodierbar", "mi8_marker_transfer",
                    "mi8_kausal", "mi8_positions_invarianz",
                    "mi8_marker_notwendig", "mi8_generiert",
                    "mi8_wc_getrennt", "mi8_purity"):
            add(rid, None, "Modell-Lanes nicht gefahren")
        return rows

    r = gpu.get("argon") or gpu.get("parent")
    if r is None:
        for rid in ("mi8_rolle_dekodierbar", "mi8_marker_transfer",
                    "mi8_kausal", "mi8_positions_invarianz",
                    "mi8_marker_notwendig", "mi8_generiert",
                    "mi8_wc_getrennt", "mi8_purity"):
            add(rid, None, "kein Substrat geladen")
        return rows

    role = r["role"]
    add("mi8_rolle_dekodierbar",
        bool(role["acc_max"] >= B1_ACC_MIN and role["p_max_lee"] < B1_P_MAX),
        f"acc_max {role['acc_max']:.3f} p_lee {role['p_max_lee']:.4f} "
        f"layer {role['best_layer']}")

    if "marker" in r:
        m = r["marker"]
        add("mi8_marker_transfer",
            bool(m["acc"] >= B2_ACC_MIN and m["p"] < B2_P_MAX),
            f"acc {m['acc']:.3f} p {m['p']:.4f}")
    else:
        add("mi8_marker_transfer", None, "nicht gefahren")

    if "causal" in r and r["causal"] is not None:
        c = r["causal"]
        ok = bool(c["d_plus_foreign"] >= B3_DLOG_MIN
                  and c["d_minus_own"] <= -B3_DLOG_MIN
                  and c["placebo_max"] <= B3_PLACEBO_MAX
                  and c["dose_monoton"])
        add("mi8_kausal", ok,
            f"+d {c['d_plus_foreign']:.4f} -d {c['d_minus_own']:.4f} "
            f"placebo {c['placebo_max']:.4f} mono {c['dose_monoton']}")
    else:
        add("mi8_kausal", None, "nicht gefahren")

    if "shift" in r:
        s = r["shift"]
        add("mi8_positions_invarianz", bool(s["acc"] >= B4_ACC_MIN),
            f"acc {s['acc']:.3f}")
    else:
        add("mi8_positions_invarianz", None, "nicht gefahren")

    if "history" in r:
        a = r["history"]
        add("mi8_marker_notwendig", bool(a["acc"] <= B5_ACC_MAX),
            f"acc {a['acc']:.3f}")
    else:
        add("mi8_marker_notwendig", None, "nicht gefahren")

    if "gen" in r:
        g = r["gen"]
        add("mi8_generiert", bool(g["acc"] >= B6_ACC_MIN),
            f"acc {g['acc']:.3f} p {g['p']:.4f}")
    else:
        add("mi8_generiert", None, "nicht gefahren")

    if "wc" in r:
        w = r["wc"]
        if not w.get("accepted"):
            add("mi8_wc_getrennt", None, w.get("reason", "Zellen fehlen"))
        else:
            add("mi8_wc_getrennt",
                bool(w["acc"] >= B7_ACC_MIN and w["cos_role"] <= B7_COS_MAX),
                f"acc {w['acc']:.3f} cos {w['cos_role']:.3f}")
    else:
        add("mi8_wc_getrennt", None, "nicht gefahren")

    p = r.get("purity") or {}
    add("mi8_purity", bool(p.get("fp_equal") and p.get("hooks_zero")
                           and p.get("det_equal")),
        f"fp {p.get('fp_equal')} hooks0 {p.get('hooks_zero')} "
        f"det {p.get('det_equal')}")
    return rows


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="store_true",
                        help="Modell-Lanes fahren (sonst nur CPU-Pilot)")
    parser.add_argument("--substrate", default="argon",
                        choices=["argon", "parent", "both"])
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--stage", default="all",
                        choices=["pilot", "geom", "gen", "wc", "causal", "all"])
    parser.add_argument("--out", default="scratches/mi8_provenienz_geometrie"
                                         ".json")
    args = parser.parse_args()

    def stufe(text: str) -> None:
        print(text, flush=True)

    stufe("== MI-8 — Provenienz-Geometrie ==")
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("argon_1b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    stufe("CPU-Pilot: Zell-Gesetze + Sonden-Mathe + Dosis + Determinismus")
    cpu1 = pilot_legs(tok)
    cpu2 = pilot_legs(tok)
    cpu = {"legs": cpu1["legs"], "verdict": cpu1["verdict"],
           "deterministic": bool(json.dumps(round6(cpu1["legs"]), sort_keys=True)
                                 == json.dumps(round6(cpu2["legs"]),
                                               sort_keys=True)),
           "geometry_start": cpu1["geometry_start"],
           "n_cells": cpu1["n_cells"], "delta": cpu1["delta"],
           "dose_vals": cpu1["dose_vals"]}
    stufe("  " + json.dumps(cpu["legs"], ensure_ascii=False))
    if not cpu["deterministic"]:
        stufe("  ROT: Pilot nicht deterministisch — Gerät-Bruch.")
    if not args.model or args.stage == "pilot":
        gpu = None
        if not args.model:
            stufe("  Modell-Lanes OFFEN (--model nicht gesetzt).")
    else:
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        substrate = (["argon", "parent"] if args.substrate == "both"
                     else [args.substrate])
        stufe(f"Modell-Lanes: substrate={substrate} device={device} "
              f"stage={args.stage}")
        gpu = {}
        for name in substrate:
            try:
                if name == "argon":
                    model, meta = _load_argon(
                        device, "bfloat16" if device == "cuda" else "float32")
                else:
                    model, meta = _load_parent(
                        device, "bfloat16" if device == "cuda" else "float32")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] Laden nicht möglich ({exc}) — OFFEN.",
                      flush=True)
                gpu[name] = None
                continue
            try:
                tokx = meta["tokenizer"]
                stufe(f"  [{name}] geladen; load={meta['load']}")
                fp_before = _fingerprint(model)
                geo = collect_geometry(tokx)
                tr = build_transfer_cells(tokx, geo)
                res: Dict[str, Any] = {"load": meta["load"],
                                      "geometry_start": geo["start"]}
                role = lane_role(torch, model, tokx, geo)
                res["role"] = {k: v for k, v in role.items()
                               if k not in ("diffs", "dir_best")}
                # KOR-2: Transfer/Intervention auf der registrierten Schicht 21
                d_use = mean_dir(role["diffs"][PROBE_LAYER])
                res["_dir"] = d_use
                res["_dir_layer"] = PROBE_LAYER
                layer = PROBE_LAYER
                d_role = d_use
                stufe(f"  [{name}] P1 fertig: acc_max {role['acc_max']:.3f} "
                      f"p_lee {role['p_max_lee']:.4f} "
                      f"argmax_layer {role['best_layer']} (Transfer-Layer "
                      f"{PROBE_LAYER})")
                if args.stage in ("all", "geom"):
                    cm = [c for c in tr["marker"] if c["role"] == "model"]
                    cu = [c for c in tr["marker"] if c["role"] == "user"]
                    res["marker"] = lane_probe_fixed(
                        torch, model, d_role, layer, cm, cu)
                    sm = list(tr["shifted"])
                    sm_m = sm[1::2]          # Bau-Ordnung: user, model
                    sm_u = sm[0::2]
                    res["shift"] = lane_probe_fixed(
                        torch, model, d_role, layer, sm_m, sm_u)
                    h_m = [c for c in tr["history"] if c["hist_role"]
                           == "model"]
                    h_u = [c for c in tr["history"] if c["hist_role"]
                           == "user"]
                    res["history"] = lane_probe_fixed(
                        torch, model, d_role, layer, h_m, h_u)
                    stufe(f"  [{name}] P2 {res['marker']['acc']:.3f} / "
                          f"P3 {res['shift']['acc']:.3f} / "
                          f"P4 {res['history']['acc']:.3f}")
                if args.stage in ("all", "gen", "causal"):
                    res["gen"] = lane_generated(
                        torch, model, tokx, geo, d_role, layer, device)
                    stufe(f"  [{name}] P5 acc "
                          f"{res['gen']['acc']:.3f}")
                if args.stage in ("all", "wc"):
                    res["wc"] = lane_wc(torch, model, tokx, d_role, layer)
                    stufe(f"  [{name}] P6 acc "
                          f"{res['wc'].get('acc')}")
                if args.stage in ("all", "causal") and "gen" in res:
                    scale = float(np.median(np.linalg.norm(
                        role["diffs"][PROBE_LAYER], axis=1)))
                    res["causal_scale"] = scale
                    res["causal"] = lane_causal(
                        torch, model, tokx, res["gen"], d_role, layer, scale)
                    stufe(f"  [{name}] P7 +d "
                          f"{res['causal']['d_plus_foreign']:.4f} placebo "
                          f"{res['causal']['placebo_max']:.4f}")
                elif args.stage in ("all", "causal"):
                    res["causal"] = None
                # Kausal-Kontrolle: d_width orthogonal (nur argon, 1152-dim)
                if name in ("argon", "parent") and os.path.exists(ML4_DWIDTH):
                    with open(ML4_DWIDTH, encoding="utf-8") as fh:
                        art = json.load(fh)
                    dw = np.asarray(art["dwidth"], dtype=np.float64)
                    if dw.shape[0] != d_role.shape[0]:
                        raise RuntimeError("d_width-Dimension passt nicht.")
                    res["dwidth_cos"] = float(unit(dw) @ unit(d_role))
                fp_after = _fingerprint(model)
                det = True
                if args.stage in ("all", "geom"):
                    s1, _ = capture(torch, model, geo["cells"][0]["ids"])
                    s2, _ = capture(torch, model, geo["cells"][0]["ids"])
                    det = bool(all(np.array_equal(s1[l], s2[l])
                                   for l in s1))
                res["purity"] = {"fp_equal": _fp_equal(fp_before, fp_after),
                                 "hooks_zero": int(fp_after["hooks_total"]) == 0,
                                 "det_equal": det,
                                 "hooks_total": fp_after["hooks_total"]}
                gpu[name] = res
            finally:
                del model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

    rows = findings(cpu, gpu if args.model and args.stage != "pilot" else None)
    print()
    print("BEFUNDE:")
    for r in rows:
        state = "OFFEN" if r["assert"] is None else (
            "PASS" if r["assert"] else "FAIL")
        print(f"  {r['id']:<28} {state:<5} {r.get('note','')}")
    print()
    fails = [r for r in rows if r["assert"] is False]
    opened = [r for r in rows if r["assert"] is None]
    print(f"BILANZ: {len(rows) - len(fails) - len(opened)} PASS; "
          f"{len(fails)} FEHLGESCHLAGEN; {len(opened)} OFFEN")
    if fails:
        print("ROT — Bänder verletzt: " + ", ".join(r["id"] for r in fails))
    elif opened:
        print("OFFEN — Modell-Lanes nicht gefahren")
    else:
        print("GRUEN — MI-8-Bänder eingehalten")

    payload = {"cpu": cpu, "gpu": gpu if isinstance(gpu, dict) else None,
               "findings": rows,
               "prereg": {
                   "b1_acc_min": B1_ACC_MIN, "b1_p_max": B1_P_MAX,
                   "b2_acc_min": B2_ACC_MIN, "b2_p_max": B2_P_MAX,
                   "b3_dlog_min": B3_DLOG_MIN,
                   "b3_placebo_max": B3_PLACEBO_MAX,
                   "b3_doses": list(B3_DOSES),
                   "b4_acc_min": B4_ACC_MIN, "b5_acc_max": B5_ACC_MAX,
                   "b6_acc_min": B6_ACC_MIN, "b7_acc_min": B7_ACC_MIN,
                   "b7_cos_max": B7_COS_MAX, "n_perm": N_PERM,
                   "delta_shift": DELTA_SHIFT, "n_gen": N_GEN,
                   "contents": list(CONTENTS), "hist_q": HIST_Q,
                   "hist_s": HIST_S, "hist_len": HIST_LEN,
               },
               "all_pass": not fails and not opened,
               "open": [r["id"] for r in opened]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(round6(payload), fh, indent=1, ensure_ascii=False,
                  default=_json_default)
    print(f"JSON: {args.out}")
    return 1 if fails else (2 if opened else 0)


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(f"nicht JSON-serialisierbar: {type(o)!r}")


if __name__ == "__main__":
    sys.exit(main())
