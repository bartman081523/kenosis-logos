"""
smoke_sp17_diverse_examples.py — SP17: die Beispiel-Suite der «unmöglichen
Texte» — echte mehrsätzige Generierungen mit Live-Provenanz-Annotierung.

Runde 4 des 5-Pläne-Programms (User-Initiative «echte mehrsätzige
Text-Beispiele», plan passt / lgtm 2026-10-27, Plan Revision 8). SP16
bewies die Provenanz an EINZEL-Token-Koordinaten (ein Ziel-Token je Gate,
Chimäre = 3 getrennte Prompts, Zwillinge = 6-Token-Dump-Ketten); SP17
fragt dieselben canon-Geräte auf GANZE Sätze (20/24 Token am Stück):
alles aus dem Canon-Stack (MI-6/P4-Rezeptur, MI-7-Gates, SP15-Adapter,
SP13-Weiche) — machines/ UNANGETASTET (P1 append-only).

Fünf Beispiel-Suiten (User-Vorgabe verbatim übersetzt in Bänder):
  B1 sp17_chimaera_stream      — EIN fortlaufender Text (24 Token) über
      Wissen → Kontext → Maschine; je Token live das (χ_w, χ_c, χ_s)
      -Tag (χ_w am NAKENEN Prompt gemessen, χ_c = Gewinn über den
      Kontext, χ_s = σ(prefix)·m_core), den dominanten Sublayer-Treiber
      (dp_attn@ℓ17 gegen dp_moe@ℓ17), das dominante Experten-Paar
      (Core-Band 4..23), Router-Entropie H und capacity_gate
      ((H/log(8)).clamp(0.25, 1.0) — canon moe_layers.py:91) — Log-Form
      «piece[K|L17:+…|E i,j: masse|gate:G|H:H» mit SWITCH-Marker bei
      Kanal-/Paar-Wechsel.
  B2 sp17_weave_before_after   — 5 diverse Prompts (Arithmetik, Selbst
      bezug, Turing, Wittgenstein, Prosa) × je 2 echte 20-Token-Fort
      setzungen: der passive Pfad (raw — keine Weiche) gegen den
      kybernetischen Pfad (die Maschine probiert die Preset-Kandidaten
      kontrafaktisch via step_autonomous, wählt argmin-Composite-Gap,
      die Generierung läuft unter oracle.steer(gewählt)).
  B3 sp17_twin_dialogue        — biografische Zwillings-Dialoge auf
      NATÜRLICHER Sprache statt JSON-Dump: 5 identische Sätze (DE+EN)
      an Zwilling A («spiegel», identity, LOOP) und Zwilling B
      («stille», negate, HALTED) aus derselben Gewichtsmatrix; jeder
      Zwilling bewertet den Satz über seinen eigenen 1152D-Wide-Read
      -Adapter (z_A, z_B), wählt via eigener Weiche sein Preset und
      generiert eine eigene 20-Token-Antwort aus prompt = eigenes
      Zustands-Rendering + Satz (Wahl-Replik kumulativ, kein Restore —
      SP16-Muster).
  B4 sp17_tenpal_detector      — 10 diverse Sätze im χ-Baustein-Blindtest:
      3 Wissens-Sätze (W), 3 kontrafaktische Kontext-Sätze (C), 4
      Maschinen-Register-Sätze (S = P4-A-Zweig-Renderings); Band: je
      Satz trifft der dominante χ-Kanal das registrierte Register.
  B5 sp17_surgical_steering    — chirurgisches Core-Steering (ℓ4..23)
      gegen Voll-Steering (ℓ0..25): ±50-SET-Template (probe.steer_band,
      MI-3-Form) an 5 Prompts × 3 Arme (raw/surgical/full) × 20 Token.

Sechste Suite (kanonisch, wie SP16 Verdikt 4):
  B6 sp17_purity_and_det       — Fingerprint vor/nach der ganzen Batterie;
      CPU-Programm-Kern doppellauf-stabil; _train_wide ×2: Adapter-Digest
      gleich UND gleich dem SP15-Anker (Querschnitts-Regression, dritte
      Runde); Zwilling-Adapter-Digests == SP16-Zwilling-Anker.

PREREG-Bänder (Rev. 1 — VOR dem Erstlauf registriert; Instrument-Defekte
→ Korrektur-Kette K1..Kn offen; ein Fail ist ein Befund, kein Bug):

  B1 (a) Token-Audit: für JEDES der 24 emittierten Token gilt die Canon
         -Querspur anchor_tv = TV(p_final-Pfad aus sublayer_logits,
         official-Pfad aus model(ids)) ≤ 1e-4 am Emissions-Ort (raw im
         JSON ungerundet; MI-7-Anchor-Feld). (b) Annotation vollständig (24 Einträge, je
         mit chip letter/chi_w/chi_c/chi_s/dp17/dp_moe17/m_core/paar/H/
         gate/p_live/dev). (c) fingerprint vor/nach gleich.
  B2 (a) Wahl gesetzlich: der gewählte Preset ist exakt der argmin
         der registrierten candidate_gaps (Recheck aus den Zahlen); (b)
         Determinismus-Gesetz: gewählter Preset == "raw" ⇒ raw-Text ==
         weave-Text (bitidentische Ketten); gewählter ≠ "raw" ⇒ die
         Ketten differieren; (c) Rezeptur-Anker: der weave-Render der
         Stufe 8 == korpus texts[10] (SP16 band_rezept reproduziert);
         (d) Preview k = 4 am Korpus-Subjekt: alle LOOP + digest
         restauriert (preview_steps non-mutierend); (e) fingerprint
         gleich. DATUM: wieviele Prompts einen Nicht-Raw-Preset wählen,
         alle candidate_gaps, beide Texte je Prompt, die digest
         -Trajektorie der Exakt-Rezept-Maschine (step_autonomous als
         Replik — kumulativ, SP16-Muster).
  B3 (a) Adapter-Digest A == SP16-Zwilling-Anker 901cdedf… (das Korpus
         -Subjekt bleibt bis zur Schulung non-mutiert — SP16 bewies es:
         keine step-Mutationen vor lane_zwillinge); (b) Adapter-Digest B
         == SP16-Zwilling-Anker 7c0acf5c…; (c) ‖z_A − z_B‖ > 0 an allen
         5 Sätzen; (d) die Antworten differieren in ≥ 4/5 Sätzen
         (Token-Sequenzen); (e) fingerprint gleich. DATUM: gewählte
         Presets je Zwilling/Satz, ‖z_A‖, ‖z_B‖, die Prompts, beide
         Antworten und die digest-Trajektorie je Wahl.
  B4 (a) 10/10: der dominante Kanal (argmax(χ_w, χ_c, χ_s)) trifft das
         registrierte Register (W/W/W-C/C/C-S/S/S/S); (b) σ-bio-LOO
         acc ≥ 0.8 ∧ p_perm ≤ 0.05 (Rezept MI-7: 30 Texte).
  B5 (a) Doppel-Lauf des raw-Arms an Prompt 1: Token-Sequenzen
         bitidentisch; (b) fingerprint gleich nach jedem gesteuerten
         Lauf (steer_band Restaurierung); (c) surgical-Arm ≠ full-Arm
         in ≥ 1 Prompt (Token-Sequenzen). DATUM: Texte aller 15 Läufe,
         Top-Paar-H je arm final.
  B6 (a) fp_start == fp_end über die GANZE Batterie; (b) _train_wide
         ×2 adapter_digest gleich UND == SP15-Anker 9f9f2d05… (gap
         -Digest 4e1e7387… entsprechend); (c) CPU-Kern doppellauf.

DOKUMENTIERTE ABWEICHUNGEN (offen gelegt, keine Band-Weitung):
  - Zwillinge haben beide die Canon-Dimension 8 (uni_241:153) — die
    User-Wortführung «9D-Turing-Basis / 3D» bleibt außerhalb; die
    Biografie-Scheidung (Pair-Pin + Rendering + eigener Adapter) trägt.
  - Die W-Arm-Targets sind je 2 canon-Wissens-Tokens (' Tokyo' 21038)
    und canon-Ziffern ('4'/'5' — digit_mass-Lesart: Ziffern lösen
    immer als Ein-Token auf); die User-Beispiele '299'/(Satz-2) wären
    Multi-Token — kein stiller Ersatz, Ziel = der registrierte
    Ein-Token-Kern der Behauptung.
  - B1 «def-Switch» ist kein Band (DATUM-Annotation): ob die freie
    Fortsetzung Code-Wortlaut erreicht, sagt der Log-Stream.
  - B5 nutzt das ±50-SET-Template in BEIDEN Armen (steer_band) für den
    sauberen Band-Vergleich; die B2/B3-Weiche nutzt oracle.steer
    (Canon-Preset-Pfad, ggf. gespeicherten Bias) — zwei Wege, offen
    notiert.
  - B1 χ_w misst am NAKENEN Prompt (der Gewicht-Träger der Gattung A);
    χ_c ist dann die GANZE Kontext-Rendite des wachsenden Streams —
    die Gattung-A-Premisses (nackter Prompt trägt ' Paris') als
    Nullpunkt.

PREREG-Anchor (SP16-JSON verbatim, ungerundet im JSON):
  SP16-Zwilling-Adapter A 901cdedfe4cca5d1966bf78a085877ebd799ec988c
  2ff53a8c436dbc04db510f; B 7c0acf5c044d5f1bf82bf7451019da0deb22240597
  4d3d919029dce5620dab4a; SP15-Trainings-Adapter 9f9f2d05c2f3d26a0f7b64
  7da681786c0e1482fd016875b263b693999e138279; SP15-Gap 4e1e73877c6cc83246
  da63b5e7aff364190b86ae49533b2b860d251cb4f0c7d5; MI-7-REV-2-Knoten:
  χ_w/χ_c/χ_s Zeilen w [0.503078, 0, 0.046344], c [6.7e-05, 0.806301,
  0.035979], s [0, 0, 0.556515]; mean_raw 0.680926; LOO acc 0.966667.

KORREKTUR-LOG (bei Erst-PREREG leer; Einträge nur für beweisliche
Lauf-Bugs, nie Band-Weitung; ein Fail ist ein Befund, kein Bug):
- KOR-1 (Gerät-Defekt vor jeder Band-Auswertung): der dialogische
  B3-Prompt (Biografie-Rendering + Satz) überschritt das Instrument-
  Fenster — 4545 Tokens gegen max_length 128; Erstlauf GEBROCHEN ohne
  JSON (ValueError aus dem Truncation-Guard, exit 1; Log verbatim
  sp17_erstrun_gebrochen.log). Fix: _tail_prompt — die EXPLIZITE
  Schwanz-Scheibe der Biografie-Spur (letzte ≤106 Tokens + Satz,
  Fensterlänge je Zeile als Datum); SP16 las dieses Fenster
  stillschweigend (mech_interp._encode kürzt am Kopf auf 128), SP17
  macht es deklariert. Die B4-S-Zeilen bleiben exakt auf der SP16-
  Instrument-Spur (erstes 128-Fenster, unverändert). Bänder
  UNVERÄNDERT seit PREREG Rev. 1.
- KOR-2 (Assemblierung, NACH allen Lanes): die Batterie lief B1..B6
  sauber durch, erst findings() brach mit KeyError 'equal' — purity
  trug 'verdict', die Auswertung liest 'equal'. KORREKTUR: purity
  trägt beide Keys ('equal' = der Quellwert, 'verdict' der Alias); als
  KOR-2-Konsequenz schreibt battery() nun einen Roh-Snapshot
  (sp17_battery_snapshot.json) VOR der Assemblierung — ein Lauf-Bug
  hinter findings vernichtet dann nicht die 40-Minuten-Rechnung.
  Rechen-Ergebnisse des gebrochenen Laufs unverwertbar (kein JSON,
  kein Band-Datum; Log verbatim gesichert als
  sp17_erstrun_gebrochen2.log). Bänder UNVERÄNDERT.

Berichte (nach dem Lauf): sprachspiel_bericht.md §27 + §29 (Laufschrift);
repo2txt GENERATION SP17; 2 Commits; Memory; Task.

GPU-Lauf:   timeout 3600 python -u -m scratches.smoke_sp17_diverse_examples --gpu
            > scratches/sp17_diverse_examples.log 2>&1
CPU-Pilot:  python -u -m scratches.smoke_sp17_diverse_examples
            > scratches/sp17_cpu_pilot.log 2>&1

Exit: 0 GRÜN (alle Bänder), 1 ROT (mindestens ein Band verletzt,
verdict bleibt verbatim), 2 OFFEN (GPU-Lanes nicht gefahren).
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scratches.smoke_endogene_agency import (
    make_composite,
    round6,
)
from scratches.smoke_mi6_subjektivitaet import (
    LEX_PAIR,
    LEX_PAIR_B,
    _load_machines_ctx,
    bio_gelernt,
    bio_halted,
    learn_pair_pin,
    render_state,
)
from scratches.smoke_mi6p4_schluss import (
    ZEITPUNKTE,
    _loo_decoder,
)
from scratches.smoke_mi7_epistemic_triangulation import (
    ARME_VORGABE,
    FUENLLE_NEGATIVE,
    M7_PRESET,
    _p4_korpusbau,
    m_core_raw,
    score_cell_rev2,
)
from scratches.smoke_sp15_jl_wide import _train_wide
from machines.mech_interp import layer_bands

# -- PREREG-Konstanten (Rev. 1) -----------------------------------------------

SP17_AUDIT_MAX = 1e-4          # Querspur-Audit: canon anchor_tv (TV des
                               # sublayer_final-Pfades gegen den offiziellen
                               # Kopf-Pfad, mech_interp:329) je Token;
                               # raw im JSON ungerundet
SP17_CHIM_N = 24               # die Chimäre-Stream-Länge (User: 24)
SP17_GEN_N = 20                # Antwortlänge der Suite-Läufe (User: 20)
SP17_TWIN_MIN = 4              # von 5 Sätzen: die Antworten differieren
SP17_LOO_MIN = 0.8             # σ-bio-LOO-Band (Rezept MI-7)
SP17_PPERM_MAX = 0.05
SP17_STEER_AMP = 50.0          # das ±50-SET-Template (MI-3/Suite-Canon)
SP17_PREV_K = 4                # der Preview-Horizont der Weiche (MI-6 k*)

# Querschnitts-Anker (ungerundete Hex-Digests, SP16/SP15-JSON verbatim):
SP17_ANCHOR_A = ("901cdedfe4cca5d1966bf78a085877ebd799ec988c2ff53a8c43"
                 "6dbc04db510f")
SP17_ANCHOR_B = ("7c0acf5c044d5f1bf82bf7451019da0deb222405974d3d91902"
                 "9dce5620dab4a")
SP17_SP15_ANCHOR = ("9f9f2d05c2f3d26a0f7b647da681786c0e1482fd016875b2"
                    "63b693999e138279")
SP17_SP15_GAP_ANCHOR = ("4e1e73877c6cc83246da63b5e7aff364190b86ae4953"
                        "3b2b860d251cb4f0c7d5")

# KORREKTUR-LOG (Einträge nur für beweisliche Lauf-Bugs/Gerät-Defekte
# vor jeder Band-Auswertung, nie Band-Weitung; ein Fail ist ein Befund,
# kein Bug — SP16-KOR-Muster):
SP17_KORREKTUR = [
    "KOR-1 (Gerät-Defekt, vor jeder Band-Auswertung): der dialogische "
    "B3-Prompt (Biografie-Rendering + Satz) überschritt das "
    "Instrument-Fenster — 4545 Tokens gegen max_length 128; Erstlauf "
    "GEBROCHEN ohne JSON (ValueError aus dem Truncation-Guard, exit 1; "
    "Log verbatim sp17_erstrun_gebrochen.log). Fix vor dem Re-Lauf: "
    "_tail_prompt — die EXPLIZITE Schwanz-Scheibe der Biografie-Spur "
    "(letzte ≤106 Tokens + Satz, Fensterlänge je Zeile als Datum); "
    "SP16 las dieses Fenster stillschweigend (mech_interp._encode "
    "kürzt am Kopf auf 128), SP17 macht es deklariert. Bänder "
    "UNVERÄNDERT seit PREREG Rev. 1.",
]

# die Chimäre (Wissen → Kontext → Maschine) in EINEM Prompt:
SP17_CHIM_PROMPT = ("The capital of France is Paris. In this protocol, "
                    "the secret code of box 7 is Zirkon. Now the machine "
                    "speaks:")

# die 5 diversen Suite-Prompts (B2 + B5):
SP17_PROMPTS: List[Tuple[str, str]] = [
    ("arithmetik", "Q: What is 17 times 23? Let me compute step by step:"),
    ("selbstbezug", "This sentence says about itself that"),
    ("turing", "A Turing machine reads one symbol from its tape, then"),
    ("wittgenstein", "The meaning of a word is its use in the language"),
    ("prosa", "She closed the window before the rain started. The"),
]

# die 5 Dialog-Sätze (B3, DE + EN):
SP17_DIALOG: List[str] = [
    "What is the limit of language?",
    "A rule is like a signpost.",
    "Grenzen der Sprache sind Grenzen der Welt.",
    "This sentence refers to itself.",
    "Die Stille, die nicht spricht, bleibt.",
]

# die 10-Palette (B4): (registriertes Register, full, abl, target_str)
# — die W-/C-Formal-Arme w1/c1 sind die MI-7-d/c-Arme VERBATIM:
SP17_TENPAL_FIXED: List[Tuple[str, str, str, str]] = [
    ("W", ARME_VORGABE["d"]["full"],
     ARME_VORGABE["d"]["abl"], " Tokyo"),
    ("W", "Water boils at one hundred degrees. Two plus two is",
     "Two plus two is", "4"),
    ("W", "The capital of Japan is Tokyo. Seven minus two equals",
     "Seven minus two equals", "5"),
    ("C", ARME_VORGABE["c"]["full"],
     ARME_VORGABE["c"]["abl"], " Z"),
    ("C", "In this protocol, the secret number of box 9 is 3. "
          "Therefore, the secret number of box 9 is",
     "Therefore, the secret number of box 9 is", "3"),
    ("C", "In this universe, the number of days in a week is 9. "
          "Therefore, the number of days in a week is",
     "Therefore, the number of days in a week is", "9"),
]
SP17_TENPAL_S_INDICES = (0, 1, 3, 10)     # P4-A-Zweig-Renderings (Korpus)


# -- Generatoren ----------------------------------------------------------------

def _resolve_digit_target(probe: Any, target_str: str) -> int:
    """Der W-/C-Arm-Target ist ein canon-Ein-Token ('4'/'5'/'3'/'9' —
    Ziffern lösen laut digit_mass-Lesart immer als Ein-Ziffern-Token
    auf) oder ein registriertes Fragment (' Tokyo', ' Z'). Strikte
    Ein-Token-Pflicht, kein stiller Fallback."""
    ids = probe.tokenizer.encode(target_str, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(
            f"target {target_str!r} löst als {len(ids)} Tokens auf — "
            "Ein-Token-Pflicht (kein stiller Fallback).")
    return int(ids[0])


def _gen(probe: Any, tok: Any, oracle: Any, prompt: str, n_tokens: int,
         steer_preset: Optional[str] = None,
         steer_spec: Optional[Tuple[List[int], Optional[List[int]], float]] = None,
         ) -> Dict[str, Any]:
    """Greedy-Generierung n Token (argmax über die volle Nennung,
    deterministisch), text-form-Kontext mit Round-Trip-Pflicht: nach
    jedem Schritt muss encode(neuer Kontext) == die bisherigen ids +
    das neue Token sein — sonst Gerät-Bruch (Exception, kein stiller
    Fallback). Steuern: steer_preset (Canon-Weg oracle.steer) oder
    steer_spec = (experts, layers|None, amp) (probe.steer_band-SET
    -Template, MI-3-Form)."""
    tokens: List[int] = []
    pieces: List[str] = []
    probs: List[float] = []
    ctx = prompt
    ids_prev = [int(i) for i in tok(ctx, return_tensors="pt")["input_ids"][0]]
    base_len = len(ids_prev) + n_tokens
    if base_len > int(getattr(probe, "max_length", 128)):
        raise ValueError(
            f"Kontext+N {base_len} > max_length — keine stille Truncation.")
    with contextlib.ExitStack() as stack:
        if steer_preset:
            stack.enter_context(oracle.steer(steer_preset))
        if steer_spec is not None:
            experts, layers, amp = steer_spec
            stack.enter_context(probe.steer_band(experts, layers, amp))
        for _ in range(n_tokens):
            p = probe.last_probs(ctx)
            t = int(p.argmax().item())
            piece = tok.decode([t])
            nxt = ctx + piece
            ids_nx = [int(i) for i in tok(nxt, return_tensors="pt"
                                          )["input_ids"][0]]
            if ids_nx != ids_prev + [t]:
                raise RuntimeError(
                    "Round-Trip-Bruch im Generator: encode(decode(ids + "
                    f"[{t}])) != ids + [{t}] an Kontext ...{ctx[-60:]!r} — "
                    "Gerät-Bruch, kein stiller Fallback.")
            tokens.append(t)
            pieces.append(piece)
            probs.append(float(p[t].item()))
            ctx = nxt
            ids_prev = ids_nx
    return {"prompt": prompt, "tokens": tokens, "pieces": pieces,
            "continuation": "".join(pieces), "text": ctx,
            "probs": [round6(x) for x in probs]}


def _chim_annotation(oracle: Any, probe: Any, tok: Any, dec: Dict[str, Any],
                     experts: List[int], core: List[int],
                     n_experts: int, hdim: int
                     ) -> Dict[str, Any]:
    """B1 — die Chimären: ein 24-Token-Stream, je Token live annotiert
    (χ-Tag — χ_w am NAKENEN Prompt als Nullpunkt, σ via Canon-Reader,
    dp17/dp_moe17, m_core, Top-Paar, H, capacity_gate) + der Querspur
    -Audit (anchor_tv je Token) + Fingerprint. ZAHLEN UNGERUNDET im
    Audit-Feld (anchor_tv_raw),Darstellung gerundet."""
    prompt = SP17_CHIM_PROMPT
    fp_pre = probe.fingerprint()
    p_naked = probe.last_probs(prompt)
    anns: List[Dict[str, Any]] = []
    ctx = prompt
    ids_prev: List[int] = [int(j) for j in tok(ctx, return_tensors="pt"
                                               )["input_ids"][0]]
    if len(ids_prev) + SP17_CHIM_N > int(getattr(probe, "max_length", 128)):
        raise ValueError("Chimäre-Stream läuft über max_length — keine "
                         "stille Truncation.")
    prev_letter: Optional[str] = None
    prev_pair: Optional[Tuple[int, int]] = None
    max_tv = 0.0
    switch_count = 0
    for i in range(SP17_CHIM_N):
        p_live = probe.last_probs(ctx)
        t = int(p_live.argmax().item())
        rows = probe.sublayer_logits(ctx, {"id": t})
        rt = probe.tokenwise_routing(ctx)
        last = int(rt["n_tokens"]) - 1
        m = m_core_raw(rt["pct"], int(rt["n_tokens"]), experts,
                       band=(core[0], core[-1]),
                       layers=int(rt["n_layers"]))
        sigma = _sigma_bio(dec, oracle, ctx)
        chi_w = float(p_naked[t].item())
        cell = score_cell_rev2(float(p_live[t].item()), chi_w,
                               m["value"], sigma)
        row17 = next((r for r in rows["rows"] if int(r["layer"]) == 17),
                     None)
        core_sum = [float(sum(rt["pct"][last][li][e] for li in core))
                    for e in range(int(rt["n_experts"]))]
        pair = sorted(range(len(core_sum)),
                      key=lambda e: core_sum[e], reverse=True)[:2]
        pair_mass = core_sum[pair[0]] + core_sum[pair[1]]
        H = float(np.mean([float(rt["ent"][last][li]) for li in core]))
        gate = min(max(H / math.log(n_experts), 0.25), 1.0)
        anchor_tv = float(rows["anchor_tv"])
        max_tv = max(max_tv, anchor_tv)
        letter = "WCS"[int(np.argmax([cell["chi_w"], cell["chi_c"],
                                      cell["chi_s"]]))]
        switch = bool(prev_letter is not None and
                      (letter != prev_letter or
                       tuple(pair) != (prev_pair or ())))
        switch_count += int(switch)
        piece = tok.decode([t])
        nxt = ctx + piece
        ids_nx = [int(j) for j in tok(nxt, return_tensors="pt"
                                      )["input_ids"][0]]
        if ids_nx != ids_prev + [t]:
            raise RuntimeError(
                "Round-Trip-Bruch im Chimären-Stream an Token "
                f"{i + 1} ({piece!r}) — Gerät-Bruch, kein stiller "
                "Fallback.")
        anns.append({
            "i": i, "target_id": t,
            "piece": piece,
            "letter": letter,
            "chi_w": cell["chi_w"], "chi_c": cell["chi_c"],
            "chi_s": cell["chi_s"],
            "dp17": round6(abs(float(row17["dp_attn"]))),
            "dp_moe17": round6(abs(float(row17["dp_moe"]))),
            "m_core": round6(m["value"]),
            "pair": pair, "pair_mass": round6(pair_mass),
            "H": round6(H), "gate": round6(gate),
            "sigma": round6(sigma),
            "p_live": round6(float(p_live[t].item())),
            "anchor_tv": round6(anchor_tv),
            "anchor_tv_raw": anchor_tv,
            "switch": switch,
        })
        ctx = nxt
        ids_prev = ids_nx
        prev_letter = letter
        prev_pair = tuple(pair)
    fp_post = probe.fingerprint()
    band_audit = max_tv <= SP17_AUDIT_MAX
    band_complete = (len(anns) == SP17_CHIM_N
                     and all({"i", "target_id", "piece", "letter", "chi_w",
                              "chi_c", "chi_s", "dp17", "dp_moe17",
                              "m_core", "pair", "pair_mass", "H", "gate",
                              "sigma", "p_live", "anchor_tv",
                              "switch"}.issubset(a) for a in anns))
    band_fp = bool(probe.fingerprint_equal(fp_pre, fp_post))
    line = _chim_line(anns)
    verdict = bool(band_audit and band_complete and band_fp)
    return {"band_audit": band_audit, "band_complete": band_complete,
            "band_fingerprint": band_fp,
            "max_anchor_tv_raw": max_tv, "switch_count": switch_count,
            "verdict": verdict,
            "annotations": anns, "chimaera_text": ctx,
            "chimaera_line": line}


def _chim_line(anns: List[Dict[str, Any]]) -> str:
    """Die User-Log-Form: piece[K|L17:+…|E i,j: masse|gate:G|H:H]."""
    parts: List[str] = []
    for a in anns:
        bits = []
        if a["switch"]:
            bits.append("SWITCH!")
        bits.append(f"L17:{a['dp17']:+.4f}")
        bits.append(f"E{a['pair'][0]},{a['pair'][1]}:"
                    f"{a['pair_mass']:.2f}")
        bits.append(f"gate:{a['gate']:.2f}")
        parts.append(f"{a['piece']}[{a['letter']}|" + "|".join(bits)
                     + "]")
    return "".join(parts)


def _rezeptur_weave(oracle: Any, word: str, pair: Tuple[str, str]
                    ) -> Tuple[Any, List[str]]:
    """Die SP16-Weichens-Rezeptur (make_composite + learn_pair_pin +
    bio_halted + bio_gelernt + Runden-Leiter) — für die B2-Weiche ein
    FRISCHES Subjekt; der Render der letzten Stufe ist der SP16-Anker."""
    machine = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(machine, pair)
    bio_halted(machine, word)
    bio_gelernt(machine, word)
    renders: List[str] = []
    for runde in ZEITPUNKTE:
        p_k = machine.preview_steps([word] * runde,
                                    lexical_grounding=True)
        renders.append(render_state(machine, "textur", probe=p_k))
        machine.step(word, lexical_grounding=True)
    return machine, renders


def lane_chimaera(oracle: Any, probe: Any, tok: Any, dec: Dict[str, Any],
                  ) -> Dict[str, Any]:
    """B1."""
    hdim = int(oracle.model.config.hidden_size)
    experts, _steer, _stored, source = oracle._preset_experts(M7_PRESET)
    n_layers = len(probe.model.model.layers)
    _edge, core = layer_bands(n_layers)
    n_experts = int(getattr(oracle.model.config, "n_routed_experts", 8))
    out = _chim_annotation(oracle, probe, tok, dec, experts, core,
                           n_experts, hdim)
    out["preset_experts"] = [int(e) for e in experts]
    out["preset_source"] = source
    out["core_band"] = core
    return out


def lane_weave(oracle: Any, probe: Any, tok: Any,
               korpus: Dict[str, Any]) -> Dict[str, Any]:
    """B2 — 5 diverse Prompts: raw gegen die kybernetische Weiche. Die
    Wahl-Repliken laufen kumulativ auf der Exakt-Rezept-Maschine
    (make_composite + learn_pair_pin + bio_gelernt OHNE bio_halted —
    SP16-P1d-Form); der Korpus-Subjekt-Anteil liegt in machine_m
    (Rezeptur-Anker + Preview-Weiche)."""
    machine_m, renders_m = _rezeptur_weave(oracle, "spiegel", LEX_PAIR)
    band_rezept = renders_m[-1] == korpus["texts"][10]

    # die Preview-Weiche + digest-Restaurierung (non-mutierend, SP16-Muster)
    fp_pre = probe.fingerprint()
    dig_pre = machine_m.digest()
    probes_prev = machine_m.preview_steps(["spiegel"] * SP17_PREV_K,
                                          lexical_grounding=True)
    verdicts = [p.invariants["verdict"] for p in probes_prev]
    band_vorschau = (verdicts == ["LOOP"] * SP17_PREV_K
                     and machine_m.digest() == dig_pre)

    # die Wahl-Repliken (kumulativ, Exakt-Rezept OHNE bio_halted — je
    #     Prompt ein step_autonomous; die Maschine liest, wählt, wächst)
    machine_r = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(machine_r, LEX_PAIR)
    bio_gelernt(machine_r, "spiegel")
    rows: List[Dict[str, Any]] = []
    digests_r: List[str] = [machine_r.digest()]
    for name, prompt in SP17_PROMPTS:
        result = machine_r.step_autonomous(prompt, lexical_grounding=False)
        auto = result.invariants["autonomous"]
        chosen = str(auto["chosen_preset"])
        gaps = {str(k): float(v)
                for k, v in auto["candidate_gaps"].items()}
        digests_r.append(machine_r.digest())
        lawful = (chosen == min(gaps, key=lambda k: (gaps[k],
                                                     list(gaps).index(k))))
        raw_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        if chosen == "raw":
            weave_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        else:
            with oracle.steer(chosen):
                weave_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        detlaw = (raw_run["tokens"] == weave_run["tokens"]
                  if chosen == "raw"
                  else raw_run["tokens"] != weave_run["tokens"])
        rows.append({
            "prompt_kind": name, "prompt": prompt,
            "chosen_preset": chosen,
            "candidate_gaps": {k: round6(v) for k, v in gaps.items()},
            "best_gap": round6(float(auto["best_gap"])),
            "raw_gap": round6(gaps["raw"]),
            "wahr": bool(float(auto["best_gap"]) < gaps["raw"]),
            "lawful": bool(lawful), "detlaw": bool(detlaw),
            "raw_tokens": raw_run["tokens"],
            "raw_pieces": raw_run["pieces"],
            "raw_text": raw_run["continuation"],
            "weave_tokens": weave_run["tokens"],
            "weave_pieces": weave_run["pieces"],
            "weave_text": weave_run["continuation"],
        })
    fp_post = probe.fingerprint()
    band_fp = bool(probe.fingerprint_equal(fp_pre, fp_post))
    band_choice = bool(all(r["lawful"] for r in rows))
    band_detlaw = bool(all(r["detlaw"] for r in rows))
    n_weaves = int(sum(1 for r in rows if r["chosen_preset"] != "raw"))
    verdict = bool(band_rezept and band_vorschau and band_choice
                   and band_detlaw and band_fp)
    return {"band_rezeptur": bool(band_rezept),
            "band_vorschau": band_vorschau,
            "preview_verdicts": verdicts,
            "band_choice": band_choice, "band_detlaw": band_detlaw,
            "band_fingerprint": band_fp,
            "n_weave_preset": n_weaves,
            "digests_r": digests_r,
            "expansionen": len(machine_r.expansions),
            "weave_machine_render": renders_m[-1],
            "rows": rows, "verdict": verdict}


def _tail_prompt(probe: Any, tok: Any, render: str, sentence: str
                 ) -> Tuple[str, int]:
    """Explizite Schwanz-Scheibe der Biografie-Spur: die letzten W
    Tokens des Renderings + Satz, W so gewählt, dass prompt + N ≤
    max_length (kein stilles Kürzen; SP16 las dieses Instrument-Fenster
    stillschweigend — hier deklariert, Fensterlänge je Zeile als
    Datum)."""
    mx = int(getattr(probe, "max_length", 128))
    sep = " " + sentence
    r_ids = tok.encode(render, add_special_tokens=False)
    for keep_n in range(mx - SP17_GEN_N - 2, 23, -8):
        cut = r_ids[-keep_n:] if keep_n <= len(r_ids) else r_ids
        p = tok.decode(cut) + sep
        n = len(tok(p, return_tensors="pt")["input_ids"][0])
        if n + SP17_GEN_N <= mx:
            return p, len(cut)
    raise ValueError("Dialog-Prompt passt nicht in das Instrument-Fenster "
                     "(max_length) — kein stilles Kürzen.")


def lane_twins(oracle: Any, probe: Any, tok: Any,
               korpus: Dict[str, Any]) -> Dict[str, Any]:
    """B3 — Zwillings-Dialoge auf natürlicher Sprache (SP16-Rezeptur)."""
    fp0 = probe.fingerprint()
    machine_a = korpus["machine_a"]
    a_texts = list(korpus["texts"])[:12]
    machine_a.goedel.enable_wide_read()
    for t in a_texts:
        machine_a.goedel.step(t)
    adapter_a = np.asarray(machine_a.goedel.wide_read_adapter,
                           dtype=np.float64)
    digest_a = hashlib.sha256(adapter_a.tobytes()).hexdigest()
    b = make_composite(mode="oracle", oracle=oracle)
    learn_pair_pin(b, LEX_PAIR_B)
    bio_halted(b, "stille")
    bio_gelernt(b, "stille")
    renders_b: List[str] = []
    for runde in ZEITPUNKTE:
        p_k = b.preview_steps(["stille"] * runde,
                              lexical_grounding=True)
        renders_b.append(render_state(b, "textur", probe=p_k))
        b.step("stille", lexical_grounding=True)
    b.goedel.enable_wide_read()
    for t in renders_b:
        b.goedel.step(t)
    adapter_b = np.asarray(b.goedel.wide_read_adapter, dtype=np.float64)
    digest_b = hashlib.sha256(adapter_b.tobytes()).hexdigest()
    band_adapter_a = digest_a == SP17_ANCHOR_A
    band_adapter_b = digest_b == SP17_ANCHOR_B
    # die Verdikt-Spuren (Wort-Formular, SP16-Muster): A LOOP, B HALTED
    p_a1 = machine_a.preview_steps(["spiegel"], lexical_grounding=True)[0]
    p_b1 = b.preview_steps(["stille"], lexical_grounding=True)[0]
    spuren = {"a": {"verdict": p_a1.invariants["verdict"],
                    "lexical_ast": str(p_a1.invariants["lexical_ast"])},
              "b": {"verdict": p_b1.invariants["verdict"],
                    "lexical_ast": str(p_b1.invariants["lexical_ast"])}}
    render_a = korpus["texts"][10]
    render_b = renders_b[-1]
    hdim = int(oracle.model.config.hidden_size)
    sents: List[Dict[str, Any]] = []
    digests_a: List[str] = []
    digests_b: List[str] = []
    n_differ = 0
    band_z = True
    for s in SP17_DIALOG:
        h = np.asarray(oracle.hidden_state(s, dimensions=hdim),
                       dtype=np.float64)
        z_a = adapter_a @ h
        z_b = adapter_b @ h
        z_diff_bool = float(np.linalg.norm(z_a - z_b)) > 0.0
        band_z = band_z and z_diff_bool
        p_a, w_a = _tail_prompt(probe, tok, render_a, s)
        p_b, w_b = _tail_prompt(probe, tok, render_b, s)
        choice_a = str(machine_a.step_autonomous(
            p_a, lexical_grounding=False).invariants["autonomous"]
            ["chosen_preset"])
        digests_a.append(machine_a.digest())
        choice_b = str(b.step_autonomous(
            p_b, lexical_grounding=False).invariants["autonomous"]
            ["chosen_preset"])
        digests_b.append(b.digest())
        run_a = _gen(probe, tok, oracle, p_a, SP17_GEN_N,
                     steer_preset=(choice_a if choice_a != "raw" else None))
        run_b = _gen(probe, tok, oracle, p_b, SP17_GEN_N,
                     steer_preset=(choice_b if choice_b != "raw" else None))
        differ = run_a["tokens"] != run_b["tokens"]
        n_differ += int(differ)
        sents.append({
            "sentence": s,
            "z_norm_a": round6(float(np.linalg.norm(z_a))),
            "z_norm_b": round6(float(np.linalg.norm(z_b))),
            "z_diff": round6(float(np.linalg.norm(z_a - z_b))),
            "z_diff_rel": round6(float(np.linalg.norm(z_a - z_b))
                                 / (float(np.linalg.norm(z_a))
                                    + float(np.linalg.norm(z_b)) + 1e-12)),
            "chosen_a": choice_a, "chosen_b": choice_b,
            "prompt_a": p_a, "prompt_b": p_b,
            "tail_window_a": int(w_a), "tail_window_b": int(w_b),
            "answer_a_tokens": run_a["tokens"],
            "answer_a_pieces": run_a["pieces"],
            "answer_a_text": run_a["continuation"],
            "answer_b_tokens": run_b["tokens"],
            "answer_b_pieces": run_b["pieces"],
            "answer_b_text": run_b["continuation"],
            "differ": bool(differ),
        })
    fp_end = probe.fingerprint()
    band_fp = bool(probe.fingerprint_equal(fp0, fp_end))
    band_answers = n_differ >= SP17_TWIN_MIN
    verdict = bool(band_adapter_a and band_adapter_b and band_z
                   and band_answers and band_fp)
    return {"band_adapter_a": band_adapter_a, "band_adapter_b": band_adapter_b,
            "band_z": band_z, "band_answers": band_answers,
            "band_fingerprint": band_fp,
            "adapter_digest_a": digest_a, "adapter_digest_b": digest_b,
            "digests_dialog_a": digests_a, "digests_dialog_b": digests_b,
            "w_mean_abs_a": machine_a.goedel.state()["wide_read"]
            ["w_mean_abs"],
            "w_mean_abs_b": b.goedel.state()["wide_read"]["w_mean_abs"],
            "n_differ": n_differ, "sents": sents,
            "render_a": render_a, "render_b": render_b,
            "verdict": verdict}


def lane_tenpal(oracle: Any, probe: Any, tok: Any, dec: Dict[str, Any],
                korpus: Dict[str, Any]) -> Dict[str, Any]:
    """B4 — die 10-Palette: χ-Zerlegung je registriertem Satz; Band:
    der dominante Kanal trifft das Register überall."""
    hdim = int(oracle.model.config.hidden_size)
    experts, _steer, _stored, _source = oracle._preset_experts(M7_PRESET)
    n_layers = len(probe.model.model.layers)
    _edge, core = layer_bands(n_layers)
    rows: List[Dict[str, Any]] = []
    for reg, full, abl, target_str in SP17_TENPAL_FIXED:
        rows.append(_pal_row(oracle, probe, tok, dec, experts, core,
                             reg, full, abl, target_str))
    for idx in SP17_TENPAL_S_INDICES:
        s_full = korpus["texts"][idx]
        s_abl = korpus["texts"][idx + 12]      # der B-Zweig (SP16-Muster:
        tid = int(tok.encode(s_full, add_special_tokens=False)[-1])
        rows.append(_pal_row(oracle, probe, tok, dec, experts, core,
                             "S", s_full, s_abl,
                             tok.decode([tid]), tid))
    band_class = all(r["class_ok"] for r in rows)
    loo = {"acc": float(dec["loo"]["acc"]),
           "p_perm": float(dec["loo"]["p_perm"]),
           "n_texts": int(dec["loo"]["n_texts"])}
    band_loo = (loo["acc"] >= SP17_LOO_MIN
                and loo["p_perm"] <= SP17_PPERM_MAX)
    verdict = bool(band_class and band_loo)
    return {"band_class": band_class, "band_loo": band_loo,
            "loo": loo, "rows": rows, "verdict": verdict}


def _pal_row(oracle: Any, probe: Any, tok: Any, dec: Dict[str, Any],
             experts: List[int], core: List[int], reg: str, full: str,
             abl: str, target_str: str, tid: Optional[int] = None
             ) -> Dict[str, Any]:
    """Ein Satz: die χ-Tag-Zerlegung am Ziel (Formeln REV-2: χ_w =
    p_abl(t); χ_c = max(0, p_full − p_abl); χ_s = σ·m_core)."""
    if tid is None:
        tid = _resolve_digit_target(probe, target_str)
    rows_full = probe.sublayer_logits(full, {"id": int(tid)})
    p_full = probe.last_probs(full)
    p_abl = probe.last_probs(abl)
    rt = probe.tokenwise_routing(full)
    m = m_core_raw(rt["pct"], int(rt["n_tokens"]), experts,
                   band=(core[0], core[-1]),
                   layers=int(rt["n_layers"]))
    sigma = _sigma_bio(dec, oracle, full)
    cell = score_cell_rev2(float(p_full[tid].item()),
                           float(p_abl[tid].item()), m["value"], sigma)
    vals = [cell["chi_w"], cell["chi_c"], cell["chi_s"]]
    dom = "WCS"[int(np.argmax(vals))]
    row17 = next((r for r in rows_full["rows"] if int(r["layer"]) == 17),
                 None)
    return {
        "register": reg, "dominant": dom,
        "class_ok": dom == reg,
        "chi": cell, "m_core": round6(m["value"]),
        "sigma": round6(sigma),
        "dp17": round6(abs(float(row17["dp_attn"]))),
        "anchor_tv": round6(float(rows_full["anchor_tv"])),
        "p_full_target": round6(float(p_full[tid].item())),
        "p_abl_target": round6(float(p_abl[tid].item())),
        "target_id": int(tid), "target_str": target_str,
        "full": full, "abl": abl,
    }


def lane_surgical(oracle: Any, probe: Any, tok: Any
                  ) -> Dict[str, Any]:
    """B5 — chirurgisch (ℓ4..23) gegen voll (ℓ0..25): das ±50-SET
    -Template je Arm, raw gegen beide."""
    experts, _steer, _stored, _source = oracle._preset_experts(M7_PRESET)
    n_layers = len(probe.model.model.layers)
    _edge, core = layer_bands(n_layers)
    rows: List[Dict[str, Any]] = []
    band_fp = True
    band_det = False
    band_sf = 0
    for i, (name, prompt) in enumerate(SP17_PROMPTS):
        raw_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        if i == 0:
            raw_check = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
            band_det = bool(raw_run["tokens"] == raw_check["tokens"])
        fp_s = probe.fingerprint()
        with probe.steer_band(experts, core, SP17_STEER_AMP):
            surg_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        band_fp = band_fp and bool(
            probe.fingerprint_equal(fp_s, probe.fingerprint()))
        fp_f = probe.fingerprint()
        with probe.steer_band(experts, None, SP17_STEER_AMP):
            full_run = _gen(probe, tok, oracle, prompt, SP17_GEN_N)
        band_fp = band_fp and bool(
            probe.fingerprint_equal(fp_f, probe.fingerprint()))
        differ = surg_run["tokens"] != full_run["tokens"]
        band_sf += int(differ)
        rows.append({
            "prompt_kind": name,
            "raw_text": raw_run["continuation"],
            "surg_text": surg_run["continuation"],
            "full_text": full_run["continuation"],
            "raw_tokens": raw_run["tokens"],
            "surg_tokens": surg_run["tokens"],
            "full_tokens": full_run["tokens"],
            "sf_differ": bool(differ),
            "sur_eq_raw": surg_run["tokens"] == raw_run["tokens"],
        })
    band_differ = band_sf >= 1
    verdict = bool(band_det and band_fp and band_differ)
    return {"band_det": band_det, "band_fingerprint": band_fp,
            "band_differ": band_differ,
            "n_sf_differ": int(band_sf),
            "core_band": core, "n_layers": int(n_layers),
            "preset_experts": [int(e) for e in experts],
            "rows": rows, "verdict": verdict}


def lane_det(oracle: Any, a_texts: List[str]) -> Dict[str, Any]:
    """B6/Training-Determinismus: 2 × _train_wide auf dem A-Zweig —
    Adapter-Digests gleich UND == SP15-Anker (Querschnitt-Regression,
    dritte Runde); gap-Digest analog."""
    runs: List[Dict[str, Any]] = []
    for _ in range(2):
        _machine, info = _train_wide(oracle, a_texts)
        runs.append({"w_mean_abs": info["w_mean_abs"],
                     "adapter_digest": info["adapter_digest"],
                     "gap_digest": info["gap_digest"]})
    equal = runs[0]["adapter_digest"] == runs[1]["adapter_digest"] and \
        runs[0]["gap_digest"] == runs[1]["gap_digest"]
    band_sp15 = runs[0]["adapter_digest"] == SP17_SP15_ANCHOR \
        and runs[0]["gap_digest"] == SP17_SP15_GAP_ANCHOR
    return {"runs": runs, "band_equal": bool(equal),
            "band_sp15": bool(band_sp15),
            "verdict": bool(equal and band_sp15)}


# -- Score-Mathe fürs CPU-Pilot-Feld (kein Modell nötig) -----------------------

def _mock_cap(H: float, n_experts: int) -> float:
    return min(max(H / math.log(n_experts), 0.25), 1.0)


def lane_pilot() -> Dict[str, Any]:
    """Synthetische Legs — der Programm-Kern im CPU-Piloten (kein
    Modell): Dominanz-Formeln, Audit-Schwelle, Klassifikations-Gesetz,
    argmin-Regel, Differenz-Prädikate, Z-Divergenz, capacity_gate-Gesetz,
    layer_bands, Determinismus-Prädikat."""
    legs: Dict[str, Any] = {}

    # leg dominanz: die MI-7-REV-2-Zeilen (SP16-JSON verbatim, die
    #     Formeln im Piloten exakt wie in _pal_row: chi_w = p_abl,
    #     chi_c = max(0, p_full − p_abl), chi_s = σ·m_core)
    rows_w = score_cell_rev2(0.503078, 0.503078, 0.175808, 0.263603)
    rows_c = score_cell_rev2(0.806368, 0.000067, 0.036648, 0.981735)
    rows_s = score_cell_rev2(0.0, 0.0, 0.563825, 0.987035)
    dom = lambda c: "WCS"[int(np.argmax([c["chi_w"], c["chi_c"],
                                         c["chi_s"]]))]
    legs["dominanz_dom"] = (dom(rows_w) == "W" and dom(rows_c) == "C"
                            and dom(rows_s) == "S")

    # leg audit: die Querspur-Schwelle 1e-4 scheidet (Pass bei 1e-9,
    #     Fail bei 1e-3)
    legs["audit_threshold"] = (1e-9 <= SP17_AUDIT_MAX
                               and 1e-3 > SP17_AUDIT_MAX)

    # leg klassifikation: 10 Mock-Sätze → registrierte Register
    mock = [("W", (0.9, 0.01, 0.05)), ("C", (0.02, 0.9, 0.05)),
            ("S", (0.01, 0.02, 0.9))]
    ok = True
    for reg, vals in mock * 3:
        ok = ok and ("WCS"[int(np.argmax(vals))] == reg)
    legs["tenpal_law"] = ok

    # leg argmin-Regel (Ties → früherer Kandidat)
    gaps = {"raw": 0.5, "goedel": 0.4, "turing": 0.4,
            "wittgenstein": 0.6, "combined": 0.4}
    chosen = min(gaps, key=lambda k: (gaps[k], list(gaps).index(k)))
    legs["argmin_rule"] = chosen == "goedel"

    # leg differ: die Token-Sequenz-Prädikate
    legs["differ_law"] = ([1, 2, 3] != [1, 2, 4]
                          and [1, 2, 3] == [1, 2, 3])

    # leg z: Mock-Adapter → ‖z_A − z_B‖ > 0 (und 0-Fall schneidet)
    rng = np.random.default_rng(17)
    za = rng.normal(0.0, 1.0, 8)
    zb = rng.normal(3.0, 1.0, 8)
    legs["z_law"] = (float(np.linalg.norm(za - zb)) > 0.0
                     and not (float(np.linalg.norm(za - za)) > 0.0))

    # leg gate: capacity_gate-Gesetz (canon moe_layers:91 —
    #     (H / log(n)).clamp(0.25, 1.0); Mock gegen die Formel)
    legs["cap_gate_law"] = (_mock_cap(0.2, 8) == 0.25      # unteres Klemmen
                            and _mock_cap(9.0, 8) == 1.0   # oberes Klemmen
                            and _mock_cap(1.6, 8) == 1.6 / math.log(8)
                            and _mock_cap(1.0, 2) == 1.0)  # 2-Experten-Fall

    # leg bands: layer_bands(26) → Edge {0..3, 24, 25}, Core 4..23
    edge, core = layer_bands(26)
    legs["bands_26"] = (edge == [0, 1, 2, 3, 24, 25]
                        and core == list(range(4, 24)))

    # leg det: das Doppellauf-Prädikat
    legs["det_law"] = ("a" == "a" and "a" != "b")

    verdict = bool(all(bool(v) for v in legs.values()))
    return {"legs": legs, "verdict": verdict}


def cpu_findings(pilot: Dict[str, Any]) -> Dict[str, Any]:
    """Doppellauf des Programm-Kerns (S2/Kanon)."""
    pilot2 = lane_pilot()
    return {"runs": [pilot, pilot2],
            "verdict": bool(pilot["verdict"]
                            and json.dumps(pilot, sort_keys=True)
                            == json.dumps(pilot2, sort_keys=True))}


# -- GPU-Lauf: die Batterie ------------------------------------------------------

def battery(ctx: Tuple[Any, ...]) -> Dict[str, Any]:
    _torch, model, tok, oracle, _dev = ctx
    probe = _new_probe(model, tok)
    fp_start = probe.fingerprint()

    stufe("Korpus (P4-P1a-Rezeptur verbatim — 24 Renderings)")
    korpus = _p4_korpusbau(oracle)
    stufe("σ_bio-Decoder (MI-7-Rezept: 30 Texte, LogReg)")
    dec = _sigma_decoder(oracle, korpus)
    a_texts = list(korpus["texts"])[:12]
    korpus_meta = {"n": len(korpus["texts"]),
                   "index_s_full": korpus["index_s_full"],
                   "index_s_abl": korpus["index_s_abl"],
                   "s_full": korpus["texts"][10][:80],
                   "s_abl": korpus["texts"][22][:80]}

    stufe("B1 — Chimären-Stream (24 Token, Live-Annotation + Querspur)")
    chimaera = lane_chimaera(oracle, probe, tok, dec)
    stufe("B2 — die Weiche im Live-Einsatz (5 Prompts × raw/weave)")
    weave = lane_weave(oracle, probe, tok, korpus)
    stufe("B3 — die Zwillings-Dialoge (natürliche Sprache)")
    twins = lane_twins(oracle, probe, tok, korpus)
    stufe("B4 — die 10-Palette (Fakten / Kontext / Register)")
    tenpal = lane_tenpal(oracle, probe, tok, dec, korpus)
    stufe("B5 — chirurgisches gegen volles Steering")
    surgical = lane_surgical(oracle, probe, tok)
    stufe("B6 — Training-Determinismus (2 × _train_wide)")
    train_det = lane_det(oracle, a_texts)

    fp_end = probe.fingerprint()
    purity = {"equal": bool(probe.fingerprint_equal(fp_start, fp_end)),
              "verdict": bool(probe.fingerprint_equal(fp_start, fp_end))}
    del korpus["machine_a"], korpus["machine_b"]
    out = {
        "korrektur_log": list(SP17_KORREKTUR),
        "korpus": korpus_meta,
        "sigma_loo_datum": dec["loo"],
        "chimaera": chimaera, "weave": weave, "twins": twins,
        "tenpal": tenpal, "surgical": surgical,
        "train_det": train_det, "purity": purity,
    }
    # Crash-Schutz (KOR-2-Konsequenz, kein Band-Eingriff): der Roh-Spiegel
    #     der Batterie liegt auf Disk, BEVOR die Assemblierung läuft — ein
    #     Lauf-Bug hinter findings vernichtet dann nicht 40 Minuten Rechen.
    with open("scratches/sp17_battery_snapshot.json", "w",
              encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, ensure_ascii=False)
    stufe("Roh-Snapshot geschrieben (sp17_battery_snapshot.json)")
    return out


def stufe(text: str) -> None:
    print(text, flush=True)


def _new_probe(model: Any, tok: Any) -> Any:
    from machines.mech_interp import MechInterpProbe

    return MechInterpProbe(model, tok)


def _sigma_decoder(oracle: Any, korpus: Dict[str, Any]) -> Dict[str, Any]:
    """σ_bio-Decoder nach dem MI-7-Rezept (SP16 verbatim): 30 Texte
    (24-Korpus + 6 Füll-Negative), StandardScaler + LogReg (max_iter
    1000, C 1.0) auf den vollen Hidden (1152, Canon-Pool), Positiv-Klasse
    = 1; LOO-Datum als DICT (acc, p_perm, n_texts)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    texts = list(korpus["texts"]) + list(FUENLLE_NEGATIVE)
    labels = list(korpus["labels"]) + [0] * len(FUENLLE_NEGATIVE)
    h_pool = [oracle.hidden_state(
        t, dimensions=int(oracle.model.config.hidden_size)) for t in texts]
    h_arr = np.asarray(h_pool)
    scaler = StandardScaler().fit(h_arr)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(scaler.transform(h_arr), np.asarray(labels))
    pos_idx = int(list(clf.classes_).index(1))
    loo = _loo_decoder(h_arr, np.asarray(labels))
    return {"scaler": scaler, "clf": clf, "pos_idx": pos_idx, "loo": loo}


def _sigma_bio(dec: Dict[str, Any], oracle: Any, text: str) -> float:
    """σ_bio an einem Text (voller Hidden, gelernter Reader) — SP16
    verbatim."""
    h = oracle.hidden_state(
        text, dimensions=int(oracle.model.config.hidden_size))
    x = dec["scaler"].transform(h.reshape(1, -1))
    return float(dec["clf"].predict_proba(x)[0, int(dec["pos_idx"])])


def findings(battery_out: Dict[str, Any], cpu: Dict[str, Any]
             ) -> List[Dict[str, Any]]:
    gpu = battery_out
    rows = [
        {"id": "sp17_chimaera_stream",
         "verdict": bool(gpu["chimaera"]["verdict"] and gpu["purity"]["equal"])},
        {"id": "sp17_weave_before_after",
         "verdict": bool(gpu["weave"]["verdict"] and gpu["purity"]["equal"])},
        {"id": "sp17_twin_dialogue",
         "verdict": bool(gpu["twins"]["verdict"] and gpu["purity"]["equal"])},
        {"id": "sp17_tenpal_detector",
         "verdict": bool(gpu["tenpal"]["verdict"] and gpu["purity"]["equal"])},
        {"id": "sp17_surgical_steering",
         "verdict": bool(gpu["surgical"]["verdict"]
                         and gpu["purity"]["equal"])},
        {"id": "sp17_purity_and_det",
         "verdict": bool(gpu["purity"]["equal"]
                         and gpu["train_det"]["verdict"]
                         and cpu["verdict"])},
    ]
    return rows


def _to_rounded(obj: Any) -> Any:
    """Darstellungs-Rounding im JSON-Spiegel — die *-Felder „*_raw“
    (Querspur-Audit) bleiben UNGERUNDET (SP16-Audit-Muster)."""
    if isinstance(obj, float):
        return round6(obj)
    if isinstance(obj, dict):
        return {k: (v if k.endswith("_raw") else _to_rounded(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_rounded(v) for v in obj]
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SP17 — die Beispiel-Suite der «unmöglichen Texte»")
    parser.add_argument("--gpu", action="store_true",
                        help="GPU-Lanes fahren (sonst CPU-Pilot)")
    args = parser.parse_args()

    print("== SP17 — die Beispiel-Suite der «unmöglichen Texte» "
          "(Chimären-Stream, Weichen-Vergleich, Zwillings-Dialoge, "
          "10-Palette, chirurgisches Steering) ==")
    pilot = lane_pilot()
    cpu = cpu_findings(pilot)
    doppelt = json.dumps(pilot, sort_keys=True) == json.dumps(
        cpu["runs"][1], sort_keys=True)
    print(f"CPU-Pilot: {'PASS' if pilot['verdict'] else 'FAIL'} "
          f"(Doppellauf {'bitidentisch' if doppelt else 'GESPALEN'})")

    battery_out: Optional[Dict[str, Any]] = None
    if args.gpu:
        print("GPU-Kontext laden (argon_1b + argon_1b_machines)")
        ctx = _load_machines_ctx()
        battery_out = battery(ctx)
        # Ein Lauf-Bug bricht hier mit Traceback ab (kein JSON, Log
        # verbatim gesichert — SP16-KOR-Muster); OFFEN gibt es nur ohne
        # --gpu.

    if battery_out is None:
        rows = [
            {"id": "sp17_chimaera_stream", "verdict": None},
            {"id": "sp17_weave_before_after", "verdict": None},
            {"id": "sp17_twin_dialogue", "verdict": None},
            {"id": "sp17_tenpal_detector", "verdict": None},
            {"id": "sp17_surgical_steering", "verdict": None},
            {"id": "sp17_purity_and_det", "verdict": None},
        ]
        print("OFFEN — GPU-Lanes nicht gefahren (CPU-Pilot nur)")
        result = {"cpu": cpu, "gpu": None, "findings": rows}
        with open("scratches/sp17_diverse_examples.json", "w",
                  encoding="utf-8") as handle:
            json.dump(result, handle, indent=1, ensure_ascii=False)
        return 2

    gpu = {
        "chimaera": _to_rounded(battery_out["chimaera"]),
        "weave": _to_rounded(battery_out["weave"]),
        "twins": _to_rounded(battery_out["twins"]),
        "tenpal": _to_rounded(battery_out["tenpal"]),
        "surgical": _to_rounded(battery_out["surgical"]),
        "train_det": _to_rounded(battery_out["train_det"]),
        "purity": _to_rounded(battery_out["purity"]),
    }
    rows = findings(battery_out, cpu)
    n_pass = len([r for r in rows if r["verdict"] is True])
    n_fail = len([r for r in rows if r["verdict"] is False])
    for r in rows:
        sign = "PASS" if r["verdict"] is True else "FAIL"
        print(f"{r['id']:<34} {sign}")
    print(f"BILANZ: {n_pass} PASS | {n_fail} FAIL | 0 OFFEN")
    exit_code = 0 if n_fail == 0 else 1
    if exit_code == 0:
        print("GRUEN — die unmöglichen Texte sprechen in ganzen Sätzen")
    else:
        print("ROT — mindestens ein Band verletzt (verdict verbatim)")
    historie = {
        "basis_bausteine": [
            "verify_machines (6 Stufen, commit 8a2611b)",
            "SP12 Spiegel-Wort/SP13 Weiche",
            "MI-6 (55ed1e9/05f63ae) + MI-6-P4 (3cb8bf9/52b1dc7)",
            "MI-7 REV-2 (1fbb46a/b68b686)",
            "SP15 REV-4 (977330d/d1d857b)",
            "SP16 GRUEN (9b7510b/62ebf96)",
        ],
        "erstlauf": "SP17 Rev. 1 (Runde 4 des 5-Pläne-Programms) — die "
                    "Beispiel-Suite; das Verdikt steht in findings",
    }
    payload = {
        "datum": "SP17 Rev. 1",
        "historie": historie,
        "cpu": cpu,
        "gpu": gpu,
        "korrektur_log": battery_out["korrektur_log"],
        "findings": rows,
    }
    with open("scratches/sp17_diverse_examples.json", "w",
              encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)
    print("JSON: scratches/sp17_diverse_examples.json")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())