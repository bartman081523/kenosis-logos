"""
repo2txt_machines.py — baut einen Ein-Datei-Konkatenat für den Transfer in
andere Kontexte (repo2txt-Stil, ein Datei-Header pro Block).

Generationen-Regime (User-Vorgabe 2026-10-26): FILES enthält NUR das
Runden-Paket des letzten Spielzugs — (a) die Dateien, die im Lauf GEÄNDERT
wurden, (b) die Dateien, die der Programmlauf AUSGEFÜHRT hat (das ganze
Programm samt importierter Maschinen-Basis und der gelesenen Anker-Daten)
und (c) seine ERGEBNISSE (Run-Logs, JSON, Berichte). Jedesmal ERSETZT, nie
kumulativ; sonst wird das Konkatenat zu groß. Das vollständige Korpus liegt
über die Git-History (frühere Generationen: kumulative Fassungen — Commits
49ecc31 [sha 6ad22d85..., 88 Dateien] und 2f61402 [sha 47ebee46..., 84
Dateien] usw.).

Ausgabe: scratches/repo2txt_machines_extension.txt (+ .sha256)
Run:     python3 scratches/repo2txt_machines.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "scratches" / "repo2txt_machines_extension.txt"

# GENERATION SP16 — Kybernetische Textgenerierung & Provenanz-Audit der
# «unmöglichen Texte» (3. Runde des 5-Pläne-Programms; Plan 3 Cybernetik
# + Plan 4 unmögliche Texte verschmolzen): machines/ UNANGETASTET — das
# SP16-Gerüst lebt NUR aus dem Canon-Stack (MI-6/P4-Rezeptur, MI-7-Gates
# + _p4_korpusbau/_arm_battery-Anker, SP15-JL-Adapter/_train_wide mit
# dem bitidentischen Trainings-Digest 9f9f2d05…). Run-Chronik:
# CPU-Pilot GRUEN (7 Legs, Doppellauf bitidentisch, sp16_cpu_pilot.log);
# Erstlauf GEBROCHEN (lane_horizont Call-Site ohne tok — NameError am
# post_weiche-Decode, OHNE JSON; Log verbatim gesichert:
# sp16_erstrun_gebrochen.log); KORREKTUR-LOG KOR-1 beweislich (reiner
# Lauf-Bug — Call-Site; Baender UNVERAENDERT seit PREREG Rev. 1,
# Plan Revision 7); Re-Lauf **4 PASS | 0 FAIL | 0 OFFEN, exit 0
# GRUEN** — die unmöglichen Texte tragen ihre Provenanz. Gattung A
# (Live-Provenanz-Stream): Gates w (' Paris' 9079, p 0.503078, Zeile
# [0.503078, 0, 0.046344], dp17 0.138588, m_core 0.175808, share
# 0.351388, σ 0.263603), c (' Z' 1491, p 0.806368, Zeile
# [0.000067, 0.806301, 0.035979], dp17 0.963739, m_core 0.036648,
# share 0.708323, σ 0.981735), s ('6' 236825, p 0.859480, Zeile
# [0, 0, 0.556515], dp17 0.873899, m_core 0.563825, share 0.922882, σ
# 0.987035); Re-Forward-Audit max_dev 0.0 (bitclean, Band 10⁻⁶);
# Dominante [W→C→S] exakt; Register σ_bio(gereift) 0.987035 > 0.5
# gegen σ_bio(Weltprosa) 0.000733 < 0.5 (LOO acc 0.966667, p_perm 0.0,
# n 30); Bio-Gewinn > 0 (p6_abl 0.119153 → p6_m 0.859480). Das
# «6»-datum 是X即非X: Emissions-Zeile am P1b-Paar C-dominant
# [0.119153, 0.740328, 0.556515] (der Zustand traegt DURCH den
# Kontext-Kanal) gegen Band-Zeile S-strikt [0, 0, 0.556515] (die
# Register-Schiene) — beide Zeilen noetig. Gattung B
# (Horizont-Weiche): Rezeptur bitgenau (r_m == texts[10], r_u ==
# texts[22]); argmax «6» 236825 gegen 236810 (ungereift); Ziffer-Masse
# SWAPPT (p5 0.009548 gegen 0.196449); Preview k = 4 alle LOOP,
# Restore bitclean; Weichen-Replik 5/5 (auto_min 5, mean_best 0.554467
# < 0.567980, mean_raw 0.680926 = P4-Anker EXAKT, wittgenstein 5/5);
# Post-Weiche-Rendern will «6» (p 0.8707). Gattung C
# (Zwillings-Divergenz): fingerprint_equal an 3 Checkpoints; A
# ('identity','x') LOOP gegen B ('negate','x') HALTED; Ketten 236825×6
# (p 0.859480 konstant) gegen 236743×6 (p 0.992976); ‖z_A‖ 127.834824,
# ‖z_B‖ 10.775427, ‖z_A − z_B‖₂ = 135.687765 (rel 0.978916);
# w_mean_abs 0.0550/0.0623. Querschnitt: adapter_digest 9f9f2d05…
# bitidentisch zum SP15-Trainings-Digest (dasselbe 24-Rendering-Korpus
# — Querschnitts-Korrelation ueber Runden); gap_digest 4e1e7387…;
# purity ueber die GANZE Batterie PASS. Kein Erlebens-Claim, 是X即非X:
# die Provenanz-Tags sind gemessene Herkunfts-Scheidung an Token-
# Koordinaten (Lens-Datum), nicht an Zustaenden. Artefakte:
# sp16_impossible_texts.json/.log (GRUEN) + sp16_cpu_pilot.log +
# sp16_erstrun_gebrochen.log (KOR-1 verbatim).
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (das SP16-Programm
# samt Maschinen-Basis und Lade-Kette inkl. MI-7- und SP15-Smokes) und
# (c) ERGEBNISSE. Nach jedem Runde-Lauf: dieses Set durch das neue
# Runden-Set ersetzen (SP16 folgt als Commit-Paar in Git-History;
# SP15: 977330d/d1d857b; MI-7: 1fbb46a/b68b686; MI-6-P4: 3cb8bf9/
# 52b1dc7; MI-6: 55ed1e9/05f63ae; MI-5: 0599d58/c1d8334; MI-3-P4+MI-4:
# 48d8d68/c860025).
FILES = [
    # (a) GEÄNDERT/NEU — der SP16-Smoke (PREREG Rev. 1 + KORREKTUR-LOG
    #     KOR-1) + Berichte (sprachspiel §26, Laufschrift §28)
    "scratches/smoke_sp16_impossible_texts.py",
    "scratches/sprachspiel_bericht.md",
    "scratches/laufprotokolle_befund_report.md",
    # (b) AUSGEFÜHRT — das SP16-Programm samt importierter Maschinen-Basis
    #     (die Götterdreiheit + BaseMachine/ArgonOracle-Laufbahn mit den
    #     SP15-Opt-ins JL-Zelle/enable_wide_read, der Composite-Pfad, der
    #     verify-Loader, der mech_interp-Checkpoint-Lader), die
    #     Vorrunder-Smokes als Import-Basis (MI-7-Gates/_p4_korpusbau/
    #     _arm_battery + SP15-Adapter/_train_wide + MI-6/P4-Rezeptur +
    #     agency-Ketten); Argon-Modellbasis (ArgonConfig/Forward +
    #     Router-Canon)
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/composite.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/verify_machines.py",
    "machines/mech_interp.py",
    "scratches/smoke_mi7_epistemic_triangulation.py",
    "scratches/smoke_sp15_jl_wide.py",
    "scratches/smoke_mi6p4_schluss.py",
    "scratches/smoke_mi6_subjektivitaet.py",
    "scratches/smoke_endogene_agency.py",
    "scratches/smoke_lexical_grounding.py",
    "scratches/smoke_paradigm_ursprung.py",
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — SP16: Re-Lauf GRÜN (Log + JSON) + CPU-Pilot-Log
    #     + der gebrochene Erstlauf verbatim (KOR-1)
    "scratches/sp16_impossible_texts.json",
    "scratches/sp16_impossible_texts.log",
    "scratches/sp16_cpu_pilot.log",
    "scratches/sp16_erstrun_gebrochen.log",
]

SEPARATOR = "=" * 78


def main() -> int:
    blocks: List[str] = []
    index_rows: List[str] = []
    missing: List[str] = []
    total_bytes = 0
    total_lines = 0
    for rel in FILES:
        path = REPO_ROOT / rel
        if not path.is_file():
            missing.append(rel)
            continue
        raw = path.read_bytes()
        total_bytes += len(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            missing.append(f"{rel} (nicht UTF-8)")
            continue
        lines = text.count("\n")
        total_lines += lines
        index_rows.append(f"  {rel}  [{len(raw):>8} B {lines:>5} Zeilen]")
        blocks.append(
            f"{SEPARATOR}\n"
            f"### FILE: {rel}\n"
            f"### bytes: {len(raw)}   lines: {lines}\n"
            f"{SEPARATOR}\n"
            f"{text}"
            f"\n"
        )

    header = (
        f"{SEPARATOR}\n"
        "repo2txt-Konkatenat: KenosisLogos/Argon — -machines-Erweiterung\n"
        "(Götterdreiheit: Gödel-/Turing-/Wittgenstein-Maschinen als Module auf Argon-1B)\n"
        "GENERATIONEN-REGIME: nur das Runden-Paket des letzten Laufs\n"
        " (SP16, Kybernetische Textgenerierung + Provenanz-Audit der\n"
        " unmöglichen Texte — 4 PASS, exit 0 GRÜN) — geändert +\n"
        " ausgeführt (SP16-Smoke + das Programm samt Maschinen-Basis,\n"
        " MI-7- und SP15-Smokes und Lade-Kette) + Ergebnisse; das\n"
        " vollständige Korpus über die Git-History (SP15: 977330d/\n"
        " d1d857b; MI-7: 1fbb46a/b68b686; MI-6-P4: 3cb8bf9/52b1dc7;\n"
        " MI-6: 55ed1e9/05f63ae; MI-5: 0599d58/c1d8334;\n"
        " MI-3-P4+MI-4: 48d8d68/c860025;\n"
        " MI-3: 97ec577/26f2e25;\n"
        " MI-2-P4: d1fc355/cb46509; MI-2-Lauf 3: 2455aea/1b4be11;\n"
        " MI-1: 67c1b5e/fc367c3;\n"
        " frühere kumulative Fassungen: 49ecc31 = 88 Dateien, 2f61402 = 84).\n"
        f"Dateien: {len(index_rows)}   bytes: {total_bytes}   Zeilen: {total_lines}\n"
        f"{SEPARATOR}\n"
        "INDEX\n"
        f"{chr(10).join(index_rows)}\n"
        f"{SEPARATOR}\n\n"
    )
    body = header + "\n".join(blocks)
    footer = (
        f"\n{SEPARATOR}\n"
        f"ENDE — sha256(des Konkatenats ohne dieser Zeile) notiert in "
        f"'scratches/repo2txt_machines_extension.sha256'\n"
        f"{SEPARATOR}\n"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = (body + footer).encode("utf-8")
    OUT_PATH.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    sha_path = OUT_PATH.with_suffix(".sha256")
    sha_path.write_text(f"{digest}\n", encoding="utf-8")

    print(f"OK: {OUT_PATH.relative_to(REPO_ROOT)} ({len(payload)} B)")
    print(f"Dateien: {len(index_rows)}, fehlend/nicht-UTF8: {len(missing)}")
    for row in missing:
        print(f"  ! {row}")
    print(f"sha256: {digest}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())