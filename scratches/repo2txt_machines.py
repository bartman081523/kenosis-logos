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

# GENERATION MI-6-P4 — der Subjekt-Schluss (Revision 3): die drei
# Kanal-Flaschenhälse aus MI-6 (5 PASS | 3 FAIL | 0 OFFEN, exit 1 ROT —
# s. Git-History 55ed1e9/05f63ae) als NEUE, vorab registrierte Bänder
# einlösen: **5 PASS | 0 FAIL | 0 OFFEN, exit 0 GRÜN.** P1a
# mi6p4_schwelle_weit — wide-1152 gepoolt acc 0.958333, p_perm 0.0 auf 24
# ECHTEN Zeitpunkten (canon-8 blind 0.0/1.0 — die 144er-Blockmittelung
# machine_base.py:362 am Code verifiziert; je-Stil textur 1.0, numerisch
# 0.833333/p 0.025); P1b mi6p4_selbstfremd_gereift — der Gap-Kanal schließt
# durch Reifung: gereifte Transplant split 0.130883 (gap 0.548455 vs
# 0.679338) bei ungereifter V3-Kontrolle 0.0 exakt (MI-6-Konstante
# 0.47851), Verdikt-Paare 5× LOOP/HALTED; Wort-Formular: „spiegel" →
# identity-Formel/LOOP, „stille" → negate-Formel/HALTED (identitäts- vs
# negations-förmig); CPU gereift 2.937263 vs ungereift 0.0. P1c
# mi6p4_oracle_sonder — machines Rang 4/4 in 6/6 Zweigen (Band ≥5/6;
# KORREKTUR-LOG IV: die perm_null-Annotation „≤9e-4" war arithmetisch
# falsch, wahr 19/4096 ≈ 4.64e-3; V6-Verdikt bleibt FAIL).
# P1d mi6p4_dissonanz — der Dissonanz-Hauptsatz: 5/5 best_gap < raw_gap,
# mean_best 0.554467 < Vater-Niveau 0.567980, Preset wittgenstein 5/5
# (mean_raw 0.680926). KORREKTUR-LOG ×4 (Erstrun-Import-Lücke ohne JSON;
# Re-Lauf-1 P1b-Arm-Defekt — beide ohne learn_pair_pin gereift, Teil-JSON
# nicht verwertet; extern gestoppte Start-Instanz; Arithmetik). MI-6-Bands
# V1/V3/V6 stehen UNVERÄNDERT als FAIL — kein Relabeling (是X即非X).
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (das MI-6-Programm samt
# Maschinen-Basis und Lade-Kette) und (c) ERGEBNISSE. Nach jedem Runde-Lauf:
# dieses Set durch das neue Runden-Set ersetzen (MI-6-Paket in Git-History:
# 55ed1e9/05f63ae; MI-5: 0599d58/c1d8334; MI-3-P4+MI-4: 48d8d68/c860025).
FILES = [
    # (a) GEÄNDERT/NEU — der P4-Schluss-Smoke (4 neue PREREG-Bänder +
    #     Determinismus, KORREKTUR-LOG ×4) + Bericht §23.9–23.12,
    #     Laufschrift §25-P4
    "scratches/smoke_mi6p4_schluss.py",
    "scratches/sprachspiel_bericht.md",
    "scratches/laufprotokolle_befund_report.md",
    # (b) AUSGEFÜHRT — das MI-6-Programm samt importierter Maschinen-Basis:
    #     Composite (preview_steps/step_autonomous-Schienen), die
    #     Götterdreiheit, die BaseMachine/ArgonOracle-Laufbahn (hidden_state
    #     -Reduktion verifiziert), der verify-Loader, der mech_interp
    #     -Checkpoint-Lader, die Agent-/Lexikon-Import-Quellen und der
    #     MI-6-Smoke als Import-Basis der P4-Lanes; Argon-Modellbasis
    #     (ArgonConfig/Forward + Router-Canon, funktions-Level gezogen)
    "machines/composite.py",
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/verify_machines.py",
    "machines/mech_interp.py",
    "scratches/smoke_mi6_subjektivitaet.py",
    "scratches/smoke_endogene_agency.py",
    "scratches/smoke_lexical_grounding.py",
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — P4: Re-Lauf-2-Log + Ergebnis-JSON (4 Verdikte +
    #     Determinismus; alle Repliken/splits/decoder-Scores/Ränge) +
    #     die beiden GEBROCHENEN Vorläufer-Logs (Erstrun-Import-Lücke,
    #     Re-Lauf-1-P1b-Arm-Defekt) + CPU-Pilot-Log als KORREKTUR-Belege;
    #     MI-6-Roh (mi6_subjektivitaet.*) bleibt in Git-History
    "scratches/mi6p4_schluss.log",
    "scratches/mi6p4_schluss.json",
    "scratches/mi6p4_erstrun_gebrochen.log",
    "scratches/mi6p4_relauf1_p1b_defekt.log",
    "scratches/mi6p4_cpu_pilot.log",
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
        " (MI-6-P4, Subjekt-Schluss) — geändert + ausgeführt (P4-Smoke +\n"
        " das MI-6-Programm samt Maschinen-Basis und Lade-Kette) +\n"
        " Ergebnisse; das vollständige Korpus über die Git-History\n"
        " (MI-6: 55ed1e9/05f63ae; MI-5: 0599d58/c1d8334;\n"
        " MI-3-P4+MI-4: 48d8d68/c860025; MI-3: 97ec577/26f2e25;\n"
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