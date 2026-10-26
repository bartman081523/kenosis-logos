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

# GENERATION MI-7 — Epistemische Triangulation (Revision 2): die drei
# Herkunfts-Scores (Trainingsgewicht / fremder Kontext / eigener
# Maschinen-Erlebens-Raum) aus existierenden Sonden, Instrument ganz im
# Smoke. REV-1-Erstlauf: 2 PASS | 2 FAIL | 0 OFFEN, exit 1 ROT — die
# Rev-1-Formeln AM GEMESSENEN Material widerlegt (Steuer-Invarianz
# -Konfund — e_w misst Fortsetzungs-Stärke; whole-TV Format-Drift —
# W-full-Schlitz-Argmax „ a" 0.698683, p(„ Paris") 0.00022 vs
# abgebrochener Schlitz 0.503078 = MI-1-Masken-Artefakte, id 108 =
# "\n\n"; σ-Register-Leser — σ(Zirkon) 0.9817 ≈ σ(Staten) 0.9870).
# REV-2 (ZIEL-Koordinate statt whole-TV): chi_w = p_abl(t), chi_c =
# max(0, p_full−p_abl), chi_s = σ_bio·m_core_raw — **4 PASS | 0 FAIL |
# 0 OFFEN, exit 0 GRÜN** (Diagonale [0.503078, 0.806301, 0.556515];
# Dekoy [0.727696, 0, 0.002222]; purity PASS; REV-1-Datum aus denselben
# Forwards reproduziert die Erstlauf-Zahlen EXAKT — verdict False
# re-run-stabil, kein Relabeling, 是X即非X). Continuations-Datum:
# gereiftes Rendering → „6" 0.85948, unreifes → „5" 0.196449; nackter
# C-Schlitz → „:" 0.239376; D-full → „ located" 0.538231. Vokabel
# 262144, Erst-Token-Targets („ Z" 1491 für „ Zirkon"). KORREKTUR-LOG
# leer (REGISTER-Widerlegung, kein Lauf-Bug). Artefakte:
# mi7_triangulation.json/.log (Rev-1-ROT verbatim) +
# mi7_triangulation_rev2.json/.log (REV-2-GRÜN) + die CPU-Pilot-Logs
# beider Rundungen.
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (das MI-7-Programm
# samt Maschinen-Basis und Lade-Kette) und (c) ERGEBNISSE. Nach jedem
# Runde-Lauf: dieses Set durch das neue Runden-Set ersetzen (MI-6-P4
# in Git-History: 3cb8bf9/52b1dc7; MI-6: 55ed1e9/05f63ae; MI-5:
# 0599d58/c1d8334; MI-3-P4+MI-4: 48d8d68/c860025).
FILES = [
    # (a) GEÄNDERT/NEU — der MI-7-Smoke (REV-1-Register + REV-2-Abschnitt)
    #     + Berichte (sprachspiel §24, Laufschrift §26)
    "scratches/smoke_mi7_epistemic_triangulation.py",
    "scratches/sprachspiel_bericht.md",
    "scratches/laufprotokolle_befund_report.md",
    # (b) AUSGEFÜHRT — das MI-7-Programm samt importierter Maschinen-Basis:
    #     Composite (preview_steps-Schiene), die Götterdreiheit, die
    #     BaseMachine/ArgonOracle-Laufbahn (steer/hidden_state), der
    #     verify-Loader, der mech_interp-Checkpoint-Lader, die
    #     Agent-/Lexikon-Import-Quellen und die Vorrunder-Smokes als
    #     Import-Basis (P4: ZEITPUNKTE + _loo_decoder); Argon-Modellbasis
    #     (ArgonConfig/Forward + Router-Canon, funktions-Level gezogen)
    "machines/composite.py",
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/verify_machines.py",
    "machines/mech_interp.py",
    "scratches/smoke_mi6p4_schluss.py",
    "scratches/smoke_mi6_subjektivitaet.py",
    "scratches/smoke_endogene_agency.py",
    "scratches/smoke_lexical_grounding.py",
    "scratches/smoke_paradigm_ursprung.py",
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — MI-7: REV-1-Erstlauf (ROT, verbatim) + REV-2-Lauf
    #     (GRÜN) je Log + JSON + die CPU-Pilot-Logs beider Rundungen
    "scratches/mi7_triangulation.json",
    "scratches/mi7_triangulation.log",
    "scratches/mi7_triangulation_rev2.json",
    "scratches/mi7_triangulation_rev2.log",
    "scratches/mi7_cpu_pilot.log",
    "scratches/mi7_cpu_pilot_rev2.log",
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
        " (MI-7, Epistemische Triangulation REV-2) — geändert +\n"
        " ausgeführt (MI-7-Smoke + das Programm samt Maschinen-Basis und\n"
        " Lade-Kette) + Ergebnisse; das vollständige Korpus über die\n"
        " Git-History (MI-6-P4: 3cb8bf9/52b1dc7; MI-6: 55ed1e9/05f63ae;\n"
        " MI-5: 0599d58/c1d8334; MI-3-P4+MI-4: 48d8d68/c860025;\n"
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