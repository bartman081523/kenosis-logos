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

# GENERATION MI-6 — Algorithmische Subjektivität über die Spiegel-Linie
# (SP8/SP12/SP13 + MI-Lens; SP14 einlöst): 5 PASS | 3 FAIL | 0 OFFEN,
# exit 1 ROT. Die Kanal-Diagnose: der Composite-Gap-Kanal trägt den
# Biografie-Kanal NICHT (V3-Split 0.0 exakt bei perfekter Verdikt-Scheidung
# 5×LOOP/5×HALTED + Dekoy tot; V6-Zweig-Konstanz — Gap scheidet Checkpoints,
# nicht Biografien, Vater Rang 1/6; V1-Canon-8-Zelle acc 0.0), während die
# Biografie satt trägt im Verdikt (V3/V5), in der breiten 1152-dim Hidden
# (V1-wide acc 0.916667, p_perm 0.0) und kausal am Describe-Dezimal-Kanal
# (V4-Knockouts > 0, Restore bitrein). SP14: baugleiche Maschinen (Build
# 41a3d3d83d619788 doppelsitzig), verschiedene Biografien — dasselbe Wort
# „spiegel" bekommt LOOP vs HALTED; drift 2.990587/1.620920. Die symbolische
# Schicht verdeckt den Schaltkreis — dreifach quantifiziert.
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (die Götterdreiheit
# samt Composite, der verify-Loader, der mech_interp-Checkpoint-Lader, die
# Agent-/Lexikon-Import-Quellen und die Argon-Modellbasis) und (c)
# ERGEBNISSE. Nach jedem Runde-Lauf: dieses Set durch das neue Runden-Set
# ersetzen (MI-5-Paket liegt in Git-History: Commits 0599d58/c1d8334;
# MI-3-P4+MI-4-Paket: 48d8d68/c860025; MI-3: 97ec577/26f2e25;
# MI-2-P4: d1fc355/cb46509).
FILES = [
    # (a) GEÄNDERT/NEU — Composite mit preview_steps (MI-6-Helper, rein
    #     additiv) + der neue Spiegel-Smoke (8 Verdikte + Datums-Zusatz,
    #     6 R2-Piloten, PREREG + PREREG-ERGÄNZUNG, KORREKTUR-LOG)
    "machines/composite.py",
    "scratches/smoke_mi6_subjektivitaet.py",
    # (b) AUSGEFÜHRT — das Programm samt Maschinen-Basis und Anker:
    #     die Götterdreiheit (Goedel self_model/uni_241, Turing Verdikt,
    #     Wittgenstein Lexikon), die BaseMachine/ArgonOracle-Laufbahn,
    #     der verify-Loader (_load_machines_model) und der mech_interp
    #     -Checkpoint-Lader (load_any_checkpoint STRICT), die Agent-
    #     Import-Quelle (gpu_context/bias_fingerprint/fingerprint/
    #     make_composite/learn_and_pin) und die Lexikon-Import-Quelle
    #     (LEX_PAIR/POOL6); Argon-Modellbasis (ArgonConfig/Forward +
    #     Router-Canon, funktions-Level gezogen)
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/verify_machines.py",
    "machines/mech_interp.py",
    "scratches/smoke_endogene_agency.py",
    "scratches/smoke_lexical_grounding.py",
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — MI-6: Run-Log + Ergebnis-JSON (Roh-Felder:
    #     Fingerprints je Proben-Stufe, Gap-Matrizen self/fremd,
    #     Knockout-Divergenzen, Decoder-Scores canon-8/wide, Rangprofile,
    #     8 Verdikte; MI-1..MI-5-JSONs bleiben in Git-History)
    "scratches/mi6_subjektivitaet.log",
    "scratches/mi6_subjektivitaet.json",
    # (c) ERGEBNISSE — Berichte des Laufs (§23, Laufschrift §25)
    "scratches/sprachspiel_bericht.md",
    "scratches/laufprotokolle_befund_report.md",
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
        " (MI-6, Spiegel-Linie) — geändert + ausgeführt (Composite + Smoke\n"
        " + die Götterdreiheit samt Loader/Lens-Basis) + Ergebnisse;\n"
        " das vollständige Korpus über die Git-History\n"
        " (MI-5: 0599d58/c1d8334; MI-3-P4+MI-4: 48d8d68/c860025;\n"
        " MI-3: 97ec577/26f2e25; MI-2-P4: d1fc355/cb46509;\n"
        " MI-2-Lauf 3: 2455aea/1b4be11; MI-1: 67c1b5e/fc367c3;\n"
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