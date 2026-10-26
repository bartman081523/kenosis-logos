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

# GENERATION MI-3-P4 + MI-4 — Argmax-Gate/Sequenz-TV/Capacity-Gate-Schild
# (5 PASS | 1 OFFEN | 3 FAIL, exit 1 ROT) + Token-Level Circuit Tracing
# (2 PASS GRÜN, exit 0): der Stummkanal ist Argmax-Ebene, die Domänen-
# Notwendigkeit sitzt tokenweise am Subspace, das Entropy-Gate ist ein
# echter, aber nicht absoluter Prosa-Schirm.
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (das vollstaendige
# Programm samt Modul-Level-importierter Maschinen-Basis: verify_machines
# importiert machine_base + 3 Maschinen + composite + build_machine_corpus
# auf Modul-Ebene; lisp_parity/lisp_bridge sind funktions-Level und laufen
# NICHT) und (c) ERGEBNISSE. Nach jedem Runde-Lauf: dieses Set durch das
# neue Runden-Set ersetzen (MI-3-Paket liegt in Git-History: Commits
# 97ec577/26f2e25; MI-2-P4: d1fc355/cb46509).
FILES = [
    # (a) GEÄNDERT/NEU — MI-3-Smoke mit P4-Erweiterung (V7-V9 Re-Reg +
    #     KORREKTUR-LOG: Lauf-1-Klassifikations-Bug + P4-CE-Gather-Bug),
    #     die Sonde mit MI-4-Zugängen (token_probs/tokenwise_routing/
    #     assemble_pct) und die neue MI-4-Circuit-Trace-Smoke
    "scratches/smoke_mech_interp_interventions.py",
    "machines/mech_interp.py",
    "scratches/smoke_mech_interp_circuit_trace.py",
    # (b) AUSGEFÜHRT — die Modul-Level-importierte Maschinen-Basis
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/composite.py",
    "machines/build_machine_corpus.py",
    # (b) AUSGEFÜHRT — GPU-Kontext (_load_machines_model) und die
    #     Argon-Modellbasis (moe_layers Router-Canon ist der Anker der
    #     P4-Verdikte)
    "machines/verify_machines.py",
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — MI-3-P4: Run-Log + Ergebnis-JSON (Roh-Felder:
    #     9 Verdikte inkl. argmax_gap/seq-TVs/l11-Gates; Baseline
    #     mi3_interventions.json bleibt in Git-History) — MI-4:
    #     pct/ent-Voll-Bahn + SEQ-TV-Profile
    "scratches/mi3_interventions_p4.log",
    "scratches/mi3_interventions_p4.json",
    "scratches/mi4_circuit_trace.log",
    "scratches/mi4_circuit_trace.json",
    # (c) ERGEBNISSE — Berichte des Laufs (§20+§21, Laufschrift §22+§23)
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
        " (MI-3-P4 + MI-4) — geändert + ausgeführt (Programm samt Basis\n"
        " und Anker) + Ergebnisse; das vollständige Korpus über die\n"
        " Git-History (MI-3: 97ec577/26f2e25; MI-2-P4: d1fc355/cb46509;\n"
        " MI-2-Lauf 3: 2455aea/1b4be11; MI-1: 67c1b5e/fc367c3; frühere\n"
        " kumulative Fassungen: 49ecc31 = 88 Dateien, 2f61402 = 84).\n"
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