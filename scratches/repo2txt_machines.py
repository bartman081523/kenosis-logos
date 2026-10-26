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

# GENERATION MI-5 — Cross-Model-MI (gemma-3-1b-it + 3 Argon-Schwestern):
# 4 PASS | 2 FAIL | 0 OFFEN, exit 1 ROT. Substrat-Vererbung (Vokabel 9079,
# Paris-Recall, ℓ17-Attention-Träger 0.83, Kristallisation, anchor_tv 0)
# gegen Charakter-Erziehung (M_digit = machines-Exklusiv; Prosa-Schild =
# Lernprodukt, Bias-Zero-Vater beweist die Kontrafaktik); die Router-
# Spezialisierung bewegt den prose-Kanal am stärksten (TV 0.621/0.603).
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (Sonde + Smoke + Scout
# + die Argon-Modellbasis, die load_any_checkpoint funktions-Level
# zieht: modeling_argon -> moe_layers; verify_machines laeuft NICHT,
# lisp-Kanal laeuft NICHT) und (c) ERGEBNISSE. Nach jedem Runde-Lauf:
# dieses Set durch das neue Runden-Set ersetzen (MI-3-P4+MI-4-Paket liegt
# in Git-History: Commits 48d8d68/c860025; MI-3: 97ec577/26f2e25;
# MI-2-P4: d1fc355/cb46509).
FILES = [
    # (a) GEÄNDERT/NEU — Sonde mit MI-5-Zugängen (is_moe_model/
    #     load_any_checkpoint STRICT/sublayer_logits_dense), der neue
    #     Cross-Model-Smoke (6 Verdikte + Datums-Zusatz, 6 Piloten) und
    #     der Instrument-Scout (gemma-Struktur, KEINE Messung)
    "machines/mech_interp.py",
    "scratches/smoke_mech_interp_cross_models.py",
    "scratches/scout_gemma_structure.py",
    # (b) AUSGEFÜHRT — die via load_any_checkpoint funktions-Level
    #     gezogene Argon-Modellbasis (ArgonConfig/Forward + Router-Canon,
    #     Anker der Batterie-Felder)
    "modeling_argon.py",
    "moe_layers.py",
    # (c) ERGEBNISSE — MI-5: Run-Log + Ergebnis-JSON (Roh-Felder:
    #     routing_pct 6×[26×8] je Schwester, bias_std_26, ent_l11,
    #     l17/dense-Fields, Chimären-Voll-Bahnen, star-Trajektorien,
    #     6 Verdikte; MI-1..MI-4-JSONs bleiben in Git-History)
    "scratches/mi5_cross_models.log",
    "scratches/mi5_cross_models.json",
    # (c) ERGEBNISSE — Berichte des Laufs (§22, Laufschrift §24)
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
        " (MI-5) — geändert + ausgeführt (Sonde, Smoke, Scout, Argon-Basis)\n"
        " + Ergebnisse; das vollständige Korpus über die Git-History\n"
        " (MI-3-P4+MI-4: 48d8d68/c860025; MI-3: 97ec577/26f2e25;\n"
        " MI-2-P4: d1fc355/cb46509; MI-2-Lauf 3: 2455aea/1b4be11;\n"
        " MI-1: 67c1b5e/fc367c3; frühere kumulative Fassungen:\n"
        " 49ecc31 = 88 Dateien, 2f61402 = 84).\n"
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