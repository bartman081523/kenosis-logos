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

# GENERATION SP15 — 1152D-Breitband-Selbstmodell (2. Runde des
# 5-Pläne-Programms): der 144:1-Block-Mittel-Flaschenhals öffnet sich
# opt-in. Maschinen-Berührungen ADDITIV (L0-Anker-Pflicht): hidden
# _state projection="orthogonal_jl" (QR Seed 42, orthonormal, je d
# gecached) + GoedelMachine.enable_wide_read (W_read 8x1152, Delta
# uni_241:220 ohne Zerfall). Run-Chronik: Erstlauf 3 PASS | 2 FAIL,
# exit 1 ROT (K1: composite.py:140 gibt dem Goedel das Trace-Outcome,
# describe()-Kanal uniform; K2: Training lernte nie auf Text-Hiddens);
# REV-2 4 PASS | 1 FAIL, exit 1 — V2 traegt (crossing [256], jl256
# 0.958333/0.0 = das wide-gepoolte-Datum EXAKT), V3 FAIL mit K3 am
# Geraet bewiesen (uni_204-encode der 24 P4-Renderings = 0 — 2-/5
# -Valenz-Akkumulation ueber die 100-Position-Zyklen mod 10**15 →
# _text_seed 0 → Ist text-unabhaengig; Canon-DATUM) und K4 bewiesen
# (tanh-Saettigung: pred-Norm √8 = 2.828427, div 0.0); REV-3 5 PASS |
# 0 FAIL, exit 0 GRUEN (LINEAR-Lesungen, Band unveraendert;
# Fold-Eingabe fuer die tanh-DATUM-Arme) mit K5 beweislich (spec-DATUM
# 0.285075: Block-Seite mass eine konsekutive Ersatzmatrix, Canon-Zelle
# ist STRIDED); **REV-4 (K5-Korrektur, Baender UNVERAENDERT): 5 PASS |
# 0 FAIL | 0 OFFEN, exit 0 GRUEN** — sens_lin_wide 0.92931 (64.5x die
# MI-6-Konstante 0.0014406), sens_lin_block 0.003159, ratio_lin
# 294.178538; tanh-DATUM re-run-stabil (ko == raw exakt je Text);
# 24 distinct raw bei Fold-Eingabe gegen 1 bei uni_204-Eingabe;
# Canon-Seeds [0]; Trainings-Digests 3x bitidentisch; L0-Exakt,
# crossing [256], purity/det PASS. Kein Erlebens-Claim, 是X即非X; REV-3
# bleibt GRUEN-mit-K5-Befund verbatim. Artefakte: sp15_wide.json/.log
# (ROT verbatim) + sp15_wide_rev2/3/4.json/.log + die CPU-Pilot-Logs
# (sp15_cpu_pilot.log/_rev2/_rev3/_rev4.log).
# Runden-Paket = (a) geaendert/neu, (b) AUSGEFUEHRT (das SP15-Programm
# samt Maschinen-Basis und Lade-Kette) und (c) ERGEBNISSE. Nach jedem
# Runde-Lauf: dieses Set durch das neue Runden-Set ersetzen (SP15
# folgt als Commit-Paar in Git-History; MI-7: 1fbb46a/b68b686;
# MI-6-P4: 3cb8bf9/52b1dc7; MI-6: 55ed1e9/05f63ae; MI-5:
# 0599d58/c1d8334; MI-3-P4+MI-4: 48d8d68/c860025).
FILES = [
    # (a) GEÄNDERT/NEU — Maschinen-Opt-ins (hidden_state-JL-Zelle +
    #     enable_wide_read), der SP15-Smoke (REV-1..REV-4-Register)
    #     + Berichte (sprachspiel §25, Laufschrift §27)
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "scratches/smoke_sp15_jl_wide.py",
    "scratches/sprachspiel_bericht.md",
    "scratches/laufprotokolle_befund_report.md",
    # (b) AUSGEFÜHRT — das SP15-Programm samt importierter Maschinen-Basis
    #     (Composite: der K1-Anker composite.py:140 steht verbatim im
    #     KORREKTUR-LOG), die Götterdreiheit, die BaseMachine/
    #     ArgonOracle-Laufbahn (steer/hidden_state), der verify-Loader,
    #     der mech_interp-Checkpoint-Lader, die Vorrunder-Smokes als
    #     Import-Basis (_l0_korpus/_loo_decoder/_bio-Maschinen); Argon-
    #     Modellbasis (ArgonConfig/Forward + Router-Canon)
    "machines/composite.py",
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
    # (c) ERGEBNISSE — SP15: die volle Run-Chronik (Erstlauf ROT + REV-2
    #     ROT + REV-3 GRÜN-mit-K5 + REV-4 GRÜN) je Log + JSON + die
    #     CPU-Pilot-Logs aller Rundungen
    "scratches/sp15_wide.json",
    "scratches/sp15_wide.log",
    "scratches/sp15_wide_rev2.json",
    "scratches/sp15_wide_rev2.log",
    "scratches/sp15_wide_rev3.json",
    "scratches/sp15_wide_rev3.log",
    "scratches/sp15_wide_rev4.json",
    "scratches/sp15_wide_rev4.log",
    "scratches/sp15_cpu_pilot.log",
    "scratches/sp15_cpu_pilot_rev2.log",
    "scratches/sp15_cpu_pilot_rev3.log",
    "scratches/sp15_cpu_pilot_rev4.log",
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
        " (SP15, 1152D-Breitband-Selbstmodell REV-4) — geändert +\n"
        " ausgeführt (SP15-Smoke + das Programm samt Maschinen-Basis und\n"
        " Lade-Kette) + Ergebnisse; das vollständige Korpus über die\n"
        " Git-History (MI-7: 1fbb46a/b68b686; MI-6-P4: 3cb8bf9/52b1dc7;\n"
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