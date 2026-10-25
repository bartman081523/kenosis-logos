"""
repo2txt_machines.py — baut einen Ein-Datei-Konkatenat aller relevanten Dateien
der -machines-Erweiterung (Götterdreiheit: Gödel/Turing/Wittgenstein) für den
Transfer in andere Kontexte (repo2txt-Stil, ein Datei-Header pro Block).

Ausgabe: scratches/repo2txt_machines_extension.txt
Run:     python3 scratches/repo2txt_machines.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "scratches" / "repo2txt_machines_extension.txt"

# Reihenfolge bewusst: Ports -> Lisp-Schicht (Spiegel + Canon) -> Daten ->
# Presets/Meta -> Trainings-/Verifikations-Glue -> Doku -> Protokolle.
FILES: List[str] = [
    # -- Maschinenports (Python) ------------------------------------------------
    "machines/__init__.py",
    "machines/machine_base.py",
    "machines/goedel_machine.py",
    "machines/turing_machine.py",
    "machines/wittgenstein_machine.py",
    "machines/composite.py",
    # -- Lisp-Spiegel + SBCL-Brücke ----------------------------------------------
    "machines/lisp_bridge.py",
    "machines/lisp_parity.py",
    "machines/lisp/README.md",
    "machines/lisp/bridge-protocol.lisp",
    "machines/lisp/run-bridge.lisp",
    "machines/lisp/parity/lisp_fixtures.json",
    # -- Vendor-Canon (Lisp, unverändert, MIT) -----------------------------------
    "machines/lisp/vendor/README.md",
    "machines/lisp/vendor/lisp-cognito-philosopher.asd",
    "machines/lisp/vendor/package.lisp",
    "machines/lisp/vendor/utils.lisp",
    "machines/lisp/vendor/goedel-encoder.lisp",
    "machines/lisp/vendor/geometric-tokenizer.lisp",
    "machines/lisp/vendor/symbolic-operator.lisp",
    "machines/lisp/vendor/cognito-construct.lisp",
    "machines/lisp/vendor/philosopher.lisp",
    "machines/lisp/vendor/reflection-engine.lisp",
    "machines/lisp/vendor/dialectic-macros.lisp",
    "machines/lisp/vendor/living-logos.lisp",
    "machines/lisp/vendor/language-learning.lisp",
    "machines/lisp/vendor/letter-speech.lisp",
    "machines/lisp/vendor/persistence.lisp",
    "machines/lisp/vendor/interactive-repl.lisp",
    "machines/lisp/vendor/run-demo.lisp",
    "machines/lisp/vendor/demo.lisp",
    "machines/lisp/vendor/simple-test.lisp",
    "machines/lisp/vendor/test-harness.lisp",
    "machines/lisp/vendor/test-load.lisp",
    "machines/lisp/vendor/TEST_RESULTS.md",
    "machines/lisp/vendor/determinism_log.txt",
    "machines/lisp/vendor/stability_signature.txt",
    "machines/lisp/vendor/tests/run-tests.lisp",
    "machines/lisp/vendor/tests/test-goedel-encoder.lisp",
    "machines/lisp/vendor/tests/test-geometric-tokenizer.lisp",
    "machines/lisp/vendor/tests/test-symbolic-operator.lisp",
    "machines/lisp/vendor/tests/test-philosopher.lisp",
    "machines/lisp/vendor/tests/test-dialectic-macros.lisp",
    # -- Daten (Korpus, Goldwerte, Registry) --------------------------------------
    "machines/corpus/machine_corpus.jsonl",
    "machines/parity/python_fixtures.json",
    "machine_registry.json",
    # -- Presets + Save-Meta (argon_1b_machines, sonst gitignored) ----------------
    "argon_1b_machines/machine_presets/goedel.json",
    "argon_1b_machines/machine_presets/turing.json",
    "argon_1b_machines/machine_presets/wittgenstein.json",
    "argon_1b_machines/machine_presets/combined.json",
    "argon_1b_machines/router_diff_meta.json",
    # -- Trainings- + Verifikations-Glue ------------------------------------------
    "train_moe_router_specialization.py",
    "machines/verify_machines.py",
    # -- Doku ---------------------------------------------------------------------
    "README.md",
    ".gitignore",
    # -- Protokolle + Smoke-Skripte ------------------------------------------------
    "scratches/smoke_machines_train.py",
    "scratches/argon_1b_machines_train.log",
    "scratches/build_corpus_re_sync.log",
    "scratches/verify_machines_stage5.log",
    "scratches/verify_machines_full.log",
    # -- Composite-Experiment-Zyklus (§7-§9, Identity-Hebel + Basisleiter) ----------
    "scratches/smoke_composite_oracle.py",
    "scratches/composite_oracle_experiment.log",
    "scratches/composite_oracle_experiment.json",
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
        "(Götterdreiheit: Gödel-/Turing-/Wittgenstein-Maschinen als Module auf Argon-1B;\n"
        " inkl. Lisp-Vendor-Canon, Brücke, Korpus, Goldwerte, Presets, Protokolle)\n"
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