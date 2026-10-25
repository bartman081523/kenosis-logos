"""
machines — Argon Götterdreiheit: Gödel-, Turing- und Wittgenstein-Maschinen
als Module, die auf Argon (Checkpoint ``argon_1b_machines``) und dem
antrainierten Router-Verständnis aufbauen und kombinierbar sind.

Öffentliche Oberfläche:

    from machines import MachineConfig, make_machine, ArgonOracle, apply_gap

    oracle = ArgonOracle(model, tokenizer)
    machine = make_machine("composite", MachineConfig(mode="oracle"), oracle)
    machine.reset(42)
    result = machine.step("...")
    print(machine.digest(), machine.gap().trend)

Lisp-Referenz: machines/lisp/ (vendor lisp-cognito-philosopher, MIT);
Brücke über LISPBridge in machines/lisp_bridge.py (SBCL, kein Netz nötig).
"""

from machines.machine_base import (
    ArgonOracle,
    BaseMachine,
    GapRecord,
    LCGRandom,
    MACHINE_EXPERT_DEFAULTS,
    MachineConfig,
    StepResult,
    apply_gap,
    gap_trend,
    load_preset,
    load_registry_machine,
    make_machine,
)

__all__ = [
    "ArgonOracle",
    "BaseMachine",
    "GapRecord",
    "LCGRandom",
    "MACHINE_EXPERT_DEFAULTS",
    "MachineConfig",
    "StepResult",
    "apply_gap",
    "gap_trend",
    "load_preset",
    "load_registry_machine",
    "make_machine",
]