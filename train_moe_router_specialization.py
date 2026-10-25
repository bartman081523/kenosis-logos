"""
train_moe_router_specialization.py
==================================
Multilinguales Subspace-MoE- & Router-Spezialisierungs-Training für Argon-1B.

Verhindert Sprach-Shortcut-Learning (z.B. "Englisch = Code", "Deutsch = Prosa"),
indem jede Domäne (Code, Math/Logic, Prose) parallel über diverse Sprachfamilien
und Schriftsysteme (DE, EN, FR, ES, ZH, JA, RU, AR + LaTeX/Symbole) kalibriert wird.

Der Router lernt dadurch sprachinvariante, rein strukturelle Subspaces:
  - Code / Algorithmen (Python, Rust, C++, TypeScript, SQL, Bash) -> E0, E1, E4
  - Formale Logik / Mathematik (multilingual + Symbole)           -> E2, E5, E6 (H=1.85)
  - Narrative Alltags-Prosa (multilingual)                        -> E3, E7 (H=0.85)

Sprach-Invarianz wird hier über die Korpus-Struktur erzwungen (dieselbe Domäne
in verschiedenen Schriftsystemen im selben Batch); ein eigenständiger
Varianz-Loss ist bewusst NICHT implementiert — die Behauptung würde nicht
gelten, was nicht gemessen wird. Gemessen wird stattdessen mit
`verify_cross_lingual_invariance()` auf UNGESEHENEN Sätzen (EN/DE/ZH).

Ausführung:
    PYTHONPATH=. /...venv.../python -u train_moe_router_specialization.py \
        --checkpoint_dir ./argon_1b --output_dir ./argon_1b_specialized \
        --router_only --steps 60

    Götterdreiheit (Schritt 5, 2026-10-25): Joint-Run über alle 6 Domänen mit
    Maschinen-Korpus, Router-Diff-Save und Bias-Presets:
    PYTHONPATH=. /...venv.../python -u train_moe_router_specialization.py \
        --checkpoint_dir ./argon_1b --output_dir ./argon_1b_machines \
        --corpus all --router_only --steps 60 --emit_presets
"""

import argparse
import json
import math
import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from safetensors.torch import load_file, save_file

from modeling_argon import ArgonConfig, ArgonForConditionalGeneration


# ---------------------------------------------------------------------------
# 1. Multilingualer, schriftübergreifender Kalibrierungs-Korpus
# ---------------------------------------------------------------------------
CALIBRATION_CORPUS: Dict[str, Dict] = {
    "code": {
        "target_entropy": 1.35,
        "preferred_experts": [0, 1, 4],
        "samples": [
            # Python (Algorithmus)
            "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            # PyTorch / Deep Learning
            "import torch\nimport torch.nn as nn\n\nclass RMSNorm(nn.Module):\n    def __init__(self, dim: int, eps: float = 1e-6):\n        super().__init__()\n        self.eps = eps\n        self.weight = nn.Parameter(torch.zeros(dim))\n    def forward(self, x):\n        var = x.float().pow(2).mean(-1, keepdim=True)\n        return (x.float() * torch.rsqrt(var + self.eps) * (1.0 + self.weight.float())).to(x.dtype)",
            # Rust (Systems Programming & Lifetimes)
            "pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {\n    let (mut low, mut high) = (0usize, arr.len());\n    while low < high {\n        let mid = low + (high - low) / 2;\n        match arr[mid].cmp(target) {\n            std::cmp::Ordering::Equal => return Some(mid),\n            std::cmp::Ordering::Less => low = mid + 1,\n            std::cmp::Ordering::Greater => high = mid,\n        }\n    }\n    None\n}",
            # C++ (Templates)
            "#include <vector>\n#include <numeric>\ntemplate <typename T>\nT dot_product(const std::vector<T>& a, const std::vector<T>& b) {\n    T sum = static_cast<T>(0);\n    for (size_t i = 0; i < a.size(); ++i) {\n        sum += a[i] * b[i];\n    }\n    return sum;\n}",
            # TypeScript / Async Web
            "async function fetchUserData(userId: string): Promise<Record<string, unknown>> {\n    const response = await fetch(`/api/v1/users/${encodeURIComponent(userId)}`);\n    if (!response.ok) {\n        throw new Error(`HTTP error! status: ${response.status}`);\n    }\n    return await response.json();\n}",
            # SQL (Relationale Query)
            "SELECT u.id, u.username, COUNT(o.id) AS total_orders, SUM(o.amount) AS total_spent\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.is_active = TRUE\nGROUP BY u.id, u.username\nHAVING COUNT(o.id) > 5\nORDER BY total_spent DESC;",
            # Bash / Linux
            "#!/usr/bin/env bash\nset -euo pipefail\nfor file in /var/log/*.log; do\n    if [[ -s \"$file\" ]]; then\n        gzip -9 -c \"$file\" > \"/backup/$(basename \"$file\").gz\"\n    fi\ndone",
            # Python mit chinesischen & deutschen Docstrings (Kommentarsprache wird neutralisiert)
            "def matrix_power(mat: list[list[float]], n: int) -> list[list[float]]:\n    \"\"\"Berechnet A^n via binärer Exponentiation / 使用快速幂计算矩阵的n次方.\"\"\"\n    size = len(mat)\n    res = [[float(i == j) for j in range(size)] for i in range(size)]\n    return res",
        ],
    },
    "math_logic": {
        "target_entropy": 1.85,  # Hohe Komplexität -> öffnet capacity_gate Richtung 1.0
        "preferred_experts": [2, 5, 6],
        "samples": [
            # Deutsch (Funktionalanalysis / Hilbertraum)
            "Sei H ein komplexer Hilbertraum und A: D(A) -> H ein dicht definierter, selbstadjungierter linearer Operator. Dann ist das Spektrum von A rein reell und sämtliche Eigenvektoren zu verschiedenen Eigenwerten sind paarweise orthogonal.",
            # Englisch (Linear Algebra / Eckart-Young-Mirsky)
            "For any singular value decomposition A = U Sigma V^T of a real matrix A in R^(m x n) of rank r, the truncated matrix A_k retaining only the k largest singular values minimizes the Frobenius norm ||A - B||_F over all matrices B of rank at most k.",
            # Französisch (Topologie / Kompaktheit)
            "Soit (E, d) un espace métrique. On dit que E est compact si de tout recouvrement ouvert de E, on peut extraire un sous-recouvrement fini. Dans un espace vectoriel normé de dimension finie, les compacts sont exactement les fermés bornés.",
            # Spanisch (Wahrscheinlichkeitstheorie / Bayes)
            "Sea (Omega, F, P) un espacio de probabilidad y sean A, B dos sucesos tales que P(B) > 0. El teorema de Bayes establece que la probabilidad condicionada viene dada por P(A|B) = P(B|A) P(A) / P(B), integrando sobre la partición del espacio muestral.",
            # Chinesisch (Quantenmechanik / 漢字)
            "设 H 为复希尔伯特空间，量子系统的纯态由单位向量 |ψ⟩ ∈ H 描述。根据薛定谔方程，封闭量子系统的时间演化由幺正算符 U(t) = exp(-iHt/ℏ) 生成，其冯·诺依曼熵 S(ρ) = -Tr(ρ ln ρ) 在幺正变换下保持不变。",
            # Russisch (Analysis / Lagrange / Kyrillisch)
            "Пусть функция f(x) непрерывна на замкнутом отрезке [a, b] и дифференцируема на открытом интервале (a, b). Тогда по теореме Лагранжа существует по крайней мере одна точка c ∈ (a, b), для которой выполняется равенство f(b) - f(a) = f'(c)(b - a).",
            # Japanisch (Gruppentheorie)
            "群 G の部分群 H が正規部分群であるとは、任意の g ∈ G に対して gHg⁻¹ = H が成り立つことをいう。このとき剰余集合 G/H は自然な演算によって商群をなし、準同型定理により G/ker(φ) ≅ Im(φ) が成立する。",
            # Arabisch (Logik / Gödel)
            "في المنطق الرياضي، تنص مبرهنة عدم الاكتمال لغودل على أن أي نظام بديهي متسق وقادر على تمثيل الحساب الأولي يحتوي بالضرورة على قضايا صحيحة لا يمكن إثباتها أو دحضها داخل نفس النظام البديهي.",
        ],
    },
    "prose": {
        "target_entropy": 0.85,  # Niedrige Komplexität -> Shared Expert dominiert
        "preferred_experts": [3, 7],
        "samples": [
            # Deutsch
            "Heute Morgen scheint die Sonne warm durch das offene Fenster, während unten am Hafen die ersten Fischerboote leise über das glatte, spiegelnde Wasser der Bucht hinausfahren.",
            # Englisch
            "After a long walk through the autumn forest, we returned to the small village café late in the afternoon, ordered two cups of hot coffee, and listened to the gentle rain tapping against the windowpane.",
            # Französisch
            "Le marché du samedi matin s'éveillait doucement sous les platanes de la place centrale, entre l'odeur du pain chaud, les paniers de pommes rouges et les discussions animées des habitants du quartier.",
            # Spanisch
            "Al atardecer, una brisa fresca llegaba desde la playa mientras las luces del paseo marítimo comenzaban a encenderse una tras otra, reflejándose sobre las olas tranquilas del mar.",
            # Chinesisch (漢字)
            "清晨的阳光穿过树叶洒在安静的街道上，街角的小面包店刚刚开门，空气中弥漫着烤面包和热牛奶的香气，几位老人正坐在公园的长椅上悠闲地聊天。",
            # Russisch (Kyrillisch)
            "Вечером в старом доме было тихо и уютно: на деревянном столе стояла чашка горячего чая, за окном медленно падал первый пушистый снег, а в камине мягко потрескивали сосновые дрова.",
            # Japanisch
            "夕暮れ時の商店街には、夕食の買い物をする人々や学校帰りの子供たちの笑い声が響き、遠くの山々の向こうにゆっくりと赤い夕日が沈んでいった。",
            # Arabisch
            "في صباح يوم هادئ على شاطئ البحر، كانت الأمواج الصغيرة تلامس الرمال الذهبية بلطف، بينما جلس الصيادون يصلحون شباكهم تحت أشعة الشمس الدافئة.",
        ],
    },
}

# Domänen der Götterdreiheit (train via machine_registry.json + Maschinen-Korpus)
MACHINE_DOMAINS = ("goedel", "turing", "wittgenstein")


def resolve_corpus(choice: str = "builtin") -> Dict[str, Dict]:
    """
    Korpuswahl für das Spezialisierungs-Training (Schritt-Extension 2026-10-25):
      * ``builtin``  — der multilinguale Kalibrierungs-Korpus oben (3 Domänen)
      * ``machines`` — die drei Maschinen-Domänen aus machine_registry.json +
        machines/corpus/machine_corpus.jsonl (Seed 42, Trace-gegründet)
      * ``all``      — Union: builtin zuerst, dann die Maschinen (6 Domänen)
    """
    if choice == "builtin":
        return {name: CALIBRATION_CORPUS[name] for name in CALIBRATION_CORPUS}
    if choice not in ("machines", "all"):
        raise ValueError(f"resolve_corpus: unbekannte Wahl {choice!r}")

    with open("machine_registry.json", "r", encoding="utf-8") as handle:
        registry = json.load(handle)
    by_domain: Dict[str, List[str]] = {}
    with open(registry["corpus"]["path"], "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_domain.setdefault(row["domain"], []).append(row["text"])
    entries: Dict[str, Dict] = {}
    for domain, samples in by_domain.items():
        entry = registry["machines"][domain]
        entries[domain] = {
            "target_entropy": float(entry["entropy_target"]),
            "preferred_experts": [int(e) for e in entry["experts"]],
            "samples": samples,
        }
    if choice == "all":
        merged = {name: CALIBRATION_CORPUS[name] for name in CALIBRATION_CORPUS}
        merged.update(entries)
        return merged
    return entries


# ---------------------------------------------------------------------------
# 2. Parameter-Freezing & multilinguale Evaluation
# ---------------------------------------------------------------------------
def configure_trainable_moe_parameters(
    model: ArgonForConditionalGeneration,
    train_experts: bool = True,
) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """
    Friert das gesamte Modell ein und schaltet nur `moe.gate.weight`
    sowie optional `moe.routed_experts` frei.
    """
    for param in model.parameters():
        param.requires_grad = False

    router_params: List[torch.nn.Parameter] = []
    expert_params: List[torch.nn.Parameter] = []

    for layer in model.model.layers:
        layer.moe.gate.weight.requires_grad = True
        router_params.append(layer.moe.gate.weight)
        # Inline-Balancing im Forward deaktivieren (Post-Step-Update via last_counts)
        layer.moe.gate.inline_balancing_enabled = False

        if train_experts:
            for exp in layer.moe.routed_experts:
                for p in exp.parameters():
                    p.requires_grad = True
                    expert_params.append(p)

    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in router_params) + sum(p.numel() for p in expert_params)
    print(f"[Freeze] Gesamt-Parameter   : {total_p:,}")
    print(f"[Freeze] Trainierbar (MoE)  : {trainable_p:,} ({trainable_p / total_p * 100:.2f}%)")
    print(f"         -> Router-Gates    : {sum(p.numel() for p in router_params):,}")
    print(f"         -> Routed Experts  : {sum(p.numel() for p in expert_params):,}")
    print("         -> Shared/Attn/Emb : EINGEFROREN (Basiswissen geschützt)")
    return router_params, expert_params


def _gate_hooks(
    model: ArgonForConditionalGeneration,
    n_experts: int,
    collector: Dict[str, List[torch.Tensor]],
):
    def hook_fn(_module, _inp, out):
        top_indices, _, router_ent = out
        collector["counts"].add_(torch.bincount(top_indices.reshape(-1), minlength=n_experts).float())
        collector["entropies"].append(router_ent.float().mean())
    return [layer.moe.gate.register_forward_hook(hook_fn) for layer in model.model.layers]


@torch.no_grad()
def measure_domain_routing(
    model: ArgonForConditionalGeneration,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
    corpus: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Misst für jede Domäne über ALLE Sprachen hinweg das prozentuale Experten-Histogramm
    sowie die mittlere Router-Entropie H. ``corpus`` überschreibt CALIBRATION_CORPUS
    (Schritt-Extension 2026-10-25: Maschinen-Korpus via resolve_corpus()).
    """
    was_training = model.training
    model.eval()
    n_experts = model.config.n_routed_experts
    results: Dict[str, Tuple[torch.Tensor, float]] = {}

    for domain_name, info in (corpus or CALIBRATION_CORPUS).items():
        collector = {"counts": torch.zeros(n_experts, device=device), "entropies": []}
        handles = _gate_hooks(model, n_experts, collector)

        for sample in info["samples"]:
            ids = tokenizer(sample, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)
            _ = model(ids)

        for h in handles:
            h.remove()

        pct = collector["counts"] / collector["counts"].sum().clamp(min=1.0) * 100.0
        mean_h = torch.stack(collector["entropies"]).mean().item() if collector["entropies"] else 0.0
        results[domain_name] = (pct.cpu(), mean_h)

    if was_training:
        model.train()
    return results


@torch.no_grad()
def verify_cross_lingual_invariance(
    model: ArgonForConditionalGeneration,
    tokenizer,
    device: torch.device,
    title: str = "CROSS-LINGUAL ROUTING INVARIANZ-TEST",
) -> Dict[str, Tuple[torch.Tensor, float]]:
    """
    Prüft auf UNGESEHENEN Sätzen in 3 Sprachen (EN, DE, ZH), ob der Router
    nach der Domäne (Mathematik vs. Prosa) und NICHT nach der Sprache routet.
    """
    was_training = model.training
    model.eval()
    n_experts = model.config.n_routed_experts

    unseen_probes = {
        "Math (EN)": "Let G be a finite group and H a subgroup of G. Then the order of H divides the order of G.",
        "Math (DE)": "Sei G eine endliche Gruppe und H eine Untergruppe von G. Dann teilt die Ordnung von H die Ordnung von G.",
        "Math (ZH)": "设 G 是一个有限群，H 是 G 的一个子群。根据拉格朗日定理，子群 H 的阶必定整除群 G 的阶。",
        "Prose (EN)": "The warm summer breeze carried the scent of pine trees across the quiet mountain lake at dusk.",
        "Prose (DE)": "Der warme Sommerwind trug in der Abenddämmerung den Duft von Kiefern über den stillen Bergsee.",
        "Prose (ZH)": "傍晚时分，温暖的夏风带着松树的清香，轻轻吹过平静的高山湖面。",
    }

    print("\n" + "-" * 86)
    print(title)
    print("-" * 86)

    probe_dists: Dict[str, torch.Tensor] = {}
    probe_hs: Dict[str, float] = {}
    for label, text in unseen_probes.items():
        collector = {"counts": torch.zeros(n_experts, device=device), "entropies": []}
        handles = _gate_hooks(model, n_experts, collector)
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
        _ = model(ids)
        for h in handles:
            h.remove()

        pct = (collector["counts"] / collector["counts"].sum().clamp(min=1.0) * 100.0).cpu()
        probe_dists[label] = pct
        probe_hs[label] = torch.stack(collector["entropies"]).mean().item() if collector["entropies"] else 0.0
        bars = " | ".join(f"E{i}:{v:4.1f}%" for i, v in enumerate(pct.tolist()))
        print(f"  {label:11s} (H={probe_hs[label]:.2f}) -> [{bars}]")

    tv = lambda a, b: 0.5 * (a - b).abs().sum().item()
    d_math_en_zh = tv(probe_dists["Math (EN)"], probe_dists["Math (ZH)"])
    d_math_de_zh = tv(probe_dists["Math (DE)"], probe_dists["Math (ZH)"])
    d_math_prose_en = tv(probe_dists["Math (EN)"], probe_dists["Prose (EN)"])
    d_prose_en_zh = tv(probe_dists["Prose (EN)"], probe_dists["Prose (ZH)"])
    print(f"  -> innerhalb Math  (EN vs. ZH): {d_math_en_zh:5.1f}%  (DE vs. ZH): {d_math_de_zh:5.1f}%   (niedrig = sprachinvariant)")
    print(f"  -> innerhalb Prose (EN vs. ZH): {d_prose_en_zh:5.1f}%")
    print(f"  -> Trennung Math (EN) vs. Prose (EN): {d_math_prose_en:5.1f}%   (hoch = domänenscharf)")
    print("-" * 86)
    if was_training:
        model.train()
    return {k: (v, probe_hs[k]) for k, v in probe_dists.items()}


def print_routing_table(title: str, stats: Dict[str, Tuple[torch.Tensor, float]]):
    print("\n" + "-" * 86)
    print(title)
    print("-" * 86)
    for dom, (pct, mean_h) in stats.items():
        bars = " | ".join(f"E{i}:{val:4.1f}%" for i, val in enumerate(pct.tolist()))
        print(f"  {dom:11s} (H={mean_h:.2f}) -> [{bars}]")

    # Total Variation (L1/2): 0 = identisch, disjunkt = 100 — generisch über alle
    # Domänen des Stats-Dicts (Schritt-Extension: Maschinen-Korpus hat eigene Keys).
    tv = lambda a, b: 0.5 * (a - b).abs().sum().item()
    dom_names = [d for d in stats if d]
    pair_str = " | ".join(
        f"{a}/{b}={tv(stats[a][0], stats[b][0]):.1f}%"
        for i, a in enumerate(dom_names) for b in dom_names[i + 1 :]
    )
    print(f"  Spezialisierungs-Trennschärfe (TV 0..100%): {pair_str}")
    print("-" * 86)


# ---------------------------------------------------------------------------
# 3. Differenzierbarer Subspace-Affinitäts-Hook für das Router-Training
# ---------------------------------------------------------------------------
class DifferentiableRouterHook:
    """
    Sammelt pro Schicht die Sigmoid-Wahrscheinlichkeiten (vor dem
    nicht-differenzierbaren topk), damit der Router über den kontrastiven
    Domänen-Loss klare Spezialisten-Cluster ausprägt.
    """
    def __init__(self, model: ArgonForConditionalGeneration):
        self.layer_probs: List[torch.Tensor] = []
        self.handles = []
        for idx, layer in enumerate(model.model.layers):
            # Mittlere bis späte Synthese-Schichten bei großen Modellen,
            # alle Schichten bei Mini-Modellen (smoke)
            if (4 <= idx <= 23 and len(model.model.layers) > 10) or len(model.model.layers) <= 10:
                h = layer.moe.gate.register_forward_hook(self._make_hook(layer.moe.gate))
                self.handles.append(h)

    def _make_hook(self, gate_module):
        def hook(_mod, inp, _out):
            x_flat = inp[0]  # [bsz * seq_len, hidden_size]
            logits = F.linear(x_flat, gate_module.weight)
            scores = torch.sigmoid(logits)
            probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
            self.layer_probs.append(probs.mean(dim=0))  # [n_routed_experts]
        return hook

    def clear(self):
        self.layer_probs.clear()

    def remove(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------------------
# 4. Haupt-Trainingsschleife
# ---------------------------------------------------------------------------
def train_moe_specialization(
    checkpoint_dir: str = "./argon_1b",
    output_dir: str = "./argon_1b_specialized",
    steps: int = 60,
    lr_router: float = 3e-3,
    lr_experts: float = 1e-4,
    entropy_reg_weight: float = 0.35,
    domain_sep_weight: float = 0.30,
    bias_update_rate: float = 0.002,
    bias_update_interval: int = 1,
    train_experts: bool = False,
    max_length: int = 128,
    corpus_choice: str = "builtin",
    save_mode: str = "router_diff",
    emit_presets: bool = False,
    prebuilt_model: Optional[ArgonForConditionalGeneration] = None,
    tokenizer=None,
    sanity_check: bool = True,
    steer: float = 50.0,
):
    print("=" * 86)
    print("ARGON-1B MULTILINGUALES SUBSPACE-MoE & ROUTER SPEZIALISIERUNGS-TRAINING")
    print("=" * 86)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calib = resolve_corpus(corpus_choice)
    print(f"[Setup] Device: {device} | Korpus: {corpus_choice} ({len(calib)} Domänen)")

    cfg_dict = None
    if prebuilt_model is not None:
        model = prebuilt_model.to(dtype=torch.bfloat16, device=device)
        config = model.config
    else:
        print(f"[Setup] | Lade Modell aus {checkpoint_dir}...")
        with open(os.path.join(checkpoint_dir, "config.json"), "r") as f:
            cfg_dict = json.load(f)
        config = ArgonConfig(**cfg_dict)

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = ArgonForConditionalGeneration(config).to(dtype=torch.bfloat16, device=device)
        state_dict = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
        model.load_state_dict(state_dict, strict=False)
    n_experts = config.n_routed_experts

    # 1. Baseline-Messung + Cross-Lingual Invarianz VOR dem Training
    stats_before = measure_domain_routing(model, tokenizer, device, max_length, corpus=calib)
    print_routing_table("VOR DEM TRAINING (Breite Initialisierung über 8 Sprachen)", stats_before)
    verify_cross_lingual_invariance(
        model, tokenizer, device, title="CROSS-LINGUAL INVARIANZ VOR DEM TRAINING (Baseline)"
    )

    # 2. Parameter einfrieren (nur Router + optional Routed Experts aktiv)
    router_params, expert_params = configure_trainable_moe_parameters(model, train_experts=train_experts)

    param_groups = [{"params": router_params, "lr": lr_router, "weight_decay": 0.0}]
    if expert_params:
        param_groups.append({"params": expert_params, "lr": lr_experts, "weight_decay": 1e-4})
    optimizer = torch.optim.AdamW(param_groups)

    # 3. Tokenisiere die multilingualen Domänen-Batches vorab
    domain_names = list(calib.keys())
    tokenized_batches: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]] = {}

    for d_name, info in calib.items():
        # Mikro-Batch-Chunking (Schritt 5): Maschinen-Korpora tragen 24 Samples
        # statt 8 — ein einziger Padded-Batch verdreifacht die Backward-Aktivierungen
        # (OOM der ersten Real-Lauf-Fassung, GPU 12 GiB). Chunk = bewährte 8er-Lane;
        # bei Domänen ≤ 8 Samples bytegleiche Semantik (ein Chunk).
        max_chunk = 8
        sample_groups = [info["samples"][i : i + max_chunk]
                         for i in range(0, len(info["samples"]), max_chunk)]
        chunk_tensors = []
        for chunk_samples in sample_groups:
            enc = tokenizer(
                chunk_samples,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attn_mask = enc["attention_mask"].to(device)
            labels = input_ids.clone()
            labels[attn_mask == 0] = -100
            chunk_tensors.append((input_ids, attn_mask, labels))

        # Ziel-Verteilungsvektor für sanfte Symmetriebrechung der Subspaces
        pref_mask = torch.full((n_experts,), 0.05, device=device, dtype=torch.float32)
        for e_idx in info["preferred_experts"]:
            pref_mask[e_idx] = 1.0
        pref_dist = pref_mask / pref_mask.sum()

        tokenized_batches[d_name] = (chunk_tensors, pref_dist, info["target_entropy"])

    router_hook = DifferentiableRouterHook(model)

    print(f"\n[Train] Starte {steps} Spezialisierungs-Schritte über {len(domain_names)} Domänen: {', '.join(domain_names)}...")
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()

        # Akkumulator für das globale DeepSeek-V3-Balancing über alle Sprachen & Domänen
        step_expert_counts = [torch.zeros(n_experts, device=device) for _ in range(config.num_hidden_layers)]
        total_ce_loss = 0.0
        total_sep_loss = 0.0

        # Jeder Optimizer-Schritt iteriert über alle Domänen, damit das globales
        # Load-Balancing fair bleibt, während jede Domäne ihre Nische sucht.
        for d_name in domain_names:
            chunk_tensors, pref_dist, target_h = tokenized_batches[d_name]
            n_chunks = len(chunk_tensors)
            for input_ids, attn_mask, labels in chunk_tensors:
                router_hook.clear()

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    labels=labels,
                    router_entropy_weight=entropy_reg_weight,
                    router_target_entropy=target_h,
                )

                # Kontrastiver Subspace-Separations-Loss (KL(Präferenz || gelernt)),
                # pro Chunk auf dem Chunk-Mittel (Approximation an die Domänen-Mittel;
                # nur der 1-Chunk-Fall = das alte bytegleiche Verhalten). reduction="sum":
                # batchmean würde hier durch V=n_experts teilen, weil wir einen EINZIGEN
                # Verteilungsvektor (nicht einen Batch) haben.
                mean_domain_probs = torch.stack(router_hook.layer_probs, dim=0).mean(dim=0)  # [n_experts]
                sep_loss = domain_sep_weight * F.kl_div(
                    (mean_domain_probs + 1e-8).log(),
                    pref_dist,
                    reduction="sum",
                )

                domain_loss = (outputs.loss + sep_loss) / (len(domain_names) * n_chunks)
                domain_loss.backward()

                # Reale Top-K-Counts aus allen Schichten für das Post-Step-Balancing
                with torch.no_grad():
                    for l_idx, layer in enumerate(model.model.layers):
                        if hasattr(layer.moe.gate, "last_counts"):
                            step_expert_counts[l_idx].add_(layer.moe.gate.last_counts)

                total_ce_loss += outputs.loss.item() / (len(domain_names) * n_chunks)
                total_sep_loss += sep_loss.item() / (len(domain_names) * n_chunks)

        torch.nn.utils.clip_grad_norm_(router_params + expert_params, max_norm=1.0)
        optimizer.step()

        # Post-Step Aux-loss-freies Balancing (DeepSeek-V3-Update NACH optimizer.step):
        # expert_bias += rate * sign(avg_count - count_i) -> untergenutzte Experten
        # angeben. Additiver Bias ist state-dict-intakt (buffer).
        if step % bias_update_interval == 0:
            with torch.no_grad():
                for l_idx, layer in enumerate(model.model.layers):
                    counts = step_expert_counts[l_idx]
                    avg_c = counts.mean()
                    layer.moe.gate.expert_bias.add_(bias_update_rate * torch.sign(avg_c - counts))

        if step == 1 or step % 10 == 0 or step == steps:
            elapsed = time.time() - t0
            print(
                f"  Step {step:03d}/{steps:03d} | "
                f"CE+Ent-Loss: {total_ce_loss:.4f} | "
                f"Subspace-Sep-Loss: {total_sep_loss:.4f} | "
                f"Zeit: {elapsed:.1f}s"
            )

    router_hook.remove()

    # Re-aktiviere das Standard-Flag
    for layer in model.model.layers:
        layer.moe.gate.inline_balancing_enabled = True

    # 4. Messung nach dem Training + Cross-Lingual Invarianz-Prüfung
    stats_after = measure_domain_routing(model, tokenizer, device, max_length, corpus=calib)
    print_routing_table("NACH DEM TRAINING (Multilinguale Domänen-Cluster & kalibrierte Entropie)", stats_after)
    verify_cross_lingual_invariance(
        model, tokenizer, device, title="CROSS-LINGUAL INVARIANZ NACH DEM TRAINING (Ungesehene Sätze)"
    )

    # 5. Wissens-Erhalt-Check in mehreren Sprachen (Mini-Smoke: deaktiviert,
    #    Zufalls-Gewichte scheitern sinnigerweise am Paris-Assert)
    if sanity_check:
        model.eval()
        for test_p, expected in [
            ("The capital of France is", "Paris"),
            ("Die Hauptstadt von Frankreich ist", "Paris"),
        ]:
            enc_p = tokenizer(test_p, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=enc_p["input_ids"],
                    attention_mask=enc_p["attention_mask"],
                    max_new_tokens=12,
                    do_sample=False,
                    use_cache=True,
                )
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            print(f"[Sanity-Check] {test_p!r} -> {gen_text!r}")
            assert expected in gen_text, f"Sanity-Check fehlgeschlagen für {test_p}!"

    # 6. Speichere spezialisierten Checkpoint — Default router_diff: mit
    #    Router-Only-Freeze ändern sich exakt gate.weight + expert_bias; das Diff
    #    gegen die Base-Safetensors ist klein (≈1 MB statt 5.1 GB Full-Save).
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n[Save] Speichere spezialisierten Checkpoint nach {output_dir} (Mode: {save_mode})...")
        if cfg_dict is None and hasattr(config, "to_dict"):
            # prebuilt-Lane (Mini-Smoke): Config aus dem lebenden Modell ableiten
            cfg_dict = json.loads(json.dumps(config.to_dict(), default=str))
        cur_state = model.state_dict()
        if save_mode == "router_diff":
            base_state = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
            diff_state = {
                k: v.cpu().contiguous()
                for k, v in cur_state.items()
                if k not in base_state or not torch.equal(v.detach().cpu(), base_state[k])
            }
            save_file(diff_state, os.path.join(output_dir, "router_diff.safetensors"))
            meta = {
                "base_checkpoint": os.path.abspath(checkpoint_dir),
                "corpus_choice": corpus_choice,
                "steps": steps,
                "n_keys": len(diff_state),
                "apply": "construct base model + load_state_dict(base) + load_state_dict(router_diff, strict=False)",
            }
            with open(os.path.join(output_dir, "router_diff_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            # Config + Tokenizer immer mitspeichern — Verify konstruiert daraus.
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(cfg_dict, f, indent=2)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
            print(f"  [Save] router_diff: {len(diff_state)} geänderte Keys nach router_diff.safetensors.")
        elif save_mode == "full":
            # .cpu() je Parameter erzeugt unabhängige Kopien (tied weights teilen
            # keine Speicher-Identität im Export) -> safetensors ohne Shared-Memory-Konflikt
            save_state = {k: v.cpu().contiguous() for k, v in cur_state.items()}
            save_file(save_state, os.path.join(output_dir, "model.safetensors"))
            tokenizer.save_pretrained(output_dir)
            with open(os.path.join(output_dir, "config.json"), "w") as f:
                json.dump(cfg_dict, f, indent=2)
            print(f"Fertig. Du kannst jetzt `compare_gemma3_vs_argon.py` auf {output_dir} ausführen.")

        # 7. Preset-Emission (Schritt-Extension der Götterdreiheit): Bias-Profile
        #    je Maschine, gemessen aus der NACH-Dem-Training-Verteilung.
        if emit_presets:
            _emit_machine_presets(
                output_dir,
                stats_after,
                config,
                entropy_targets={d: calib[d]["target_entropy"] for d in MACHINE_DOMAINS},
                steer=steer,
            )
            print("  [Save] machine_presets/{goedel,turing,wittgenstein,combined}.json geschrieben.")


def _emit_machine_presets(
    output_dir: str,
    stats_after: Dict[str, Tuple[torch.Tensor, float]],
    config,
    entropy_targets: Dict[str, float],
    steer: float = 50.0,
) -> None:
    """
    Bias-Profile aus der NACH-Dem-Training-Verteilung: +steer auf die zwei
    top-gelenkten Experten, -steer auf den Rest (Suite-Muster aus
    compare_gemma3_vs_argon.py:249-260). Nur Maschinen-Domänen; combined = Union.
    """
    presets_dir = os.path.join(output_dir, "machine_presets")
    os.makedirs(presets_dir, exist_ok=True)
    n_layers = config.num_hidden_layers
    n_experts = config.n_routed_experts
    combined: List[int] = []
    for domain in MACHINE_DOMAINS:
        if domain not in stats_after:
            continue
        pct = stats_after[domain][0]
        experts = sorted(range(n_experts), key=lambda e: (-pct[e].item(), e))[:2]
        bias_values: List[float] = []
        for _ in range(n_layers):
            for e in range(n_experts):
                bias_values.append(steer if e in experts else -steer)
        preset = {
            "name": f"argon-{domain}",
            "domain": domain,
            "entropy_target": float(entropy_targets[domain]),
            "mean_h": float(stats_after[domain][1]),  # gemessene Router-Entropie nach dem Training
            "experts": experts,
            "steer": steer,
            "expert_bias": {"layers": n_layers, "values": bias_values},
            "routing_pct": [float(v) for v in pct.tolist()],
        }
        with open(os.path.join(presets_dir, f"{domain}.json"), "w") as f:
            json.dump(preset, f, indent=2)
        combined.extend(experts)
    combined_unique = sorted(set(combined))
    combined_bias: List[float] = []
    for _ in range(n_layers):
        for e in range(n_experts):
            combined_bias.append(steer if e in combined_unique else -steer)
    combined_preset = {
        "name": "argon-gotterdreiheit-combined",
        "domain": "combined",
        "entropy_target": None,
        "mean_h": None,      # Union: je Maschinen-Preset maßgeblich
        "experts": combined_unique,
        "steer": steer,
        "expert_bias": {"layers": n_layers, "values": combined_bias},
        "routing_pct": None,  # Union: je Maschinen-Preset maßgeblich
    }
    with open(os.path.join(presets_dir, "combined.json"), "w") as f:
        json.dump(combined_preset, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argon-1B Multilinguales Subspace-MoE Spezialisierungs-Training")
    parser.add_argument("--checkpoint_dir", type=str, default="./argon_1b")
    parser.add_argument("--output_dir", type=str, default="./argon_1b_specialized")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--max_length", type=int, default=128, help="Sequenzlänge der Domänen-Batches")
    parser.add_argument("--lr_router", type=float, default=3e-3)
    parser.add_argument("--lr_experts", type=float, default=1e-4)
    parser.add_argument("--router_only", action="store_true",
                        help="Trainiert nur die 26 Router-Matrizen; routed_experts bleiben eingefroren")
    parser.add_argument("--corpus", type=str, default="builtin", choices=["builtin", "machines", "all"],
                        help="Kalibrierungs-Korpus: builtin (3 Domänen) | machines (Götterdreiheit) | all (6)")
    parser.add_argument("--save_mode", type=str, default="router_diff", choices=["router_diff", "full"],
                        help="router_diff speichert nur geänderte gate.weight/expert_bias-Keys")
    parser.add_argument("--emit_presets", action="store_true",
                        help="Schreibt machine_presets/{goedel,turing,wittgenstein,combined}.json")
    args = parser.parse_args()

    train_moe_specialization(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        steps=args.steps,
        max_length=args.max_length,
        lr_router=args.lr_router,
        lr_experts=args.lr_experts,
        train_experts=not args.router_only,
        corpus_choice=args.corpus,
        save_mode=args.save_mode,
        emit_presets=args.emit_presets,
    )