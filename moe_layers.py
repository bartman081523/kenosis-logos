"""
moe_layers.py
=============
Sparsely-Gated Mixture-of-Experts (MoE) mit GeGLU-Experten und
Entropie-adaptivem Sigmoid-Routing für die Argon-Architektur.
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArgonGeGLUExpert(nn.Module):
    """
    Feed-Forward-Experte mit GeGLU-Aktivierung (tanh-Approximation).
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class ArgonEntropyAdaptiveRouter(nn.Module):
    """
    Entropie-adaptives Sigmoid-Routing:
    1. Berechnet unabhängige Sigmoid-Affinitäten statt konkurrierendem Softmax.
    2. Aux-loss-freies Load-Balancing über additiven expert_bias.
    3. Misst die Shannon-Entropie H der Router-Verteilung pro Token:
       - Bei niedriger Entropie (Routine-Sequenzen) werden sekundäre Expertenränge gedämpft
         und die Gesamt-Routed-Amplitude auf 25% skaliert (Shared Expert dominiert).
       - Bei hoher Entropie (komplexe Mehrdeutigkeit) öffnen die Spezialisten bis 100%.
    """
    inline_balancing_enabled: bool = True

    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 1152)
        self.n_routed_experts = getattr(config, "n_routed_experts", 8)
        self.base_k = min(getattr(config, "num_experts_per_tok", 2), self.n_routed_experts)
        self.entropy_threshold = getattr(config, "router_entropy_threshold", 1.8)

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.register_buffer("expert_bias", torch.zeros(self.n_routed_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(x, self.weight)
        scores = torch.sigmoid(logits)

        # Shannon-Entropie der normalisierten Affinitäten
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
        router_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # Top-K Auswahl über bias-korrigierte Scores
        biased_scores = scores + self.expert_bias.unsqueeze(0)
        _, top_indices = torch.topk(biased_scores, k=self.base_k, dim=-1, sorted=False)

        # Aux-loss-freies Load-Balancing im Training (Gradient-Checkpointing-sicher).
        # last_counts bleibt der Snapshot für die ausgelagerte Variante
        # (inline_balancing_enabled = False -> Update nach optimizer.step()).
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                counts = torch.bincount(top_indices.reshape(-1), minlength=self.n_routed_experts).float()
                self.last_counts = counts.detach().clone()
                if self.inline_balancing_enabled:
                    avg_count = counts.mean()
                    self.expert_bias.add_(0.001 * torch.sign(avg_count - counts))

        top_weights = torch.gather(scores, dim=-1, index=top_indices)

        # Sekundäre Experten dämpfen, wenn Router-Entropie unter dem Schwellenwert liegt
        if self.base_k > 1:
            low_entropy_mask = (router_entropy < self.entropy_threshold).unsqueeze(-1)
            rank_mask = torch.arange(self.base_k, device=x.device).unsqueeze(0) >= 1
            top_weights = torch.where(low_entropy_mask & rank_mask, top_weights * 0.15, top_weights)

        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # Kapazitäts-Skalierung der Routed-Amplitude relativ zum Shared Expert
        max_entropy = math.log(self.n_routed_experts)
        capacity_gate = (router_entropy / max_entropy).clamp(min=0.25, max=1.0).unsqueeze(-1)
        top_weights = top_weights * capacity_gate
        return top_indices, top_weights, router_entropy


class ArgonMoELayer(nn.Module):
    """
    MoE-Layer mit persistenten Shared Experts und spärlich aktivierten Routed Experts.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 1152)
        self.moe_inter_dim = getattr(config, "moe_inter_dim", 1728)
        self.shared_inter_dim = getattr(config, "shared_intermediate_size", getattr(config, "intermediate_size", 6912))
        self.n_shared_experts = getattr(config, "n_shared_experts", 1)
        self.n_routed_experts = getattr(config, "n_routed_experts", 8)

        self.shared_experts = nn.ModuleList([
            ArgonGeGLUExpert(self.hidden_size, self.shared_inter_dim)
            for _ in range(self.n_shared_experts)
        ])

        self.routed_experts = nn.ModuleList([
            ArgonGeGLUExpert(self.hidden_size, self.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])

        self.gate = ArgonEntropyAdaptiveRouter(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])

        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        top_indices, top_weights, router_entropy = self.gate(x_flat)
        routed_out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.routed_experts):
            token_mask, k_pos = torch.where(top_indices == i)
            if token_mask.numel() > 0:
                selected_x = x_flat[token_mask]
                expert_out = expert(selected_x)
                weights = top_weights[token_mask, k_pos].unsqueeze(-1)
                routed_out.index_add_(0, token_mask, expert_out * weights)

        total_out = (shared_out + routed_out).view(orig_shape)
        entropy_reshaped = router_entropy.view(orig_shape[0], orig_shape[1])
        return total_out, entropy_reshaped