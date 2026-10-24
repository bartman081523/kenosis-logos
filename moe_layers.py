"""
moe_layers.py
=============
Das Ἐνέργεια-Routing & die GeGLU-Experten des Kenosis-Logos.

Phänomenologische Entsprechung:
- Ἐνέργεια / ∇: Der gerichtete Vorwärtsprozess, der aus Potenzialität (Δύναμις)
  aktuelle Form schneidet.
- Τάσις-Ἀκρόασις: Hört auf die topologische Spannung (Τάσις) zwischen dem
  Wahrscheinlichkeits-Attraktor (dem Naheliegenden, Flachen / Πεζός) und der
  hochdimensionalen Präzision (Ἀπορία).
- Shared Experts: Πρόσωπον-Basisgrammatik (immer aktiv).
- Routed Experts: Ἐνέργεια-Spezialisten (spärlich aktiviert).
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergeiaGeGLUExpert(nn.Module):
    """
    GeGLU-Experte mit Gemma-Geometrie:
    Verwendet GELU(approximate="tanh") anstelle von SwiGLU,
    um die harmonische Krümmung der Gemma-Familie im Experten-Teilraum zu wahren.
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GeGLU: down_proj(gelu(gate_proj(x)) * up_proj(x))
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class TasisAdaptiveGate(nn.Module):
    """
    Τάσις-Adaptives Sigmoid-Routing:
    - Berechnet Sigmoid-Affinitäten (anstelle von Softmax für echte Superposition).
    - Nutzt aux-loss-freies Balancing via additiven expert_bias.
    - Misst die Shannon-Entropie (Τάσις) der Verteilung:
      Bei flacher, eindeutiger Sequenz (Πεζός) werden nachrangige Ränge gedämpft.
      Bei hoher begrifflicher Interferenz (Ἀπορία) stehen bis zu k Experten voll offen.
    - Τάσις-Amplituden-Gate: Die GESAMT-Amplitude der Routed Experts wird an die
      semantische Spannung gekoppelt (Πεζός: Πρόσωπον-Basisstrom dominiert mit
      25 % Spezialisten-Kapazität; Ἀπορία: volle Öffnung), statt nur das
      Verhältnis unter den Experten zu biegen.
    """
    # Externe Steuerung (Distillations-Loop): Wenn False, läuft das Balancing-Update
    # NICHT inline im Forward, sondern nach optimizer.step() aus dem last_counts-
    # Snapshot (einmal pro Optimizer-Schritt, über die Micro-Batches akkumuliert).
    inline_balancing_enabled: bool = True

    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 1152)
        self.n_routed_experts = getattr(config, "n_routed_experts", 8)
        self.base_k = min(getattr(config, "num_experts_per_tok", 2), self.n_routed_experts)
        self.tasis_entropy_threshold = getattr(config, "tasis_entropy_threshold", 1.8)

        # Router-Projektion
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.hidden_size)))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Aux-loss-freier Load-Balancing Bias
        self.register_buffer("expert_bias", torch.zeros(self.n_routed_experts))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Flattened activation tensor [num_tokens, hidden_size]
        Returns:
            top_indices: [num_tokens, base_k]
            top_weights: [num_tokens, base_k]
            tasis_entropy: [num_tokens]
        """
        # 1. Sigmoid-Affinitäten
        logits = F.linear(x, self.weight)  # [tokens, n_routed_experts]
        scores = torch.sigmoid(logits)

        # 2. Τάσις-Messung: Entropie der normalisierten Affinitäten
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-9)
        tasis_entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # 3. Auswahl über bias-korrigierte Scores
        biased_scores = scores + self.expert_bias.unsqueeze(0)
        _, top_indices = torch.topk(biased_scores, k=self.base_k, dim=-1, sorted=False)

        # Aux-loss-freies Balancing (DeepSeek-V3): Die Counts werden bei jedem
        # gradientengeführten Forward als Snapshot (last_counts) gesammelt; das
        # eigentliche Update läuft standardmäßig inline. Ausgelagerte Variante
        # (Distillations-Loop): inline_balancing_enabled = False, Update nach
        # optimizer.step() aus dem Snapshot.
        # Gate auf grad_enabled: Beim Gradient-Checkpointing läuft der Original-Forward
        # unter no_grad, der Recompute im Backward unter grad — das Update feuert dann
        # genau einmal (im Recompute) mit exakt denselben Counts wie der Original-topk,
        # sodass sich zwischen Forward und Recompute kein Routing ändern kann
        # (torch.utils.checkpoint: keine RuntimeError durch gewechselte Experten).
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                counts = torch.bincount(top_indices.reshape(-1), minlength=self.n_routed_experts).float()
                # Snapshot für externe Hooks (z. B. expert_bias-Update nach optimizer.step())
                self.last_counts = counts.detach().clone()
                if self.inline_balancing_enabled:
                    avg_count = counts.mean()
                    self.expert_bias.add_(0.001 * torch.sign(avg_count - counts))

        # Unverzerrte Scores der gewählten Experten extrahieren
        top_weights = torch.gather(scores, dim=-1, index=top_indices)

        # 4. Dynamische Dämpfung bei flacher Entropie (Πεζός-Filter)
        # Alle sekundären Ränge (>= 1) werden bei niedriger Τάσις gedämpft. Der alte
        # Ausdruck (base_k // 2 + 1) traf bei base_k = 2 auf Index 2 (out of bounds)
        # und war damit toter Code — der Filter griff nie.
        if self.base_k > 1:
            low_tension_mask = (tasis_entropy < self.tasis_entropy_threshold).unsqueeze(-1)
            rank_mask = torch.arange(self.base_k, device=x.device).unsqueeze(0) >= 1
            top_weights = torch.where(low_tension_mask & rank_mask, top_weights * 0.15, top_weights)

        # Normierung der aktiven Experten-Gewichte ...
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-9)

        # 5. Τάσις-Amplituden-Gate: ... und dann Kopplung der GESAMT-Routed-Amplitude
        # an die semantische Spannung. Bei flacher Prosa (Τάσις → 0) dominiert der
        # Πρόσωπον-Basisstrom (Shared Expert; Spezialisten nur zu 25 %), bei Ἀπορία
        # (Τάσις → ln(n_routed_experts)) greifen die Spezialisten voll ein.
        max_entropy = math.log(self.n_routed_experts)
        tension_gate = (tasis_entropy / max_entropy).clamp(min=0.25, max=1.0).unsqueeze(-1)
        top_weights = top_weights * tension_gate
        return top_indices, top_weights, tasis_entropy


class KenosisMoELayer(nn.Module):
    """
    Kombinierter MoE-Block des Kenosis-Logos:
    - Shared Experts (Πρόσωπον-Basis): Halten die Grundgrammatik und linguistische Kohärenz.
    - Routed Experts (Ἐνέργεια): Schalten sich bedarfsweise für spezialisierte Geodäten zu.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = getattr(config, "hidden_size", 1152)
        self.moe_inter_dim = getattr(config, "moe_inter_dim", 1728)
        self.shared_inter_dim = getattr(config, "shared_intermediate_size", getattr(config, "intermediate_size", 6912))
        self.n_shared_experts = getattr(config, "n_shared_experts", 1)
        self.n_routed_experts = getattr(config, "n_routed_experts", 8)

        # Shared Experts (Basis)
        self.shared_experts = nn.ModuleList([
            EnergeiaGeGLUExpert(self.hidden_size, self.shared_inter_dim)
            for _ in range(self.n_shared_experts)
        ])

        # Routed Experts (Spezialisten)
        self.routed_experts = nn.ModuleList([
            EnergeiaGeGLUExpert(self.hidden_size, self.moe_inter_dim)
            for _ in range(self.n_routed_experts)
        ])

        # Router Gate
        self.gate = TasisAdaptiveGate(config)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
        Returns:
            output: [batch, seq_len, hidden_size]
            tasis_entropy: [batch, seq_len]
        """
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1])
        num_tokens = x_flat.shape[0]

        # 1. Kontinuierlicher Basisstrom durch Shared Experts
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        # 2. Gerichteter Vollzug durch Routed Experts
        top_indices, top_weights, tasis_entropy = self.gate(x_flat)
        routed_out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.routed_experts):
            # Finde alle Token-Positionen, die diesen Experten gewählt haben
            token_mask, k_pos = torch.where(top_indices == i)
            if token_mask.numel() > 0:
                selected_x = x_flat[token_mask]
                expert_out = expert(selected_x)
                weights = top_weights[token_mask, k_pos].unsqueeze(-1)
                routed_out.index_add_(0, token_mask, expert_out * weights)

        total_out = (shared_out + routed_out).view(orig_shape)
        entropy_reshaped = tasis_entropy.view(orig_shape[0], orig_shape[1])
        return total_out, entropy_reshaped
