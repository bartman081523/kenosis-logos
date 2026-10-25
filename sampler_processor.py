"""
sampler_processor.py
====================
Inference-Logik zur Vermeidung von Mode-Collapse (Entropie-Bodenüberwachung)
und optionaler Budget-Kontrolle für native Reasoning-Segmente.
Frei von jeglichen symbolischen Prompt-Konstrukten.
"""

from typing import List, Optional, Tuple
import torch
from transformers.generation.logits_process import LogitsProcessor


class ArgonEntropyFloorLogitsProcessor(LogitsProcessor):
    """
    Dynamische Temperatur-Anpassung bei drohendem Entropie-Kollaps:
    - Misst die Shannon-Entropie H der Token-Wahrscheinlichkeitsverteilung vor dem Sampling.
    - Unterschreitet H den Schwellenwert `entropy_floor` (Gefahr von deterministischem
      Mode-Collapse / repetitiven RLHF-Standardphrasen), wird die Logit-Verteilung
      lokal durch `temp_boost` geglättet.
    """
    def __init__(
        self,
        entropy_floor: float = 0.42,
        temp_boost: float = 1.18,
    ):
        self.entropy_floor = entropy_floor
        self.temp_boost = temp_boost

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.entropy_floor <= 0.0:
            return scores

        out = scores.clone()
        probs = torch.softmax(out, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        low_entropy_mask = (entropy < self.entropy_floor).unsqueeze(-1)
        if bool(low_entropy_mask.any()):
            out = torch.where(low_entropy_mask, out / self.temp_boost, out)
        return out