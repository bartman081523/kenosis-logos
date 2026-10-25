"""
modeling_argon.py
=================
Hauptarchitektur für Argon (1B Hybrid MLA-MoE Multimodal Transformer).

Kernkomponenten:
1. ArgonMultimodalEmbedding: Early-Fusion für Text, Vision-Patches und Audio-Frames;
   bidirektionale Attention-Maske und 2D-RoPE für zusammenhängende Bild-Regionen.
2. ArgonHybridAttention:
   - Lokale Sliding-Window-Schichten (5:1): MQA mit rope_theta = 10.000 und KV-Ringpuffer.
   - Globale Schichten (layer % 6 == 5): Multi-Head Latent Attention (MLA) mit
     rope_theta = 1.000.000 und komprimiertem Latent-KV-Cache ([c_kv, k_rope] = 320 Dim).
3. ArgonMoELayer: Shared + Routed GeGLU-Experten mit Entropie-adaptiver Kapazität.
4. Gauß-gewichtete Router-Entropie-Regularisierung über die Netzwerktiefe.
"""

import math
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin

from moe_layers import ArgonMoELayer


class ArgonConfig(PretrainedConfig):
    model_type = "argon"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 1152,
        intermediate_size: int = 6912,
        num_hidden_layers: int = 26,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta_local: float = 10000.0,
        rope_theta_global: float = 1000000.0,
        sliding_window: int = 512,
        sliding_window_pattern: int = 6,
        attn_logit_softcapping: Optional[float] = None,
        final_logit_softcapping: Optional[float] = None,
        use_mla: bool = True,
        kv_lora_rank: int = 256,
        q_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        use_moe: bool = True,
        n_shared_experts: int = 1,
        n_routed_experts: int = 8,
        num_experts_per_tok: int = 2,
        moe_inter_dim: int = 1728,
        router_entropy_threshold: float = 1.8,
        router_entropy_target: float = 1.0,
        router_depth_peak: float = 0.55,
        router_depth_sigma: float = 0.30,
        vision_bidirectional: bool = True,
        vision_2d_rope: bool = True,
        shared_intermediate_size: Optional[int] = 6912,
        vision_config: Optional[Dict[str, Any]] = None,
        audio_config: Optional[Dict[str, Any]] = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta_local = rope_theta_local
        self.rope_theta_global = rope_theta_global
        self.sliding_window = sliding_window
        self.sliding_window_pattern = sliding_window_pattern
        self.attn_logit_softcapping = attn_logit_softcapping
        self.final_logit_softcapping = final_logit_softcapping
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.use_moe = use_moe
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_inter_dim = moe_inter_dim
        self.router_entropy_threshold = router_entropy_threshold
        self.router_entropy_target = router_entropy_target
        self.router_depth_peak = router_depth_peak
        self.router_depth_sigma = router_depth_sigma
        self.vision_bidirectional = vision_bidirectional
        self.vision_2d_rope = vision_2d_rope
        self.shared_intermediate_size = shared_intermediate_size or intermediate_size
        self.vision_config = vision_config or {"vision_hidden_size": 1152, "image_size": 384}
        self.audio_config = audio_config or {"audio_hidden_size": 896}


class ArgonRMSNorm(nn.Module):
    """
    Zero-centered RMSNorm mit (1.0 + weight) Skalierung.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        norm_x = x_float * torch.rsqrt(variance + self.eps)
        out = norm_x * (1.0 + self.weight.float())
        return out.to(orig_dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class ArgonMultimodalEmbedding(nn.Module):
    """
    Early-Fusion Eingangsprojektion für Text, Vision-Patches und Audio-Frames.
    """
    def __init__(self, config: ArgonConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        vision_dim = config.vision_config.get("vision_hidden_size", config.hidden_size)
        audio_dim = config.audio_config.get("audio_hidden_size", 896)
        self.vision_proj = nn.Linear(vision_dim, config.hidden_size, bias=False)
        self.audio_proj = nn.Linear(audio_dim, config.hidden_size, bias=False)
        self.norm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.text_embed(input_ids) * math.sqrt(self.hidden_size)

        if modality_mask is not None:
            if vision_patches is not None:
                v_emb = self.norm(self.vision_proj(vision_patches))
                h = torch.where(modality_mask.unsqueeze(-1) == 1, v_emb, h)
            if audio_frames is not None:
                a_emb = self.norm(self.audio_proj(audio_frames))
                h = torch.where(modality_mask.unsqueeze(-1) == 2, a_emb, h)
        return h


class ArgonHybridAttention(nn.Module):
    """
    Hybride Attention-Schicht mit Latent-KV-Cache-Unterstützung:
    - Lokale Sliding-Window-Schichten: MQA, cached (k, v) begrenzt auf sliding_window.
    - Globale MLA-Schichten: cached ausschließlich den komprimierten 320-Dim Vektor (c_kv, k_rope_raw).
    """
    def __init__(self, config: ArgonConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.softcapping = config.attn_logit_softcapping
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        pattern = getattr(config, "sliding_window_pattern", 6)
        self.is_global = (layer_idx % pattern == (pattern - 1))
        self.use_mla = config.use_mla and self.is_global

        self.q_norm = ArgonRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = ArgonRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_mla:
            self.kv_lora_rank = config.kv_lora_rank
            self.qk_nope_dim = config.qk_nope_head_dim
            self.qk_rope_dim = config.qk_rope_head_dim
            self.v_head_dim = config.v_head_dim

            self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_dim, bias=False)
            self.kv_up = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False)
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = x.shape

        if self.use_mla:
            # 1. Query projizieren, normieren und auf die letzten 64 Kanäle RoPE für die aktuellen Positionen anwenden
            q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
            q = self.q_norm(q).transpose(1, 2)
            q_nope = q[..., :self.qk_nope_dim]
            q_rope = q[..., self.qk_nope_dim:]
            cos_q = cos[:, :, -q_len:, :]
            sin_q = sin[:, :, -q_len:, :]
            q_rope = apply_rotary_pos_emb(q_rope, cos_q, sin_q)
            q = torch.cat([q_nope, q_rope], dim=-1)

            # 2. Latente KV-Kompression berechnen: [bsz, q_len, 320]
            kv_comp = self.kv_down(x)
            c_kv_new = kv_comp[..., :self.kv_lora_rank]
            k_rope_new = kv_comp[..., self.kv_lora_rank:]

            # 3. Komprimierten Latent-KV-Cache aktualisieren (nur 320 Floats pro Token!)
            if past_key_value is not None:
                c_kv = torch.cat([past_key_value[0], c_kv_new], dim=1)
                k_rope_raw = torch.cat([past_key_value[1], k_rope_new], dim=1)
            else:
                c_kv, k_rope_raw = c_kv_new, k_rope_new

            present_key_value = (c_kv, k_rope_raw) if use_cache else None
            kv_len = c_kv.shape[1]

            # 4. KV aus dem Latent-Vektor entfalten, normieren und RoPE über die gesamte KV-Sequenz anwenden
            kv_up_out = self.kv_up(c_kv).view(bsz, kv_len, self.num_heads, self.qk_nope_dim + self.v_head_dim)
            k_nope = kv_up_out[..., :self.qk_nope_dim]
            v = kv_up_out[..., self.qk_nope_dim:]

            k_rope_expanded = k_rope_raw.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope, k_rope_expanded], dim=-1)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)

            k_nope_norm = k[..., :self.qk_nope_dim]
            k_rope_norm = k[..., self.qk_nope_dim:]
            k_rope_rot = apply_rotary_pos_emb(k_rope_norm, cos[:, :, :kv_len, :], sin[:, :, :kv_len, :])
            k = torch.cat([k_nope_norm, k_rope_rot], dim=-1)

            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        else:
            # Lokaler MQA Sliding-Window-Pfad
            q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(bsz, q_len, self.num_kv_heads, self.head_dim)

            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)

            cos_q = cos[:, :, -q_len:, :]
            sin_q = sin[:, :, -q_len:, :]
            q = apply_rotary_pos_emb(q, cos_q, sin_q)
            k = apply_rotary_pos_emb(k, cos_q, sin_q)

            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)

            present_key_value = (k, v) if use_cache else None

            if self.num_kv_heads < self.num_heads:
                k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

        if self.softcapping is not None:
            scores = self.softcapping * torch.tanh(scores / self.softcapping)

        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_out), present_key_value


class ArgonDecoderLayer(nn.Module):
    def __init__(self, config: ArgonConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        pattern = getattr(config, "sliding_window_pattern", 6)
        self.is_global = (layer_idx % pattern == (pattern - 1))

        self.input_layernorm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = ArgonHybridAttention(config, layer_idx)
        self.post_attention_layernorm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = ArgonMoELayer(config)
        self.post_feedforward_layernorm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        norm_states = self.input_layernorm(hidden_states)
        attn_out, present_kv = self.self_attn(
            norm_states,
            cos=cos,
            sin=sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        ffn_in = self.pre_feedforward_layernorm(h)
        moe_out, router_entropy = self.moe(ffn_in)
        out = residual + self.post_feedforward_layernorm(moe_out)
        return out, router_entropy, present_kv


class ArgonModel(nn.Module):
    def __init__(self, config: ArgonConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = ArgonMultimodalEmbedding(config)
        self.layers = nn.ModuleList([
            ArgonDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = ArgonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_rotary_emb(self, seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0).unsqueeze(0), emb.sin().unsqueeze(0).unsqueeze(0)

    def _build_rotary_emb_2d(
        self,
        coords_x: torch.Tensor,
        coords_y: torch.Tensor,
        head_dim: int,
        theta: float,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        split = half // 2
        freqs_x = coords_x.to(torch.float32).unsqueeze(-1) * inv_freq[:split].unsqueeze(0)
        freqs_y = coords_y.to(torch.float32).unsqueeze(-1) * inv_freq[split:].unsqueeze(0)
        angles = torch.cat([freqs_x, freqs_y], dim=-1)
        emb = torch.cat((angles, angles), dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def _build_2d_coords(self, modality_mask: torch.Tensor, device: torch.device):
        bsz, seq_len = modality_mask.shape
        coords_x = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(0).expand(bsz, -1).clone()
        coords_y = coords_x.clone()
        has_vision = False
        mods_rows = modality_mask.tolist()
        for i in range(bsz):
            mods = mods_rows[i]
            p = 0
            while p < seq_len:
                if mods[p] == 1:
                    s = p
                    while p < seq_len and mods[p] == 1:
                        p += 1
                    run_len = p - s
                    G = max(1, int(round(math.sqrt(run_len))))
                    base = torch.arange(run_len, device=device, dtype=torch.float32)
                    coords_x[i, s:p] = s + (base % G)
                    coords_y[i, s:p] = s + torch.div(base, G, rounding_mode="floor")
                    has_vision = True
                else:
                    p += 1
        return coords_x, coords_y, has_vision

    def _prepare_masks(
        self,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        device: torch.device,
        dtype: torch.dtype,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Erzeuge vollständige [kv_len, kv_len] Kausalmaske und schneide die letzten q_len Zeilen für Decode/Prefill
        full_causal = torch.triu(torch.full((kv_len, kv_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
        full_sliding = full_causal.clone()
        if self.config.sliding_window is not None:
            band = torch.tril(torch.full((kv_len, kv_len), float("-inf"), device=device, dtype=dtype), diagonal=-self.config.sliding_window)
            full_sliding = full_sliding + band

        causal_mask = full_causal[-q_len:, :].unsqueeze(0).unsqueeze(0)
        sliding_mask = full_sliding[-q_len:, :].unsqueeze(0).unsqueeze(0)

        if attention_mask is not None:
            if attention_mask.dim() == 2 and attention_mask.shape[-1] == kv_len:
                pad_mask = (1.0 - attention_mask.to(dtype=dtype)) * -1e9
                pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
                global_mask = causal_mask + pad_mask
                sliding_mask = sliding_mask + pad_mask
            elif attention_mask.dim() == 4:
                global_mask = causal_mask + attention_mask
                sliding_mask = sliding_mask + attention_mask
            else:
                global_mask = causal_mask
        else:
            global_mask = causal_mask

        # Bidirektionale Aufmerksamkeit für zusammenhängende Vision-Patch-Blöcke im Prefill
        if q_len == kv_len and modality_mask is not None and getattr(self.config, "vision_bidirectional", True):
            n_rows = modality_mask.shape[0]
            runs_per_row = []
            for i in range(n_rows):
                mods = modality_mask[i].tolist()
                runs, p = [], 0
                while p < kv_len:
                    if mods[p] == 1:
                        s = p
                        while p < kv_len and mods[p] == 1:
                            p += 1
                        runs.append((s, p))
                    else:
                        p += 1
                runs_per_row.append(runs)

            if any(runs_per_row):
                if global_mask.shape[0] != n_rows:
                    expand_shape = (n_rows,) + global_mask.shape[1:]
                    global_mask = global_mask.expand(expand_shape).clone()
                    sliding_mask = sliding_mask.expand(expand_shape).clone()
                for i, runs in enumerate(runs_per_row):
                    for (s, e) in runs:
                        global_mask[i, ..., s:e, s:e] = 0.0
                        sliding_mask[i, ..., s:e, s:e] = 0.0

        return global_mask, sliding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        bsz, q_len = input_ids.shape

        # transformers 5.x Cache-Objekt (DynamicCache aus generate()) entpacken:
        # Die Latent-KV-Cache-Tensoren liegen unverändert in cache.layers[].keys/.values.
        # Nicht initialisierte Layer (generierter Leer-Cache im Prefill-Schritt) werden
        # über get_seq_length() == 0 herausgefiltert.
        if past_key_values is not None and not isinstance(past_key_values, (list, tuple)):
            past_key_values = [
                (layer.keys, layer.values)
                for layer in past_key_values.layers
                if layer.get_seq_length() > 0
            ]

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None:
            # In Schicht 0 (lokales MQA) liegt k auf [bsz, kv_heads, past_len, head_dim]
            past_len = past_key_values[0][0].shape[2] if past_key_values[0][0].dim() == 4 else past_key_values[0][0].shape[1]
        kv_len = past_len + q_len

        h = self.embed_tokens(
            input_ids=input_ids,
            modality_mask=modality_mask,
            vision_patches=vision_patches,
            audio_frames=audio_frames,
        )

        device = h.device
        dtype = h.dtype

        modality_mask_2d = modality_mask if (modality_mask is not None and modality_mask.dim() == 2 and q_len == kv_len) else None
        coords_x = coords_y = None
        has_vision = False
        if modality_mask_2d is not None and getattr(self.config, "vision_2d_rope", True):
            coords_x, coords_y, has_vision = self._build_2d_coords(modality_mask_2d, device)
        has_vision = has_vision and coords_x is not None

        if has_vision:
            cos_local, sin_local = self._build_rotary_emb_2d(
                coords_x, coords_y, self.config.head_dim, self.config.rope_theta_local, device
            )
            rope_dim_global = self.config.qk_rope_head_dim if self.config.use_mla else self.config.head_dim
            cos_global, sin_global = self._build_rotary_emb_2d(
                coords_x, coords_y, rope_dim_global, self.config.rope_theta_global, device
            )
        else:
            cos_local, sin_local = self._build_rotary_emb(kv_len, self.config.head_dim, self.config.rope_theta_local, device)
            rope_dim_global = self.config.qk_rope_head_dim if self.config.use_mla else self.config.head_dim
            cos_global, sin_global = self._build_rotary_emb(kv_len, rope_dim_global, self.config.rope_theta_global, device)

        cos_local, sin_local = cos_local.to(dtype), sin_local.to(dtype)
        cos_global, sin_global = cos_global.to(dtype), sin_global.to(dtype)

        global_mask, sliding_mask = self._prepare_masks(
            attention_mask, q_len, kv_len, device, dtype, modality_mask=modality_mask_2d
        )

        all_entropies = []
        next_decoder_cache = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if (past_key_values is not None and idx < len(past_key_values)) else None
            if layer.is_global:
                h, layer_entropy, present_kv = layer(
                    h, cos=cos_global, sin=sin_global, attention_mask=global_mask, past_key_value=past_kv, use_cache=use_cache
                )
            else:
                h, layer_entropy, present_kv = layer(
                    h, cos=cos_local, sin=sin_local, attention_mask=sliding_mask, past_key_value=past_kv, use_cache=use_cache
                )
            all_entropies.append(layer_entropy)
            if use_cache:
                next_decoder_cache.append(present_kv)

        h = self.norm(h)
        if use_cache:
            # Rückgabe als echtes Cache-Objekt: die Generate-Schleife von transformers
            # 5.x ruft jede Iteration `cache.get_seq_length()` auf. DynamicCache ohne
            # config erzeugt reine DynamicLayer, die die 3-dim MLA-Latenztensoren
            # (c_kv, k_rope) unverändert speichern — kein Sliding-Crop der globalen
            # Latent-KV-Caches.
            next_decoder_cache = DynamicCache(ddp_cache_data=tuple(next_decoder_cache))
        return h, all_entropies, next_decoder_cache


class ArgonForConditionalGeneration(PreTrainedModel, GenerationMixin):
    config_class = ArgonConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.text_embed.weight"}

    def __init__(self, config: ArgonConfig):
        super().__init__(config)
        self.model = ArgonModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_softcap = config.final_logit_softcapping

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.text_embed.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens.text_embed

    def set_input_embeddings(self, value):
        self.model.embed_tokens.text_embed = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        # Wenn der Latent-KV-Cache bereits gefüllt ist, übergebe nur das neueste Token.
        # transformers 5.x Cache-Objekte erlauben kein Indexieren — Länge via get_seq_length().
        if past_key_values is not None:
            cache_len = (
                past_key_values.get_seq_length()
                if isinstance(past_key_values, Cache)
                else len(past_key_values)
            )
            if cache_len > 0:
                input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        router_entropy_weight: float = 0.0,
        router_target_entropy: Optional[float] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        use_cache = use_cache if use_cache is not None else False
        hidden_states, all_entropies, next_cache = self.model(
            input_ids=input_ids,
            modality_mask=modality_mask,
            vision_patches=vision_patches,
            audio_frames=audio_frames,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = self.lm_head(hidden_states)

        if self.final_softcap is not None:
            logits = self.final_softcap * torch.tanh(logits / self.final_softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if all_entropies and router_entropy_weight != 0.0:
            stacked_entropies = torch.stack(list(all_entropies), dim=0).float()
            n_layers = stacked_entropies.shape[0]
            layer_pos = torch.arange(n_layers, device=stacked_entropies.device, dtype=stacked_entropies.dtype)
            center = float(getattr(self.config, "router_depth_peak", 0.55)) * (n_layers - 1)
            sigma = max(float(getattr(self.config, "router_depth_sigma", 0.30)) * (n_layers - 1), 1e-6)
            layer_weights = torch.exp(-0.5 * ((layer_pos - center) / sigma) ** 2)
            layer_weights = (layer_weights / layer_weights.sum().clamp(min=1e-9)).view(n_layers, 1, 1)

            if labels is not None:
                bsz, seq_len = labels.shape
                valid_mask = torch.zeros((bsz, seq_len), device=stacked_entropies.device, dtype=stacked_entropies.dtype)
                valid_mask[:, :-1] = (labels[:, 1:] != -100).to(stacked_entropies.dtype)
                weighted_mask = valid_mask.unsqueeze(0).expand_as(stacked_entropies) * layer_weights
                mean_entropy = (stacked_entropies * weighted_mask).sum() / weighted_mask.sum().clamp(min=1e-9)
            else:
                mean_entropy = (stacked_entropies * layer_weights).mean()

            target = (
                router_target_entropy
                if router_target_entropy is not None
                else float(getattr(self.config, "router_entropy_target", 1.0))
            )
            reg_loss = router_entropy_weight * (mean_entropy - target) ** 2
            loss = reg_loss if loss is None else loss + reg_loss

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_cache,
            hidden_states=None,
            attentions=None,
        )
        output.router_entropies = all_entropies
        return output


ArgonForCausalLM = ArgonForConditionalGeneration