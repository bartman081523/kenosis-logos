"""
modeling_kenosis.py
===================
Hauptmodell für Kenosis-Logos (Open-Kenosis-Logos).

Mathematisch und technisch korrigiert:
1. Χώρα: UnifiedChoraEmbedding (Text + orthogonale Vision/Audio-Projektoren);
   bidirektionale AttentionMask für kontiguierende Vision-Regionen (modality == 1)
   und mRoPE-artiges 2D-RoPE für Bild-Patches (Text reproduziert (t, t) EXAKT das
   alte 1D-Spektrum).
2. Συνήχησις: SynechesisHybridAttention
   - Lokale Sliding-Schichten: rope_theta = 10.000, MQA, q_norm/k_norm.
   - Globale Schichten (layer % 6 == 5): rope_theta = 1.000.000, MLA mit
     exakter Ausrichtung: Nope (0..192) und RoPE (192..256).
   - Einheitliche Vor-Attention q_norm und k_norm auf allen 26 Schichten.
   - Optionales Attention Logit Soft-Capping (config-gesteuert; Gemma-3-Parität: null).
   - Offen (nächster Schritt): Latent-KV-Cache (past_key_values speichert [c_kv, k_rope]),
     damit die Συνήχησις als stehende Welle ruht statt bei jedem Token neu zu rechnen.
3. Ἐνέργεια: KenosisMoELayer (GeGLU-Experten; Τάσις-Filter auf sekundären Rängen
   und Τάσις-Amplituden-Gate auf der GESAMT-Routed-Kapazität).
4. Κρίσις: Optionales Final Logit Soft-Capping im Vorwärtspfad (Gemma-3-Parität: null);
   Τάσις-Entropie-Kopplung im Loss (Off-by-Default über tasis_entropy_weight) mit
   glockenförmigem Depth-Profil ω_l (Synthese-Maximum in den mittleren Schichten).
5. Inferenz-Support: 4D Attention-Masking, prepare_inputs_for_generation.
"""

import math
from typing import Optional, Tuple, List, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation import GenerationMixin

from moe_layers import KenosisMoELayer


class KenosisConfig(PretrainedConfig):
    model_type = "kenosis_logos"

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
        attn_logit_softcapping: Optional[float] = 50.0,
        final_logit_softcapping: Optional[float] = 30.0,
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
        tasis_entropy_threshold: float = 1.8,
        tasis_entropy_target: float = 1.0,
        tasis_depth_peak: float = 0.55,
        tasis_depth_sigma: float = 0.30,
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
        self.tasis_entropy_threshold = tasis_entropy_threshold
        self.tasis_entropy_target = tasis_entropy_target
        self.tasis_depth_peak = tasis_depth_peak
        self.tasis_depth_sigma = tasis_depth_sigma
        self.vision_bidirectional = vision_bidirectional
        self.vision_2d_rope = vision_2d_rope
        self.shared_intermediate_size = shared_intermediate_size or intermediate_size
        self.vision_config = vision_config or {"vision_hidden_size": 1152, "image_size": 384}
        self.audio_config = audio_config or {"audio_hidden_size": 896}


class KenosisRMSNorm(nn.Module):
    """
    Gemma3-kompatible RMSNorm mit Parameter-Offset (1.0 + weight).
    Initialisierung mit Nullen; Skalierung um Eins zentriert.
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
    """
    Wendet RoPE auf Tensoren an.
    x: [batch, heads, seq_len, head_dim]
    cos, sin: [1, 1, seq_len, head_dim]
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotated * sin)


class UnifiedChoraEmbedding(nn.Module):
    """
    Χώρα-Eingangsprojektion:
    Vereinigt Text, Vision-Patches und Audio-Frames im selben Hilbertraum H (Early Fusion).
    """
    def __init__(self, config: KenosisConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        vision_dim = config.vision_config.get("vision_hidden_size", config.hidden_size)
        audio_dim = config.audio_config.get("audio_hidden_size", 896)
        self.vision_proj = nn.Linear(vision_dim, config.hidden_size, bias=False)
        self.audio_proj = nn.Linear(audio_dim, config.hidden_size, bias=False)
        self.norm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Gemma spezifische Embedding-Skalierung: sqrt(hidden_size)
        h = self.text_embed(input_ids) * math.sqrt(self.hidden_size)

        if modality_mask is not None:
            if vision_patches is not None:
                v_emb = self.norm(self.vision_proj(vision_patches))
                h = torch.where(modality_mask.unsqueeze(-1) == 1, v_emb, h)
            if audio_frames is not None:
                a_emb = self.norm(self.audio_proj(audio_frames))
                h = torch.where(modality_mask.unsqueeze(-1) == 2, a_emb, h)
        return h


class SynechesisHybridAttention(nn.Module):
    """
    Συνήχησις-Attention mit präziser Kanal- und Normierungs-Ausrichtung:
    - Lokale Sliding-Schichten: Gemma-MQA mit q_norm, k_norm, Sliding Window (512).
    - Globale MLA-Schichten:
      * q_proj -> [num_heads, 256]
      * kv_down -> c_kv [256] + k_rope [64]
      * kv_up(c_kv) -> k_nope [num_heads, 192] + v [num_heads, 256]
      * Zusammensetzen von k = [k_nope, k_rope] -> [num_heads, 256]
      * q_norm(q) und k_norm(k) auf vollem 256er-Raum vor RoPE!
      * RoPE auf den letzten 64 Kanälen (192..256) beider Tensoren!
      * Optionales Attention Logit Soft-Capping (config-gesteuert).
    """
    def __init__(self, config: KenosisConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim  # 256
        self.sliding_window = config.sliding_window
        self.softcapping = config.attn_logit_softcapping
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        pattern = getattr(config, "sliding_window_pattern", 6)
        self.is_global = (layer_idx % pattern == (pattern - 1))
        self.use_mla = config.use_mla and self.is_global

        # q_norm und k_norm existieren auf ALLEN Schichten
        self.q_norm = KenosisRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = KenosisRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        if self.use_mla:
            # Multi-Head Latent Attention
            self.kv_lora_rank = config.kv_lora_rank  # 256
            self.qk_nope_dim = config.qk_nope_head_dim  # 192
            self.qk_rope_dim = config.qk_rope_head_dim  # 64
            self.v_head_dim = config.v_head_dim  # 256

            # kv_down: 256 (latenter KV) + 64 (k_rope) = 320
            self.kv_down = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_dim, bias=False)
            # kv_up: expandiert 256 -> 4 * (192 + 256) = 1792
            self.kv_up = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_dim + self.v_head_dim), bias=False)
            # q_proj: 4 * 256 = 1024
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
            # o_proj: 4 * 256 -> 1152
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        else:
            # Gemma 3 Standard Sliding Window MQA
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
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        if self.use_mla:
            # 1. Q projizieren: [bsz, seq_len, num_heads, 256]
            q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)

            # 2. KV komprimieren & unkomprimieren
            kv_comp = self.kv_down(x)  # [bsz, seq_len, 320]
            c_kv = kv_comp[..., :self.kv_lora_rank]  # [bsz, seq_len, 256]
            k_rope = kv_comp[..., self.kv_lora_rank:]  # [bsz, seq_len, 64]

            kv_up_out = self.kv_up(c_kv).view(bsz, seq_len, self.num_heads, self.qk_nope_dim + self.v_head_dim)
            k_nope = kv_up_out[..., :self.qk_nope_dim]  # [bsz, seq_len, num_heads, 192]
            v = kv_up_out[..., self.qk_nope_dim:]  # [bsz, seq_len, num_heads, 256]

            # 3. K rekonstruieren auf vollen 256 Dimensionen (Nope auf 0..192, RoPE auf 192..256)
            k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
            k = torch.cat([k_nope, k_rope_expanded], dim=-1)  # [bsz, seq_len, num_heads, 256]

            # 4. Gemma-3 q_norm und k_norm auf dem vollen 256er-Raum VOR RoPE anwenden!
            q = self.q_norm(q).transpose(1, 2)  # [bsz, heads, seq_len, 256]
            k = self.k_norm(k).transpose(1, 2)  # [bsz, heads, seq_len, 256]
            v = v.transpose(1, 2)  # [bsz, heads, seq_len, 256]

            # 5. RoPE exakt auf den hinteren 64 Kanälen (192..256) anwenden
            q_nope = q[..., :self.qk_nope_dim]
            q_rope = q[..., self.qk_nope_dim:]
            k_nope = k[..., :self.qk_nope_dim]
            k_rope = k[..., self.qk_nope_dim:]

            # cos und sin haben für MLA bereits exakt die Dimension qk_rope_head_dim (64)
            q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
            k_rope = apply_rotary_pos_emb(k_rope, cos, sin)

            q = torch.cat([q_nope, q_rope], dim=-1)
            k = torch.cat([k_nope, k_rope], dim=-1)

            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        else:
            # Gemma 3 Standard MQA Pfad
            q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)

            # Gemma-3 q_norm und k_norm VOR RoPE
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)

            # RoPE auf vollen 256 Dimensionen (Gemma 3 Standard)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

            # MQA Broadcast über Query Heads
            if self.num_kv_heads < self.num_heads:
                k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

            scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling

        # Attention Logit Soft-Capping (50.0)
        if self.softcapping is not None:
            scores = self.softcapping * torch.tanh(scores / self.softcapping)

        # Masking anwenden (4D additive Maske)
        if attention_mask is not None:
            scores = scores + attention_mask

        weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(weights, v)  # [bsz, heads, seq_len, 256]
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_out)


class KenosisDecoderLayer(nn.Module):
    """
    Ein einzelner Transformer-Block des Kenosis-Logos.
    """
    def __init__(self, config: KenosisConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        pattern = getattr(config, "sliding_window_pattern", 6)
        self.is_global = (layer_idx % pattern == (pattern - 1))

        self.input_layernorm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SynechesisHybridAttention(config, layer_idx)
        self.post_attention_layernorm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = KenosisMoELayer(config)
        self.post_feedforward_layernorm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        norm_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(norm_states, cos=cos, sin=sin, attention_mask=attention_mask)
        h = residual + self.post_attention_layernorm(attn_out)

        residual = h
        ffn_in = self.pre_feedforward_layernorm(h)
        moe_out, tasis_entropy = self.moe(ffn_in)
        out = residual + self.post_feedforward_layernorm(moe_out)
        return out, tasis_entropy


class KenosisModel(nn.Module):
    """
    Transformer-Backbone mit differenzierten RoPE-Frequenzen
    und robuster 4D Causal/Sliding-Maskengenerierung.
    """
    def __init__(self, config: KenosisConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = UnifiedChoraEmbedding(config)
        self.layers = nn.ModuleList([
            KenosisDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = KenosisRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        """
        mRoPE-artiges 2D-RoPE für die Χώρα: Die unique Frequenzen werden in eine
        x- und eine y-Partie geteilt. Für Text-Koordinaten (x = y = t) reproduziert
        dies EXAKT das alte 1D-Spektrum (cat(inv_freq[:split], inv_freq[split:]) * t
        = inv_freq * t), für Vision-Patches rotiert die Frequenz-Partie in der
        Bild-Ebene — Nachbarschaft im Bild ist auch im RoPE-Spektrum Nachbarschaft.
        Normerhaltung bleibt gegeben (Paar-Rotation auf konsistenten Kanalpaaren).
        Returns: cos, sin vom Shape [bsz, 1, seq_len, head_dim].
        """
        half = head_dim // 2  # Anzahl der unique Frequenzen
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        split = half // 2
        freqs_x = coords_x.to(torch.float32).unsqueeze(-1) * inv_freq[:split].unsqueeze(0)
        freqs_y = coords_y.to(torch.float32).unsqueeze(-1) * inv_freq[split:].unsqueeze(0)
        angles = torch.cat([freqs_x, freqs_y], dim=-1)  # [bsz, seq_len, half]
        emb = torch.cat((angles, angles), dim=-1)       # [bsz, seq_len, head_dim]
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def _build_2d_coords(self, modality_mask: torch.Tensor, device: torch.device):
        """
        Koordinaten-Feld für die 2D-Positionierung: Text- und Audio-Symbole laufen
        1D weiter (x = y = t); jede kontiguierende Vision-Region (modality == 1)
        wird als Bild-Region gelesen und erhält (Zeile, Spalte)-Koordinaten mit
        Offset s (Sequenzposition der Region) — Nachbarn im Bild liegen im RoPE
        eng beieinander, statt nur 'k Wörter weiter hinten' im 1D-Faden.
        """
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
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Erstellt additive 4D-Masken:
        - global_mask: Reine Kausalmaske (+ Padding).
        - sliding_mask: Kausalmaske + Sliding Window Limit (+ Padding).
        - Χώρα-Δίπτυχο: Kontiguierende Vision-Regionen (modality == 1) sehen
          INNERHALB derselben Bild-Region bidirektional (beide Richtungen offen),
          so wie Gemma 3 / Gemini die Bild-Patches voll miteinander verschränken;
          Text bleibt strikt kausal, Bild → Zukunft bleibt unterdreieckig verboten.
        """
        # Basis-Kausalmaske: [1, 1, seq_len, seq_len]
        causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Sliding Window Maske
        sliding_window = self.config.sliding_window
        sliding_mask = causal_mask.clone()
        if sliding_window is not None:
            band_mask = torch.tril(torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype), diagonal=-sliding_window)
            sliding_mask = sliding_mask + band_mask.unsqueeze(0).unsqueeze(0)

        # Wenn HF-Attention-Mask übergeben wurde [batch_size, seq_len] (1 für Beachten, 0 für Pad)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Konvertiere [bsz, seq_len] -> additive Maske [bsz, 1, 1, seq_len]
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

        if modality_mask is not None and getattr(self.config, "vision_bidirectional", True):
            n_rows = modality_mask.shape[0]
            runs_per_row = []
            for i in range(n_rows):
                mods = modality_mask[i].tolist()
                runs, p = [], 0
                while p < seq_len:
                    if mods[p] == 1:
                        s = p
                        while p < seq_len and mods[p] == 1:
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
                        # Aufhebung der Kausalität (und des Sliding-Bands) NUR
                        # innerhalb der Bild-Region, beide Richtungen.
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
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsz, seq_len = input_ids.shape
        h = self.embed_tokens(
            input_ids=input_ids,
            modality_mask=modality_mask,
            vision_patches=vision_patches,
            audio_frames=audio_frames,
        )

        device = h.device
        dtype = h.dtype

        # 1. Differenzierte RoPE-Frequenzen berechnen
        modality_mask_2d = modality_mask if (modality_mask is not None and modality_mask.dim() == 2) else None
        coords_x = coords_y = None
        has_vision = False
        if modality_mask_2d is not None and getattr(self.config, "vision_2d_rope", True):
            coords_x, coords_y, has_vision = self._build_2d_coords(modality_mask_2d, device)
        has_vision = has_vision and coords_x is not None

        if has_vision:
            # Χώρα-2D: Vision-Patches drehen in der Bild-Ebene (Text reproduziert
            # mit (t, t) exakt das alte 1D-Spektrum).
            cos_local, sin_local = self._build_rotary_emb_2d(
                coords_x, coords_y, self.config.head_dim, self.config.rope_theta_local, device
            )
            rope_dim_global = self.config.qk_rope_head_dim if self.config.use_mla else self.config.head_dim
            cos_global, sin_global = self._build_rotary_emb_2d(
                coords_x, coords_y, rope_dim_global, self.config.rope_theta_global, device
            )
            cos_local, sin_local = cos_local.to(dtype), sin_local.to(dtype)
            cos_global, sin_global = cos_global.to(dtype), sin_global.to(dtype)
        else:
            # Lokale Schichten: theta = 10.000 (Gemma 3 Sliding Standard, 256 Dim)
            cos_local, sin_local = self._build_rotary_emb(seq_len, self.config.head_dim, self.config.rope_theta_local, device)
            cos_local, sin_local = cos_local.to(dtype), sin_local.to(dtype)

            # Globale Schichten: theta = 1.000.000 (Gemma 3 Full Standard, 64 Dim für MLA)
            rope_dim_global = self.config.qk_rope_head_dim if self.config.use_mla else self.config.head_dim
            cos_global, sin_global = self._build_rotary_emb(seq_len, rope_dim_global, self.config.rope_theta_global, device)
            cos_global, sin_global = cos_global.to(dtype), sin_global.to(dtype)

        # 2. Additive 4D-Masken vorbereiten
        global_mask, sliding_mask = self._prepare_masks(
            attention_mask, seq_len, device, dtype, modality_mask=modality_mask_2d
        )

        all_entropies = []
        for layer in self.layers:
            if layer.is_global:
                h, layer_entropy = layer(h, cos=cos_global, sin=sin_global, attention_mask=global_mask)
            else:
                h, layer_entropy = layer(h, cos=cos_local, sin=sin_local, attention_mask=sliding_mask)
            all_entropies.append(layer_entropy)

        h = self.norm(h)
        return h, all_entropies


class KenosisLogosForConditionalGeneration(PreTrainedModel, GenerationMixin):
    """
    Vollständiges Modell mit Final Logit Soft-Capping (30.0) und Generation-Schnittstellen.
    """
    config_class = KenosisConfig
    base_model_prefix = "model"
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.text_embed.weight"}

    def __init__(self, config: KenosisConfig):
        super().__init__(config)
        self.model = KenosisModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.final_softcap = config.final_logit_softcapping

        # Word embedding tying
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
        **kwargs,
    ) -> Dict[str, Any]:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
        vision_patches: Optional[torch.Tensor] = None,
        audio_frames: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        tasis_entropy_weight: float = 0.0,
        tasis_target_entropy: Optional[float] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states, all_entropies = self.model(
            input_ids=input_ids,
            modality_mask=modality_mask,
            vision_patches=vision_patches,
            audio_frames=audio_frames,
            attention_mask=attention_mask,
        )

        logits = self.lm_head(hidden_states)

        # Final Logit Soft-Capping (nur wenn in config definiert)
        if self.final_softcap is not None:
            logits = self.final_softcap * torch.tanh(logits / self.final_softcap)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        # Τάσις-Κόλλα (Kondensations-Kanal): Kopplung der Router-Entropie an die
        # Ziel-Komplexität. Spreizt den Router bei Ἀπορία (hohes Ziel), bündelt
        # bei Πεζός (niedriges Ziel). Off by default; die Destillations-Loop
        # setzt tasis_entropy_weight > 0 (Ziel per Prompt-Komplexität).
        # Außerhalb des Label-Gates, damit auch KL-Destillation (logits-basiert
        # ohne labels) den Term in outputs.loss übernehmen kann.
        if all_entropies and tasis_entropy_weight != 0.0:
            stacked_tasis = torch.stack(list(all_entropies), dim=0).float()  # [layers, bsz, seq]
            n_layers = stacked_tasis.shape[0]
            # Τάσις-Depth-Profil ω_l (Glocke über die Tiefe): Die frühen Schichten
            # sortieren lexikalisch/syntaktisch (Τάσις niedrig), die späten verengen
            # bereits auf Κρίσις-Wortkandidaten; die begriffliche Synthese (Θέα ἐν
            # Ἐνεργείᾳ) liegt in den mittleren Schichten um tasis_depth_peak.
            # Ein flaches Mittel bestraft frühe/späte Schichten für ihre Klarheit.
            layer_pos = torch.arange(n_layers, device=stacked_tasis.device, dtype=stacked_tasis.dtype)
            center = float(getattr(self.config, "tasis_depth_peak", 0.55)) * (n_layers - 1)
            sigma = max(float(getattr(self.config, "tasis_depth_sigma", 0.30)) * (n_layers - 1), 1e-6)
            layer_weights = torch.exp(-0.5 * ((layer_pos - center) / sigma) ** 2)
            layer_weights = layer_weights / layer_weights.sum().clamp(min=1e-9)
            layer_weights = layer_weights.view(n_layers, 1, 1)
            if labels is not None:
                # CE-alignierte Padding-Maske: Entropie an Position t beurteilt das
                # Vorhersagen von Token t+1; Padding (labels = -100) fließt nicht ein.
                # Zähler und Nenner tragen dieselbe Layer-Gewichtung (ω_l).
                bsz, seq_len = labels.shape
                valid_mask = torch.zeros((bsz, seq_len), device=stacked_tasis.device, dtype=stacked_tasis.dtype)
                valid_mask[:, :-1] = (labels[:, 1:] != -100).to(stacked_tasis.dtype)
                weighted_mask = valid_mask.unsqueeze(0).expand_as(stacked_tasis) * layer_weights
                mean_tasis = (stacked_tasis * weighted_mask).sum() / weighted_mask.sum().clamp(min=1e-9)
            else:
                mean_tasis = (stacked_tasis * layer_weights).sum()
            target = (
                tasis_target_entropy
                if tasis_target_entropy is not None
                else float(getattr(self.config, "tasis_entropy_target", 1.0))
            )
            tasis_loss = tasis_entropy_weight * (mean_tasis - target) ** 2
            loss = tasis_loss if loss is None else loss + tasis_loss

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
        output.router_entropies = all_entropies
        return output


KenosisLogosForCausalLM = KenosisLogosForConditionalGeneration
