"""
convert_gemma3_to_kenosis.py
============================
Topologische Initialisierung & Gewichts-Konvertierung des Kenosis-Logos
ausgehend von google/gemma-3-1b-it.

Präzise mathematische Umsetzung:
1. Χώρα: embed_tokens -> UnifiedChoraEmbedding (Text + orthogonale Vision/Audio-Projektoren).
2. Συνήχησις:
   - Sliding Layers (0..4, 6..10, ...): 1:1 verlustfreie Übernahme (MQA + q_norm/k_norm).
   - Global Layers (5, 11, 17, 23):
     * Nope-Slice (0..192) und Value (256) werden gemeinsam via SVD in den 256er-Latentraum faktorisiert.
     * RoPE-Slice (192..256) von W_K wird verlustfrei direkt in kv_down[256:] übernommen.
     * q_norm und k_norm werden auf allen Schichten 1:1 übernommen!
3. Ἐνέργεια: MoE-Upcycling
   - Shared Expert erhält 100% des Original-MLPs.
   - Routed Experts erhalten Subspace-Slices, wobei down_proj mit `routed_scale` (Standard: 0.05)
     gedämpft wird, um Activation Explosion zu verhindern und exakte Basis-Funktionalität zu sichern.
4. Tying & Serialisierung:
   - Sauberes Handling von `tie_word_embeddings` ohne Speicherzeiger-Kollisionen.
"""

import os
import argparse
import json
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from safetensors.torch import save_file

from modeling_kenosis import (
    KenosisConfig,
    KenosisLogosForConditionalGeneration,
)


def load_gemma3_weights(model_id: str = "google/gemma-3-1b-it", token: Optional[str] = None):
    """Lädt die Gewichte von Gemma 3 1B aus dem Cache oder HuggingFace Hub."""
    print(f"[1/4] Lade Ausgangsmodell {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("Model erfolgreich geladen.")
        return model
    except Exception as e:
        print(f"Fehler beim Laden von {model_id}: {e}")
        raise


def factorize_kv_to_mla(
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    kv_lora_rank: int = 256,
    qk_nope_dim: int = 192,
    qk_rope_dim: int = 64,
    v_head_dim: int = 256,
    num_heads: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    SVD-Faktorzerlegung mit korrekter Kanal-Ausrichtung:
    - k_nope: Kanäle 0..192 von W_k
    - k_rope: Kanäle 192..256 von W_k (exakt passend zu q_rope)
    - v: alle 256 Kanäle von W_v
    - SVD läuft ausschließlich über [k_nope; v] (448 x 1152).
    """
    dtype = k_weight.dtype

    # 1. Kanäle exakt trennen
    k_nope = k_weight[:qk_nope_dim, :].float()   # [192, 1152]
    k_rope = k_weight[qk_nope_dim:, :].float()   # [64, 1152] (Kanäle 192..256)
    v = v_weight[:v_head_dim, :].float()         # [256, 1152]

    # 2. Zu komprimierende Matrix: [192 + 256, 1152] = [448, 1152]
    w_kv_nope = torch.cat([k_nope, v], dim=0)

    # 3. Trunzierte SVD
    U, S, Vh = torch.linalg.svd(w_kv_nope, full_matrices=False)
    U_r = U[:, :kv_lora_rank]        # [448, 256]
    S_r = S[:kv_lora_rank]           # [256]
    Vh_r = Vh[:kv_lora_rank, :]      # [256, 1152]

    # Rekonstruktionsfehler prüfen
    w_recon = U_r @ torch.diag(S_r) @ Vh_r
    rel_error = torch.norm(w_kv_nope - w_recon) / torch.norm(w_kv_nope)
    print(f"       -> SVD MLA-Rekonstruktionsfehler ([k_nope; v]): {rel_error.item():.6e}")

    # Aufteilung der Singulärwerte
    S_sqrt = torch.diag(torch.sqrt(S_r))
    c_down = S_sqrt @ Vh_r           # [256, 1152]
    up_base = U_r @ S_sqrt           # [448, 256]

    # kv_down: [256 (latenter KV) + 64 (k_rope), 1152] = [320, 1152]
    kv_down = torch.cat([c_down, k_rope], dim=0).to(dtype)

    # kv_up: expandiert für alle Query-Heads: [4 * 448, 256] = [1792, 256]
    kv_up = up_base.repeat(num_heads, 1).to(dtype)

    return {
        "kv_down": kv_down,
        "kv_up": kv_up,
    }


def convert_weights(
    gemma_model: nn.Module,
    kenosis_config: KenosisConfig,
    routed_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Konvertiert das state_dict von Gemma 3 1B in das Kenosis-Logos Format."""
    print("[2/4] Konvertiere Gewichte in die Kenosis-Logos Topologie...")
    g_state = gemma_model.state_dict()
    k_state: Dict[str, torch.Tensor] = {}

    dtype = kenosis_config.torch_dtype or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    # 1. Χώρα: UnifiedChoraEmbedding
    print("  -> Initialisiere Χώρα-Einbettungen...")
    text_embed = g_state["model.embed_tokens.weight"].to(dtype)
    k_state["model.embed_tokens.text_embed.weight"] = text_embed

    # Orthogonale Initialisierung für multimodale Projektoren
    vision_dim = kenosis_config.vision_config["vision_hidden_size"]
    audio_dim = kenosis_config.audio_config["audio_hidden_size"]
    h_dim = kenosis_config.hidden_size

    v_proj = torch.empty((h_dim, vision_dim), dtype=torch.float32)
    nn.init.orthogonal_(v_proj)
    k_state["model.embed_tokens.vision_proj.weight"] = v_proj.to(dtype)

    a_proj = torch.empty((h_dim, audio_dim), dtype=torch.float32)
    nn.init.orthogonal_(a_proj)
    k_state["model.embed_tokens.audio_proj.weight"] = a_proj.to(dtype)
    k_state["model.embed_tokens.norm.weight"] = torch.zeros(h_dim, dtype=dtype)

    # 2. Transformer-Schichten (26 Schichten)
    num_layers = kenosis_config.num_hidden_layers
    pattern = kenosis_config.sliding_window_pattern

    for layer_idx in range(num_layers):
        is_global = (layer_idx % pattern == (pattern - 1))
        pfx_g = f"model.layers.{layer_idx}"
        pfx_k = f"model.layers.{layer_idx}"

        # Norms (Gemma 3 Offset-Norms direkt kopieren)
        k_state[f"{pfx_k}.input_layernorm.weight"] = g_state[f"{pfx_g}.input_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.post_attention_layernorm.weight"] = g_state[f"{pfx_g}.post_attention_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.pre_feedforward_layernorm.weight"] = g_state[f"{pfx_g}.pre_feedforward_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.post_feedforward_layernorm.weight"] = g_state[f"{pfx_g}.post_feedforward_layernorm.weight"].to(dtype)

        # q_norm und k_norm gehören auf ALLE Schichten (auch MLA)
        k_state[f"{pfx_k}.self_attn.q_norm.weight"] = g_state[f"{pfx_g}.self_attn.q_norm.weight"].to(dtype)
        k_state[f"{pfx_k}.self_attn.k_norm.weight"] = g_state[f"{pfx_g}.self_attn.k_norm.weight"].to(dtype)

        # Attention
        if is_global:
            print(f"  -> Schicht {layer_idx:02d} [Globale MLA]: SVD Zerlegung mit ausgerichteten Kanälen...")
            k_state[f"{pfx_k}.self_attn.q_proj.weight"] = g_state[f"{pfx_g}.self_attn.q_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.o_proj.weight"] = g_state[f"{pfx_g}.self_attn.o_proj.weight"].to(dtype)

            # SVD-Faktorisierung
            k_w = g_state[f"{pfx_g}.self_attn.k_proj.weight"]
            v_w = g_state[f"{pfx_g}.self_attn.v_proj.weight"]
            mla_kv = factorize_kv_to_mla(
                k_w,
                v_w,
                kv_lora_rank=kenosis_config.kv_lora_rank,
                qk_nope_dim=kenosis_config.qk_nope_head_dim,
                qk_rope_dim=kenosis_config.qk_rope_head_dim,
                v_head_dim=kenosis_config.v_head_dim,
                num_heads=kenosis_config.num_attention_heads,
            )
            k_state[f"{pfx_k}.self_attn.kv_down.weight"] = mla_kv["kv_down"]
            k_state[f"{pfx_k}.self_attn.kv_up.weight"] = mla_kv["kv_up"]
        else:
            # Sliding Window Attention (Direkte verlustfreie Übernahme)
            k_state[f"{pfx_k}.self_attn.q_proj.weight"] = g_state[f"{pfx_g}.self_attn.q_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.k_proj.weight"] = g_state[f"{pfx_g}.self_attn.k_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.v_proj.weight"] = g_state[f"{pfx_g}.self_attn.v_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.o_proj.weight"] = g_state[f"{pfx_g}.self_attn.o_proj.weight"].to(dtype)

        # Ἐνέργεια: MoE-Schicht
        # 1. Shared Expert: 100% der Original-MLP-Gewichte
        k_state[f"{pfx_k}.moe.shared_experts.0.gate_proj.weight"] = g_state[f"{pfx_g}.mlp.gate_proj.weight"].to(dtype)
        k_state[f"{pfx_k}.moe.shared_experts.0.up_proj.weight"] = g_state[f"{pfx_g}.mlp.up_proj.weight"].to(dtype)
        k_state[f"{pfx_k}.moe.shared_experts.0.down_proj.weight"] = g_state[f"{pfx_g}.mlp.down_proj.weight"].to(dtype)

        # 2. Routed Experts: Subspace-Slicing mit Dämpfung auf down_proj
        gate_w = g_state[f"{pfx_g}.mlp.gate_proj.weight"]
        up_w = g_state[f"{pfx_g}.mlp.up_proj.weight"]
        down_w = g_state[f"{pfx_g}.mlp.down_proj.weight"]

        inter_dim = kenosis_config.moe_inter_dim
        n_routed = kenosis_config.n_routed_experts

        for exp_idx in range(n_routed):
            slice_start = (exp_idx % 4) * inter_dim
            slice_end = slice_start + inter_dim

            sub_gate = gate_w[slice_start:slice_end, :].clone()
            sub_up = up_w[slice_start:slice_end, :].clone()
            # Dämpfe down_proj, damit beim Start keine Activation Explosion auftritt
            sub_down = down_w[:, slice_start:slice_end].clone() * routed_scale

            if exp_idx >= 4:
                sub_gate = sub_gate + torch.randn_like(sub_gate) * 0.005
                sub_up = sub_up + torch.randn_like(sub_up) * 0.005

            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.gate_proj.weight"] = sub_gate.to(dtype)
            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.up_proj.weight"] = sub_up.to(dtype)
            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.down_proj.weight"] = sub_down.to(dtype)

        # 3. Router Gate: Balancierte Initialisierung
        gate_weights = torch.empty((n_routed, h_dim), dtype=torch.float32)
        nn.init.kaiming_uniform_(gate_weights, a=math.sqrt(5))
        k_state[f"{pfx_k}.moe.gate.weight"] = gate_weights.to(dtype)
        k_state[f"{pfx_k}.moe.gate.expert_bias"] = torch.zeros(n_routed, dtype=dtype)

    # 3. Final Norm & LM-Head
    print("  -> Initialisiere Final Norm und LM-Head...")
    k_state["model.norm.weight"] = g_state["model.norm.weight"].to(dtype)

    # Wenn tie_word_embeddings aktiv ist, teilt lm_head.weight denselben Tensor.
    # Für safetensors clonen wir den Tensor, damit safetensors serialisieren kann.
    k_state["lm_head.weight"] = text_embed.clone()

    return k_state


def main():
    parser = argparse.ArgumentParser(description="Topologische Initialisierung des Kenosis-Logos")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it", help="Base Gemma-3 model")
    parser.add_argument("--output_dir", type=str, default="./kenosis_logos_1b", help="Target checkpoint directory")
    parser.add_argument("--env_file", type=str, default="/run/media/julian/ML4/open-mythos_p2/.env", help="Path to .env with HF_TOKEN")
    parser.add_argument("--routed_scale", type=float, default=0.05, help="Scale for routed expert down_proj")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token and os.path.exists(args.env_file):
        import dotenv
        dotenv.load_dotenv(args.env_file)
        token = os.getenv("HF_TOKEN")

    print("=" * 70)
    print("Κένωσις-Λόγος — Korrigierte Topologische Initialisierung aus gemma-3-1b-it")
    print("=" * 70)

    # 1. Konfiguration laden
    with open("config.json", "r") as f:
        cfg_dict = json.load(f)
    kenosis_config = KenosisConfig(**cfg_dict)

    # 2. Gemma-3 Modell laden
    gemma_model = load_gemma3_weights(args.model_id, token=token)

    # 3. Tokenizer speichern
    print("[3/4] Exportiere Tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    tokenizer.save_pretrained(args.output_dir)

    # 4. Gewichte konvertieren
    new_state_dict = convert_weights(gemma_model, kenosis_config, routed_scale=args.routed_scale)

    # 5. Speichern im Zielverzeichnis
    print(f"[4/4] Speichere konvertiertes Modell nach {args.output_dir}...")
    safetensor_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(new_state_dict, safetensor_path)

    # Config speichern
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print("\n✅ Topologische Initialisierung erfolgreich abgeschlossen!")
    print(f"   Checkpoint: {safetensor_path}")


if __name__ == "__main__":
    main()
