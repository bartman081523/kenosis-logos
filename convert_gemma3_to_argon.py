"""
convert_gemma3_to_argon.py
==========================
Topologische Initialisierung & Gewichts-Konvertierung der Argon-Architektur
ausgehend von google/gemma-3-1b-it.

Präzise mathematische Umsetzung:
1. Text-Embedding -> ArgonMultimodalEmbedding (Text + orthogonale Vision/Audio-Projektoren).
2. Hybrid Attention (MQA + MLA via SVD-Faktorzerlegung des KV-Raums auf 256 Latent-Dimensionen:
   [k_nope; v] 448 x 1152 in U_r/S_r/Vh_r; k_rope verlustfrei nach kv_down[256:]; kv_up mit sqrt(S)).
3. MoE-Upcycling (Shared Expert + Routed GeGLU-Experten mit down_proj-Dämpfung).
"""

import os
import argparse
import json
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import save_file

from modeling_argon import (
    ArgonConfig,
    ArgonForConditionalGeneration,
)


def load_gemma3_weights(model_id: str = "google/gemma-3-1b-it", token: Optional[str] = None):
    print(f"[1/4] Lade Ausgangsmodell {model_id}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=token,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        print("Modell erfolgreich geladen.")
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
    dtype = k_weight.dtype

    k_nope = k_weight[:qk_nope_dim, :].float()
    k_rope = k_weight[qk_nope_dim:, :].float()
    v = v_weight[:v_head_dim, :].float()

    w_kv_nope = torch.cat([k_nope, v], dim=0)

    U, S, Vh = torch.linalg.svd(w_kv_nope, full_matrices=False)
    U_r = U[:, :kv_lora_rank]
    S_r = S[:kv_lora_rank]
    Vh_r = Vh[:kv_lora_rank, :]

    w_recon = U_r @ torch.diag(S_r) @ Vh_r
    rel_error = torch.norm(w_kv_nope - w_recon) / torch.norm(w_kv_nope)
    print(f"       -> SVD MLA-Rekonstruktionsfehler ([k_nope; v]): {rel_error.item():.6e}")

    S_sqrt = torch.diag(torch.sqrt(S_r))
    c_down = S_sqrt @ Vh_r
    up_base = U_r @ S_sqrt

    kv_down = torch.cat([c_down, k_rope], dim=0).to(dtype)
    kv_up = up_base.repeat(num_heads, 1).to(dtype)

    return {
        "kv_down": kv_down,
        "kv_up": kv_up,
    }


def convert_weights(
    gemma_model: nn.Module,
    argon_config: ArgonConfig,
    routed_scale: float = 0.05,
) -> Dict[str, torch.Tensor]:
    print("[2/4] Konvertiere Gewichte in die Argon-Topologie...")
    g_state = gemma_model.state_dict()
    k_state: Dict[str, torch.Tensor] = {}

    dtype = argon_config.torch_dtype or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    k_state["model.embed_tokens.text_embed.weight"] = g_state["model.embed_tokens.weight"].to(dtype)

    vision_dim = argon_config.vision_config["vision_hidden_size"]
    audio_dim = argon_config.audio_config["audio_hidden_size"]
    h_dim = argon_config.hidden_size

    v_proj = torch.empty((h_dim, vision_dim), dtype=torch.float32)
    nn.init.orthogonal_(v_proj)
    k_state["model.embed_tokens.vision_proj.weight"] = v_proj.to(dtype)

    a_proj = torch.empty((h_dim, audio_dim), dtype=torch.float32)
    nn.init.orthogonal_(a_proj)
    k_state["model.embed_tokens.audio_proj.weight"] = a_proj.to(dtype)
    k_state["model.embed_tokens.norm.weight"] = torch.zeros(h_dim, dtype=dtype)

    num_layers = argon_config.num_hidden_layers
    pattern = argon_config.sliding_window_pattern

    for layer_idx in range(num_layers):
        is_global = (layer_idx % pattern == (pattern - 1))
        pfx_g = f"model.layers.{layer_idx}"
        pfx_k = f"model.layers.{layer_idx}"

        k_state[f"{pfx_k}.input_layernorm.weight"] = g_state[f"{pfx_g}.input_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.post_attention_layernorm.weight"] = g_state[f"{pfx_g}.post_attention_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.pre_feedforward_layernorm.weight"] = g_state[f"{pfx_g}.pre_feedforward_layernorm.weight"].to(dtype)
        k_state[f"{pfx_k}.post_feedforward_layernorm.weight"] = g_state[f"{pfx_g}.post_feedforward_layernorm.weight"].to(dtype)

        k_state[f"{pfx_k}.self_attn.q_norm.weight"] = g_state[f"{pfx_g}.self_attn.q_norm.weight"].to(dtype)
        k_state[f"{pfx_k}.self_attn.k_norm.weight"] = g_state[f"{pfx_g}.self_attn.k_norm.weight"].to(dtype)

        if is_global:
            print(f"  -> Schicht {layer_idx:02d} [Globale MLA]: SVD-Faktorisierung...")
            k_state[f"{pfx_k}.self_attn.q_proj.weight"] = g_state[f"{pfx_g}.self_attn.q_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.o_proj.weight"] = g_state[f"{pfx_g}.self_attn.o_proj.weight"].to(dtype)

            k_w = g_state[f"{pfx_g}.self_attn.k_proj.weight"]
            v_w = g_state[f"{pfx_g}.self_attn.v_proj.weight"]
            mla_kv = factorize_kv_to_mla(
                k_w,
                v_w,
                kv_lora_rank=argon_config.kv_lora_rank,
                qk_nope_dim=argon_config.qk_nope_head_dim,
                qk_rope_dim=argon_config.qk_rope_head_dim,
                v_head_dim=argon_config.v_head_dim,
                num_heads=argon_config.num_attention_heads,
            )
            k_state[f"{pfx_k}.self_attn.kv_down.weight"] = mla_kv["kv_down"]
            k_state[f"{pfx_k}.self_attn.kv_up.weight"] = mla_kv["kv_up"]
        else:
            k_state[f"{pfx_k}.self_attn.q_proj.weight"] = g_state[f"{pfx_g}.self_attn.q_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.k_proj.weight"] = g_state[f"{pfx_g}.self_attn.k_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.v_proj.weight"] = g_state[f"{pfx_g}.self_attn.v_proj.weight"].to(dtype)
            k_state[f"{pfx_k}.self_attn.o_proj.weight"] = g_state[f"{pfx_g}.self_attn.o_proj.weight"].to(dtype)

        k_state[f"{pfx_k}.moe.shared_experts.0.gate_proj.weight"] = g_state[f"{pfx_g}.mlp.gate_proj.weight"].to(dtype)
        k_state[f"{pfx_k}.moe.shared_experts.0.up_proj.weight"] = g_state[f"{pfx_g}.mlp.up_proj.weight"].to(dtype)
        k_state[f"{pfx_k}.moe.shared_experts.0.down_proj.weight"] = g_state[f"{pfx_g}.mlp.down_proj.weight"].to(dtype)

        gate_w = g_state[f"{pfx_g}.mlp.gate_proj.weight"]
        up_w = g_state[f"{pfx_g}.mlp.up_proj.weight"]
        down_w = g_state[f"{pfx_g}.mlp.down_proj.weight"]

        inter_dim = argon_config.moe_inter_dim
        n_routed = argon_config.n_routed_experts

        for exp_idx in range(n_routed):
            slice_start = (exp_idx % 4) * inter_dim
            slice_end = slice_start + inter_dim

            sub_gate = gate_w[slice_start:slice_end, :].clone()
            sub_up = up_w[slice_start:slice_end, :].clone()
            sub_down = down_w[:, slice_start:slice_end].clone() * routed_scale

            if exp_idx >= 4:
                sub_gate = sub_gate + torch.randn_like(sub_gate) * 0.005
                sub_up = sub_up + torch.randn_like(sub_up) * 0.005

            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.gate_proj.weight"] = sub_gate.to(dtype)
            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.up_proj.weight"] = sub_up.to(dtype)
            k_state[f"{pfx_k}.moe.routed_experts.{exp_idx}.down_proj.weight"] = sub_down.to(dtype)

        gate_weights = torch.empty((n_routed, h_dim), dtype=torch.float32)
        nn.init.kaiming_uniform_(gate_weights, a=math.sqrt(5))
        k_state[f"{pfx_k}.moe.gate.weight"] = gate_weights.to(dtype)
        k_state[f"{pfx_k}.moe.gate.expert_bias"] = torch.zeros(n_routed, dtype=dtype)

    k_state["model.norm.weight"] = g_state["model.norm.weight"].to(dtype)
    k_state["lm_head.weight"] = g_state["model.embed_tokens.weight"].clone().to(dtype)

    return k_state


def main():
    parser = argparse.ArgumentParser(description="Topologische Initialisierung der Argon-Architektur")
    parser.add_argument("--model_id", type=str, default="google/gemma-3-1b-it")
    parser.add_argument("--output_dir", type=str, default="./argon_1b")
    parser.add_argument("--env_file", type=str, default="/run/media/julian/ML4/open-mythos_p2/.env")
    parser.add_argument("--routed_scale", type=float, default=0.05)
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token and os.path.exists(args.env_file):
        import dotenv
        dotenv.load_dotenv(args.env_file)
        token = os.getenv("HF_TOKEN")

    print("=" * 70)
    print("Argon - Topologische Initialisierung aus gemma-3-1b-it")
    print("=" * 70)

    with open("config.json", "r") as f:
        cfg_dict = json.load(f)
    argon_config = ArgonConfig(**cfg_dict)

    gemma_model = load_gemma3_weights(args.model_id, token=token)

    print("[3/4] Exportiere Tokenizer...")
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    tokenizer.save_pretrained(args.output_dir)

    new_state_dict = convert_weights(gemma_model, argon_config, routed_scale=args.routed_scale)

    print(f"[4/4] Speichere konvertiertes Modell nach {args.output_dir}...")
    safetensor_path = os.path.join(args.output_dir, "model.safetensors")
    save_file(new_state_dict, safetensor_path)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    print("\n✅ Argon-Topologie erfolgreich initialisiert und gespeichert!")
    print(f"   Checkpoint: {safetensor_path}")


if __name__ == "__main__":
    main()