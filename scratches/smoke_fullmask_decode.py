"""
smoke_fullmask_decode.py — Verifikation des Vollmasken-Decode-Robustheits-Fixes.

Szenario: Prefill mit modality_mask (Vision-Patches), dann Decode-Schritte, bei
denen dem Aufruf NOCHMALS die volle Maske (Länge kv_len) als modality_mask
übergeben wird — ohne separates kv_modality_mask. Vor dem Fix wurde die
Maske (shape[1] > q_len) stillschweigend ignoriert (Embedding-Verstoß + 1D-RoPE);
nach dem Fix wird sie in modality_mask_in (q_len) und modality_mask_kv (kv_len)
gespalten und muss bitgenau dem kv_modality_mask-Pfad entsprechen.
"""

import torch

from modeling_argon import ArgonConfig, ArgonForConditionalGeneration


def mini_config() -> ArgonConfig:
    return ArgonConfig(
        vocab_size=2000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=48,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        kv_lora_rank=48,
        q_lora_rank=64,
        v_head_dim=48,
        moe_inter_dim=32,
        shared_intermediate_size=256,
        n_routed_experts=8,
        num_experts_per_tok=2,
        sliding_window=512,
        sliding_window_pattern=6,
        tie_word_embeddings=True,
    )


@torch.no_grad()
def main() -> None:
    torch.manual_seed(0)
    model = ArgonForConditionalGeneration(mini_config()).eval()
    Pv = 8
    mmv = torch.zeros((1, Pv), dtype=torch.long)
    mmv[0, 2:6] = 1  # 4 Vision-Patches
    visv = torch.randn(1, Pv, model.config.vision_config["vision_hidden_size"]) * 0.5
    ids = torch.randint(100, 1500, (1, Pv + 2))  # zwei Text-Gap-Tokens + Text

    # Referenz: Prefill + zwei Decode-Schritte mit kv_modality_mask (getesteter Pfad)
    pre = model(ids[:, :Pv], modality_mask=mmv, vision_patches=visv[:, :Pv], use_cache=True)
    reference_ids = ids.clone()
    ref_logits = []
    past_ref = pre.past_key_values
    full_mask = torch.cat([mmv, torch.zeros((1, 2), dtype=torch.long)], dim=1)
    full_vis = torch.cat([visv, torch.zeros(1, 2, visv.shape[2])], dim=1)
    for step in range(2):
        T = Pv + step
        out = model(reference_ids[:, T:T + 1], past_key_values=past_ref, use_cache=True,
                    kv_modality_mask=full_mask[:, :T + 1])
        ref_logits.append(out.logits[:, -1, :])
        past_ref = out.past_key_values

    # Kandidat: identische Cache-Spur, aber VOLL-Maske als modality_mask übergeben
    pre2 = model(ids[:, :Pv], modality_mask=mmv, vision_patches=visv[:, :Pv], use_cache=True)
    cand_logits = []
    past_cand = pre2.past_key_values
    for step in range(2):
        T = Pv + step
        out = model(ids[:, T:T + 1], past_key_values=past_cand, use_cache=True,
                    modality_mask=full_mask[:, :T + 1])
        cand_logits.append(out.logits[:, -1, :])
        past_cand = out.past_key_values

    for step, (r, c) in enumerate(zip(ref_logits, cand_logits)):
        d = (r - c).abs().max().item()
        assert d < 1e-4, f"Vollmasken-Decode weicht vom kv_modality_mask-Zweig ab (Step {step}): {d:.2e}"
        print(f"  Decode-Step {step +1}: max|Δ| gegen kv_modality_mask-Pfad = {d:.2e}")

    # Ohne jede Masken-Info muss der Fehlerpfad sichtbar bleiben (1D-Rekalkulation)
    pre3 = model(ids[:, :Pv], modality_mask=mmv, vision_patches=visv[:, :Pv], use_cache=True)
    plain = model(ids[:, Pv:Pv + 1], past_key_values=pre3.past_key_values, use_cache=True)
    d = (plain.logits[:, -1, :] - ref_logits[0]).abs().max().item()
    assert d > 1e-3, f"Maskenloser Pfad identisch mit 2D-Pfad — Vision-RoPE-Verlust unsichtbar: {d:.2e}"
    print(f"  Maskenloser Pfad (1D-Rekalkulation): Δ {d:.2e} (erwartet groß — Vision-Grid geht verloren)")

    print("SMOKE VOLLMASKEN-DECODE: ALLE ASSERTS GRÜN")


if __name__ == "__main__":
    main()