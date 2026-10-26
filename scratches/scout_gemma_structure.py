"""scout_gemma_structure.py — MI-5 Instrument-Scout (S1/S2, KEINE Messung).

Prüft NUR Struktur-Fakten von google/gemma-3-1b-it lokal aus dem HF-Cache:
Layer-Attribut-Namen (moe? self_attn? mlp?), Kopf-Pfad-Module
(model.model.norm / model.lm_head / model.model.embed_tokens),
Hook-Ausgabeformen für den geplanten dense-arm (sublayer_logits_dense via
h_mid = h_in + attn_out, gemma3 pre-norm residual). Keine Top-1s, keine
Wahrscheinlichkeiten, keine digit-Massen — die Meßdati entstehen erst im
registrierten MI-5-Batterie-Lauf (P2: Registrierung vor Messung).

Run: python scratches/scout_gemma_structure.py     (GPU, venv)
"""

import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"


def main() -> int:
    facts: dict = {"model_id": MODEL_ID}

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    facts["type"] = type(model).__name__
    facts["inner_type"] = type(model.model).__name__
    facts["vocab_size"] = int(model.config.vocab_size)
    facts["tie_word_embeddings"] = bool(model.config.tie_word_embeddings)
    for k in ("final_logit_softcapping", "attn_logit_softcapping"):
        facts[f"config_{k}"] = getattr(model.config, k, None)

    layer = model.model.layers[0]
    facts["n_layers"] = len(model.model.layers)
    facts["layer_attr"] = {
        name: hasattr(layer, name)
        for name in ("moe", "self_attn", "mlp", "input_layernorm",
                     "post_attention_layernorm", "pre_feedforward_layernorm",
                     "post_feedforward_layernorm")}
    facts["submodules"] = sorted(k for k, _ in layer.named_children())

    for name, mod in (("norm", model.model.norm),
                      ("lm_head", model.lm_head),
                      ("embed_tokens", model.model.embed_tokens)):
        facts[f"has_{name}"] = mod is not None
        facts[f"type_{name}"] = type(mod).__name__

    # Hook-Formen-Verifikation (kleiner Forward, nur Shapes/dtypes):
    h_in, attn_out, mlp_out, layer_out = [], [], [], []
    handle_in = layer.register_forward_pre_hook(
        lambda _m, args: h_in.append(args[0].detach()))
    handle_attn = layer.self_attn.register_forward_hook(
        lambda _m, _i, o: attn_out.append(
            (o[0] if isinstance(o, tuple) else o).detach()))
    handle_mlp = layer.mlp.register_forward_hook(
        lambda _m, _i, o: mlp_out.append(
            (o[0] if isinstance(o, tuple) else o).detach()))
    handle_out = layer.register_forward_hook(
        lambda _m, _i, o: layer_out.append(
            (o[0] if isinstance(o, tuple) else o).detach()))
    try:
        ids = tok("Paris is the capital of", return_tensors="pt",
                  add_special_tokens=True).input_ids.to("cuda")
        with torch.no_grad():
            _ = model(ids)
    finally:
        for h in (handle_in, handle_attn, handle_mlp, handle_out):
            h.remove()

    def shape(t: torch.Tensor) -> str:
        return f"{tuple(t.shape)}/{t.dtype}"

    facts["hook_shapes"] = {
        "layer_input[0]": shape(h_in[0]),
        "self_attn_out[0]": shape(attn_out[0]),
        "mlp_out[0]": shape(mlp_out[0]),
        "layer_out[0]": shape(layer_out[0]),
        "n_captures": {"in": len(h_in), "attn": len(attn_out),
                       "mlp": len(mlp_out), "out": len(layer_out)},
    }
    facts["mid_shape_ok"] = bool(
        h_in[0].shape == attn_out[0].shape and
        h_in[0].dtype == attn_out[0].dtype and
        h_in[0].shape == layer_out[0].shape)

    print(json.dumps(facts, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())