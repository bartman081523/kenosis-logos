"""
machines/mech_interp.py
=======================
MechInterpProbe — Sonden-API für die mechanistische Interpretierbarkeit
der Argon-Maschinen (Spur B, MI-1..MI-4). Rein lesend; Interventionen
(Steering/Ablation) kommen in MI-3 über die Preset-Mechanik von
machine_base hinzu.

Canon-Anker (voll gelesen 2026-10-26):
 - moe_layers.py:56-93   ArgonEntropyAdaptiveRouter.forward
                         -> (top_indices, top_weights, router_entropy);
                         Top-K über scores + expert_bias (k = num_experts_per_tok);
                         last_counts nur im Training belegt — im Eval
                         verlassen, Capture läuft über Forward-Hooks.
 - moe_layers.py:128     gate-Aufruf innerhalb ArgonMoELayer.forward.
 - modeling_argon.py:601-611  Schicht-Schleife: (h, layer_entropy, _) = layer(...).
 - modeling_argon.py:615 h = self.norm(h)   (Final-Norm, Hook-Punkt).
 - modeling_argon.py:733 logits = self.lm_head(hidden_states)
 - modeling_argon.py:735-736 final_softcap — Checkpoint argon_1b_machines:
                         final_logit_softcapping = None  -> Lens-Pfad exakt
                         softmax(lm_head(norm(h))).
 - modeling_argon.py:188-190  ArgonHybridAttention: pattern=6,
                         is_global = (layer_idx % 6 == 5) -> globale
                         MLA-Schichten L5, L11, L17, L23 (auch DecoderLayer
                         :317, hookbar am Layer-Objekt).
 - modeling_argon.py:326-351  ArgonDecoderLayer.forward:
                         h_mid = h_in + post_attention_layernorm(attn_out)
                         (:345; attn auf input_layernorm(h_in) :336);
                         h_out = h_mid + post_feedforward_layernorm(moe_out)
                         (:350; MoE-Eingang pre_feedforward_layernorm :348).
 - train_moe_router_specialization.py:239  Kanon-Pct-Konvention:
                         ungewichtete Assignment-Bincounts über top_indices,
                         normalisiert counts/counts.sum().clamp(min=1.0)*100.
 - train_moe_router_specialization.py:292  Kanon-TV: 0.5 * (a-b).abs().sum().
 - train_moe_router_specialization.py:341  Trainings-Hook-Bereich
                         4 <= idx <= 23 (Begründung der Bänder).
 - machine_base.py:224-235  _encode/_gate_hooks-Muster (Vorgänger).

Prinzip (MechanisticCircuitMixMind v1.0, EINHEIT T1): Mittelwert ist kein
Beweis — diese API liefert FELDER (26x8-Matrizen, 26-Vektoren,
Layer-Trajektorien), nie Aggregate ohne Label. Pct-Zeilen sind
Fraction-Konvention (Summe 1.0 je Schicht); Prozent = x100.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def canon_tv(a: torch.Tensor, b: torch.Tensor) -> float:
    """Kanon-TV auf Fraction-Skala (train_moe_router_specialization:292):
    0.5 * sum(|a-b|), Wertebereich [0,1]."""
    return 0.5 * float((a - b).abs().sum().item())


def ent_mean(t: torch.Tensor) -> float:
    """Mittlere Router-Entropie eines Schicht-Slices (float-Ausgabe)."""
    return float(t.detach().float().mean().item()) if t.numel() else 0.0


def layer_bands(num_layers: int) -> Tuple[List[int], List[int]]:
    """Registrierte Bänder der Tiefen-Hypothese: Edge {0..3, L-2, L-1},
    Core {4..L-3} — Training-Hook-Bereich 4 <= idx <= 23
    (train_moe_router_specialization.py:341)."""
    edge = [0, 1, 2, 3, num_layers - 2, num_layers - 1]
    core = list(range(4, num_layers - 2))
    return edge, core


class MechInterpProbe:
    """Rein-lesende Messeinheit: layerwise Routing-Felder + Logit-Lens."""

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None,
                 max_length: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        self.max_length = int(max_length)
        self.n_layers = len(model.model.layers)
        self.n_experts = int(getattr(model.config, "n_routed_experts", 8))
        self.model.eval()

    # -- Encoding ----------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        )["input_ids"].to(self.device)

    # -- Reinheits-Fingerprint (EINHEIT S2) --------------------------------
    def fingerprint(self) -> Dict[str, Any]:
        """expert_bias + gate.weight je Schicht — torch.equal-vergleichbar."""
        return {
            "expert_bias": [layer.moe.gate.expert_bias.data.cpu().clone()
                            for layer in self.model.model.layers],
            "gate_weight": [layer.moe.gate.weight.data.cpu().clone()
                            for layer in self.model.model.layers],
        }

    def fingerprint_equal(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ok = all(torch.equal(x, y) for x, y in
                 zip(a["expert_bias"], b["expert_bias"]))
        return ok and all(torch.equal(x, y) for x, y in
                          zip(a["gate_weight"], b["gate_weight"]))

    # -- MI-1: layerwise Routing-Tomographie -------------------------------
    @torch.no_grad()
    def layerwise_routing(self, texts: Sequence[str]) -> Dict[str, Any]:
        """26x8 Assignment-Counts (Kanon-Konvention, ungewichtet) +
        mittlere Router-Entropie je Schicht über alle Tokens aller Texte."""
        L, E = self.n_layers, self.n_experts
        counts = torch.zeros((L, E), dtype=torch.float64)
        ent_sum = torch.zeros(L, dtype=torch.float64)
        ent_tok = torch.zeros(L, dtype=torch.float64)

        def _hook(layer_idx: int):
            def hook_fn(_module, _inp, out):
                top_indices, _, router_ent = out
                counts[layer_idx] += torch.bincount(
                    top_indices.reshape(-1), minlength=E).double().cpu()
                ent_sum[layer_idx] += ent_mean(router_ent) * float(router_ent.numel())
                ent_tok[layer_idx] += float(router_ent.numel())
            return hook_fn

        handles = [self.model.model.layers[idx].moe.gate.register_forward_hook(
            _hook(idx)) for idx in range(L)]
        try:
            for text in texts:
                ids = self._encode(text)
                _ = self.model(ids)
        finally:
            for handle in handles:
                handle.remove()

        row_sums = counts.sum(dim=1, keepdim=True).clamp(min=1.0)
        pct_frac = (counts / row_sums).float()
        ent = (ent_sum / ent_tok.clamp(min=1.0)).float()
        return {
            "counts_26x8": counts.float(),
            "pct_26x8": pct_frac,           # Zeilen summen 1.0 (Kanon: x100 fuer %)
            "pct_percent_26x8": pct_frac * 100.0,
            "entropy_26": ent,
            "n_tokens": int(ent_tok.sum().item()),
        }

    # -- MI-1: Logit-Lens über alle Schichten ------------------------------
    @torch.no_grad()
    def layerwise_logits(
        self, text: str, answer_targets: Optional[Dict[str, Sequence[int]]] = None,
    ) -> Dict[str, Any]:
        """Residualstrom-Lens: für jedes h in [Emb, Layer0..Layer{n-1}] der
        offizielle Kopf-Pfad norm -> lm_head (Softcap des Checkpoints
        beachten; argon_1b_machines: None) -> P; Top-1-Trajektorie +
        Antwort-Token-Wahrscheinlichkeit + Gap-TV gegen die fp32-Referenz
        der letzten Schicht. Official-bf16-Logits werden als Anker
        mitgeloggt (anchor_tv, fp32-Lens-vs-bf16-Kopf-Kanal)."""
        L = self.n_layers
        captured: List[torch.Tensor] = []

        handles = [self.model.model.layers[idx].register_forward_hook(
            lambda _m, _i, o: captured.append(_out_layer(o)))
            for idx in range(L)]
        emb_handle = self.model.model.embed_tokens.register_forward_hook(
            lambda _m, _i, o: captured.insert(0, o.detach()))
        official_logits = None
        ids = None
        try:
            ids = self._encode(text)
            outputs = self.model(ids)
            official_logits = outputs.logits[0, -1].detach()
        finally:
            for handle in handles:
                handle.remove()
            emb_handle.remove()

        norm_mod = self.model.model.norm
        head = self.model.lm_head

        def _lens(h: torch.Tensor) -> torch.Tensor:
            # Exakt der offizielle Kopf-Pfad (modeling_argon.py:615/:733,
            # bf16-Kette), Softmax über die fp32-Streuwerte der logits:
            # keine dtype-Neuerfindung; anchor_tv deckt die Parität.
            logits = head(norm_mod(h[0, -1]))
            return torch.softmax(logits.float(), dim=-1)

        p_final = _lens(captured[-1])
        official_p = torch.softmax(official_logits.float(), dim=-1)
        official_top1 = int(torch.argmax(official_p).item())

        rows: List[Dict[str, Any]] = []
        for idx, h in enumerate(captured):
            p = _lens(h)
            top1 = int(torch.argmax(p).item())
            row: Dict[str, Any] = {
                "layer": -1 if idx == 0 else idx - 1,
                "kind": "emb" if idx == 0 else "layer",
                "top1_id": top1,
                "top1_str": self.tokenizer.decode([top1], skip_special_tokens=True),
                "top1_prob": float(p[top1].item()),
                "gap_tv": canon_tv(p, p_final),
            }
            if answer_targets:
                traj: Dict[str, float] = {}
                for label, cands in answer_targets.items():
                    vals = [float(p[t].item()) for t in cands if 0 <= t < p.numel()]
                    traj[label] = max(vals) if vals else 0.0
                row["answer_prob"] = traj
            rows.append(row)

        return {
            "rows": rows,
            "p_final_top1_id": int(torch.argmax(p_final).item()),
            "p_final_top1_str": self.tokenizer.decode(
                [int(torch.argmax(p_final).item())], skip_special_tokens=True),
            "official_top1_id": official_top1,
            "official_top1_str": self.tokenizer.decode(
                [official_top1], skip_special_tokens=True),
            "official_top1_prob": float(official_p[official_top1].item()),
            "anchor_tv": canon_tv(p_final, official_p),
            "n_tokens": int(ids.shape[1]),
        }

    # -- MI-2: Sublayer-Zerlegung (Attention vs MoE) ------------------------
    @torch.no_grad()
    def sublayer_logits(self, text: str, answer_target: Dict[str, Any]) -> Dict[str, Any]:
        """Halbschichten-Lens nach modeling_argon.py:326-351:

        h_mid(l) = h_out(l-1) + post_attention_layernorm(attn_out)   (:345)
        h_out(l) = h_mid(l) + post_feedforward_layernorm(moe_out)    (:350)

        Capture exakt: forward-hook auf layer.post_attention_layernorm
        (Output = norm(attn_out), bf16) + Layer-Output-Hook (h_out, MI-1-
        Muster) + Emb-Hook (h_out(-1)). h_mid ist die ORIGINAL-bf16-Addition
        :345 — keine Re-Normierung, keine dtype-Neuerfindung (EINHEIT S3).
        P je sublayer über den offiziellen Kopf-Pfad (_lens: norm -> lm_head,
        softmax fp32). dp_attn/dp_moe auf der registrierten target-Koordinate;
        tv_in_mid = canon_tv(P_in, P_mid) als Architektur-Profil-Feld.
        """
        L = self.n_layers
        h_out_all: List[torch.Tensor] = []      # h_out(l) fuer l = -1..L-1
        normed_attn: List[Optional[torch.Tensor]] = [None] * L

        def _norm_hook(li: int):
            def hook_fn(_module, _inp, out):
                normed_attn[li] = out.detach()
            return hook_fn

        handles = [self.model.model.layers[idx].register_forward_hook(
            lambda _m, _i, o: _store_layer(h_out_all, _out_layer(o)))
            for idx in range(L)]
        emb_handle = self.model.model.embed_tokens.register_forward_hook(
            lambda _m, _i, o: _store_layer(h_out_all, o.detach(), first=True))
        handles += [self.model.model.layers[idx].post_attention_layernorm
                    .register_forward_hook(_norm_hook(idx)) for idx in range(L)]
        official_logits = None
        ids = None
        try:
            ids = self._encode(text)
            outputs = self.model(ids)
            official_logits = outputs.logits[0, -1].detach()
        finally:
            for handle in handles:
                handle.remove()
            emb_handle.remove()

        if any(v is None for v in normed_attn):
            raise RuntimeError(
                "sublayer_logits: post_attention_layernorm-Hook fehlte "
                f"({sum(v is None for v in normed_attn)} Schichten) — "
                "Gerät-Bruch, kein stiller Fallback.")
        if len(h_out_all) != L + 1:
            raise RuntimeError(
                f"sublayer_logits: {len(h_out_all)} statt {L + 1} h_out-"
                "Captures — Gerät-Bruch, kein stiller Fallback.")

        norm_mod = self.model.model.norm   # ArgonModel-Submodul (Canon :362;
        head = self.model.lm_head          # :626-634 trägt nur lm_head)

        def _lens(h: torch.Tensor) -> torch.Tensor:
            # Exakt der offizielle Kopf-Pfad (modeling_argon.py:615/:733,
            # bf16-Kette), Softmax über die fp32-Streuwerte der logits:
            logits = head(norm_mod(h[0, -1]))
            return torch.softmax(logits.float(), dim=-1)

        target_id = int(answer_target["id"])
        official_p = torch.softmax(official_logits.float(), dim=-1)
        p_final = _lens(h_out_all[-1])
        p_emb = _lens(h_out_all[0])

        rows: List[Dict[str, Any]] = []
        for li in range(L):
            h_mid = h_out_all[li] + normed_attn[li]     # exakt :345 (bf16)
            p_in = _lens(h_out_all[li])                 # P(h_out(l-1))
            p_mid = _lens(h_mid)                        # P(h_mid(l))
            p_out = _lens(h_out_all[li + 1])            # P(h_out(l))
            top1_out_id = int(torch.argmax(p_out).item())
            layer = self.model.model.layers[li]
            rows.append({
                "layer": li,
                "is_global": bool(layer.is_global),
                "p_in": float(p_in[target_id].item()),
                "p_mid": float(p_mid[target_id].item()),
                "p_out": float(p_out[target_id].item()),
                "dp_attn": float(p_mid[target_id].item() - p_in[target_id].item()),
                "dp_moe": float(p_out[target_id].item() - p_mid[target_id].item()),
                "tv_in_mid": canon_tv(p_in, p_mid),
                # Registered gap-Lesart (MI-1-Definition, canon_tv): Lens-TV
                # der Schicht-p_out-Verteilung gegen die finale Verteilung
                # p_final (h_out(25)) — Roh-Feld für die Tail-Kontraktion.
                "gap_out_tv": canon_tv(p_out, p_final),
                "top1_out_id": top1_out_id,
                "top1_out_str": self.tokenizer.decode(
                    [top1_out_id], skip_special_tokens=True),
                "top1_out_prob": float(p_out[top1_out_id].item()),
            })

        telescope = sum(r["dp_attn"] + r["dp_moe"] for r in rows)
        return {
            "rows": rows,
            "target_id": target_id,
            "target_str": answer_target.get("str", ""),
            "p_emb": float(p_emb[target_id].item()),
            "p_final": float(p_final[target_id].item()),
            "p_final_top1_id": int(torch.argmax(p_final).item()),
            "official_top1_id": int(torch.argmax(official_p).item()),
            "anchor_tv": canon_tv(p_final, official_p),
            "telescope_target": telescope,
            "endpoints_target": float(p_final[target_id].item() - p_emb[target_id].item()),
            "telescope_broken": bool(abs(telescope - (p_final[target_id].item()
                                                      - p_emb[target_id].item())) > 1e-6),
            "n_tokens": int(ids.shape[1]),
        }

    # -- Kristallisations-Finder (EINHEIT T2: Trajektorie statt Mittelwert) --
    def crystallization_start(self, rows: Sequence[Dict[str, Any]]) -> Optional[int]:
        """Kleinsten stabilen Layer-Start lesen (Modulfunktion)."""
        return crystallization_start(rows)


CRYSTAL_MIN_TAIL = 2  # Registrierung: stabiler TAIL >= 2 Schichten gilt als Fenster


def crystallization_start(rows: Sequence[Dict[str, Any]],
                          min_tail: int = CRYSTAL_MIN_TAIL) -> Optional[int]:
    """Kleinster Layer-Index, ab dem top1 stabil bis zum ENDE bleibt und der
    stabile TAIL mindestens ``min_tail`` Schichten spannt; None, wenn kein
    stabiler Start existiert (Oszillation, Via-Negativa-Datum)."""
    layer_rows = [r for r in rows if r["kind"] == "layer"]
    if not layer_rows:
        return None
    final_top1 = layer_rows[-1]["top1_id"]
    for start_idx in range(len(layer_rows)):
        span = len(layer_rows) - start_idx
        if span < min_tail:
            break
        if all(r["top1_id"] == final_top1 for r in layer_rows[start_idx:]):
            return layer_rows[start_idx]["layer"]
    return None


def _out_layer(o: Any) -> torch.Tensor:
    """Decoder-Layer-Output (h, layer_entropy, present_kv) -> Residual h."""
    return o[0] if isinstance(o, tuple) else o.detach()


def _store_layer(store: List[torch.Tensor], h: torch.Tensor,
                 first: bool = False) -> None:
    """Geordnete h_out-Capture-Einheit (EINHEIT S2): Emb-Hook schiebt an
    index 0 (h_out(-1)); Layer-Hooks feuern je Layer in Forward-Reihenfolge
    und haengen an — die Liste ist dann [-1, 0, 1, ..., L-1]."""
    if first:
        store.insert(0, h)
    else:
        store.append(h)