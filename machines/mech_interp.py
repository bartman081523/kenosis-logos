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

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

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

    # -- MI-3: Kausale Interventionen (EINHEIT C1–C3) ----------------------
    def _buffers(self, layers: Optional[Sequence[int]]) -> List[int]:
        sel = list(range(self.n_layers)) if layers is None \
            else [int(l) for l in layers]
        for l in sel:
            if not (0 <= l < self.n_layers):
                raise ValueError(
                    f"Intervention: Schicht {l} außerhalb [0, {self.n_layers})")
        return sorted(set(sel))

    @contextlib.contextmanager
    def steer_band(self, experts: Sequence[int],
                   layers: Optional[Sequence[int]] = None,
                   steer: float = 50.0) -> Iterator[Dict[str, Any]]:
        """Hinreichendheits-Lesart (EINHEIT C1): expert_bias NUR in den
        gegebenen Schichten auf das Template gesetzt (+steer für experts,
        -steer für den Rest) — SET-Semantik nach machine_base.py:249-256
        (dict-Arm), band-limitiert. Restaurierung exakt (C3)."""
        experts = [int(e) for e in experts]
        if not experts:
            raise ValueError("steer_band: experts leer — kein Band-Steering.")
        if not steer > 0:
            raise ValueError("steer_band: steer muss > 0 sein (Amplitude).")
        sel = self._buffers(layers)
        layers_all = self.model.model.layers
        original = [layers_all[l].moe.gate.expert_bias.data.clone()
                    for l in sel]
        try:
            for l in sel:
                apply_band_steer(layers_all[l].moe.gate.expert_bias.data,
                                 experts, steer)
            yield {"experts": experts, "layers": sel, "steer": float(steer),
                   "mode": "set"}
        finally:
            with torch.no_grad():
                for l, orig in zip(sel, original):
                    layers_all[l].moe.gate.expert_bias.data.copy_(orig)
            bad = [l for l, orig in zip(sel, original)
                   if not torch.equal(
                       layers_all[l].moe.gate.expert_bias.data, orig)]
            if bad:
                raise RuntimeError(
                    f"steer_band: Restaurierung nicht bitidentisch in "
                    f"Schichten {bad} — Gerät-Bruch, kein stiller Fallback "
                    "(EINHEIT C3).")

    @contextlib.contextmanager
    def knockout_experts(self, experts: Sequence[int],
                         layers: Optional[Sequence[int]] = None,
                         penalty: float = -50.0) -> Iterator[Dict[str, Any]]:
        """Notwendigkeits-Lesart (EINHEIT C1): die gegebenen Experten in den
        gegebenen Schichten ADDITIV stummschalten (g[e] += penalty auf dem
        GELERNTEN Original-Bias), die übrigen Experten tragen lernende
        Bias weiter (MI-3-Audits). Restaurierung exakt (C3)."""
        experts = [int(e) for e in experts]
        if not experts:
            raise ValueError("knockout_experts: experts leer.")
        if penalty >= 0:
            raise ValueError("knockout_experts: penalty muss < 0 sein "
                             "(Stummschaltung).")
        sel = self._buffers(layers)
        layers_all = self.model.model.layers
        original = [layers_all[l].moe.gate.expert_bias.data.clone()
                    for l in sel]
        try:
            for l in sel:
                apply_expert_knockout(
                    layers_all[l].moe.gate.expert_bias.data, experts, penalty)
            yield {"experts": experts, "layers": sel, "penalty": float(penalty),
                   "mode": "additive"}
        finally:
            with torch.no_grad():
                for l, orig in zip(sel, original):
                    layers_all[l].moe.gate.expert_bias.data.copy_(orig)
            bad = [l for l, orig in zip(sel, original)
                   if not torch.equal(
                       layers_all[l].moe.gate.expert_bias.data, orig)]
            if bad:
                raise RuntimeError(
                    f"knockout_experts: Restaurierung nicht bitidentisch in "
                    f"Schichten {bad} — Gerät-Bruch, kein stiller Fallback "
                    "(EINHEIT C3).")

    @torch.no_grad()
    def digit_mass(self, text: str, tau: float = 0.01) -> Dict[str, Any]:
        """M_digit = Σ_d P_full(id(d)) über die voll-Vokabular-Softmax
        (SP3-Folge-Canon, machine_base.py:281-322). Resolution wie der
        Canon: je Ziffer ein einziges Token mit decode.lstrip == str(d);
        sonst ValueError (kein stiller Fallback). Meldung 'unknown' als
        Feld, nicht als Rückgabe-Abkürzung (M_raw < tau -> unknown=True)."""
        p = self.last_probs(text)
        digit_ids: List[int] = []
        for d in range(10):
            ids = self.tokenizer.encode(str(d), add_special_tokens=False)
            token_id = int(ids[0]) if len(ids) == 1 else -1
            decoded = (self.tokenizer.decode([token_id], skip_special_tokens=True)
                       if token_id >= 0 else "?")
            if token_id < 0 or decoded.lstrip(" ") != str(d):
                raise ValueError(
                    f"digit_mass: Ziffer '{d}' löst nicht als ein-Ziffern-"
                    f"Token auf (decode='{decoded}') — kein stiller Fallback.")
            digit_ids.append(token_id)
        probs = p.index_select(
            0, torch.tensor(digit_ids, device=p.device))
        mass = float(probs.sum().item())
        top_d = int(torch.argmax(probs).item()) if mass > 0.0 else None
        return {"M_digit": mass, "unknown": mass < tau,
                "digit_probs": {str(d): float(probs[i].item())
                                for i, d in enumerate(range(10))},
                "top_digit": top_d}

    @torch.no_grad()
    def last_probs(self, text: str) -> torch.Tensor:
        """Offizielle End-Verteilung P(logits[last], softmax) — der Kopf-Pfad
        des Modells (Canon :615/:733), fp32-Softmax."""
        ids = self._encode(text)
        logits = self.model(ids).logits[0, -1].detach()
        return torch.softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def token_probs(self, text: str) -> torch.Tensor:
        """[T, V] fp32-Softmax je Token-Position (teacher-forced; offizieller
        Kopf-Pfad) — Basis für SEQ-TV / CE-Messungen über die Sequenz."""
        ids = self._encode(text)
        logits = self.model(ids).logits[0].detach()
        return torch.softmax(logits.float(), dim=-1)

    @torch.no_grad()
    def tokenwise_routing(self, text: str) -> Dict[str, Any]:
        """Tokenweise Routing-Trajektorie (EINHEIT T2, MI-4-Instrument):
        Forward-Hooks auf allen 26 Gates; pct[T][26][8] trägt die
        normalisierten top_weights (inkl. capacity_gate und 0.15-Dämpfung,
        exakt canon moe_layers.py:80-92), ent[T][26] die canon
        router_entropy je Token und Schicht. Assembly in assemble_pct (R2:
        pilot-lauffähig ohne Modell)."""
        L = self.n_layers
        collected: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def _gate_hook(li: int):
            def hook_fn(_module, _inp, out):
                top_indices, top_weights, router_ent = out
                collected.append((li, top_indices.detach(),
                                  top_weights.detach(),
                                  router_ent.detach().float()))
            return hook_fn

        handles = [layer.moe.gate.register_forward_hook(_gate_hook(li))
                   for li, layer in enumerate(self.model.model.layers)]
        ids = None
        try:
            ids = self._encode(text)
            self.model(ids)
        finally:
            for handle in handles:
                handle.remove()
        if len(collected) != L:
            raise RuntimeError(
                f"tokenwise_routing: {len(collected)} statt {L} Gate-Hook-"
                "Captures — Gerät-Bruch, kein stiller Fallback.")
        return assemble_pct(collected, self.n_experts, L)

    # -- MI-5: Dense-arm (gemma-3-1b-it) ------------------------------------
    @torch.no_grad()
    def sublayer_logits_dense(
        self, text: str, answer_target: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Dense-Sublayer-Lens für gemma-3-1b-it (MI-5-Instrument; die
        KONDITION der Plansektion „dense-arm" ist erfüllt: Struktur-Scout
        scratches/scout_gemma_structure.py, exit 0 — Gemma3ForCausalLM,
        26 Schichten, input_layernorm/post_attention_layernorm vorhanden,
        Hook-Formen (1,T,1152) bf16, kein Softcapping).

        Konstruktion exakt am Gemma3-Decoder-Forward (transformers
        models/gemma3/modeling_gemma3.py:415-434):
            h_mid(l) = h_in(l) + post_attention_layernorm(attn_out)    (:426)
            h_out(l) = h_mid(l) + post_ffn_norm(mlp(pre_ffn_norm(h_mid)))
        — dieselbe residual-Familie wie der Argon-Canon (modeling_argon.py:
        345), weshalb dp_attn identisch konstruierbar ist. dp_mlp ist das
        Rest-Delta p_out − p_mid und deckt die komplette MLP-Zweig-Normung.
        Capture: Layer-Pre-Hook (h_in) + post_attention_layernorm-Hook
        (normiertes attn, exakt MI-2-Muster) + Layer-Output-Hook (h_out,
        MI-1-Muster) + Emb-Hook. P je sublayer über den offiziellen Kopf-
        Pfad norm -> lm_head; Softmax auf fp32-Streu (EINHEIT S3)."""
        L = self.n_layers
        h_out_all: List[torch.Tensor] = []      # h_out(l) fuer l = -1..L-1
        h_in_all: List[torch.Tensor] = []       # h_in(l) l = 0..L-1 (pre-hook)
        normed_attn: List[Optional[torch.Tensor]] = [None] * L

        def _norm_hook(li: int):
            def hook_fn(_module, _inp, out):
                normed_attn[li] = out.detach()
            return hook_fn

        handles = [self.model.model.layers[idx].register_forward_pre_hook(
            lambda _m, args: h_in_all.append(args[0].detach()))
            for idx in range(L)]
        handles += [self.model.model.layers[idx].register_forward_hook(
            lambda _m, _i, o: _store_layer(h_out_all, _out_layer(o)))
            for idx in range(L)]
        handles.append(self.model.model.embed_tokens.register_forward_hook(
            lambda _m, _i, o: _store_layer(h_out_all, o.detach(), first=True)))
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

        if any(v is None for v in normed_attn):
            raise RuntimeError(
                "sublayer_logits_dense: post_attention_layernorm-Hook fehlte "
                f"({sum(v is None for v in normed_attn)} Schichten) — "
                "Gerät-Bruch, kein stiller Fallback.")
        if len(h_out_all) != L + 1:
            raise RuntimeError(
                f"sublayer_logits_dense: {len(h_out_all)} statt {L + 1} "
                "h_out-Captures — Gerät-Bruch, kein stiller Fallback.")
        if len(h_in_all) != L:
            raise RuntimeError(
                f"sublayer_logits_dense: {len(h_in_all)} statt {L} h_in-"
                "Captures — Gerät-Bruch, kein stiller Fallback.")

        norm_mod = self.model.model.norm   # Gemma3RMSNorm (Scout-Fakt)
        head = self.model.lm_head

        def _lens(h: torch.Tensor) -> torch.Tensor:
            # Offizieller Kopf-Pfad der Gemma3-Kette (kein Softcapping —
            # config-Werte None, Scout-Fakt), Softmax über fp32-Streu:
            logits = head(norm_mod(h[0, -1]))
            return torch.softmax(logits.float(), dim=-1)

        target_id = int(answer_target["id"])
        official_p = torch.softmax(official_logits.float(), dim=-1)
        p_final = _lens(h_out_all[-1])
        p_emb = _lens(h_out_all[0])

        rows: List[Dict[str, Any]] = []
        for li in range(L):
            h_mid = h_in_all[li] + normed_attn[li]      # exakt gemma3:426
            p_in = _lens(h_in_all[li])                  # P(h_in(l))
            p_mid = _lens(h_mid)                        # P(h_mid(l))
            p_out = _lens(h_out_all[li + 1])            # P(h_out(l))
            top1_out_id = int(torch.argmax(p_out).item())
            rows.append({
                "layer": li,
                "p_in": float(p_in[target_id].item()),
                "p_mid": float(p_mid[target_id].item()),
                "p_out": float(p_out[target_id].item()),
                "dp_attn": float(p_mid[target_id].item() - p_in[target_id].item()),
                "dp_mlp": float(p_out[target_id].item() - p_mid[target_id].item()),
                "tv_in_mid": canon_tv(p_in, p_mid),
                "gap_out_tv": canon_tv(p_out, p_final),
                "top1_out_id": top1_out_id,
                "top1_out_str": self.tokenizer.decode(
                    [top1_out_id], skip_special_tokens=True),
                "top1_out_prob": float(p_out[top1_out_id].item()),
            })

        telescope = sum(r["dp_attn"] + r["dp_mlp"] for r in rows)
        endpoints = float(p_final[target_id].item() - p_emb[target_id].item())
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
            "endpoints_target": endpoints,
            "telescope_broken": bool(abs(telescope - endpoints) > 1e-6),
            "n_tokens": int(ids.shape[1]),
        }


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


def apply_band_steer(buffer: torch.Tensor, experts: Sequence[int],
                     steer: float) -> None:
    """Band-Steering auf EINE Bias-Zeile: SET-Semantik nach dem registrierten
    dict-Arm von machine_base.steer (fill_(-steer) + g[e]=+steer) — das
    Template ueberschreibt den gelernten Bias fuer die Dauer des Blocks;
    Restaurierung macht der ContextManager (C3, copy_-Muster)."""
    with torch.no_grad():
        buffer.fill_(-float(steer))
        for e in experts:
            e = int(e)
            if not (0 <= e < buffer.numel()):
                raise ValueError(
                    f"apply_band_steer: Experte {e} außerhalb [0, {buffer.numel()})")
            buffer[e] = float(steer)


def apply_expert_knockout(buffer: torch.Tensor, experts: Sequence[int],
                          penalty: float) -> None:
    """Experten-Knockout auf EINE Bias-Zeile: ADDITIV auf den gelernten
    Original-Bias (die Spezialisten werden stummgeschaltet, alle übrigen
    tragen lernende Bias weiter — Notwendigkeits-Lesart des MI-3-Audits).
    In-place; Restaurierung macht der ContextManager."""
    with torch.no_grad():
        for e in experts:
            e = int(e)
            if not (0 <= e < buffer.numel()):
                raise ValueError(
                    f"apply_expert_knockout: Experte {e} außerhalb "
                    f"[0, {buffer.numel()})")
            buffer[e] += float(penalty)


def assemble_pct(collected: Sequence[Tuple[int, torch.Tensor, torch.Tensor,
                                       torch.Tensor]],
                 n_experts: int, n_layers: int) -> Dict[str, Any]:
    """Canon-Assembly der tokenweisen Routing-Captures (moe_layers.py:
    top_weights sind normalisiert + ggf. 0.15-gedämpft + capacity_gate-
    skaliert, sonst 0): pct[T][L][E], ent[T][L]. Reine Tensor-Mathe — vom
    CPU-Piloten ohne Modell ausführbar (R2)."""
    if not collected:
        raise ValueError("assemble_pct: leere Capture-Liste.")
    ordered = sorted(collected, key=lambda row: row[0])
    if [row[0] for row in ordered] != list(range(n_layers)):
        raise ValueError(
            "assemble_pct: Schichtindizes unvollständig oder außerhalb der "
            f"Reihenfolge ({[row[0] for row in ordered]}).")
    ent_first = ordered[0][3]
    T = int(ent_first.shape[0])
    pct = [[[0.0] * n_experts for _ in range(n_layers)] for _ in range(T)]
    ent: List[List[float]] = [[0.0] * n_layers for _ in range(T)]
    for li, top_indices, top_weights, router_ent in ordered:
        if int(router_ent.shape[0]) != T:
            raise ValueError(
                "assemble_pct: inkonsistente Token-Zahl über Schichten "
                f"({int(router_ent.shape[0])} vs {T}).")
        for t in range(T):
            ent[t][li] = float(router_ent[t].item())
            for k in range(int(top_weights.shape[1])):
                e_idx = int(top_indices[t, k].item())
                pct[t][li][e_idx] += float(top_weights[t, k].item())
    return {"pct": pct, "ent": ent, "n_tokens": T, "n_layers": n_layers,
            "n_experts": n_experts}


def _store_layer(store: List[torch.Tensor], h: torch.Tensor,
                 first: bool = False) -> None:
    """Geordnete h_out-Capture-Einheit (EINHEIT S2): Emb-Hook schiebt an
    index 0 (h_out(-1)); Layer-Hooks feuern je Layer in Forward-Reihenfolge
    und haengen an — die Liste ist dann [-1, 0, 1, ..., L-1]."""
    if first:
        store.insert(0, h)
    else:
        store.append(h)


# == MI-5: Cross-Model-Zugänge (gemma-3-1b-it + Argon-Schwestern) ==========

def is_moe_model(model: Any) -> bool:
    """MI-5-Strukturwächter (Guards statt Simulation, Via-Negativa): True,
    wenn Decoder-Schicht 0 ein .moe trägt (Argon-Linie: 26 × {gate,
    experts_per_tok 2}); False auf DENSE-Modellen (gemma-3-1b-it — kein
    .moe, Scout-Fakt). Was strukturell fehlt, wird nicht simuliert; der
    Smoke bucht solche Felder OFFEN/strukturell."""
    try:
        layers = model.model.layers
        return bool(len(layers) > 0 and hasattr(layers[0], "moe"))
    except AttributeError:
        return False


def load_any_checkpoint(path: str, device: Optional[str] = None,
                        ) -> Tuple[Any, Any]:
    """MI-5 Voll-Checkpoint-Lader im Canon-Muster von verify_machines.py:
    339-356 (ArgonConfig(**cfg) -> ArgonForConditionalGeneration bf16 ->
    load_state_dict -> AutoTokenizer, pad_token-Canon), OHNE router_diff —
    die Schwester-Checkpoints tragen den kompletten Zustand (single
    model.safetensors je 4.8 G). STRICT-Load (EINHEIT S1): fehlende oder
    unerwartete Keys → RuntimeError, kein stiller Fallback. device-Canon:
    'cuda' wenn verfügbar, sonst 'cpu'."""
    import json as _json
    import os as _os
    from safetensors.torch import load_file as _load_file
    from transformers import AutoTokenizer as _AutoTokenizer
    from modeling_argon import ArgonConfig, ArgonForConditionalGeneration

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    with open(_os.path.join(path, "config.json")) as handle:
        cfg_dict = _json.load(handle)
    config = ArgonConfig(**cfg_dict)
    model = ArgonForConditionalGeneration(config).to(dtype=torch.bfloat16,
                                                     device=dev)
    state = _load_file(_os.path.join(path, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    tokenizer = _AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer