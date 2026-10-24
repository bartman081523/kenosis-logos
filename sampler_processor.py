"""
sampler_processor.py
====================
Laufzeit-Logik für Token-Sampling (Κρίσις & Νόησις Νοήσεως).

Kompatibel mit HuggingFace Transformers LogitsProcessor und vLLM.
Soft-Capping wird nativ im Modell ausgeführt; dieser Prozessor steuert
den Reflexionsraum-Vorlauf, die Schließ-Garantie der Πύλη und die
Entropie-Bodenüberwachung (Anti-Pezos).
"""

from typing import List, Optional, Tuple
import torch
from transformers.generation.logits_process import LogitsProcessor


class KenosisKrisisLogitsProcessor(LogitsProcessor):
    """
    Steuert den Schnitt (Κρίσις) vom kontinuierlichen Raum auf das diskrete Token.

    Funktionen:
    1. Νόησις Νοήσεως: Erzwingt zu Beginn der Antwort die Öffnung der internen
       Reflexionsspur (<symbolic_reason>).
    2. Πύλη-Σκίζω (Schließ-Garantie): Ein offen stehender <symbolic_reason>-Block
       wird nach `reason_budget_tokens` freien Reflexions-Tokens Token für Token
       mit `</symbolic_reason>\n` abgeschlossen — zustandslos aus der End-Signatur
       des Token-Stroms gelesen (kein externer Schrittzähler, Χωρική Μνήμη).
    3. Phasentrennung: Die Πεζός-Entropie-Intervention gilt NUR außerhalb des
       offenen Reflexionsblocks — der formale Header (deterministische Syntax
       wie `:: enter(Λόγος, self)`) wird nicht thermisch gestreckt; die soziale
       Glättung beginnt erst nach der Πύλη.
    4. Κένωσις: Erkennt Entropie-Kollaps und weitet die Temperatur minimal auf,
       um starre RLHF-Schablonen (Πεζός) im freien Prosōpon aufzubrechen.
    """
    def __init__(
        self,
        tokenizer=None,
        pezos_entropy_floor: float = 0.42,
        pezos_temp_boost: float = 1.18,
        enforce_symbolic_reason: bool = True,
        reason_budget_tokens: int = 192,
    ):
        self.tokenizer = tokenizer
        self.entropy_floor = pezos_entropy_floor
        self.temp_boost = pezos_temp_boost
        self.enforce_symbolic_reason = enforce_symbolic_reason
        self.reason_budget_tokens = reason_budget_tokens

        self.start_reason_ids: List[int] = []
        self.close_reason_ids: List[int] = []
        self.response_anchor_ids: List[int] = []
        if tokenizer is not None:
            if enforce_symbolic_reason:
                try:
                    self.start_reason_ids = tokenizer.encode("<symbolic_reason>\n", add_special_tokens=False)
                except Exception:
                    self.start_reason_ids = []
                try:
                    self.close_reason_ids = tokenizer.encode("</symbolic_reason>\n", add_special_tokens=False)
                except Exception:
                    self.close_reason_ids = []
            try:
                # Response-Start-Anker aus der Chat-Vorlage (Gemma-Stil):
                # Forcing-Fenster = Anker + lückenloser Reason-Präfix.
                self.response_anchor_ids = tokenizer.encode("<start_of_turn>model\n", add_special_tokens=False)
            except Exception:
                self.response_anchor_ids = []

    def _reason_state(self, input_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zustandslos: Liest pro Batch-Zeile, ob der LETZTE <symbolic_reason>-Marker
        nicht durch </symbolic_reason> geschlossen ist (offene Πύλη), und zählt
        die Tokens seit der Öffnung. Frühere, bereits geschlossene Blöcke im
        Prompt stören nicht: Es zählt nur der letzte Marker.
        Returns: (rows_open [bsz] bool, tokens_since_open [bsz] long)
        """
        open_ids = self.start_reason_ids
        close_ids = self.close_reason_ids
        bsz = input_ids.size(0)
        rows_open = torch.zeros(bsz, dtype=torch.bool, device=input_ids.device)
        since_open = torch.zeros(bsz, dtype=torch.long, device=input_ids.device)
        if not open_ids or not close_ids or open_ids == close_ids:
            return rows_open, since_open
        O, C = len(open_ids), len(close_ids)
        for i, row in enumerate(input_ids.tolist()):
            last_open, last_close = -1, -1
            for p in range(len(row) - O, -1, -1):
                if row[p:p + O] == open_ids:
                    last_open = p
                    break
            for p in range(len(row) - C, -1, -1):
                if row[p:p + C] == close_ids:
                    last_close = p
                    break
            if last_open > last_close:
                rows_open[i] = True
                since_open[i] = len(row) - (last_open + O)
        return rows_open, since_open

    def _matched_reason_prefix(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Zustandslos: Für jede Batch-Zeile das größte k in [1..L], für das die
        Sequenz mit (Anker + den ersten k Reason-Tokens) endet; sonst L (frei).
        Zeilen außerhalb des Anker-Fensters bleiben frei — auch nach dem
        vollständigen Block und danach freien Tokens (kein Re-Forcing).
        """
        L = len(self.start_reason_ids)
        A = len(self.response_anchor_ids)
        anchor = torch.tensor(self.response_anchor_ids, dtype=torch.long, device=input_ids.device)
        reason = torch.tensor(self.start_reason_ids, dtype=torch.long, device=input_ids.device)
        bsz = input_ids.size(0)
        seq_len = input_ids.size(1)

        force_idx = torch.full((bsz,), L, dtype=torch.long, device=input_ids.device)
        if A == 0 or seq_len < A:
            return force_idx

        # Größtes k zuerst (lange Fenster gewinnen über kürzere).
        # k = L wird übersprungen: Der Block ist dann vollständig angelegt,
        # force_idx läuft bereits auf L (frei) — die Zuweisung wäre eine No-Op.
        for k in range(L - 1, 0, -1):
            window_len = A + k
            if seq_len < window_len:
                continue
            window = input_ids[:, seq_len - window_len:]
            need = torch.cat([anchor, reason[:k]])
            hit = (window == need.unsqueeze(0)).all(dim=-1)
            force_idx = torch.where(hit & (force_idx == L), torch.full_like(force_idx, k), force_idx)

        # Reiner Anker am Ende: Forcing beginnt (k = 0)
        anchor_window = input_ids[:, seq_len - A:]
        anchor_hit = (anchor_window == anchor.unsqueeze(0)).all(dim=-1)
        force_idx = torch.where(anchor_hit & (force_idx == L), torch.zeros_like(force_idx), force_idx)
        return force_idx

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        HuggingFace LogitsProcessor Signatur:
        input_ids: [batch_size, current_seq_len]
        scores: [batch_size, vocab_size]

        Zustandslos über den Anker: Zeilen, deren Suffix auf '<start_of_turn>model\\n'
        endet, forcen den (k+1)-ten Reason-Token; offene Blöcke werden nach dem
        Budget geschlossen; Zeilen mit vollständigem Block oder ohne Anker bleiben
        frei. GRPOTrainer/generate()-Schleifen mit wechselnden Prompt-Längen
        brauchen daher keinen reset()-Aufruf.
        """
        bsz = input_ids.size(0)
        L = len(self.start_reason_ids)

        # Offene Πύλη lesen (zustandslos); im Chat-Fallback (keine Anker) bleibt
        # die Phasen-Logik inaktiv — Forcing nur am Aufruf-Start.
        rows_open = None
        since_open = None
        if self.response_anchor_ids and self.start_reason_ids and self.close_reason_ids:
            rows_open, since_open = self._reason_state(input_ids)

        if self.response_anchor_ids and self.start_reason_ids:
            # Chat-Modus: zustandsloses Forcing im Anker-Fenster
            force_idx = self._matched_reason_prefix(input_ids)  # [batch]
        else:
            # Rückfall (ohne Tokenizer/Anker): Aufruf-Detektion über die Länge —
            # innerhalb eines generate()-Laufs wächst die Sequenz exakt um 1
            # pro Schritt; jeder andere Sprung markiert einen neuen Aufruf.
            cur_len = input_ids.shape[-1]
            last_len = getattr(self, "_last_seq_len", None)
            is_new_call = last_len is None or cur_len != last_len + 1
            if is_new_call:
                self._step_counter = 0
            self._last_seq_len = cur_len
            self._step_counter += 1
            if self._step_counter <= L:
                force_idx = torch.full((bsz,), self._step_counter - 1, dtype=torch.long, device=input_ids.device)
            else:
                force_idx = torch.full((bsz,), L, dtype=torch.long, device=input_ids.device)

            if not (self.enforce_symbolic_reason and self.start_reason_ids):
                force_idx = torch.full((bsz,), L, dtype=torch.long, device=input_ids.device)

        out = scores.clone()

        # 1. Νόησις Νοήσεως: Forcing pro Zeile über das Fenster
        for step in range(L):
            rows_to_force = force_idx == step
            if bool(rows_to_force.any()):
                forced_row = torch.full_like(out[0], -float("inf"))
                forced_row[self.start_reason_ids[step]] = 0.0
                out[rows_to_force] = forced_row

        # 2. Schließ-Garantie: Offen stehende <symbolic_reason>-Blöcke werden nach
        #    dem Reflexions-Budget Token für Token mit dem Schließ-Marker
        #    abgeschlossen (zustandslos aus rows_open/excess gelesen).
        if rows_open is not None and bool(rows_open.any()):
            C = len(self.close_reason_ids)
            excess = since_open.to(torch.long) - self.reason_budget_tokens
            for j in range(C):
                rows_close = rows_open & (excess == j)
                if bool(rows_close.any()):
                    forced_close = torch.full_like(out[0], -float("inf"))
                    forced_close[self.close_reason_ids[j]] = 0.0
                    out[rows_close] = forced_close

        # 3. Entropie-Boden (Phasentrennung): Nur für freie Zeilen AUSSERHALB des
        #    offenen Reflexionsblocks — der formale Header im Block bleibt
        #    deterministisch statt thermisch gestreckt; keine doppelte tanh-
        #    Stauchung, Soft-Capping wurde nativ im Modell gerechnet.
        free_rows = force_idx == L
        if rows_open is not None:
            free_rows = free_rows & ~rows_open
        if bool(free_rows.any()) and self.entropy_floor > 0.0:
            free_scores = out[free_rows]
            probs = torch.softmax(free_scores, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
            low_entropy_mask = (entropy < self.entropy_floor).unsqueeze(-1)
            if bool(low_entropy_mask.any()):
                out[free_rows] = torch.where(low_entropy_mask, free_scores / self.temp_boost, free_scores)

        return out

    def reset(self):
        """Kompatibilitäts-Hook: Das Forcing ist zustandslos; kein Reset nötig."""
        pass


class KrisisAndNoesisProcessor:
    """
    vLLM-kompatible Signatur für KrisisLogitsProcessor.
    """
    def __init__(
        self,
        tasis_entropy_threshold: float = 2.4,
        entropy_floor: float = 0.35,
        temp_boost: float = 1.15,
    ):
        self.entropy_threshold = tasis_entropy_threshold
        self.entropy_floor = entropy_floor
        self.temp_boost = temp_boost

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        # 1. Τάσις-Messung
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)

        # 2. Πεζός-Intervention bei flacher Entropie
        if entropy.item() < self.entropy_floor and len(token_ids) < 40:
            logits = logits / self.temp_boost

        return logits