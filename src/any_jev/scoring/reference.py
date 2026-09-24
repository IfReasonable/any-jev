"""Batched full-forward teacher-forced continuation scoring."""

from collections.abc import Sequence

import torch

from any_jev.errors import UnsupportedModelError
from any_jev.scoring.outputs import CandidateScores


class ReferenceCandidateScorer:
    """Score token continuations with one full forward pass per candidate batch."""

    def __init__(self, model: torch.nn.Module, tokenizer: object | None = None) -> None:
        config = getattr(model, "config", None)
        if config is not None and getattr(config, "is_encoder_decoder", False):
            raise UnsupportedModelError("Only decoder-only CausalLM models are supported")
        self.model = model.eval()
        self.tokenizer = tokenizer

    def score_token_candidates(
        self,
        prefix_input_ids: torch.LongTensor,
        candidate_input_ids: Sequence[torch.LongTensor],
        *,
        attention_mask: torch.LongTensor | None = None,
    ) -> CandidateScores:
        """Compute log P(candidate tokens | prefix tokens) with next-token logits.

        A one-dimensional prefix may have right padding indicated by
        ``attention_mask``. Candidate tensors must be one-dimensional and nonempty.
        The prefix must contain at least one real token because its last logit
        predicts the first candidate token.
        """
        if prefix_input_ids.ndim == 2 and prefix_input_ids.shape[0] == 1:
            prefix_input_ids = prefix_input_ids[0]
        if prefix_input_ids.ndim != 1 or prefix_input_ids.dtype != torch.long:
            raise ValueError("prefix_input_ids must be a 1D LongTensor")
        if attention_mask is not None:
            if attention_mask.ndim == 2 and attention_mask.shape[0] == 1:
                attention_mask = attention_mask[0]
            if attention_mask.shape != prefix_input_ids.shape:
                raise ValueError("attention_mask must match prefix_input_ids")
            length = int(attention_mask.sum().item())
            if not torch.all(attention_mask[:length] == 1) or not torch.all(
                attention_mask[length:] == 0
            ):
                raise ValueError("attention_mask must indicate a contiguous right-padded prefix")
            prefix_input_ids = prefix_input_ids[:length]
        if prefix_input_ids.numel() == 0:
            raise ValueError("prefix must contain at least one token")
        if not candidate_input_ids:
            raise ValueError("candidates must not be empty")
        candidates = list(candidate_input_ids)
        for candidate in candidates:
            if candidate.ndim != 1 or candidate.dtype != torch.long or candidate.numel() == 0:
                raise ValueError("each candidate must be a nonempty 1D LongTensor")

        device = next(self.model.parameters()).device
        prefix = prefix_input_ids.to(device)
        candidates = [candidate.to(device) for candidate in candidates]
        prefix_length = prefix.numel()
        lengths = [prefix_length + candidate.numel() for candidate in candidates]
        max_length = max(lengths)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.model.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = 0
        batch = torch.full((len(candidates), max_length), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(batch)
        for index, candidate in enumerate(candidates):
            length = lengths[index]
            batch[index, :prefix_length] = prefix
            batch[index, prefix_length:length] = candidate
            mask[index, :length] = 1

        with torch.inference_mode():
            outputs = self.model(input_ids=batch, attention_mask=mask)
            logits = getattr(outputs, "logits", None)
            if logits is None or logits.ndim != 3:
                raise UnsupportedModelError(
                    "Model forward must return [batch, position, vocab] logits"
                )
            token_scores = []
            for index, candidate in enumerate(candidates):
                # Position prefix_length-1 predicts the first candidate token.
                selected_logits = logits[index, prefix_length - 1 : lengths[index] - 1]
                logprobs = torch.log_softmax(selected_logits.float(), dim=-1)
                token_scores.append(logprobs.gather(-1, candidate.unsqueeze(-1)).squeeze(-1))
            sequence_scores = torch.stack([score.sum() for score in token_scores])
        return CandidateScores(
            candidates=tuple(str(candidate.tolist()) for candidate in candidates),
            token_logprobs=token_scores,
            sequence_logprobs=sequence_scores,
        )
