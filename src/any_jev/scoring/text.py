"""String continuation tokenization with a checked context boundary."""

from collections.abc import Sequence

import torch

from any_jev.errors import CandidateBoundaryError
from any_jev.scoring.outputs import CandidateScores
from any_jev.scoring.reference import ReferenceCandidateScorer


class TextCandidateScorer(ReferenceCandidateScorer):
    """Add safe string tokenization to the reference token scorer."""

    def score_text(
        self,
        context: str,
        candidates: Sequence[str],
        *,
        add_special_tokens: bool = True,
    ) -> CandidateScores:
        """Score strings only when context tokens survive concatenation unchanged."""
        if self.tokenizer is None:
            raise ValueError("score_text requires a tokenizer")
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not candidates:
            raise ValueError("candidates must not be empty")
        if any(not isinstance(candidate, str) or not candidate for candidate in candidates):
            raise ValueError("each candidate must be a nonempty string")
        prefix_ids = self.tokenizer(context, add_special_tokens=add_special_tokens)["input_ids"]
        if not prefix_ids:
            raise ValueError("context tokenization is empty; provide a nonempty token prefix")
        continuation_ids = []
        for candidate in candidates:
            full_ids = self.tokenizer(context + candidate, add_special_tokens=add_special_tokens)[
                "input_ids"
            ]
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise CandidateBoundaryError(
                    f"Token boundary changed for candidate {candidate!r}; try a leading space or "
                    "newline, or use score_token_candidates()"
                )
            ids = full_ids[len(prefix_ids) :]
            if not ids:
                raise ValueError(f"candidate {candidate!r} produced no continuation tokens")
            continuation_ids.append(torch.tensor(ids, dtype=torch.long))
        scores = self.score_token_candidates(
            torch.tensor(prefix_ids, dtype=torch.long), continuation_ids
        )
        scores.candidates = tuple(candidates)
        return scores
