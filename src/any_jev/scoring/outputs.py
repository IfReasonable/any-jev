"""Results of scoring a supplied candidate set."""

from dataclasses import dataclass

import torch


@dataclass
class CandidateScores:
    """Unnormalized conditional log probabilities for candidate continuations.

    ``probabilities`` are normalized only among the supplied candidates and are
    not calibrated probabilities of correctness.
    """

    candidates: tuple[str, ...]
    token_logprobs: list[torch.Tensor]
    sequence_logprobs: torch.Tensor

    def normalized_probabilities(self, temperature: float = 1.0) -> torch.Tensor:
        """Return a softmax over sequence scores at positive ``temperature``."""
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and greater than zero")
        return torch.softmax(self.sequence_logprobs / temperature, dim=0)
