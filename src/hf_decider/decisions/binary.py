"""Binary decision convenience wrapper around Choice."""

from dataclasses import dataclass

from hf_decider.decisions.choice import Choice
from hf_decider.scoring.text import TextCandidateScorer


@dataclass(frozen=True)
class BinaryResult:
    """Yes/no decision and relative probabilities among the two labels."""

    choice: str
    index: int
    logprobs: dict[str, float]
    probabilities: dict[str, float]
    probability_true: float
    probability_false: float


class Binary:
    """A yes/no question scored through the Choice implementation."""

    def __init__(self, instruction: str) -> None:
        self._choice = Choice(["yes", "no"], instruction=instruction)

    def decide(self, state: str, scorer: TextCandidateScorer) -> BinaryResult:
        """Score the yes/no Choice and expose named probabilities."""
        result = self._choice.decide(state, scorer)
        return BinaryResult(
            result.choice,
            result.index,
            result.logprobs,
            result.probabilities,
            result.probabilities["yes"],
            result.probabilities["no"],
        )
