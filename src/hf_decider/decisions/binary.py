"""Binary decision convenience wrapper around Choice."""

from collections.abc import Sequence
from dataclasses import dataclass

from hf_decider.decisions.choice import Choice, ChoiceResult
from hf_decider.scoring.text import TextCandidateScorer
from hf_decider.scoring.vision import VisionCandidateScorer


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
        return self._from_choice(result)

    def decide_vision(
        self, state: str, images: Sequence[object], scorer: VisionCandidateScorer
    ) -> BinaryResult:
        """Score the same yes/no Choice with a VLM and images."""
        return self._from_choice(self._choice.decide_vision(state, images, scorer))

    @staticmethod
    def _from_choice(result: ChoiceResult) -> BinaryResult:
        return BinaryResult(
            result.choice,
            result.index,
            result.logprobs,
            result.probabilities,
            result.probabilities["yes"],
            result.probabilities["no"],
        )
