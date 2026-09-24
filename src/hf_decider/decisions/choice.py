"""Choice decision over two to twenty-six options."""

from collections.abc import Sequence
from dataclasses import dataclass

from hf_decider.compiler.labels import label_candidates
from hf_decider.compiler.prompt import choice_prompt
from hf_decider.decisions.base import option_scores
from hf_decider.errors import CandidateBoundaryError
from hf_decider.scoring.text import TextCandidateScorer
from hf_decider.scoring.vision import VisionCandidateScorer


@dataclass(frozen=True)
class ChoiceResult:
    """Winning option and relative probabilities among supplied options."""

    choice: str
    index: int
    logprobs: dict[str, float]
    probabilities: dict[str, float]


class Choice:
    """A question answered by choosing one of at most 26 distinct options."""

    def __init__(
        self, options: Sequence[str], instruction: str = "Which option best matches the state?"
    ) -> None:
        if not 2 <= len(options) <= 26:
            raise ValueError("Choice requires between 2 and 26 options")
        if any(not isinstance(option, str) or not option for option in options):
            raise ValueError("Choice options must be nonempty strings")
        if len(set(options)) != len(options):
            raise ValueError("Choice options must be distinct")
        if not isinstance(instruction, str) or not instruction:
            raise ValueError("instruction must be a nonempty string")
        self.options = tuple(options)
        self.instruction = instruction

    def decide(self, state: str, scorer: TextCandidateScorer) -> ChoiceResult:
        """Compile a prompt, score its labels, and select the highest score."""
        prompt = choice_prompt(state, self.instruction, self.options)
        labels = label_candidates(scorer.tokenizer, prompt, len(self.options))
        scores = scorer.score_text(prompt, labels)
        logprobs, probabilities, index = option_scores(scores, self.options)
        return ChoiceResult(self.options[index], index, logprobs, probabilities)

    def decide_vision(
        self, state: str, images: Sequence[object], scorer: VisionCandidateScorer
    ) -> ChoiceResult:
        """Score labels as VLM assistant continuations conditioned on images."""
        prompt = choice_prompt(state, self.instruction, self.options)
        for separator in (" ", "\n", "\n "):
            labels = tuple(separator + chr(65 + index) for index in range(len(self.options)))
            try:
                scores = scorer.score_image_text(prompt, images, labels)
            except CandidateBoundaryError:
                continue
            logprobs, probabilities, index = option_scores(scores, self.options)
            return ChoiceResult(self.options[index], index, logprobs, probabilities)
        raise CandidateBoundaryError("No stable token boundary for VLM option labels")
