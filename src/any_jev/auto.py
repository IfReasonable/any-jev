"""Convenient Hugging Face model loaders and multi-question decisions."""

from collections.abc import Mapping, Sequence

from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

from any_jev.decisions.binary import Binary, BinaryResult
from any_jev.decisions.choice import Choice, ChoiceResult
from any_jev.errors import UnsupportedModelError
from any_jev.scoring.text import TextCandidateScorer
from any_jev.scoring.vision import VisionCandidateScorer


class AutoCandidateScorer(TextCandidateScorer):
    """A CausalLM candidate scorer loaded from a Hub ID or local directory."""

    @classmethod
    def from_pretrained(cls, model_id: str, **model_kwargs: object) -> "AutoCandidateScorer":
        """Load a tokenizer and decoder-only model with Transformers Auto classes."""
        tokenizer_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
        }
        tokenizer_kwargs = {
            key: value for key, value in model_kwargs.items() if key in tokenizer_keys
        }
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        except ValueError as error:
            raise UnsupportedModelError(f"{model_id!r} is not a supported CausalLM") from error
        return cls(model, tokenizer)


class AutoVisionCandidateScorer(VisionCandidateScorer):
    """A multimodal candidate scorer loaded from a Hub ID or local directory."""

    @classmethod
    def from_pretrained(cls, model_id: str, **model_kwargs: object) -> "AutoVisionCandidateScorer":
        """Load a VLM and its processor with Transformers Auto classes."""
        processor_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "revision",
            "subfolder",
            "token",
            "trust_remote_code",
        }
        processor_kwargs = {
            key: value for key, value in model_kwargs.items() if key in processor_keys
        }
        processor = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
        except ValueError as error:
            raise UnsupportedModelError(
                f"{model_id!r} is not a supported image-text model"
            ) from error
        return cls(model, processor)


class AutoDecider:
    """Apply independent Choice and Binary questions to the same state."""

    def __init__(self, scorer: TextCandidateScorer | VisionCandidateScorer) -> None:
        self.scorer = scorer

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, vision: bool = False, **model_kwargs: object
    ) -> "AutoDecider":
        """Load an LLM or, with ``vision=True``, a VLM and processor."""
        if vision:
            return cls(AutoVisionCandidateScorer.from_pretrained(model_id, **model_kwargs))
        return cls(AutoCandidateScorer.from_pretrained(model_id, **model_kwargs))

    def decide(
        self,
        state: str,
        questions: Mapping[str, Choice | Binary],
        *,
        images: Sequence[object] | None = None,
    ) -> dict[str, ChoiceResult | BinaryResult]:
        """Score independent questions, optionally conditioned on VLM images."""
        if not isinstance(state, str):
            raise TypeError("state must be a string")
        is_vision = isinstance(self.scorer, VisionCandidateScorer)
        if is_vision and not images:
            raise ValueError("images are required for a VLM decider")
        if not is_vision and images is not None:
            raise ValueError("images require a VLM decider (vision=True)")
        result = {}
        for name, question in questions.items():
            if not isinstance(question, (Choice, Binary)):
                raise TypeError(f"question {name!r} must be Choice or Binary")
            if is_vision:
                result[name] = question.decide_vision(state, images, self.scorer)
            else:
                result[name] = question.decide(state, self.scorer)
        return result
