"""Convenient Hugging Face model loaders and multi-question decisions."""

from collections.abc import Mapping

from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_decider.decisions.binary import Binary, BinaryResult
from hf_decider.decisions.choice import Choice, ChoiceResult
from hf_decider.errors import UnsupportedModelError
from hf_decider.scoring.text import TextCandidateScorer


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


class AutoDecider:
    """Apply independent Choice and Binary questions to the same state."""

    def __init__(self, scorer: TextCandidateScorer) -> None:
        self.scorer = scorer

    @classmethod
    def from_pretrained(cls, model_id: str, **model_kwargs: object) -> "AutoDecider":
        """Load the underlying candidate scorer from a Hub ID or local path."""
        return cls(AutoCandidateScorer.from_pretrained(model_id, **model_kwargs))

    def decide(
        self, state: str, questions: Mapping[str, Choice | Binary]
    ) -> dict[str, ChoiceResult | BinaryResult]:
        """Score each question independently and return typed decision results."""
        if not isinstance(state, str):
            raise TypeError("state must be a string")
        result = {}
        for name, question in questions.items():
            if not isinstance(question, (Choice, Binary)):
                raise TypeError(f"question {name!r} must be Choice or Binary")
            result[name] = question.decide(state, self.scorer)
        return result
