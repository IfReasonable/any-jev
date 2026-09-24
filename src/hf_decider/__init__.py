"""Candidate scoring and typed decisions for Hugging Face causal language models."""

from hf_decider.auto import AutoCandidateScorer, AutoDecider
from hf_decider.decisions.binary import Binary, BinaryResult
from hf_decider.decisions.choice import Choice, ChoiceResult
from hf_decider.errors import CandidateBoundaryError, UnsupportedModelError
from hf_decider.scoring.outputs import CandidateScores

__version__ = "0.1.0"

__all__ = [
    "AutoCandidateScorer",
    "AutoDecider",
    "Binary",
    "BinaryResult",
    "CandidateBoundaryError",
    "CandidateScores",
    "Choice",
    "ChoiceResult",
    "UnsupportedModelError",
]
