"""Candidate scoring and typed decisions for Hugging Face causal language models."""

from any_jev.auto import AutoCandidateScorer, AutoDecider, AutoVisionCandidateScorer
from any_jev.decisions.binary import Binary, BinaryResult
from any_jev.decisions.choice import Choice, ChoiceResult
from any_jev.errors import CandidateBoundaryError, UnsupportedModelError
from any_jev.scoring.outputs import CandidateScores

__version__ = "0.2.0"

__all__ = [
    "AutoCandidateScorer",
    "AutoDecider",
    "AutoVisionCandidateScorer",
    "Binary",
    "BinaryResult",
    "CandidateBoundaryError",
    "CandidateScores",
    "Choice",
    "ChoiceResult",
    "UnsupportedModelError",
]
