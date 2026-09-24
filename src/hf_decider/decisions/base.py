"""Shared decision result conversion."""

from hf_decider.scoring.outputs import CandidateScores


def option_scores(
    scores: CandidateScores, options: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float], int]:
    """Map ordered candidate scores to ordered option scores."""
    probabilities = scores.normalized_probabilities()
    index = int(probabilities.argmax().item())
    return (
        dict(zip(options, scores.sequence_logprobs.tolist(), strict=True)),
        dict(zip(options, probabilities.tolist(), strict=True)),
        index,
    )
