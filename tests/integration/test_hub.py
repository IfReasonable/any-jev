"""Run with: pytest -m integration (requires Hub access)."""

import pytest

from hf_decider import AutoCandidateScorer, AutoDecider, Binary, Choice


@pytest.mark.integration
def test_hub_tiny_causal_lm():
    scorer = AutoCandidateScorer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    scores = scorer.score_text("The capital is", [" Paris", " London"])
    assert scores.sequence_logprobs.shape == (2,)
    assert len(scores.token_logprobs) == 2
    decisions = AutoDecider(scorer).decide(
        "Customer asks about a duplicate charge.",
        {
            "route": Choice(["billing", "technical", "sales"]),
            "refund": Binary("Is the customer requesting a refund?"),
        },
    )
    assert len(decisions["route"].probabilities) == 3
    assert 0 <= decisions["refund"].probability_true <= 1
