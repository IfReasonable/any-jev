import pytest

from hf_decider import AutoDecider, Binary, Choice
from hf_decider.scoring.text import TextCandidateScorer
from tests.scoring.test_reference import PredictableLM


class CharTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=True):
        ids = [ord(char) % 11 + 1 for char in text]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


def test_choice_and_binary():
    decider = AutoDecider(TextCandidateScorer(PredictableLM(), CharTokenizer()))
    results = decider.decide(
        "Charged twice",
        {"route": Choice(["billing", "technical", "sales"]), "refund": Binary("Refund?")},
    )
    route = results["route"]
    assert list(route.probabilities) == ["billing", "technical", "sales"]
    assert list(route.logprobs) == ["billing", "technical", "sales"]
    assert sum(route.probabilities.values()) == pytest.approx(1)
    assert route.choice == list(route.probabilities)[route.index]
    refund = results["refund"]
    assert refund.probability_true + refund.probability_false == pytest.approx(1)


def test_questions_are_independent():
    class RecordingScorer(TextCandidateScorer):
        def __init__(self):
            super().__init__(PredictableLM(), CharTokenizer())
            self.contexts = []

        def score_text(self, context, candidates, *, add_special_tokens=True):
            self.contexts.append(context)
            return super().score_text(context, candidates, add_special_tokens=add_special_tokens)

    scorer = RecordingScorer()
    decider = AutoDecider(scorer)
    route = Choice(["billing", "technical"])
    decider.decide("state", {"one": Binary("First?"), "two": route})
    first_second_context = scorer.contexts[1]
    decider.decide("state", {"one": Binary("Changed?"), "two": route})
    assert scorer.contexts[3] == first_second_context


def test_choice_validation():
    with pytest.raises(ValueError, match="26"):
        Choice([str(i) for i in range(27)])
    with pytest.raises(ValueError, match="distinct"):
        Choice(["a", "a"])
