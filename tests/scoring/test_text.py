import pytest

from any_jev import CandidateBoundaryError
from any_jev.scoring.text import TextCandidateScorer
from tests.scoring.test_reference import PredictableLM


class PairTokenizer:
    pad_token_id = 0

    def __call__(self, text, *, add_special_tokens=True):
        ids = [sum(map(ord, text[index : index + 2])) % 11 + 1 for index in range(0, len(text), 2)]
        if add_special_tokens:
            ids.insert(0, 1)
        return {"input_ids": ids}


def test_boundary_error_and_safe_continuation():
    scorer = TextCandidateScorer(PredictableLM(), PairTokenizer())
    with pytest.raises(CandidateBoundaryError, match="leading space"):
        scorer.score_text("a", ["b"])
    scores = scorer.score_text("ab", ["cd", "efgh"])
    assert scores.candidates == ("cd", "efgh")
    assert [len(x) for x in scores.token_logprobs] == [1, 2]


def test_empty_text_inputs():
    scorer = TextCandidateScorer(PredictableLM(), PairTokenizer())
    with pytest.raises(ValueError, match="candidates"):
        scorer.score_text("ab", [])
    with pytest.raises(ValueError, match="candidate"):
        scorer.score_text("ab", [""])
    with pytest.raises(ValueError, match="context tokenization"):
        scorer.score_text("", ["ab"], add_special_tokens=False)
