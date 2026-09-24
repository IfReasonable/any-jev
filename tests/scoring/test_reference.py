from types import SimpleNamespace

import pytest
import torch

from any_jev import CandidateScores, UnsupportedModelError
from any_jev.scoring.reference import ReferenceCandidateScorer


class PredictableLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(is_encoder_decoder=False, pad_token_id=0)

    def forward(self, input_ids, attention_mask):
        vocab = torch.arange(12, device=input_ids.device).float()
        logits = -(vocab[None, None, :] - input_ids[:, :, None].float() - 1).abs()
        return SimpleNamespace(logits=logits + self.weight)

    def generate(self, *args, **kwargs):
        raise AssertionError("generate must not be called")


def test_first_token_uses_last_prefix_logit():
    model = PredictableLM()
    scores = ReferenceCandidateScorer(model).score_token_candidates(
        torch.tensor([2, 4]), [torch.tensor([5])]
    )
    expected = torch.log_softmax(model(torch.tensor([[2, 4]]), torch.ones(1, 2)).logits[0, 1], -1)[
        5
    ]
    torch.testing.assert_close(scores.token_logprobs[0][0], expected)
    torch.testing.assert_close(scores.sequence_logprobs[0], expected)
    assert not model.training


@pytest.mark.parametrize("candidate", [[5], [5, 6, 7], [5, 6, 7, 8, 9, 10, 11]])
def test_teacher_forcing_and_padding(candidate):
    model = PredictableLM()
    scorer = ReferenceCandidateScorer(model)
    candidates = [torch.tensor([1]), torch.tensor([3, 4, 5]), torch.tensor(candidate)]
    scores = scorer.score_token_candidates(torch.tensor([2, 4]), candidates)
    assert [len(x) for x in scores.token_logprobs] == [len(x) for x in candidates]
    for index, ids in enumerate(candidates):
        full = torch.cat((torch.tensor([2, 4]), ids))
        logits = model(full[None, :], torch.ones_like(full)[None, :]).logits[0]
        expected = torch.log_softmax(logits[1:-1], -1).gather(1, ids[:, None]).squeeze(1)
        torch.testing.assert_close(scores.token_logprobs[index], expected)
        torch.testing.assert_close(scores.sequence_logprobs[index], expected.sum())


def test_batch_matches_five_independent_reference_calls():
    scorer = ReferenceCandidateScorer(PredictableLM())
    candidates = [torch.tensor(list(range(1, length + 1))) for length in range(1, 6)]
    batch = scorer.score_token_candidates(torch.tensor([2, 4]), candidates)
    singles = torch.stack(
        [
            scorer.score_token_candidates(torch.tensor([2, 4]), [ids]).sequence_logprobs[0]
            for ids in candidates
        ]
    )
    torch.testing.assert_close(batch.sequence_logprobs, singles, rtol=1e-5, atol=1e-5)


def test_right_padded_prefix_is_trimmed():
    scorer = ReferenceCandidateScorer(PredictableLM())
    padded = scorer.score_token_candidates(
        torch.tensor([2, 4, 0, 0]), [torch.tensor([5])], attention_mask=torch.tensor([1, 1, 0, 0])
    )
    plain = scorer.score_token_candidates(torch.tensor([2, 4]), [torch.tensor([5])])
    torch.testing.assert_close(padded.sequence_logprobs, plain.sequence_logprobs)


def test_invalid_inputs():
    scorer = ReferenceCandidateScorer(PredictableLM())
    with pytest.raises(ValueError, match="candidates"):
        scorer.score_token_candidates(torch.tensor([1]), [])
    with pytest.raises(ValueError, match="candidate"):
        scorer.score_token_candidates(torch.tensor([1]), [torch.tensor([], dtype=torch.long)])
    with pytest.raises(ValueError, match="prefix"):
        scorer.score_token_candidates(torch.tensor([], dtype=torch.long), [torch.tensor([1])])
    with pytest.raises(ValueError, match="right-padded"):
        scorer.score_token_candidates(
            torch.tensor([0, 2]), [torch.tensor([1])], attention_mask=torch.tensor([0, 1])
        )


def test_probability_semantics_and_temperature():
    scores = CandidateScores(("a", "b"), [], torch.tensor([-1.0, -2.0]))
    torch.testing.assert_close(
        scores.normalized_probabilities(), torch.softmax(scores.sequence_logprobs, 0)
    )
    with pytest.raises(ValueError, match="temperature"):
        scores.normalized_probabilities(0)


def test_seq2seq_rejected():
    model = PredictableLM()
    model.config.is_encoder_decoder = True
    with pytest.raises(UnsupportedModelError):
        ReferenceCandidateScorer(model)
