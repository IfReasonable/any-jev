"""Offline VLM tests with a processor and image-dependent logits."""

from types import SimpleNamespace

import pytest
import torch

from hf_decider import AutoDecider, Binary, CandidateBoundaryError, Choice
from hf_decider.scoring.vision import VisionCandidateScorer


class FakeVisionProcessor:
    tokenizer = object()

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt=False,
        continue_final_message=False,
        tokenize,
        return_dict,
        return_tensors,
    ):
        assert tokenize and return_dict and return_tensors == "pt"
        assert add_generation_prompt != continue_final_message
        image = messages[0]["content"][0]["image"]
        prefix = [1, 2, 3]
        if continue_final_message:
            candidate = messages[-1]["content"][0]["text"]
            if candidate.startswith("!"):
                prefix[-1] = 4
            suffix = [ord(char) % 20 + 1 for char in candidate]
        else:
            suffix = []
        ids = torch.tensor([prefix + suffix])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "pixel_values": torch.tensor([[float(image)]]),
            "image_grid_thw": torch.tensor([[1, 1, 1]]),
        }


class FakeVisionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        assert attention_mask.shape == input_ids.shape
        assert image_grid_thw.shape == (1, 3)
        vocab = torch.arange(32).float()
        logits = -(
            vocab[None, None, :] - input_ids[:, :, None].float() - pixel_values[:, None, :]
        ).abs()
        return SimpleNamespace(logits=logits + self.weight)

    def generate(self, *args, **kwargs):
        raise AssertionError("generate must not be called")


def test_image_conditioned_teacher_forcing_and_padding_free_scores():
    model = FakeVisionModel()
    scorer = VisionCandidateScorer(model, FakeVisionProcessor())
    scores = scorer.score_image_text("What is shown?", [2], [" A", " BCDE"])
    assert not model.training
    assert [len(x) for x in scores.token_logprobs] == [2, 5]
    assert scores.candidates == (" A", " BCDE")
    ids = torch.tensor([[1, 2, 3, ord(" ") % 20 + 1, ord("A") % 20 + 1]])
    logits = model(
        ids, torch.ones_like(ids), torch.tensor([[2.0]]), torch.tensor([[1, 1, 1]])
    ).logits
    expected = torch.log_softmax(logits[0, 2:4], -1).gather(1, ids[0, 3:5, None]).squeeze(1)
    torch.testing.assert_close(scores.token_logprobs[0], expected)
    torch.testing.assert_close(scores.sequence_logprobs[0], expected.sum())
    other_image = scorer.score_image_text("What is shown?", [5], [" A"])
    assert scores.sequence_logprobs[0] != other_image.sequence_logprobs[0]


def test_vision_boundary_and_invalid_inputs():
    scorer = VisionCandidateScorer(FakeVisionModel(), FakeVisionProcessor())
    with pytest.raises(CandidateBoundaryError):
        scorer.score_image_text("context", [1], ["!unstable"])
    with pytest.raises(ValueError, match="images"):
        scorer.score_image_text("context", [], [" A"])
    with pytest.raises(ValueError, match="candidate"):
        scorer.score_image_text("context", [1], [""])


def test_vision_choice_binary_share_scorer_and_questions_stay_independent():
    class RecordingScorer(VisionCandidateScorer):
        def __init__(self):
            super().__init__(FakeVisionModel(), FakeVisionProcessor())
            self.contexts = []

        def score_image_text(self, context, images, candidates):
            self.contexts.append(context)
            return super().score_image_text(context, images, candidates)

    scorer = RecordingScorer()
    decider = AutoDecider(scorer)
    route = Choice(["billing", "technical", "sales"])
    result = decider.decide("state", {"route": route, "refund": Binary("Refund?")}, images=[2])
    assert sum(result["route"].probabilities.values()) == pytest.approx(1)
    assert result["refund"].probability_true + result["refund"].probability_false == pytest.approx(
        1
    )
    first_prompt = scorer.contexts[0]
    decider.decide("state", {"route": route, "refund": Binary("Changed?")}, images=[2])
    assert scorer.contexts[2] == first_prompt
    with pytest.raises(ValueError, match="images"):
        decider.decide("state", {"route": route})
