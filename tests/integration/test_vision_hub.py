"""Small real-model check of processor inputs, logits and decisions."""

import pytest
from PIL import Image

from hf_decider import AutoDecider, AutoVisionCandidateScorer, Choice


@pytest.mark.integration
def test_tiny_llava_scores_image_conditioned_continuations(tmp_path):
    scorer = AutoVisionCandidateScorer.from_pretrained(
        "trl-internal-testing/tiny-LlavaForConditionalGeneration"
    )
    image = Image.new("RGB", (32, 32), color=(255, 0, 0))
    scores = scorer.score_image_text("What color is the square?", [image], [" red", " blue"])
    assert scores.sequence_logprobs.shape == (2,)
    assert all(len(tokens) > 0 for tokens in scores.token_logprobs)
    image_path = tmp_path / "square.png"
    image.save(image_path)
    path_scores = scorer.score_image_text("What color is the square?", [str(image_path)], [" red"])
    assert path_scores.sequence_logprobs[0] == pytest.approx(scores.sequence_logprobs[0].item())
    decision = AutoDecider.from_pretrained(
        "trl-internal-testing/tiny-LlavaForConditionalGeneration",
        vision=True,
        local_files_only=True,
    ).decide(
        "A square is shown.",
        {"color": Choice(["red", "blue"], instruction="Which color is the square?")},
        images=[image],
    )
    assert len(decision["color"].probabilities) == 2


@pytest.mark.integration
def test_tiny_qwen2_vl_opaque_image_grid_inputs():
    scorer = AutoVisionCandidateScorer.from_pretrained(
        "optimum-intel-internal-testing/tiny-random-qwen2vl"
    )
    image = Image.new("RGB", (32, 32), color=(0, 0, 255))
    scores = scorer.score_image_text("Name the color.", [image], [" blue", " red"])
    assert scores.sequence_logprobs.shape == (2,)
