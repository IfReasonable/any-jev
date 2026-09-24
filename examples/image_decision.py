"""Score candidates and make a Choice decision about a local image."""

import sys

from hf_decider import AutoDecider, AutoVisionCandidateScorer, Choice

if len(sys.argv) != 3:
    raise SystemExit("usage: python examples/image_decision.py MODEL_ID IMAGE_PATH")

model_id, image_path = sys.argv[1:]
scorer = AutoVisionCandidateScorer.from_pretrained(model_id)
scores = scorer.score_image_text(
    "What kind of image is this?", [image_path], [" document", " photo"]
)
print("candidate log probabilities:", scores.sequence_logprobs.tolist())

decider = AutoDecider(scorer)
result = decider.decide(
    "Classify this image.",
    {"type": Choice(["document", "photograph"])},
    images=[image_path],
)
print(result["type"])
