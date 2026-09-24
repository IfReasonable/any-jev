"""Run a candidate scoring example with a local or Hub CausalLM."""

import sys

from hf_decider import AutoCandidateScorer

model_id = sys.argv[1] if len(sys.argv) > 1 else "hf-internal-testing/tiny-random-gpt2"
scorer = AutoCandidateScorer.from_pretrained(model_id)
scores = scorer.score_text("The capital of France is", [" Paris", " London", " Berlin"])
for candidate, logprob, probability in zip(
    scores.candidates, scores.sequence_logprobs.tolist(), scores.normalized_probabilities().tolist()
):
    print(candidate, logprob, probability)
