# HF-Decider

Turn a compatible Hugging Face causal LM into a candidate scorer / probabilistic decider without generating text.

Install with `pip install hf-decider`. For development, run `pip install -e '.[dev]'` in the `jev` environment.

## Candidate scoring

```python
from hf_decider import AutoCandidateScorer

scorer = AutoCandidateScorer.from_pretrained("Qwen/Qwen3-4B")
scores = scorer.score_text(
    context="The capital of France is",
    candidates=[" Paris", " London", " Berlin"],
)
print(scores.sequence_logprobs)
print(scores.normalized_probabilities())
```

Scores are sums of token log probabilities, with no length normalization. The first continuation token is predicted by the last context token. The implementation uses forward logits and teacher forcing; it never calls `generate()`.

## Choice

```python
from hf_decider import AutoDecider, Choice

decider = AutoDecider.from_pretrained("Qwen/Qwen3-4B")
result = decider.decide(
    state="Customer says they were charged twice.",
    questions={"route": Choice(["billing", "technical", "sales"], instruction="Which team?")},
)
print(result["route"].choice, result["route"].probabilities)
```

## Binary

```python
from hf_decider import AutoDecider, Binary

decider = AutoDecider.from_pretrained("Qwen/Qwen3-4B")
result = decider.decide(
    state="Customer says they were charged twice.",
    questions={"refund": Binary("Is the customer requesting a refund?")},
)
print(result["refund"].probability_true)
```

The normalized probabilities mean **relative probability mass within the supplied candidate set**: `softmax(sequence_logprobs)`. They are **not calibrated probabilities that an answer is correct**. No calibration is provided in v0.1.

Supported: decoder-only Hugging Face causal language models with token logits. Not yet supported: VLMs, encoder-only or sequence-to-sequence models. `Choice` accepts 2–26 distinct options. String continuations require a stable token boundary; `CandidateBoundaryError` suggests a leading space/newline or the token-level API. Empty context tokenizations cannot predict a first candidate token. `score_token_candidates()` accepts token tensors directly and allows right-padded prefixes with an attention mask.

Roadmap: v0.2 will explore VLM input and shared-prefix caching; v0.3 may add calibration and generalized codebooks.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and [CHANGELOG.md](CHANGELOG.md) for release notes.
