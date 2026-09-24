# any-jev

Turn a compatible Hugging Face LLM or VLM into a candidate scorer / probabilistic decider without generating text.

Install with `pip install any-jev` for LLMs or `pip install 'any-jev[vision]'` for image support. For development, run `pip install -e '.[dev,vision]'` in the `jev` environment.

## Candidate scoring

```python
from any_jev import AutoCandidateScorer

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
from any_jev import AutoDecider, Choice

decider = AutoDecider.from_pretrained("Qwen/Qwen3-4B")
result = decider.decide(
    state="Customer says they were charged twice.",
    questions={"route": Choice(["billing", "technical", "sales"], instruction="Which team?")},
)
print(result["route"].choice, result["route"].probabilities)
```

## Binary

```python
from any_jev import AutoDecider, Binary

decider = AutoDecider.from_pretrained("Qwen/Qwen3-4B")
result = decider.decide(
    state="Customer says they were charged twice.",
    questions={"refund": Binary("Is the customer requesting a refund?")},
)
print(result["refund"].probability_true)
```

## Vision-language models

```python
from PIL import Image
from any_jev import AutoVisionCandidateScorer, AutoDecider, Choice

model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
image = Image.open("example.jpg").convert("RGB")

scorer = AutoVisionCandidateScorer.from_pretrained(model_id)
scores = scorer.score_image_text("What is in the picture?", [image], [" a cat", " a dog"])
print(scores.sequence_logprobs)

decider = AutoDecider(scorer)  # reuse the loaded VLM
result = decider.decide(
    "Route this image for review.",
    {"type": Choice(["document", "photograph"], instruction="What kind of image is this?")},
    images=[image],
)
print(result["type"].probabilities)
```

Images may be PIL images, local paths, or URLs understood by the model processor. For a prepared chat history, call `scorer.score_messages(messages, candidates)` using the Transformers multimodal chat format. Candidates are assistant text continuations. The processor prepares image tensors and text tokens; the scorer checks that each full candidate tokenization preserves the prompt prefix, then uses forward logits and teacher forcing. Each candidate currently needs its own VLM forward pass. The model must return logits aligned to its `input_ids` positions.

For decision-only use, `AutoDecider.from_pretrained(model_id, vision=True)` loads the VLM and processor in one call.

The normalized probabilities mean **relative probability mass within the supplied candidate set**: `softmax(sequence_logprobs)`. They are **not calibrated probabilities that an answer is correct**. No calibration is provided.

Supported: decoder-only Hugging Face causal language models and image-text-to-text VLMs with token-aligned causal logits. Not supported: encoder-only or sequence-to-sequence outputs without aligned causal logits. `Choice` accepts 2–26 distinct options. String continuations require a stable token boundary; `CandidateBoundaryError` suggests a leading space/newline or the token-level API. Empty context tokenizations cannot predict a first candidate token. `score_token_candidates()` accepts token tensors directly and allows right-padded prefixes with an attention mask.

Roadmap: shared-prefix caching, calibration, and generalized codebooks are future work.

See [CONTRIBUTING.md](CONTRIBUTING.md) for validation and [CHANGELOG.md](CHANGELOG.md) for release notes.
