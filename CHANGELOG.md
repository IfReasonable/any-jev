# Changelog

## 0.2.0 (release candidate)

- Rename the distribution to `any-jev` and the Python import package to `any_jev`.
- Score assistant text continuations conditioned on images and chat history through `AutoVisionCandidateScorer`.
- Use VLMs for Choice and Binary decisions with `AutoDecider.from_pretrained(..., vision=True)` and `decide(..., images=[...])`.
- Add a `vision` extra for Pillow and a tiny LLaVA integration test.

VLM scoring uses one processor call and full forward per candidate. A changed chat-template token boundary raises `CandidateBoundaryError`.

## 0.1.0 (release candidate)

- Exact batched continuation scoring with token-boundary checks.
- Choice and Binary decisions over supplied candidate sets.
- Relative candidate-set probabilities; no calibration or generation.

Known limits: decoder-only text CausalLMs, maximum 26 Choice options, stable string token boundaries required.

Next: v0.2 VLM input and shared-prefix caching.
