# Changelog

## 0.1.0 (release candidate)

- Exact batched continuation scoring with token-boundary checks.
- Choice and Binary decisions over supplied candidate sets.
- Relative candidate-set probabilities; no calibration or generation.

Known limits: decoder-only text CausalLMs, maximum 26 Choice options, stable string token boundaries required.

Next: v0.2 VLM input and shared-prefix caching.
