# Implementation notes

- `score_token_candidates()` is the scoring core. It supports an optional contiguous, right-padded prefix mask by trimming that padding before scoring. Left-padded or gapped prefixes are rejected because model-specific position handling could change scores.
- Candidate tokens are represented as stringified token ID lists in `CandidateScores.candidates` for the token API; `score_text()` replaces these with the original text candidates. The actual scores always come from the token API.
- The reference path uses one full forward pass for all candidates, accumulates log probabilities in float32, and returns model-device tensors. There is no cache path.
- `AutoCandidateScorer.from_pretrained()` passes loading options such as `token`, `revision`, and `local_files_only` to the tokenizer as well as the model. The remaining keyword arguments go to the model.
- The default Choice compiler tries space and newline separators, checking every label's token boundary before scoring. If none are stable, it raises `CandidateBoundaryError`.
- The release package name remains the document's provisional `hf-decider`; the owner must confirm it before publishing. No tag or remote publication has been created.
- VLM scoring uses `AutoProcessor.apply_chat_template` for both the prefix and each open assistant continuation. It checks the exact processor-produced prefix token IDs before any model forward. Extra model inputs, including image tensors and grid metadata, are passed through to the model without architecture-specific handling.
- VLM candidates are currently scored with separate full forwards because multimodal tensor batch layouts vary by architecture. This preserves scoring correctness while leaving shared-prefix and batched VLM optimization for a later version.
