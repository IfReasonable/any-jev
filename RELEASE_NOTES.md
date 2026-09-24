# v0.2.0 release notes

any-jev 0.2.0 adds image-conditioned text continuation scoring for Hugging Face image-text-to-text VLMs. Use `AutoVisionCandidateScorer.from_pretrained(...)` for direct scores or `AutoDecider.from_pretrained(..., vision=True)` for Choice and Binary decisions with images. Install with `pip install 'any-jev[vision]'`. The original text-only LLM path remains available.

Relative probabilities are a softmax over supplied candidate sequence log probabilities. They are not calibrated correctness probabilities. Choice supports up to 26 options, and string continuations require stable token boundaries.

VLM limitations: one processor and model forward per candidate, image input only, and a required token-stable assistant continuation. Shared-prefix caching and calibration remain future work.

The owner must configure Trusted Publishing, run TestPyPI installation smoke tests, and create the release tag before public publication.
