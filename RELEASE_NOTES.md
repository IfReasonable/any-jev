# v0.1.0 release notes

HF-Decider 0.1.0 adds exact teacher-forced candidate continuation scoring for Hugging Face decoder-only causal LMs, with Choice and Binary decision APIs. Install with `pip install hf-decider`.

Relative probabilities are a softmax over supplied candidate sequence log probabilities. They are not calibrated correctness probabilities. The release supports text-only models, up to 26 Choice options, and string continuations with stable token boundaries. VLM input and shared-prefix caching are planned for v0.2.

The owner must confirm the PyPI package name, configure Trusted Publishing, run TestPyPI installation smoke tests, and create the release tag before public publication.
