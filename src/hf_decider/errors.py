"""Public exceptions for unsupported scoring inputs."""


class CandidateBoundaryError(ValueError):
    """The context tokenization is not a prefix of context plus candidate."""


class UnsupportedModelError(ValueError):
    """The loaded model cannot provide decoder-only next-token logits."""
