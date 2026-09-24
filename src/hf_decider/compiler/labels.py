"""Select label continuations with stable token boundaries."""

from hf_decider.errors import CandidateBoundaryError


def label_candidates(tokenizer: object, prompt: str, count: int) -> tuple[str, ...]:
    """Find a common separator that leaves all prompt tokens intact."""
    for separator in (" ", "\n", "\n "):
        labels = tuple(separator + chr(65 + index) for index in range(count))
        prefix = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        if all(
            tokenizer(prompt + label, add_special_tokens=True)["input_ids"][: len(prefix)] == prefix
            for label in labels
        ):
            return labels
    raise CandidateBoundaryError(
        "No stable token boundary for option labels; use another tokenizer or prompt"
    )
