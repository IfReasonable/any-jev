"""Choice prompt template."""


def choice_prompt(state: str, instruction: str, options: tuple[str, ...]) -> str:
    """Compile one independent question into an explicit label prompt."""
    lines = [state, "Question:", instruction, "Options:"]
    lines.extend(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))
    lines.extend(["Choose the best option. Return only the option label.", "Answer:"])
    return "\n".join(lines)
