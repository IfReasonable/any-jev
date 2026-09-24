"""Run a three-way Choice decision."""

import sys

from any_jev import AutoDecider, Choice

model_id = sys.argv[1] if len(sys.argv) > 1 else "hf-internal-testing/tiny-random-gpt2"
decider = AutoDecider.from_pretrained(model_id)
result = decider.decide(
    "Customer says they were charged twice.",
    {"route": Choice(["billing", "technical", "sales"], instruction="Which team?")},
)
print(result["route"])
