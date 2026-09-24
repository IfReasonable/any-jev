"""Run a yes/no Binary decision."""

import sys

from any_jev import AutoDecider, Binary

model_id = sys.argv[1] if len(sys.argv) > 1 else "hf-internal-testing/tiny-random-gpt2"
decider = AutoDecider.from_pretrained(model_id)
result = decider.decide(
    "Customer says they were charged twice.",
    {"refund": Binary("Is the customer requesting a refund?")},
)
print(result["refund"])
