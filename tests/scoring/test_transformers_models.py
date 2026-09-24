"""Offline random-weight tests for two decoder-only model families."""

import pytest
import torch
from transformers import AutoModelForCausalLM, GPT2Config, LlamaConfig

from any_jev.scoring.reference import ReferenceCandidateScorer


@pytest.mark.parametrize(
    "config",
    [
        GPT2Config(
            vocab_size=32,
            bos_token_id=1,
            eos_token_id=2,
            n_positions=32,
            n_embd=32,
            n_layer=1,
            n_head=4,
        ),
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
        ),
    ],
)
def test_random_transformers_model_matches_individual_forward(config):
    torch.manual_seed(3)
    model = AutoModelForCausalLM.from_config(config)
    scorer = ReferenceCandidateScorer(model)
    prefix = torch.tensor([2, 3, 4])
    candidates = [torch.tensor([5]), torch.tensor([6, 7, 8]), torch.tensor([9, 10, 11, 12])]
    batch = scorer.score_token_candidates(prefix, candidates)
    individual = torch.stack(
        [
            scorer.score_token_candidates(prefix, [candidate]).sequence_logprobs[0]
            for candidate in candidates
        ]
    )
    torch.testing.assert_close(batch.sequence_logprobs, individual, rtol=1e-5, atol=1e-5)
