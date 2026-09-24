"""Compare batched and repeated full-forward scoring on a random tiny GPT-2.

Example: python benchmarks/benchmark_scoring.py --context 128 --candidates 8 --length 4
"""

import argparse
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, GPT2Config

from any_jev.scoring.reference import ReferenceCandidateScorer


def main() -> None:
    """Print repeatable timing and numerical parity for one benchmark shape."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    if min(args.context, args.candidates, args.length, args.iterations) < 1:
        parser.error("all dimensions and iterations must be positive")
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPT2Config(
        vocab_size=128,
        bos_token_id=1,
        eos_token_id=2,
        n_positions=args.context + args.length + 1,
        n_embd=64,
        n_layer=2,
        n_head=4,
    )
    scorer = ReferenceCandidateScorer(AutoModelForCausalLM.from_config(config).to(device))
    prefix = torch.randint(1, 128, (args.context,))
    candidates = [torch.randint(1, 128, (args.length,)) for _ in range(args.candidates)]

    def synchronize() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    def run(batched: bool) -> tuple[float, torch.Tensor, int | None]:
        scorer.score_token_candidates(prefix, candidates if batched else candidates[:1])
        synchronize()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.iterations):
            if batched:
                values = scorer.score_token_candidates(prefix, candidates).sequence_logprobs
            else:
                values = torch.stack(
                    [
                        scorer.score_token_candidates(prefix, [candidate]).sequence_logprobs[0]
                        for candidate in candidates
                    ]
                )
        synchronize()
        memory = torch.cuda.max_memory_allocated() if device == "cuda" else None
        return (time.perf_counter() - start) / args.iterations, values, memory

    repeated, reference, repeated_memory = run(False)
    batched, optimized, batched_memory = run(True)
    error = (reference - optimized).abs().max().item()
    torch.testing.assert_close(reference, optimized, rtol=1e-5, atol=1e-5)
    print(f"hardware={platform.processor() or platform.machine()} device={device}")
    print(f"torch={torch.__version__} transformers={transformers.__version__} dtype=float32")
    print(f"context={args.context} candidates={args.candidates} length={args.length}")
    print("method | latency_ms | peak_cuda_bytes")
    print(f"repeated | {repeated * 1000:.3f} | {repeated_memory}")
    print(f"batched | {batched * 1000:.3f} | {batched_memory}")
    print(f"max_abs_error={error:.9g}")


if __name__ == "__main__":
    main()
