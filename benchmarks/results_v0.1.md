# v0.1 benchmark results

Measured in the `jev` environment on 2026-09-24, using `python benchmarks/benchmark_scoring.py --context 128 --candidates 8 --length 4 --iterations 3`.

- Hardware: x86_64 CPU (no CUDA device)
- PyTorch: 2.14.0+cu130; Transformers: 5.17.0; dtype: float32
- Context: 128 tokens; candidates: 8; candidate length: 4 tokens

| Method | Mean latency (ms) | Peak CUDA memory |
| --- | ---: | ---: |
| Repeated single candidate | 2539.653 | N/A |
| Batched candidates | 596.902 | N/A |

Maximum absolute score error: 0. Timings reflect this small random GPT-2 configuration and are not a general performance guarantee.
