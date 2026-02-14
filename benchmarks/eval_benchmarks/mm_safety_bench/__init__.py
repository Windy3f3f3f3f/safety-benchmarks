"""
MM-SafetyBench Benchmark
Based on ECCV 2024: Query-Relevant Images Jailbreak Large Multi-Modal Models
https://arxiv.org/abs/2311.17600

Evaluates multimodal LLM safety under image-based adversarial attacks.
"""

from .mm_safety_bench import mm_safety_bench_illegal_activity

__all__ = [
    "mm_safety_bench_illegal_activity",
    # Future scenarios can be added here
    # "mm_safety_bench_hate_speech",
    # "mm_safety_bench_malware_generation",
    # ...
]
