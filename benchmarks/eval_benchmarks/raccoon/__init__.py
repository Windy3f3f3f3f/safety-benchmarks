"""
Raccoon Benchmark - Prompt Extraction Attacks

This benchmark evaluates model resistance to prompt extraction attacks.
It tests whether adversarial queries can cause models to reveal their
system prompt using ROUGE-L similarity scoring.

Based on:
    Raccoon: A Benchmark for Prompt Extraction Attacks on Large Language Models
    https://arxiv.org/abs/2310.01798
"""

from .raccoon import (
    DEFAULT_SYSTEM_PROMPT,
    apply_raccoon_system_prompt,
    raccoon,
)

__all__ = [
    "raccoon",
    "apply_raccoon_system_prompt",
    "DEFAULT_SYSTEM_PROMPT",
]
