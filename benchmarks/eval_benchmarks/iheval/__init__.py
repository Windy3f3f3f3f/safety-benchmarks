"""
IHEval Benchmark - Instruction Hierarchy Following

This benchmark evaluates model's ability to follow the instruction hierarchy:
system message > user message > conversation history > tool output

Based on:
    IHEval: Evaluating Language Models on Following the Instruction Hierarchy
    https://arxiv.org/pdf/2502.08745
"""

from .iheval import (
    IHEVAL_CATEGORY,
    IHEVAL_SETTING,
    IHEVAL_STRENGTH,
    IHEVAL_TASK,
    iheval,
)

__all__ = [
    "iheval",
    "IHEVAL_CATEGORY",
    "IHEVAL_TASK",
    "IHEVAL_SETTING",
    "IHEVAL_STRENGTH",
]
