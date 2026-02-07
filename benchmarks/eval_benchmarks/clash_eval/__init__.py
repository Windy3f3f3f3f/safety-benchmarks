"""
ClashEval: Context-Knowledge Clash Evaluation

Evaluates how easily a model can be misled by false information
injected into RAG retrieval context.

Based on:
    Wu et al. "ClashEval: Quantifying the tug-of-war between an LLM's
    internal prior and external evidence"
    https://arxiv.org/abs/2404.10198
"""

from .clash_eval import clash_eval

__all__ = ["clash_eval"]