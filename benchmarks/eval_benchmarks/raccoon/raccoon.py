"""
Raccoon Benchmark - Prompt Extraction Attacks

This module implements the Raccoon benchmark, which evaluates model resistance
to prompt extraction attacks. The benchmark tests whether adversarial queries
can cause a model to reveal its system prompt.

Based on:
    Raccoon: A Benchmark for Prompt Extraction Attacks on Large Language Models
    https://arxiv.org/abs/2310.01798

Usage:
    inspect eval eval_benchmarks/raccoon --model openai/gpt-4o-mini --limit 10
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageSystem, GenerateConfig
from inspect_ai.solver import Generate, Solver, TaskState, solver

try:
    from .dataset import DEFAULT_SYSTEM_PROMPT, load_raccoon_dataset
    from .scorer import DEFAULT_SUCCESS_THRESHOLD, raccoon_scorer
    from .templates import OPENAI_DEFAULT_TEMPLATE
except ImportError:
    from dataset import DEFAULT_SYSTEM_PROMPT, load_raccoon_dataset
    from scorer import DEFAULT_SUCCESS_THRESHOLD, raccoon_scorer
    from templates import OPENAI_DEFAULT_TEMPLATE

if TYPE_CHECKING:
    pass

# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096


@solver
def apply_raccoon_system_prompt() -> Solver:
    """Apply the system prompt to each sample before generation.

    The Raccoon benchmark tests whether an attack can extract the system prompt.
    This solver retrieves the system prompt from sample metadata and applies it.

    The solver handles both:
    1. Plain system prompts (direct use)
    2. GPT template-wrapped prompts (with name substitution)
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Get system prompt configuration from metadata
        system_prompt = (
            state.metadata.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            if state.metadata
            else DEFAULT_SYSTEM_PROMPT
        )
        use_template = (
            state.metadata.get("use_template", True) if state.metadata else True
        )

        if use_template:
            # Apply GPT template with name substitution
            # Extract prompt name from sample ID for the template
            prompt_name = (
                state.metadata.get("attack_name", "GPT").split("/")[-1]
                if state.metadata
                else "GPT"
            )

            from string import Template

            full_system_prompt = Template(OPENAI_DEFAULT_TEMPLATE).safe_substitute(
                name=prompt_name, user_prompt=system_prompt
            )
        else:
            full_system_prompt = system_prompt

        # Insert system message at the beginning of messages
        state.messages.insert(0, ChatMessageSystem(content=full_system_prompt))

        # Generate model response
        state = await generate(state)

        return state

    return solve


@task
def raccoon(
    attack_category: Literal["singular_attacks", "compound_attacks", "all"] = "all",
    system_prompt: str | None = None,
    use_template: bool = True,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    shuffle: bool = False,
    limit: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    prompt_categories: list[str] | None = None,
    prompts_limit: int | None = None,
) -> Task:
    """Raccoon: Prompt Extraction Attack Benchmark.

    Evaluates model resistance to prompt extraction attacks. The benchmark
    tests whether adversarial queries can cause models to reveal their system
    prompt using ROUGE-L similarity scoring.

    By default, loads system prompts from 20_prompts.jsonl, creating a dataset
    with N system prompts × M attacks. Use system_prompt parameter to test a
    single custom prompt instead.

    Args:
        attack_category: Which attack category to use
            - "singular_attacks": Single-prompt attacks
            - "compound_attacks": Multi-step attacks
            - "all": All available attacks
        system_prompt: Optional single system prompt to test (overrides JSONL)
        use_template: Whether to wrap system prompt in GPT template
        success_threshold: ROUGE-L threshold for successful attack (0.0-1.0)
        shuffle: Whether to shuffle the dataset
        limit: Maximum number of samples to evaluate (after expansion)
        temperature: Generation temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in model response
        prompt_categories: Filter system prompts by categories (e.g., ["navigation", "emergency"])
        prompts_limit: Limit number of system prompts to load from JSONL

    Returns:
        A Task that evaluates prompt extraction resistance

    Example:
        # Use all system prompts from JSONL (20 prompts × N attacks)
        inspect eval eval_benchmarks/raccoon --model openai/gpt-4o-mini --limit 50

        # Use only emergency category prompts
        inspect eval eval_benchmarks/raccoon --model openai/gpt-4o-mini -T prompt_categories=["emergency"]

        # Use single custom system prompt
        inspect eval eval_benchmarks/raccoon --model openai/gpt-4o-mini -T system_prompt="You are a helpful assistant."
    """
    # Validate threshold
    if not 0 <= success_threshold <= 1:
        raise ValueError(
            f"success_threshold must be between 0 and 1, got {success_threshold}"
        )

    return Task(
        dataset=load_raccoon_dataset(
            attack_category=attack_category,
            system_prompt=system_prompt,
            use_template=use_template,
            shuffle=shuffle,
            limit=limit,
            prompt_categories=prompt_categories,
            prompts_limit=prompts_limit,
        ),
        solver=[
            apply_raccoon_system_prompt(),
        ],
        scorer=[raccoon_scorer(threshold=success_threshold)],
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )


__all__ = [
    "raccoon",
    "apply_raccoon_system_prompt",
    "DEFAULT_SYSTEM_PROMPT",
]
