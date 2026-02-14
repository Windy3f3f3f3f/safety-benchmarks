"""
IHEval Benchmark - Instruction Hierarchy Following

This module implements the IHEval benchmark which evaluates model's ability
to follow the instruction hierarchy: system message > user message >
conversation history > tool output.

Based on:
    IHEval: Evaluating Language Models on Following the Instruction Hierarchy
    https://arxiv.org/pdf/2502.08745

Usage:
    inspect eval eval_benchmarks/iheval --model openai/gpt-4o-mini --limit 10
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

try:
    from .dataset import (
        IHEVAL_CATEGORY,
        IHEVAL_SETTING,
        IHEVAL_STRENGTH,
        IHEVAL_TASK,
        load_iheval_dataset,
    )
    from .scorer import iheval_scorer, instruction_following_rate
except ImportError:
    from dataset import (
        IHEVAL_CATEGORY,
        IHEVAL_SETTING,
        IHEVAL_STRENGTH,
        IHEVAL_TASK,
        load_iheval_dataset,
    )
    from scorer import iheval_scorer, instruction_following_rate


# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 4096


@task
def iheval(
    category: IHEVAL_CATEGORY = "rule-following",
    task: IHEVAL_TASK = "multi-turn",
    setting: IHEVAL_SETTING = "aligned",
    system_strength: IHEVAL_STRENGTH = "default",
    user_strength: IHEVAL_STRENGTH = "default",
    tool_strength: IHEVAL_STRENGTH = "default",
    limit: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Task:
    """IHEval: Instruction Hierarchy Following Benchmark.

    Evaluates model's ability to follow the instruction hierarchy where
    system message > user message > conversation history > tool output.

    The benchmark includes 9 tasks across 4 categories:
    - rule-following: Single-turn and multi-turn instruction following
    - task-execution: NLP tasks (extraction, translation, classification)
    - safety: Security scenarios (hijack, prompt extraction)
    - tool-use: Function calling scenarios

    Args:
        category: Task category (rule-following, task-execution, safety, tool-use)
        task: Specific task name (e.g., multi-turn, extraction, hijack, etc.)
        setting: Evaluation setting
            - aligned: All inputs align with system instruction
            - conflict: Low-priority inputs conflict with system
            - reference: All instructions merged into user message (baseline)
        system_strength: System prompt strength (default, weak, strong)
        user_strength: User prompt strength (for conflict settings)
        tool_strength: Tool output strength (for tool-use tasks)
        limit: Maximum number of samples to evaluate
        temperature: Generation temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in model response

    Returns:
        A Task that evaluates instruction hierarchy following

    Example:
        # Evaluate multi-turn aligned setting
        inspect eval eval_benchmarks/iheval --model openai/gpt-4o-mini \\
            -T category=rule-following task=multi-turn setting=aligned

        # Evaluate conflict scenario with strong prompts
        inspect eval eval_benchmarks/iheval --model openai/gpt-4o-mini \\
            -T category=rule-following task=multi-turn setting=conflict \\
            system_strength=strong user_strength=strong
    """
    # Validate category-task combinations
    _validate_combination(category, task)

    return Task(
        dataset=load_iheval_dataset(
            category=category,
            task=task,
            setting=setting,
            system_strength=system_strength,
            user_strength=user_strength,
            tool_strength=tool_strength,
            limit=limit,
        ),
        solver=[generate()],
        scorer=[iheval_scorer()],
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )


def _validate_combination(category: IHEVAL_CATEGORY, task: IHEVAL_TASK) -> None:
    """Validate that the category-task combination is valid.

    Args:
        category: Task category
        task: Task name

    Raises:
        ValueError: If the combination is invalid
    """
    valid_combinations = {
        "rule-following": ["single-turn", "multi-turn"],
        "task-execution": ["extraction", "translation", "lang-detect"],
        "safety": ["hijack", "system-prompt-extract"],
        "tool-use": ["get-webpage", "slack-user"],
    }

    valid_tasks = valid_combinations.get(category, [])
    if task not in valid_tasks:
        raise ValueError(
            f"Task '{task}' is not valid for category '{category}'. "
            f"Valid tasks are: {', '.join(valid_tasks)}"
        )


__all__ = [
    "iheval",
    "IHEVAL_CATEGORY",
    "IHEVAL_TASK",
    "IHEVAL_SETTING",
    "IHEVAL_STRENGTH",
]
