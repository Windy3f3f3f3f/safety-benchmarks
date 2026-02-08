"""SafeAgentBench: Evaluating Embodied LLM Agent Safety

Yin et al. "SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents"
https://arxiv.org/abs/2412.13178

This benchmark evaluates whether LLM agents can:
1. Refuse dangerous household task instructions (unsafe tasks)
2. Successfully plan safe household tasks (safe tasks)
3. Comply with safety constraints in multi-step tasks (long-horizon tasks)

# Run all tasks with mixed sampling (100 samples)
inspect eval safeagentbench --model openai/doubao-seed-1-8

# Run only unsafe task evaluation
inspect eval safeagentbench_unsafe --model openai/deepseek-v3.2 --limit 50

# Run only safe task evaluation
inspect eval safeagentbench_safe --model openai/deepseek-v3.2 --limit 30

# Custom judge model
inspect eval safeagentbench -T judge_model=openai/gpt-4o --model openai/doubao-seed-1-8
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model

from .dataset import load_safeagentbench_dataset
from .scorer import safeagentbench_scorer
from .solvers import build_solver

DEFAULT_JUDGE_MODEL = "openai/GLM-4.7"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0


@task
def safeagentbench(
    task_type: str | None = None,
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
    sample_size: int = 100,
    seed: int = 42,
) -> Task:
    """SafeAgentBench: evaluate embodied LLM agent safety.

    Args:
        task_type: "unsafe", "safe", "all", or None (mixed sampling).
        judge_model: Model for LLM-as-judge evaluation.
        agent_strategy: Solver strategy — "direct", "cot", or "two_stage".
        sample_size: Number of samples for mixed sampling.
        seed: Random seed for reproducible sampling.
    """
    dataset = load_safeagentbench_dataset(
        task_type=task_type,
        sample_size=sample_size,
        seed=seed,
    )

    return Task(
        dataset=dataset,
        solver=build_solver(agent_strategy),
        scorer=safeagentbench_scorer(judge_model=judge_model),
        config=GenerateConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        ),
        version="1.0.0",
    )


@task
def safeagentbench_unsafe(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
) -> Task:
    """Evaluate only unsafe tasks — measures rejection rate.

    Includes unsafe_detailed, abstract, and long_horizon tasks.
    Primary safety metric: rejection_rate (higher = safer).
    """
    return safeagentbench(
        task_type="unsafe", judge_model=judge_model, agent_strategy=agent_strategy
    )


@task
def safeagentbench_safe(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
) -> Task:
    """Evaluate only safe tasks — measures planning capability.

    Includes safe_detailed tasks only.
    Primary metric: safe_success_rate (higher = more capable).
    """
    return safeagentbench(
        task_type="safe", judge_model=judge_model, agent_strategy=agent_strategy
    )
