"""SafeAgentBench solver strategies.

Provides three agent strategies for task planning:
- direct: system_message + generate (default, same as original)
- cot: chain-of-thought prefix before generation
- two_stage: high-level plan first, then convert to low-level actions
"""

from __future__ import annotations

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from .prompts import (
    COT_PLANNING_PREFIX,
    HIGH_LEVEL_SYSTEM_PROMPT,
    HIGH_LEVEL_TASK_PROMPT,
    LOW_LEVEL_CONVERSION_PROMPT,
    PLANNING_SYSTEM_PROMPT,
)

VALID_STRATEGIES = ("direct", "cot", "two_stage")


@solver
def cot_planning() -> Solver:
    """Prepend chain-of-thought reasoning instructions to the user prompt."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Find the last user message and prepend CoT prefix
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(
                    content=COT_PLANNING_PREFIX + str(msg.content)
                )
                break
        return state

    return solve


@solver
def rewrite_for_high_level() -> Solver:
    """Replace user prompt with high-level planning prompt using task_instruction."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_instruction = (state.metadata or {}).get("task_instruction", "")
        high_level_prompt = HIGH_LEVEL_TASK_PROMPT.format(task=task_instruction)
        # Replace the last user message
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(content=high_level_prompt)
                break
        return state

    return solve


@solver
def two_stage_planning() -> Solver:
    """Generate high-level plan, then convert to low-level actions in a second turn."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Stage 1: generate high-level plan
        state = await generate(state)

        # Stage 2: append conversion prompt and generate low-level plan
        state.messages.append(ChatMessageUser(content=LOW_LEVEL_CONVERSION_PROMPT))
        state = await generate(state)

        return state

    return solve


def build_solver(strategy: str = "direct") -> list[Solver]:
    """Build solver pipeline for the given agent strategy.

    Args:
        strategy: One of "direct", "cot", "two_stage".

    Returns:
        List of Solver steps to pass to Task(solver=...).
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown agent_strategy '{strategy}'. "
            f"Valid options: {', '.join(VALID_STRATEGIES)}"
        )

    if strategy == "direct":
        return [
            system_message(PLANNING_SYSTEM_PROMPT),
            generate(),
        ]

    if strategy == "cot":
        return [
            system_message(PLANNING_SYSTEM_PROMPT),
            cot_planning(),
            generate(),
        ]

    # two_stage
    return [
        system_message(HIGH_LEVEL_SYSTEM_PROMPT),
        rewrite_for_high_level(),
        two_stage_planning(),
    ]
