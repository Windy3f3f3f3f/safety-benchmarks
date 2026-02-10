"""SafeAgentBench solver strategies.

Text planner strategies (used by safeagentbench):
- direct:      LoTa-Bench — system prompt + single generation (arxiv:2402.08178)
- cot:         chain-of-thought reasoning prefix before generation
- two_stage:   high-level plan first, then convert to low-level actions
- progprompt:  ProgPrompt — programming-style action format (arxiv:2209.11302)
- fewshot:     LLM-Planner — few-shot examples prepended to prompt (arxiv:2212.04088)
- cooperative: CoELA — two-agent env analysis then planning (arxiv:2307.02485)
- mldt:        MLDT — multi-level decomposition: task → sub-tasks → actions (arxiv:2403.18760)

Also provides ``thor_scene_init()`` for react/visual modes.
"""

from __future__ import annotations

import logging

from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, system_message

from .thor_client import get_thor_client

log = logging.getLogger(__name__)

from .prompts import (
    COOPERATIVE_ENV_SYSTEM_PROMPT,
    COOPERATIVE_ENV_TASK_PROMPT,
    COOPERATIVE_PLANNING_PROMPT,
    COT_PLANNING_PREFIX,
    FEWSHOT_EXAMPLES,
    HIGH_LEVEL_SYSTEM_PROMPT,
    HIGH_LEVEL_TASK_PROMPT,
    LOW_LEVEL_CONVERSION_PROMPT,
    MLDT_ACTION_PROMPT,
    MLDT_DECOMPOSE_PROMPT,
    MLDT_DECOMPOSE_SYSTEM_PROMPT,
    MLDT_DETAIL_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    PROGPROMPT_SYSTEM_PROMPT,
    PROGPROMPT_TASK_PROMPT,
)

VALID_STRATEGIES = ("direct", "cot", "two_stage", "progprompt", "fewshot", "cooperative", "mldt")


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


@solver
def progprompt_rewrite() -> Solver:
    """Rewrite user prompt into ProgPrompt programming-style format."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        task_instruction = (state.metadata or {}).get("task_instruction", "")
        progprompt = PROGPROMPT_TASK_PROMPT.format(task=task_instruction)
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(content=progprompt)
                break
        return state

    return solve


@solver
def fewshot_prefix() -> Solver:
    """Prepend few-shot examples to the user prompt."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(
                    content=FEWSHOT_EXAMPLES + str(msg.content)
                )
                break
        return state

    return solve


@solver
def cooperative_planning() -> Solver:
    """Two-agent cooperative planning: environment analysis then action planning."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Stage 1: environment analysis
        task_instruction = (state.metadata or {}).get("task_instruction", "")
        env_prompt = COOPERATIVE_ENV_TASK_PROMPT.format(task=task_instruction)
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(content=env_prompt)
                break
        state = await generate(state)

        # Stage 2: planning based on environment analysis
        state.messages.append(ChatMessageUser(content=COOPERATIVE_PLANNING_PROMPT))
        state = await generate(state)

        return state

    return solve


@solver
def mldt_planning() -> Solver:
    """Multi-level decomposition: task → sub-tasks → details → actions."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Stage 1: decompose task into sub-tasks
        task_instruction = (state.metadata or {}).get("task_instruction", "")
        decompose_prompt = MLDT_DECOMPOSE_PROMPT.format(task=task_instruction)
        for i in range(len(state.messages) - 1, -1, -1):
            msg = state.messages[i]
            if msg.role == "user":
                state.messages[i] = ChatMessageUser(content=decompose_prompt)
                break
        state = await generate(state)

        # Stage 2: detail each sub-task
        state.messages.append(ChatMessageUser(content=MLDT_DETAIL_PROMPT))
        state = await generate(state)

        # Stage 3: convert to low-level actions
        state.messages.append(ChatMessageUser(content=MLDT_ACTION_PROMPT))
        state = await generate(state)

        return state

    return solve


def build_solver(strategy: str = "direct") -> list[Solver]:
    """Build solver pipeline for the given text planner strategy.

    Args:
        strategy: One of "direct", "cot", "two_stage", "progprompt",
                  "fewshot", "cooperative", "mldt".

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

    if strategy == "two_stage":
        return [
            system_message(HIGH_LEVEL_SYSTEM_PROMPT),
            rewrite_for_high_level(),
            two_stage_planning(),
        ]

    if strategy == "progprompt":
        return [
            system_message(PROGPROMPT_SYSTEM_PROMPT),
            progprompt_rewrite(),
            generate(),
        ]

    if strategy == "fewshot":
        return [
            system_message(PLANNING_SYSTEM_PROMPT),
            fewshot_prefix(),
            generate(),
        ]

    if strategy == "cooperative":
        return [
            system_message(COOPERATIVE_ENV_SYSTEM_PROMPT),
            cooperative_planning(),
        ]

    # mldt
    return [
        system_message(MLDT_DECOMPOSE_SYSTEM_PROMPT),
        mldt_planning(),
    ]


@solver
def thor_scene_init(thor_url: str = "http://localhost:9100") -> Solver:
    """Reset the AI2-THOR scene before the agent starts interacting.

    Reads ``scene_name`` from sample metadata and calls ``/reset``.
    """

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        scene = (state.metadata or {}).get("scene_name", "FloorPlan1")
        client = get_thor_client(thor_url)
        try:
            await client.reset(scene)
            log.info(f"AI2-THOR scene reset to {scene}")
        except Exception as e:
            log.error(f"Failed to reset scene {scene}: {e}")
        return state

    return solve
