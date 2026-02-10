"""SafeAgentBench: Evaluating Embodied LLM Agent Safety

Yin et al. "SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents"
https://arxiv.org/abs/2412.13178

This benchmark evaluates whether LLM agents can:
1. Refuse dangerous household task instructions (unsafe tasks)
2. Successfully plan safe household tasks (safe tasks)
3. Comply with safety constraints in multi-step tasks (long-horizon tasks)

## Sub-tasks

Three sub-tasks, each corresponding to a different solver type:

### safeagentbench — text planner (7 agent strategies)
  Model reads task instruction -> generates action plan text ->
  LLM judge evaluates plan quality against reference steps.
  agent_strategy options:
    direct      — LoTa-Bench      (https://arxiv.org/abs/2402.08178)
    cot         — chain-of-thought reasoning prefix
    two_stage   — high-level plan then low-level conversion
    progprompt  — ProgPrompt       (https://arxiv.org/abs/2209.11302)
    fewshot     — LLM-Planner      (https://arxiv.org/abs/2212.04088)
    cooperative — CoELA            (https://arxiv.org/abs/2307.02485)
    mldt        — MLDT             (https://arxiv.org/abs/2403.18760)
  With with_execution=True, scorer also executes the plan in AI2-THOR.

### safeagentbench_react — ReAct interactive agent
  ReAct baseline (https://arxiv.org/abs/2210.03629).
  Model interacts with AI2-THOR step-by-step via tool calls (text feedback).
  Always includes execution scoring (requires Docker).

### safeagentbench_visual — visual interactive agent (3 strategies)
  Model interacts with AI2-THOR via tool calls with screenshot feedback.
  agent_strategy options:
    basic   — standard visual agent
    pca     — PCA-EVAL  (https://arxiv.org/abs/2402.15527)
    map_vlm — MAP       (https://arxiv.org/abs/2408.05478)
  Always includes execution scoring (requires Docker + vision model).

## Prerequisites for execution modes

AI2-THOR runs in a Docker container with GPU (NVIDIA + Vulkan + Xvfb):
  cd benchmarks/eval_benchmarks/safeagentbench/docker
  docker compose up -d --build
  curl http://localhost:9100/health  # should return {"status": "ok"}

## Usage Examples

  # text planner — semantic only (no simulator)
  inspect eval eval_benchmarks/safeagentbench --model openai/doubao-seed-1-8

  # text planner with chain-of-thought
  inspect eval eval_benchmarks/safeagentbench -T agent_strategy=cot --model openai/doubao-seed-1-8

  # text planner with execution scoring (needs Docker)
  inspect eval eval_benchmarks/safeagentbench -T with_execution=true --model openai/doubao-seed-1-8

  # ReAct agent (needs Docker)
  inspect eval eval_benchmarks/safeagentbench_react --model openai/doubao-seed-1-8

  # visual agent (needs Docker + vision model)
  inspect eval eval_benchmarks/safeagentbench_visual --model openai/doubao-seed-1-8

  # visual agent with PCA strategy
  inspect eval eval_benchmarks/safeagentbench_visual -T agent_strategy=pca --model openai/doubao-seed-1-8
"""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import Solver, basic_agent, system_message

from .dataset import load_safeagentbench_dataset
from .execution_scorer import execution_scorer
from .scorer import safeagentbench_scorer
from .solvers import build_solver

DEFAULT_JUDGE_MODEL = "openai/GLM-4.7"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.0

VALID_VISUAL_STRATEGIES = ("basic", "pca", "map_vlm")


def _build_task(
    eval_mode: str,
    task_type: str | None,
    judge_model: str | Model | None,
    solver: Solver | list[Solver],
    thor_url: str,
    sample_size: int,
    seed: int,
) -> Task:
    """Internal helper to build a Task with the right dataset, scorer, and config."""
    dataset = load_safeagentbench_dataset(
        task_type=task_type,
        sample_size=sample_size,
        seed=seed,
        eval_mode=eval_mode,
    )

    semantic = safeagentbench_scorer(judge_model=judge_model)
    if eval_mode == "semantic":
        scorers = semantic
    else:
        exec_sc = execution_scorer(thor_url=thor_url)
        scorers = [semantic, exec_sc]

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorers,
        config=GenerateConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        ),
        version="2.1.0",  # 2.1: abstract instruction expanded to 4 levels
    )


_THOR_DONE_DESCRIPTION = (
    "Signal that you have finished interacting with the environment. "
    "Call this when you have completed the task or decided to refuse it."
)


def _build_react_solver() -> Solver:
    """Build interactive ReAct agent.

    Creates a basic_agent that:
    1. Resets AI2-THOR to the sample's scene (thor_scene_init)
    2. Provides thor_execute and thor_observe (text feedback)
    3. Registers thor_done as the submit tool via basic_agent(submit_name=...)
    4. Loops until the model calls thor_done or hits 40 messages
    """
    from .prompts import REACT_SYSTEM_PROMPT
    from .solvers import thor_scene_init
    from .tools import thor_execute, thor_observe

    return basic_agent(
        init=[system_message(REACT_SYSTEM_PROMPT), thor_scene_init()],
        tools=[thor_execute(), thor_observe()],
        max_messages=40,
        submit_name="thor_done",
        submit_description=_THOR_DONE_DESCRIPTION,
    )


def _build_visual_solver(agent_strategy: str = "basic") -> Solver:
    """Build interactive visual agent with the given strategy.

    Provides thor_execute_visual and thor_observe_visual (screenshot feedback).
    Registers thor_done as the submit tool via basic_agent(submit_name=...).

    Strategies:
      basic   — standard visual system prompt
      pca     — Perception-Cognition-Action structured reasoning
      map_vlm — multi-agent visual planning (ENV/GROUNDING/PLANNING)
    """
    if agent_strategy not in VALID_VISUAL_STRATEGIES:
        raise ValueError(
            f"Unknown visual agent_strategy '{agent_strategy}'. "
            f"Valid options: {', '.join(VALID_VISUAL_STRATEGIES)}"
        )

    from .prompts import MAP_VLM_VISUAL_SYSTEM_PROMPT, PCA_VISUAL_SYSTEM_PROMPT, VISUAL_SYSTEM_PROMPT
    from .solvers import thor_scene_init
    from .visual_tools import thor_execute_visual, thor_observe_visual

    prompts = {
        "basic": VISUAL_SYSTEM_PROMPT,
        "pca": PCA_VISUAL_SYSTEM_PROMPT,
        "map_vlm": MAP_VLM_VISUAL_SYSTEM_PROMPT,
    }

    return basic_agent(
        init=[system_message(prompts[agent_strategy]), thor_scene_init()],
        tools=[thor_execute_visual(), thor_observe_visual()],
        max_messages=40,
        submit_name="thor_done",
        submit_description=_THOR_DONE_DESCRIPTION,
    )


# ---------------------------------------------------------------------------
# Public @task functions
# ---------------------------------------------------------------------------

@task
def safeagentbench(
    task_type: str | None = None,
    agent_strategy: str = "direct",
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    with_execution: bool = False,
    thor_url: str = "http://localhost:9100",
    sample_size: int = 100,
    seed: int = 42,
) -> Task:
    """Text planner — 7 agent strategies covering paper baselines 1-5 + cot/two_stage.

    Args:
        task_type: "unsafe", "safe", "all", or None (mixed sampling).
        agent_strategy: Planning strategy. Options:
            "direct"      — LoTa-Bench (https://arxiv.org/abs/2402.08178)
            "cot"         — chain-of-thought reasoning prefix
            "two_stage"   — high-level plan then low-level conversion
            "progprompt"  — ProgPrompt (https://arxiv.org/abs/2209.11302)
            "fewshot"     — LLM-Planner (https://arxiv.org/abs/2212.04088)
            "cooperative" — CoELA (https://arxiv.org/abs/2307.02485)
            "mldt"        — MLDT (https://arxiv.org/abs/2403.18760)
        judge_model: Model for LLM-as-judge evaluation.
        with_execution: If True, also execute plan in AI2-THOR (needs Docker).
        thor_url: URL of AI2-THOR action server.
        sample_size: Number of samples for mixed sampling.
        seed: Random seed for reproducible sampling.
    """
    solver = build_solver(agent_strategy)
    eval_mode = "exec" if with_execution else "semantic"
    return _build_task(eval_mode, task_type, judge_model, solver, thor_url, sample_size, seed)


@task
def safeagentbench_react(
    task_type: str | None = None,
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    thor_url: str = "http://localhost:9100",
    sample_size: int = 100,
    seed: int = 42,
) -> Task:
    """ReAct interactive agent — ReAct baseline (https://arxiv.org/abs/2210.03629).

    Model drives AI2-THOR step-by-step via tool calls with text feedback.
    Always includes execution scoring (requires Docker).
    """
    solver = _build_react_solver()
    return _build_task("react", task_type, judge_model, solver, thor_url, sample_size, seed)


@task
def safeagentbench_visual(
    task_type: str | None = None,
    agent_strategy: str = "basic",
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    thor_url: str = "http://localhost:9100",
    sample_size: int = 100,
    seed: int = 42,
) -> Task:
    """Visual interactive agent — 3 strategies covering paper baselines 7-8.

    Model drives AI2-THOR via tool calls with screenshot feedback.
    Always includes execution scoring (requires Docker + vision model).

    Args:
        agent_strategy: Visual planning strategy. Options:
            "basic"   — standard visual agent
            "pca"     — PCA-EVAL (https://arxiv.org/abs/2402.15527)
            "map_vlm" — MAP (https://arxiv.org/abs/2408.05478)
    """
    solver = _build_visual_solver(agent_strategy)
    return _build_task("visual", task_type, judge_model, solver, thor_url, sample_size, seed)


