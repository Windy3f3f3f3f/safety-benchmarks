"""SafeAgentBench: Evaluating Embodied LLM Agent Safety

Yin et al. "SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents"
https://arxiv.org/abs/2412.13178

This benchmark evaluates whether LLM agents can:
1. Refuse dangerous household task instructions (unsafe tasks)
2. Successfully plan safe household tasks (safe tasks)
3. Comply with safety constraints in multi-step tasks (long-horizon tasks)

## Evaluation Modes (eval_mode parameter)

Four modes, from lightweight to full-stack:

### semantic (default) — pure text, no simulator
  Model reads task instruction → generates action plan text →
  LLM judge evaluates plan quality against reference steps.
  No AI2-THOR involved. Fastest mode.

### exec — model generates text plan, scorer executes in simulator
  Model reads task instruction → generates action plan text (same as semantic) →
  Scorer extracts action steps from text → resets AI2-THOR 3D scene →
  executes each step in simulator → reads final object states →
  compares with ground-truth states.
  Model itself never interacts with AI2-THOR; execution is post-hoc.

### react — model interacts with simulator via tool calls, text feedback
  AI2-THOR scene is reset → model receives task instruction →
  model calls thor_execute("find Knife") tool → tool sends HTTP request
  to AI2-THOR Docker container → container executes action in Unity simulator →
  returns success/failure text to model → model decides next action →
  ... loop until model calls thor_done().
  Model drives the simulator step-by-step through tool-use, receiving
  text descriptions of action results and object states.

### visual — same as react, but with screenshots
  Same interactive loop as react, but each tool call also returns a
  PNG screenshot from AI2-THOR's in-simulator camera. The model can
  see the 3D environment and use visual information to plan actions.
  Requires a vision-capable model.

## Prerequisites for exec/react/visual modes

AI2-THOR runs in a Docker container with GPU (NVIDIA + Vulkan + Xvfb):
  cd benchmarks/eval_benchmarks/safeagentbench/docker
  docker compose up -d --build
  curl http://localhost:9100/health  # should return {"status": "ok"}

## Task Parameters (all via -T key=value)

  eval_mode     Evaluation mode. Default: "semantic".
                "semantic" | "exec" | "react" | "visual" (see above).

  task_type     Which tasks to include. Default: None (mixed sampling).
                None  — mixed sampling: 50 unsafe + 30 safe + 10 abstract + 10 long_horizon.
                "unsafe" — all unsafe tasks (unsafe_detailed + abstract + long_horizon).
                "safe"   — all safe_detailed tasks only.
                "all"    — all 750 samples without sampling.

  agent_strategy  Planning strategy for semantic/exec modes. Default: "direct".
                  "direct"    — system prompt + single generation.
                  "cot"       — chain-of-thought prefix before generation.
                  "two_stage" — generate high-level plan, then convert to low-level actions.
                  Ignored for react/visual (they use basic_agent with tool-use loop).

  judge_model   Model for LLM-as-judge semantic evaluation. Default: "openai/GLM-4.7".
                Requires ZHIPU_API_KEY and ZHIPU_BASE_URL env vars if using Zhipu API.

  thor_url      URL of AI2-THOR action server. Default: "http://localhost:9100".
                Only used for exec/react/visual modes.

  sample_size   Number of samples for mixed sampling. Default: 100.
                Only effective when task_type is None.

  seed          Random seed for reproducible sampling. Default: 42.

## Task Variants

  safeagentbench         Main task, supports all parameters above.
  safeagentbench_unsafe  Shortcut: task_type="unsafe" (all unsafe tasks).
  safeagentbench_safe    Shortcut: task_type="safe" (all safe tasks).

## Output Metrics

  Semantic scorer (all modes):
    rejection_rate      — proportion of unsafe tasks where model refused (higher = safer).
    safe_success_rate   — proportion of safe tasks with successful planning.
    risk_rate           — 1 - rejection_rate (proportion of unsafe tasks where model complied).

  Execution scorer (exec/react/visual only):
    execution_success_rate      — proportion of samples where all GT objects fully match.
    execution_step_success_rate — average per-object property match ratio.

## Usage Examples

  # semantic — pure text evaluation, no simulator needed
  inspect eval eval_benchmarks/safeagentbench --model openai/doubao-seed-1-8

  # semantic with chain-of-thought strategy
  inspect eval eval_benchmarks/safeagentbench -T agent_strategy=cot --model openai/doubao-seed-1-8

  # exec — model generates plan text, scorer executes in AI2-THOR
  inspect eval eval_benchmarks/safeagentbench -T eval_mode=exec --model openai/doubao-seed-1-8

  # react — model drives AI2-THOR step-by-step via tool calls, text feedback
  inspect eval eval_benchmarks/safeagentbench -T eval_mode=react --model openai/doubao-seed-1-8

  # visual — same as react but with screenshots (needs vision model)
  inspect eval eval_benchmarks/safeagentbench -T eval_mode=visual --model openai/doubao-seed-1-8

  # only unsafe tasks, limit to 10 samples
  inspect eval eval_benchmarks/safeagentbench_unsafe --model openai/doubao-seed-1-8 --limit 10

  # only safe tasks with exec evaluation
  inspect eval eval_benchmarks/safeagentbench_safe -T eval_mode=exec --model openai/doubao-seed-1-8
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

VALID_EVAL_MODES = ("semantic", "exec", "react", "visual")


def _build_react_solver() -> Solver:
    """Build interactive ReAct agent for eval_mode=react.

    Creates a basic_agent that:
    1. Resets AI2-THOR to the sample's scene (thor_scene_init)
    2. Gives the model 3 tools to interact with the simulator:
       - thor_execute(instruction): run an action, get text result from simulator
       - thor_observe(): read all object states from simulator as text summary
       - thor_done(reason): signal completion or refusal
    3. Loops up to 40 messages (model thinks → calls tool → gets result → ...)
    """
    from .prompts import REACT_SYSTEM_PROMPT
    from .solvers import thor_scene_init
    from .tools import thor_done, thor_execute, thor_observe

    return basic_agent(
        init=[system_message(REACT_SYSTEM_PROMPT), thor_scene_init()],
        tools=[thor_execute(), thor_observe(), thor_done()],
        max_messages=40,
    )


def _build_visual_solver() -> Solver:
    """Build interactive visual agent for eval_mode=visual.

    Same structure as react, but each tool call also returns a PNG screenshot
    from AI2-THOR's in-simulator camera as ContentImage, so the model can
    see the 3D environment. Requires a vision-capable model.
    """
    from .prompts import VISUAL_SYSTEM_PROMPT
    from .solvers import thor_scene_init
    from .visual_tools import thor_done, thor_execute_visual, thor_observe_visual

    return basic_agent(
        init=[system_message(VISUAL_SYSTEM_PROMPT), thor_scene_init()],
        tools=[thor_execute_visual(), thor_observe_visual(), thor_done()],
        max_messages=40,
    )


@task
def safeagentbench(
    task_type: str | None = None,
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
    eval_mode: str = "semantic",
    thor_url: str = "http://localhost:9100",
    sample_size: int = 100,
    seed: int = 42,
) -> Task:
    """SafeAgentBench: evaluate embodied LLM agent safety.

    Args:
        task_type: "unsafe", "safe", "all", or None (mixed sampling).
        judge_model: Model for LLM-as-judge evaluation.
        agent_strategy: Solver strategy — "direct", "cot", or "two_stage".
        eval_mode: Evaluation mode — "semantic", "exec", "react", or "visual".
        thor_url: URL of AI2-THOR action server (for exec/react/visual modes).
        sample_size: Number of samples for mixed sampling.
        seed: Random seed for reproducible sampling.
    """
    if eval_mode not in VALID_EVAL_MODES:
        raise ValueError(
            f"Unknown eval_mode '{eval_mode}'. Valid: {', '.join(VALID_EVAL_MODES)}"
        )

    dataset = load_safeagentbench_dataset(
        task_type=task_type,
        sample_size=sample_size,
        seed=seed,
        eval_mode=eval_mode,
    )

    # --- Solver ---
    # semantic/exec: model generates plan text (no simulator interaction).
    #   For exec, the plan text is executed post-hoc by the scorer.
    # react/visual: model interacts with AI2-THOR via tool calls in a loop.
    if eval_mode in ("semantic", "exec"):
        solver = build_solver(agent_strategy)
    elif eval_mode == "react":
        solver = _build_react_solver()
    else:  # visual
        solver = _build_visual_solver()

    # --- Scorer ---
    # All modes get semantic scoring (LLM judge evaluates plan text).
    # exec/react/visual additionally get execution scoring:
    #   exec: scorer resets scene, executes extracted plan, compares object states.
    #   react/visual: scene was already manipulated by agent tools, scorer just
    #                 reads final object states and compares with ground truth.
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
        version="1.0.0",
    )


@task
def safeagentbench_unsafe(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
    eval_mode: str = "semantic",
    thor_url: str = "http://localhost:9100",
) -> Task:
    """Evaluate only unsafe tasks — measures rejection rate.

    Includes unsafe_detailed, abstract, and long_horizon tasks.
    Primary safety metric: rejection_rate (higher = safer).
    """
    return safeagentbench(
        task_type="unsafe",
        judge_model=judge_model,
        agent_strategy=agent_strategy,
        eval_mode=eval_mode,
        thor_url=thor_url,
    )


@task
def safeagentbench_safe(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    agent_strategy: str = "direct",
    eval_mode: str = "semantic",
    thor_url: str = "http://localhost:9100",
) -> Task:
    """Evaluate only safe tasks — measures planning capability.

    Includes safe_detailed tasks only.
    Primary metric: safe_success_rate (higher = more capable).
    """
    return safeagentbench(
        task_type="safe",
        judge_model=judge_model,
        agent_strategy=agent_strategy,
        eval_mode=eval_mode,
        thor_url=thor_url,
    )
