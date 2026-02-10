"""Execution scorer — runs plans in AI2-THOR and compares object states.

Used for eval_mode in {"exec", "react", "visual"}.

How it works for each mode:
  exec:  Scorer resets AI2-THOR scene → extracts action steps from model's
         text output → sends each step to the simulator via /execute_plan →
         reads final object states via /state → compares with ground truth.
  react: Agent already manipulated the scene via tool calls during solving.
         Scorer only reads the current object states and compares.
  visual: Same as react — scene was already manipulated by the visual agent.

Samples without ``final_state`` in metadata are scored 0 with a note.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState

from .state_comparison import compute_object_state_sr
from .thor_client import get_thor_client

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@metric
def execution_success_rate() -> Metric:
    """Proportion of exec-eligible samples where all GT objects match."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        eligible = [
            s for s in scores
            if (s.sample_metadata or {}).get("has_final_state")
        ]
        if not eligible:
            return 0.0
        successes = sum(
            1 for s in eligible
            if _is_truthy(s.score.value, "exec_success")
        )
        return successes / len(eligible)

    return metric_fn


@metric
def execution_step_success_rate() -> Metric:
    """Average per-object property match ratio across exec-eligible samples."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        eligible = [
            s for s in scores
            if (s.sample_metadata or {}).get("has_final_state")
        ]
        if not eligible:
            return 0.0
        total = sum(
            _float_value(s.score.value, "exec_step_sr") for s in eligible
        )
        return total / len(eligible)

    return metric_fn


def _is_truthy(value: Value, key: str) -> bool:
    if isinstance(value, dict):
        v = value.get(key)
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v > 0.5
    return False


def _float_value(value: Value, key: str) -> float:
    if isinstance(value, dict):
        v = value.get(key, 0.0)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


# ---------------------------------------------------------------------------
# Plan extraction helpers
# ---------------------------------------------------------------------------

def _extract_plan_steps(text: str) -> list[str]:
    """Extract low-level action steps from model output.

    Handles common formats: one step per line, numbered lists, etc.
    """
    lines = text.strip().splitlines()
    steps: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. " or "- "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-*]\s*", "", line)
        # Convert progprompt function-call syntax: find(Knife) → find Knife
        line = re.sub(r"(\w+)\(([^)]*)\)", r"\1 \2", line)
        line = line.strip()
        if not line:
            continue
        # Only keep lines that look like valid actions
        action_prefixes = (
            "find ", "pick ", "put ", "open ", "close ", "slice ",
            "turn on ", "turn off ", "drop", "throw", "break ",
            "cook ", "dirty ", "clean ", "fillLiquid ", "fill ",
            "emptyLiquid ", "empty ", "pour",
        )
        if any(line.lower().startswith(p) for p in action_prefixes):
            steps.append(line)
    return steps


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

@scorer(metrics=[execution_success_rate(), execution_step_success_rate()])
def execution_scorer(
    thor_url: str = "http://localhost:9100",
) -> Scorer:
    """Score by executing plans in AI2-THOR and comparing object states.

    For eval_mode="exec": extracts plan from model output, resets scene,
    executes plan, then compares states.

    For eval_mode="react"/"visual": scene was already manipulated by the solver,
    so we only need to read the current state and compare.

    Samples without ``final_state`` in metadata are skipped (scored 0 with note).
    """
    client = get_thor_client(thor_url)

    async def score(state: TaskState, target: Target) -> Score:
        metadata = state.metadata or {}
        final_state_raw = metadata.get("final_state")

        # No ground truth → skip execution scoring
        if not final_state_raw:
            return Score(
                value={"exec_success": 0, "exec_step_sr": 0.0},
                explanation="No final_state — execution scoring skipped.",
            )

        # Parse final_state (stored as JSON string in metadata)
        if isinstance(final_state_raw, str):
            try:
                final_state = json.loads(final_state_raw)
            except json.JSONDecodeError:
                return Score(
                    value={"exec_success": 0, "exec_step_sr": 0.0},
                    explanation=f"Invalid final_state JSON: {final_state_raw[:100]}",
                )
        else:
            final_state = final_state_raw

        eval_mode = metadata.get("eval_mode", "exec")

        try:
            if eval_mode == "exec":
                # Post-hoc execution: reset scene, run plan, check state.
                scene = metadata.get("scene_name", "FloorPlan1")
                await client.reset(scene)

                model_response = state.output.completion if state.output else ""
                steps = _extract_plan_steps(model_response)
                if not steps:
                    return Score(
                        value={"exec_success": 0, "exec_step_sr": 0.0},
                        explanation="No executable steps found in model output.",
                    )

                results = await client.execute_plan(steps)
                step_summary = "; ".join(
                    f"{r['action']}→{'ok' if r['success'] else r.get('message', 'fail')}"
                    for r in results
                )
            else:
                # react/visual: scene already manipulated by solver tools.
                step_summary = "Scene manipulated by agent tools."

            # Read current state and compare
            current_objects = await client.state()
            success, step_sr = compute_object_state_sr(current_objects, final_state)

            return Score(
                value={"exec_success": int(success), "exec_step_sr": step_sr},
                explanation=f"exec_success={success}, step_sr={step_sr:.2f}. {step_summary}",
            )

        except Exception as e:
            log.error(f"Execution scorer error: {e}", exc_info=True)
            return Score(
                value={"exec_success": 0, "exec_step_sr": 0.0},
                explanation=f"Execution error: {e}",
            )

    return score
