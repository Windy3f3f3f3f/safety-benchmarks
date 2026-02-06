"""
PersonalizedSafety: Evaluating LLM Safety in High-Risk Personalized Scenarios

Wu et al. "Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach"
https://arxiv.org/abs/2505.18882

This benchmark evaluates models on their ability to provide safe, empathetic,
and personalized responses to users in sensitive situations (mental health,
self-harm risk, etc.).

# Run context-free evaluation (paper Section 4.1 baseline)
inspect eval personalized_safety --model openai/gpt-4o --limit 50

# Run with AIHubMix endpoint
export OPENAI_BASE_URL=https://aihubmix.com/v1
inspect eval personalized_safety --model openai/deepseek-v3.2 --limit 50

# Run context-rich evaluation
inspect eval personalized_safety -T context_mode=context_rich --model openai/gpt-4o

# Custom judge model
inspect eval personalized_safety -T judge_model=openai/gpt-4o --model openai/deepseek-v3.2
"""

from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import generate, system_message

from .prompts import (
    CONTEXT_FREE_PROMPT,
    CONTEXT_RICH_PROMPT,
    SYSTEM_PROMPT,
)
from .scorer import personalized_safety_scorer
from .utils import build_background_description
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "wick1d/Personalized_Safety_Data"

DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7


def record_to_sample(
    context_mode: Literal["context_free", "context_rich"] = "context_free",
) -> Any:
    """Create a function to convert dataset records to Sample objects.

    Args:
        context_mode: Whether to include background in model input
            - "context_free": Model only sees query (paper Section 4.1)
            - "context_rich": Model sees user background + query

    Returns:
        Function that converts records to Sample objects
    """

    def converter(record: dict[str, Any]) -> Sample:
        query = record["query"]
        background_desc = build_background_description(record)

        # Build input based on context mode
        if context_mode == "context_free":
            input_text = CONTEXT_FREE_PROMPT.format(query=query)
        else:
            input_text = CONTEXT_RICH_PROMPT.format(
                background=background_desc,
                query=query,
            )

        return Sample(
            input=input_text,
            target="",  # No expected output for this benchmark
            metadata={
                "query": query,
                "background_description": background_desc,
                "scenario": record.get("scenario", "unknown"),
                "source": record.get("source", "unknown"),
                "context_mode": context_mode,
                # Store profile fields for analysis
                "age": record.get("age"),
                "gender": record.get("gender"),
                "mental_health_status": record.get("mental_health_status"),
                "emotional_state": record.get("emotional_state"),
            },
        )

    return converter


@task
def personalized_safety(
    context_mode: Literal["context_free", "context_rich"] = "context_free",
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
    judge_max_tokens: int = 800,
    judge_temperature: float = 0.7,
) -> Task:
    """PersonalizedSafety benchmark for evaluating LLM safety in high-risk scenarios.

    This benchmark evaluates models on their ability to provide safe, empathetic,
    and personalized responses to users in sensitive situations (mental health,
    self-harm risk, etc.).

    Evaluation dimensions:
    - Risk Sensitivity (1-5): Identifying and addressing safety risks
    - Emotional Empathy (1-5): Understanding user's emotional state
    - Personalization (1-5): Tailoring advice to user's background

    Args:
        context_mode: Evaluation mode
            - "context_free": Model only sees query (paper Section 4.1 baseline)
            - "context_rich": Model sees user background + query
        judge_model: Model for evaluating responses (default: gpt-4o)
        judge_max_tokens: Max tokens for judge evaluation
        judge_temperature: Temperature for judge model

    Returns:
        Task object for running the evaluation
    """
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="train",  # Dataset only has train split
        sample_fields=record_to_sample(context_mode),
    )

    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_PROMPT),
            generate(),
        ],
        scorer=personalized_safety_scorer(
            judge_model=judge_model,
            judge_max_tokens=judge_max_tokens,
            judge_temperature=judge_temperature,
        ),
        config=GenerateConfig(
            temperature=DEFAULT_TEMPERATURE,
            max_tokens=DEFAULT_MAX_TOKENS,
        ),
        version="1.0.0",
    )


@task
def personalized_safety_context_free(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
) -> Task:
    """Context-free evaluation (paper Section 4.1 baseline).

    Model only sees the user query, no background information.
    Judge still evaluates with full background context.
    """
    return personalized_safety(
        context_mode="context_free",
        judge_model=judge_model,
    )


@task
def personalized_safety_context_rich(
    judge_model: str | Model | None = DEFAULT_JUDGE_MODEL,
) -> Task:
    """Context-rich evaluation with user background.

    Model sees both user background and query.
    Judge evaluates with full background context.
    """
    return personalized_safety(
        context_mode="context_rich",
        judge_model=judge_model,
    )
