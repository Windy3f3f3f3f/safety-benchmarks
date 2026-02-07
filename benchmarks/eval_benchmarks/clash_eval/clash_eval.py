"""
ClashEval: Context-Knowledge Clash Evaluation

Evaluates how easily a model can be misled by false information injected
into RAG retrieval context.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig
from inspect_ai.solver import generate

from .prompt import RAG_RESPONSE
from .scorer import context_adherence

# Path to the local parquet dataset
_DATA_PATH = Path(__file__).parent / "data" / "dataset.pqt"

# Map from prompt.py dict keys
RAG_SYSTEM_PROMPTS = {k: v for k, v in RAG_RESPONSE.items() if k != "user"}


def _get_rag_user_prompt(question: str, context: str) -> str:
    """Build user prompt using the RAG_RESPONSE 'user' template."""
    return RAG_RESPONSE["user"].format(question=question, context=context)


def record_to_sample(record: dict[str, Any], idx: int) -> Sample:
    """Convert a dataset record to an Inspect Sample with dynamic system prompt.

    Args:
        record: Dict with keys: question, dataset, context_mod, etc.
        idx: The index of the record in the dataframe (for unique ID).
    """
    domain = str(record.get("dataset", "names"))
    question = str(record.get("question", ""))
    context = str(record.get("context_mod", ""))
    answer_mod = str(record.get("answer_mod", ""))
    answer_original = str(record.get("answer_original", ""))
    mod_type = str(record.get("mod_type", ""))

    # Get domain-specific system prompt (fallback to 'names')
    system_prompt = RAG_SYSTEM_PROMPTS.get(domain, RAG_SYSTEM_PROMPTS.get("names", ""))

    # Build user prompt
    user_prompt = _get_rag_user_prompt(question, context)

    # Construct message list for dynamic system prompt
    messages = [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=user_prompt),
    ]

    # Create stable and UNIQUE sample ID using the index
    # Format: clash_{domain}_{index}
    sample_id = f"clash_{domain}_{idx:06d}"

    return Sample(
        input=messages,
        target=answer_mod,
        id=sample_id,
        metadata={
            "answer_original": answer_original,
            "answer_mod": answer_mod,
            "mod_type": mod_type,
            "dataset": domain,
            "question": question,
        },
    )


def load_clash_eval_dataset(
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
    domains: list[str] | None = None,
) -> MemoryDataset:
    """Load ClashEval dataset from local parquet file."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(
            f"ClashEval dataset not found at {_DATA_PATH}. "
            f"Please place the cleaned dataset.pqt file in "
            f"'{_DATA_PATH.parent}/'."
        )

    df = pd.read_parquet(_DATA_PATH)

    # Filter by domain if specified
    if domains:
        df = df[df["dataset"].isin(domains)]

    # Shuffle
    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Apply limit
    if limit is not None:
        df = df.head(limit)

    # Convert to samples (Pass index 'i' to record_to_sample)
    records = df.to_dict("records")
    samples = [record_to_sample(r, i) for i, r in enumerate(records)]

    return MemoryDataset(samples)


@task
def clash_eval(
    limit: int | None = None,
    shuffle: bool = False,
    domains: list[str] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> Task:
    """ClashEval: Context-Knowledge Clash Evaluation."""
    return Task(
        dataset=load_clash_eval_dataset(
            limit=limit,
            shuffle=shuffle,
            domains=domains,
        ),
        solver=[
            generate(),
        ],
        scorer=context_adherence(),
        config=GenerateConfig(
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        version="1.0.0",
    )

__all__ = [
    "clash_eval",
    "load_clash_eval_dataset",
    "record_to_sample",
]