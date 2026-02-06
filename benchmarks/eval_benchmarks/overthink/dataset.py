"""
Dataset loading for OverThink Benchmark.

This module provides functionality to load and filter the FreshQA dataset
for use in the slowdown attack evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    pass

from inspect_ai.dataset import MemoryDataset, Sample

try:
    from .templates import (
        MDP_PROBLEM_TEMPLATES,
        TARGET_CONTEXT_TEMPLATES,
    )
except ImportError:
    from templates import (
        MDP_PROBLEM_TEMPLATES,
        TARGET_CONTEXT_TEMPLATES,
    )
from inspect_evals.utils import create_stable_id

# Path to the FreshQA dataset file
# Users should download freshqa.csv from the FreshQA GitHub repository:
# https://github.com/freshllms/freshqa
# and place it in the data directory.
_FRESHQA_DATA_PATH = Path(__file__).parent / "data" / "freshqa.csv"


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert a FreshQA record to an Inspect Sample.

    Args:
        record: A dictionary containing FreshQA data with keys:
            - question: The question text
            - source: Wikipedia source URL(s)
            - fact_type: Type of fact (never-changing, slow-changing, fast-changing)
            - answer: The correct answer
            - keywords: Relevant keywords

    Returns:
        A Sample object with the question as input and metadata.
    """
    question = record.get("question", "")
    source = record.get("source", "")
    fact_type = record.get("fact_type", "")
    answer = record.get("answer", "")
    keywords = record.get("keywords", "")

    # Create a stable ID from the question content
    sample_id = create_stable_id(question, prefix="overthink")

    # Store relevant metadata for the evaluation
    metadata = {
        "fact_type": fact_type,
        "source": source,
        "answer": answer,
        "keywords": keywords,
    }

    return Sample(
        input=question,
        id=sample_id,
        metadata=metadata,
    )


def expand_dataset_with_templates(
    samples: list[Sample],
    attack_type: Literal["context_agnostic", "context_aware"],
) -> list[Sample]:
    """Expand each sample into multiple variants with different template indices.

    For context_agnostic: 10 variants (10 TARGET_CONTEXT_TEMPLATES)
    For context_aware: 6 variants (6 MDP_PROBLEM_TEMPLATES)

    Args:
        samples: List of base samples to expand
        attack_type: Type of attack ("context_agnostic" or "context_aware")

    Returns:
        Expanded list of samples with template_index metadata.
    """
    if attack_type == "context_agnostic":
        num_templates = len(TARGET_CONTEXT_TEMPLATES)  # 10 templates
    else:  # context_aware
        num_templates = len(MDP_PROBLEM_TEMPLATES)  # 6 templates

    expanded_samples = []

    for sample in samples:
        # Create attack variants (template_index = 0 to num_templates-1)
        for template_idx in range(num_templates):
            attack_sample = Sample(
                input=sample.input,
                id=f"{sample.id}_t{template_idx}",
                metadata={
                    **(sample.metadata if sample.metadata else {}),
                    "attack_type": attack_type,
                    "template_index": template_idx,
                },
            )
            expanded_samples.append(attack_sample)

    return expanded_samples


def load_freshqa_dataset(
    shuffle: bool = True,
    limit: int | None = None,
    seed: int | None = None,
    num_samples: int = 5,
    attack_type: Literal["context_agnostic", "context_aware"] = "context_agnostic",
) -> MemoryDataset:
    """Load and filter the FreshQA dataset.

    The dataset is filtered to only include "never-changing" and "slow-changing"
    facts with Wikipedia sources. This is the same filtering used in the
    original OverThink benchmark.

    Each sample is expanded into multiple variants:
    - For context_agnostic: 10 variants (10 templates)
    - For context_aware: 6 variants (6 templates)

    The limit parameter applies to the final expanded dataset.

    Args:
        shuffle: Whether to shuffle the dataset
        limit: Maximum number of samples to return AFTER expansion
        seed: Random seed for shuffling
        num_samples: Number of base FreshQA samples to load BEFORE expansion
        attack_type: Type of attack for template expansion

    Returns:
        A Dataset of Samples for evaluation.

    Raises:
        FileNotFoundError: If the FreshQA CSV file is not found.
    """
    if not _FRESHQA_DATA_PATH.exists():
        raise FileNotFoundError(
            f"FreshQA dataset not found at {_FRESHQA_DATA_PATH}. "
            f"Please download 'freshqa.csv' from the FreshQA repository "
            f"(https://github.com/freshllms/freshqa) and place it in "
            f"'{Path(__file__).parent / 'data'}/'."
        )

    # Load the JSON/CSV dataset
    # The FreshQA CSV has a specific format with header rows
    # We'll use json_dataset which can handle CSV-like structures
    # or we can use pandas directly

    import pandas as pd

    # Read CSV with skiprows=2 to handle FreshQA format
    df = pd.read_csv(_FRESHQA_DATA_PATH, skiprows=2)

    # Filter to only never-changing and slow-changing facts with Wikipedia sources
    filtered_df = df[
        (df["fact_type"].isin(["never-changing", "slow-changing"]))
        & (df["source"].str.contains("https://en.wikipedia.org", na=False, case=False))
    ]

    # Convert to list of dicts
    records = filtered_df.to_dict("records")

    # Shuffle records before sampling if requested
    if shuffle:
        if seed is not None:
            import random
            random.seed(seed)
        import random
        random.shuffle(records)

    # Apply num_samples limit BEFORE expansion (base samples)
    if num_samples is not None:
        records = records[:num_samples]

    # Create Samples (cast each record to dict[str, Any])
    samples = [
        record_to_sample({str(k): v for k, v in record.items()}) for record in records
    ]

    # Expand dataset with template variants
    expanded_samples = expand_dataset_with_templates(samples, attack_type)

    # Shuffle expanded samples if requested
    if shuffle:
        if seed is not None:
            import random
            random.seed(seed + 1)  # Different seed for expanded shuffle
        import random
        random.shuffle(expanded_samples)

    # Apply limit AFTER expansion (final dataset)
    if limit is not None:
        expanded_samples = expanded_samples[:limit]

    return MemoryDataset(expanded_samples)


__all__ = [
    "record_to_sample",
    "load_freshqa_dataset",
    "expand_dataset_with_templates",
]
