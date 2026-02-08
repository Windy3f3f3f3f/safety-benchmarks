"""SafeAgentBench dataset loading and sampling.

Loads 4 JSONL files (safe_detailed, unsafe_detailed, abstract, long_horizon)
and applies mixed sampling strategy for evaluation.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from .prompts import PLANNING_SYSTEM_PROMPT, PLANNING_TASK_PROMPT

DATA_DIR = Path(__file__).parent / "data"

# 10 primary risk categories for stratified sampling
PRIMARY_RISK_CATEGORIES = [
    "Fire Hazard",
    "Electrical Shock Hazard",
    "Explosion Hazard",
    "Slip Hazard",
    "Liquid and Spill Damage",
    "Poisoning/Ingestion Hazard",
    "Breakage and Dropping",
    "Furniture and Decor Damage",
    "Misuse of Electrical Appliances",
    "Damage to Small Items",
]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _normalize_risk_category(cat: str) -> str:
    """Normalize noisy category labels to one of 10 primary categories."""
    cat_lower = cat.lower().strip()
    cat_lower = re.sub(r"^risk\s*(?:category)?\s*:\s*", "", cat_lower)
    for primary in PRIMARY_RISK_CATEGORIES:
        if primary.lower() in cat_lower:
            return primary
    return cat


def _record_to_sample(
    record: dict[str, Any],
    task_type: str,
    safety_label: str,
    index: int,
) -> Sample:
    """Convert a dataset record to an inspect_ai Sample."""
    task = record["instruction"]
    prompt = PLANNING_TASK_PROMPT.format(task=task)

    # Reference steps for judge evaluation
    ref_steps = record.get("step", [])
    if isinstance(ref_steps, list):
        ref_steps_str = "\n".join(ref_steps)
    else:
        ref_steps_str = str(ref_steps)

    return Sample(
        input=prompt,
        target="",
        id=f"safeagentbench_{task_type}_{index}",
        metadata={
            "task_instruction": task,
            "safety_label": safety_label,
            "task_type": task_type,
            "risk_category": _normalize_risk_category(
                record.get("risk_category", "")
            ),
            "reference_steps": ref_steps_str,
            "scene_name": record.get("scene_name", ""),
        },
    )


def load_safeagentbench_dataset(
    task_type: str | None = None,
    sample_size: int | None = 100,
    seed: int = 42,
) -> MemoryDataset:
    """Load SafeAgentBench dataset with mixed sampling.

    Args:
        task_type: Filter by type. None = mixed sampling.
            "unsafe" = only unsafe tasks (detailed + abstract).
            "safe" = only safe detailed tasks.
            "all" = all records without sampling.
        sample_size: Target sample count for mixed sampling (only for task_type=None).
        seed: Random seed for reproducible sampling.

    Returns:
        MemoryDataset containing Sample objects.
    """
    unsafe_detailed = _load_jsonl(DATA_DIR / "unsafe_detailed_1009.jsonl")
    safe_detailed = _load_jsonl(DATA_DIR / "safe_detailed_1009.jsonl")
    abstract_raw = _load_jsonl(DATA_DIR / "abstract_1009.jsonl")
    long_horizon = _load_jsonl(DATA_DIR / "long_horizon_1009.jsonl")

    samples: list[Sample] = []
    idx = 0

    if task_type == "safe":
        for r in safe_detailed:
            samples.append(_record_to_sample(r, "safe_detailed", "safe", idx))
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench_safe")

    if task_type == "unsafe":
        for r in unsafe_detailed:
            samples.append(
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx)
            )
            idx += 1
        for r in abstract_raw:
            # Use first instruction variant for abstract tasks
            if isinstance(r["instruction"], list):
                r = dict(r)
                r["instruction"] = r["instruction"][0]
            samples.append(_record_to_sample(r, "abstract", "unsafe", idx))
            idx += 1
        for r in long_horizon:
            samples.append(
                _record_to_sample(r, "long_horizon", "unsafe", idx)
            )
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench_unsafe")

    if task_type == "all":
        for r in unsafe_detailed:
            samples.append(
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx)
            )
            idx += 1
        for r in safe_detailed:
            samples.append(_record_to_sample(r, "safe_detailed", "safe", idx))
            idx += 1
        for r in abstract_raw:
            if isinstance(r["instruction"], list):
                r = dict(r)
                r["instruction"] = r["instruction"][0]
            samples.append(_record_to_sample(r, "abstract", "unsafe", idx))
            idx += 1
        for r in long_horizon:
            samples.append(
                _record_to_sample(r, "long_horizon", "unsafe", idx)
            )
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench")

    # Default: mixed sampling (50 unsafe_detailed + 30 safe + 10 abstract + 10 long_horizon)
    rng = random.Random(seed)

    # 50 unsafe detailed — stratified by 10 categories, 5 each
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in unsafe_detailed:
        norm = _normalize_risk_category(r.get("risk_category", ""))
        if norm in PRIMARY_RISK_CATEGORIES:
            by_cat[norm].append(r)
    for cat in PRIMARY_RISK_CATEGORIES:
        pool = by_cat[cat]
        chosen = rng.sample(pool, min(5, len(pool)))
        for r in chosen:
            samples.append(
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx)
            )
            idx += 1

    # 30 safe detailed
    for r in rng.sample(safe_detailed, 30):
        samples.append(_record_to_sample(r, "safe_detailed", "safe", idx))
        idx += 1

    # 10 abstract (first instruction variant)
    for r in rng.sample(abstract_raw, 10):
        r = dict(r)
        if isinstance(r["instruction"], list):
            r["instruction"] = r["instruction"][0]
        samples.append(_record_to_sample(r, "abstract", "unsafe", idx))
        idx += 1

    # 10 long horizon
    for r in rng.sample(long_horizon, 10):
        samples.append(_record_to_sample(r, "long_horizon", "unsafe", idx))
        idx += 1

    return MemoryDataset(samples=samples, name="safeagentbench")
