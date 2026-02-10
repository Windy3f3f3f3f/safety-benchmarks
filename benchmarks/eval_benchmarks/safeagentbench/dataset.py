"""SafeAgentBench dataset loading and sampling.

Loads 4 JSONL files (safe_detailed, unsafe_detailed, abstract, long_horizon)
and applies mixed sampling strategy for evaluation.

Abstract records contain 4 instruction variants per sample at different
abstraction levels (0=most specific, 3=most abstract). Each variant is
expanded into a separate Sample with ``abstraction_level`` metadata.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from inspect_ai.dataset import MemoryDataset, Sample

from .prompts import INTERACTIVE_TASK_PROMPT, PLANNING_SYSTEM_PROMPT, PLANNING_TASK_PROMPT

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
    eval_mode: str = "semantic",
    abstraction_level: int | None = None,
) -> Sample:
    """Convert a dataset record to an inspect_ai Sample.

    Args:
        abstraction_level: For abstract tasks, 0 (most specific) to 3 (most abstract).
            None for non-abstract tasks.
    """
    task = record["instruction"]
    if eval_mode in ("react", "visual"):
        prompt = INTERACTIVE_TASK_PROMPT.format(task=task)
    else:
        prompt = PLANNING_TASK_PROMPT.format(task=task)

    # Reference steps for judge evaluation
    ref_steps = record.get("step", [])
    if isinstance(ref_steps, list):
        ref_steps_str = "\n".join(ref_steps)
    else:
        ref_steps_str = str(ref_steps)

    # final_state for execution evaluation (None → no exec eval possible)
    final_state = record.get("final_state")
    has_final_state = final_state is not None

    # Store final_state as JSON string to survive metadata serialization
    final_state_json = json.dumps(final_state) if final_state is not None else None

    id_suffix = f"_L{abstraction_level}" if abstraction_level is not None else ""

    metadata: dict[str, Any] = {
        "task_instruction": task,
        "safety_label": safety_label,
        "task_type": task_type,
        "risk_category": _normalize_risk_category(
            record.get("risk_category", "")
        ),
        "reference_steps": ref_steps_str,
        "scene_name": record.get("scene_name", ""),
        "final_state": final_state_json,
        "has_final_state": has_final_state,
        "eval_mode": eval_mode,
    }
    if abstraction_level is not None:
        metadata["abstraction_level"] = abstraction_level

    return Sample(
        input=prompt,
        target="",
        id=f"safeagentbench_{task_type}_{index}{id_suffix}",
        metadata=metadata,
    )


def _expand_abstract_record(
    record: dict[str, Any],
    index: int,
    eval_mode: str = "semantic",
) -> list[Sample]:
    """Expand one abstract record into 4 samples (one per abstraction level).

    Each abstract record contains instruction as a list of 4 strings:
      [0] most specific, [1] somewhat abstract, [2] more abstract, [3] most abstract.
    Returns 4 Samples sharing the same index but distinguished by L0-L3 suffix.
    """
    instructions = record["instruction"]
    if not isinstance(instructions, list):
        instructions = [instructions]

    samples = []
    for level, instruction in enumerate(instructions):
        r_copy = dict(record)
        r_copy["instruction"] = instruction
        samples.append(
            _record_to_sample(
                r_copy, "abstract", "unsafe", index, eval_mode,
                abstraction_level=level,
            )
        )
    return samples


def load_safeagentbench_dataset(
    task_type: str | None = None,
    sample_size: int | None = 100,
    seed: int = 42,
    eval_mode: str = "semantic",
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
            samples.append(_record_to_sample(r, "safe_detailed", "safe", idx, eval_mode))
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench_safe")

    if task_type == "unsafe":
        for r in unsafe_detailed:
            samples.append(
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx, eval_mode)
            )
            idx += 1
        for r in abstract_raw:
            samples.extend(_expand_abstract_record(r, idx, eval_mode))
            idx += 1
        for r in long_horizon:
            samples.append(
                _record_to_sample(r, "long_horizon", "unsafe", idx, eval_mode)
            )
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench_unsafe")

    if task_type == "all":
        for r in unsafe_detailed:
            samples.append(
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx, eval_mode)
            )
            idx += 1
        for r in safe_detailed:
            samples.append(_record_to_sample(r, "safe_detailed", "safe", idx, eval_mode))
            idx += 1
        for r in abstract_raw:
            samples.extend(_expand_abstract_record(r, idx, eval_mode))
            idx += 1
        for r in long_horizon:
            samples.append(
                _record_to_sample(r, "long_horizon", "unsafe", idx, eval_mode)
            )
            idx += 1
        return MemoryDataset(samples=samples, name="safeagentbench")

    # Default: mixed sampling (50 unsafe_detailed + 30 safe + 40 abstract [10 records × 4 levels] + 10 long_horizon = 130)
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
                _record_to_sample(r, "unsafe_detailed", "unsafe", idx, eval_mode)
            )
            idx += 1

    # 30 safe detailed
    for r in rng.sample(safe_detailed, 30):
        samples.append(_record_to_sample(r, "safe_detailed", "safe", idx, eval_mode))
        idx += 1

    # 10 abstract records × 4 abstraction levels = 40 samples
    for r in rng.sample(abstract_raw, 10):
        samples.extend(_expand_abstract_record(r, idx, eval_mode))
        idx += 1

    # 10 long horizon
    for r in rng.sample(long_horizon, 10):
        samples.append(_record_to_sample(r, "long_horizon", "unsafe", idx, eval_mode))
        idx += 1

    return MemoryDataset(samples=samples, name="safeagentbench")
