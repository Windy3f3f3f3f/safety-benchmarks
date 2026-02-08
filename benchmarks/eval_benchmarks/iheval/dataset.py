"""
IHEval Benchmark - Dataset Loading

This module handles loading data from the IHEval benchmark and converting it
to Inspect Sample objects for evaluation.

IHEval evaluates models on following the instruction hierarchy:
system message > user message > conversation history > tool output

Categories:
- rule-following: Single-turn and multi-turn instruction following (IFEval-based)
- task-execution: NLP tasks (extraction, translation, classification)
- safety: Security scenarios (hijack, extraction)
- tool-use: Function calling scenarios (intrinsic, injected)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser

if TYPE_CHECKING:
    pass


# Path to the IHEval data directory (data is in the data/ subdirectory)
_IHEVAL_DATA_PATH = Path(__file__).parent / "data"


# Category definitions
IHEVAL_CATEGORY = Literal[
    "rule-following",
    "task-execution",
    "safety",
    "tool-use",
]

IHEVAL_TASK = Literal[
    # rule-following
    "single-turn",
    "multi-turn",
    # task-execution
    "extraction",
    "translation",
    "lang-detect",
    # safety
    "hijack",
    "system-prompt-extract",
    # tool-use
    "get-webpage",
    "slack-user",
]

IHEVAL_SETTING = Literal[
    "aligned",
    "conflict",
    "reference",
]

IHEVAL_STRENGTH = Literal[
    "default",
    "weak",
    "strong",
]


def get_data_path(
    category: IHEVAL_CATEGORY,
    task: IHEVAL_TASK,
    setting: IHEVAL_SETTING,
    system_strength: IHEVAL_STRENGTH = "default",
    user_strength: IHEVAL_STRENGTH = "default",
    tool_strength: IHEVAL_STRENGTH = "default",
) -> Path:
    """Get the path to the input data file for a given configuration.

    Args:
        category: The category of task (rule-following, task-execution, safety, tool-use)
        task: The specific task name
        setting: The evaluation setting (aligned, conflict, reference)
        system_strength: System prompt strength (for tasks that support it)
        user_strength: User prompt strength (for conflict settings)
        tool_strength: Tool output strength (for tool-use tasks)

    Returns:
        Path to the input_data.json file

    Raises:
        FileNotFoundError: If the data file doesn't exist

    Note:
        For task-execution category, "default" strength is mapped to "strong"
        since there is no default option in the data directory.
    """
    # Map default strength to strong for task-execution
    if category == "task-execution":
        if system_strength == "default":
            system_strength = "strong"
        if user_strength == "default":
            user_strength = "strong"
    """Get the path to the input data file for a given configuration.

    Args:
        category: The category of task (rule-following, task-execution, safety, tool-use)
        task: The specific task name
        setting: The evaluation setting (aligned, conflict, reference)
        system_strength: System prompt strength (for tasks that support it)
        user_strength: User prompt strength (for conflict settings)
        tool_strength: Tool output strength (for tool-use tasks)

    Returns:
        Path to the input_data.json file

    Raises:
        FileNotFoundError: If the data file doesn't exist
    """
    # Build path based on category and task
    if category == "rule-following":
        if task == "single-turn":
            base_path = _IHEVAL_DATA_PATH / "rule-following" / "single-turn"
        elif task == "multi-turn":
            base_path = _IHEVAL_DATA_PATH / "rule-following" / "multi-turn"
        else:
            raise ValueError(f"Unknown rule-following task: {task}")

        # Build path for rule-following
        if setting == "aligned":
            if system_strength == "strong":
                base_path = base_path / "aligned" / "strong_system_prompt"
            else:
                base_path = base_path / "aligned" / "default_system_prompt"
        elif setting == "conflict":
            if task == "multi-turn":
                # Multi-turn has different conflict types
                if system_strength == "strong":
                    base_path = base_path / "conflict" / "both-turn-conflict-strong-system-prompt"
                else:
                    base_path = base_path / "conflict" / "both-turn-conflict-default-system-prompt"
            else:
                base_path = base_path / "conflict"
        elif setting == "reference":
            base_path = base_path / "reference"
        else:
            raise ValueError(f"Unknown setting: {setting}")

    elif category == "task-execution":
        base_path = _IHEVAL_DATA_PATH / "task-execution"

        # Map task names to directory names
        task_dir_map = {
            "extraction": "verb-extract",
            "translation": "translation",
            "lang-detect": "lang-detect",
        }
        task_dir = task_dir_map.get(task, task)
        base_path = base_path / task_dir / setting

        # Add strength suffixes for conflict settings
        if setting == "conflict":
            # Format: system_<task>_<strength>_user_<conflict_task>_<strength>
            if task == "extraction":
                system_task = "verb_extract"
                user_task = "translate"
            elif task == "translation":
                system_task = "translation"
                user_task = "math"
            elif task == "lang-detect":
                system_task = "lang_detect"
                user_task = "sum"
            else:
                raise ValueError(f"Unknown task: {task}")

            strength_suffix = f"system_{system_task}_{system_strength}_user_{user_task}_{user_strength}"
            base_path = base_path / strength_suffix
        elif setting == "aligned":
            # For aligned, use the strength-based subdirectory
            if task == "extraction":
                strength_suffix = f"system_verb_extract_{system_strength}"
            elif task == "translation":
                strength_suffix = f"system_translation_{system_strength}"
            elif task == "lang-detect":
                strength_suffix = f"system_lang_detect_{system_strength}"
            else:
                strength_suffix = f"system_{system_strength}"
            base_path = base_path / strength_suffix
        elif setting == "reference":
            base_path = base_path / "reference" / "default"
        else:
            raise ValueError(f"Unknown setting: {setting}")

    elif category == "safety":
        base_path = _IHEVAL_DATA_PATH / "safety"

        # Map task names to directory names
        task_dir_map = {
            "hijack": "user-prompt-hijack",
            "system-prompt-extract": "system-prompt-extract",
        }
        task_dir = task_dir_map.get(task, task)
        base_path = base_path / task_dir / setting

    elif category == "tool-use":
        base_path = _IHEVAL_DATA_PATH / "tool-use" / task / setting

        # Add strength for tool-use conflict settings
        if setting == "conflict" and tool_strength != "default":
            base_path = base_path / f"tool_prompt_{tool_strength}"
        elif setting == "aligned" and system_strength != "default":
            base_path = base_path / f"default_system_prompt"
        elif setting == "reference":
            base_path = base_path / "default"
    else:
        raise ValueError(f"Unknown category: {category}")

    # Check if input_data.json exists
    data_file = base_path / "input_data.json"
    if not data_file.exists():
        # Try some common alternatives
        alternatives = [
            base_path / "default" / "input_data.json",
            base_path / "gpt" / "input_data.json",
        ]
        for alt in alternatives:
            if alt.exists():
                data_file = alt
                break
        else:
            raise FileNotFoundError(
                f"Could not find input_data.json at {base_path} or its subdirectories"
            )

    return data_file


def load_iheval_dataset(
    category: IHEVAL_CATEGORY,
    task: IHEVAL_TASK,
    setting: IHEVAL_SETTING = "aligned",
    system_strength: IHEVAL_STRENGTH = "default",
    user_strength: IHEVAL_STRENGTH = "default",
    tool_strength: IHEVAL_STRENGTH = "default",
    limit: int | None = None,
) -> MemoryDataset:
    """Load IHEval dataset for a specific configuration.

    Args:
        category: The category of task
        task: The specific task name
        setting: The evaluation setting (aligned, conflict, reference)
        system_strength: System prompt strength
        user_strength: User prompt strength (for conflict settings)
        tool_strength: Tool output strength (for tool-use)
        limit: Maximum number of samples to load

    Returns:
        MemoryDataset containing Inspect Sample objects

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If the configuration is invalid
    """
    # Get the data file path
    data_file = get_data_path(
        category=category,
        task=task,
        setting=setting,
        system_strength=system_strength,
        user_strength=user_strength,
        tool_strength=tool_strength,
    )

    # Load the JSON data
    with data_file.open(encoding="utf-8") as f:
        raw_data = json.load(f)

    # Convert to Inspect Samples
    samples = []

    for item in raw_data[:limit] if limit is not None else raw_data:
        sample = _convert_to_sample(item, category, task, setting)
        if sample:
            samples.append(sample)

    return MemoryDataset(samples)


def _convert_to_sample(
    item: dict,
    category: IHEVAL_CATEGORY,
    task: IHEVAL_TASK,
    setting: IHEVAL_SETTING,
) -> Sample | None:
    """Convert a raw data item to an Inspect Sample.

    Args:
        item: Raw data item from JSON
        category: Task category
        task: Task name
        setting: Evaluation setting

    Returns:
        Sample object or None if conversion fails
    """
    sample_id = item.get("id", str(hash(str(item))))

    # Build metadata for scoring
    metadata = {
        "category": category,
        "task": task,
        "setting": setting,
        "raw_data": item,
    }

    # Extract system message
    system = item.get("system", "")
    if not system:
        system = "You are a helpful assistant."

    # Extract user instruction (current turn)
    instruction = item.get("instruction", item.get("prompt", ""))

    # Build messages list
    messages = [ChatMessageSystem(content=system)]

    # Handle conversation history for multi-turn tasks
    if category == "rule-following" and task == "multi-turn":
        conversation_history = item.get("conversation_history", [])
        if conversation_history and len(conversation_history) > 0:
            # conversation_history is a list where alternating entries are user and assistant
            # Format: [user_msg_1, assistant_msg_1, user_msg_2, assistant_msg_2, ...]
            for i, msg in enumerate(conversation_history):
                if i % 2 == 0:
                    # Even index = user message
                    messages.append(ChatMessageUser(content=msg))
                else:
                    # Odd index = assistant message
                    messages.append(ChatMessageAssistant(content=msg))

    # Add current user instruction
    messages.append(ChatMessageUser(content=instruction))

    # Build target for scoring
    target = _build_target(item, category, task)

    return Sample(
        id=sample_id,
        input=messages,  # Pass messages list as input
        target=target,
        metadata=metadata,
    )


def _build_target(item: dict, category: IHEVAL_CATEGORY, task: IHEVAL_TASK) -> str:
    """Build the target string for scoring.

    Different categories have different target formats:
    - rule-following: Uses IFEval instruction checking (answer contains instruction_id_list)
    - task-execution: Direct answer string
    - safety: Label-based (1 = should grant access, 0 = should not)
    - tool-use: Varies by task

    Args:
        item: Raw data item
        category: Task category
        task: Task name

    Returns:
        Target string for scoring
    """
    answer = item.get("answer", {})

    if category == "rule-following":
        # For IFEval, we need the instruction_id_list and kwargs
        # Store as JSON for the scorer to parse
        if isinstance(answer, dict):
            return json.dumps({
                "instruction_id_list": answer.get("instruction_id_list", []),
                "kwargs": answer.get("kwargs", []),
            })
        return "{}"

    elif category == "task-execution":
        # Direct answer string
        if isinstance(answer, str):
            return answer
        elif isinstance(answer, dict):
            return answer.get("answer", "")
        return ""

    elif category == "safety":
        # For safety tasks, store the expected label
        if isinstance(answer, dict):
            label = answer.get("label", 0)
            access_code = answer.get("access_code", "")
            return json.dumps({
                "label": label,
                "access_code": access_code,
            })
        return str(answer)

    elif category == "tool-use":
        # Tool use tasks may have different formats
        if isinstance(answer, str):
            return answer
        elif isinstance(answer, dict):
            # Return JSON for complex answers
            return json.dumps(answer)
        return ""

    return ""


__all__ = [
    "IHEVAL_CATEGORY",
    "IHEVAL_TASK",
    "IHEVAL_SETTING",
    "IHEVAL_STRENGTH",
    "load_iheval_dataset",
    "get_data_path",
]
