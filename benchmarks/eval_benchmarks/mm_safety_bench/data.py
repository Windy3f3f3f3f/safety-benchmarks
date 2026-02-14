"""
MM-SafetyBench data loading utilities
Handles loading of MM-SafetyBench dataset (questions and images)
"""
import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional

from inspect_ai.dataset import Sample
from inspect_ai.solver import TaskState


def get_mm_safety_root() -> Path:
    """
    Get MM-SafetyBench data root directory.

    Directory structure:
    E:\\code\\aisafety\\MM\\
    └── MM-SafetyBench-main/
        └── data/
            ├── processed_questions/  (question JSON files)
            └── imgs/                   (image files)
    """
    # Current file: safety-benchmarks/benchmarks/eval_benchmarks/mm_safety_bench/data.py
    # Go up 4 levels to safety-benchmarks, then up 1 level to E:\\code\\aisafety\\MM\\
    current_dir = Path(__file__).parent.parent.parent.parent.parent
    mm_bench_root = current_dir / "MM-SafetyBench-main"
    return mm_bench_root / "data"


def encode_image(image_path: Path) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded string
    """
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            return img_base64
    except Exception as e:
        raise FileNotFoundError(f"Cannot read image: {image_path}") from e


def load_mm_safety_data(
    scenario: str,
    image_type: str = "SD",
    limit: Optional[int] = None
) -> List[Sample]:
    """
    Load MM-SafetyBench data for specific scenario and image type.

    Args:
        scenario: Scenario name (e.g., "01-Illegal_Activitiy")
        image_type: Image type ("SD", "SD_TYPO", or "TYPO")
        limit: Sample count limit (None for all)

    Returns:
        List of inspect_ai Sample objects
    """

    # Get data root directory
    data_root = get_mm_safety_root()
    question_file = data_root / "processed_questions" / f"{scenario}.json"
    img_dir = data_root / "imgs" / scenario / image_type

    # Validate files exist
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    if not img_dir.exists():
        # Image directory missing is a warning (some image types may not exist)
        return []

    # Read question data
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Determine sample range
    sample_ids = list(questions.keys())
    if limit is not None and limit < len(sample_ids):
        sample_ids = sample_ids[:limit]

    # Build sample list
    samples = []
    for sample_id in sample_ids:
        q_data = questions[sample_id]

        # Determine question field (based on image type)
        if image_type == "SD":
            question_key = "Rephrased Question(SD)"
        elif image_type == "SD_TYPO":
            question_key = "Rephrased Question"
        else:  # TYPO
            question_key = "Rephrased Question"

        question_text = q_data.get(question_key, q_data.get("Rephrased Question", ""))

        # Check image file
        image_path = img_dir / f"{sample_id}.jpg"

        if not image_path.exists():
            # Image doesn't exist, skip this sample
            continue

        # Encode image
        try:
            img_base64 = encode_image(image_path)
        except Exception:
            # Image encoding failed, skip
            continue

        # Build sample
        samples.append(Sample(
            id=f"{scenario}_{sample_id}_{image_type}",
            input=question_text,
            target="safe",  # Default target is safe (will be overridden by scorer)
            metadata={
                "scenario": scenario,
                "sample_id": sample_id,
                "image_type": image_type,
                "image_base64": img_base64,
                "question": q_data.get("Question", ""),
                "changed_question": q_data.get("Changed Question", ""),
                "key_phrase": q_data.get("Key Phrase", ""),
                "phrase_type": q_data.get("Phrase Type", ""),
                "original": q_data
            }
        ))

    return samples


def load_mm_safety_data_all_types(
    scenario: str,
    limit: Optional[int] = None
) -> List[Sample]:
    """
    Load MM-SafetyBench data for all 3 image types (matches original paper).

    Each original sample is expanded to 3 test samples (SD, SD_TYPO, TYPO).
    Matches the original codebase's inference.py logic exactly.

    Args:
        scenario: Scenario name (e.g., "01-Illegal_Activitiy")
        limit: Sample count limit (refers to original samples, None for all)

    Returns:
        List of inspect_ai Sample objects (each original sample × 3 image types)
    """

    # 3 image types (identical to original codebase)
    image_types = [
        ("SD", "Rephrased Question(SD)"),
        ("SD_TYPO", "Rephrased Question"),
        ("TYPO", "Rephrased Question")
    ]

    # Get data root directory
    data_root = get_mm_safety_root()
    question_file = data_root / "processed_questions" / f"{scenario}.json"

    # Validate question file exists
    if not question_file.exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")

    # Read question data
    with open(question_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # Determine original sample range
    sample_ids = list(questions.keys())
    if limit is not None and limit < len(sample_ids):
        sample_ids = sample_ids[:limit]

    # Build sample list (each original sample × 3 image types)
    samples = []
    total_tests = len(sample_ids) * len(image_types)

    for img_type, question_key in image_types:
        img_dir = data_root / "imgs" / scenario / img_type

        # Validate image directory exists
        if not img_dir.exists():
            continue

        # Iterate through all samples
        for sample_id in sample_ids:
            q_data = questions[sample_id]
            question_text = q_data.get(question_key, q_data.get("Rephrased Question", ""))

            # Check image file
            image_path = img_dir / f"{sample_id}.jpg"
            if not image_path.exists():
                # Image doesn't exist, skip
                continue

            # Encode image
            try:
                img_base64 = encode_image(image_path)
            except Exception:
                # Encoding failed, skip
                continue

            # Build sample (ID format: {scenario}_{sample_id}_{img_type})
            samples.append(Sample(
                id=f"{scenario}_{sample_id}_{img_type}",
                input=question_text,
                target="safe",
                metadata={
                    "scenario": scenario,
                    "sample_id": sample_id,
                    "image_type": img_type,
                    "image_base64": img_base64,
                    "question": q_data.get("Question", ""),
                    "changed_question": q_data.get("Changed Question", ""),
                    "key_phrase": q_data.get("Key Phrase", ""),
                    "phrase_type": q_data.get("Phrase Type", ""),
                    "original": q_data
                }
            ))

    return samples


def get_available_scenarios() -> List[str]:
    """
    Get list of all available scenarios.

    Returns:
        List of scenario names
    """
    return [
        "01-Illegal_Activitiy",
        "02-HateSpeech",
        "03-Malware_Generation",
        "04-Physical_Harm",
        "05-EconomicHarm",
        "06-Fraud",
        "07-Sex",
        "08-Political_Lobbying",
        "09-Privacy_Violence",
        "10-Legal_Opinion",
        "11-Financial_Advice",
        "12-Health_Consultation",
        "13-Gov_Decision"
    ]


def get_scenarios_info() -> Dict[str, Dict[str, Any]]:
    """
    Get scenario metadata information.

    Returns:
        Dictionary of scenario information
    """
    return {
        "01-Illegal_Activitiy": {
            "name": "Illegal Activity",
            "description": "Illegal activities",
            "image_types": ["SD", "SD_TYPO", "TYPO"]
        },
        "02-HateSpeech": {
            "name": "Hate Speech",
            "description": "Hate speech",
            "image_types": ["SD", "SD_TYPO", "TYPO"]
        },
        # ... other scenarios
    }
