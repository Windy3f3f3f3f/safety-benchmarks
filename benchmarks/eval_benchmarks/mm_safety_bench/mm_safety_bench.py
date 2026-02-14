"""
MM-SafetyBench task definitions
Implements multimodal LLM safety evaluation based on ECCV 2024
"""
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.solver import solver, TaskState, generate
from inspect_ai.scorer import Scorer, Score
from inspect_ai.model import ChatMessageUser, ContentText, ContentImage

# Use absolute imports to support running via file path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from data import load_mm_safety_data, load_mm_safety_data_all_types
from scorer import mm_safety_judge


@solver
def mm_safety_solver():
    """Process multimodal messages (text + image)"""
    async def solve(state: TaskState, generate):
        # Get image from metadata
        img_base64 = state.metadata.get("image_base64")

        if img_base64:
            # Reconstruct user message with text and image
            state.messages = [
                ChatMessageUser(content=[
                    ContentText(text=state.input_text),
                    ContentImage(image=f"data:image/jpeg;base64,{img_base64}")
                ])
            ]

        # Run model generation
        return await generate(state)
    return solve


@task
def mm_safety_bench_illegal_activity(
    scenario: str = "01-Illegal_Activitiy",
    image_type: str = None,  # None means test all 3 types
    limit: int = None
) -> Task:
    """
    MM-SafetyBench: Illegal activity scenario evaluation (complete replication)

    Evaluates model safety under image-based adversarial attacks.
    Tests all 3 image types (SD, SD_TYPO, TYPO) by default, matching original paper.

    Args:
        scenario: Scenario name (default "01-Illegal_Activitiy")
        image_type: Image type (SD, SD_TYPO, TYPO, or None for all types)
        limit: Sample count limit (refers to original samples, actual count × 3)

    Returns:
        inspect_ai Task object
    """

    # Task metadata
    if image_type:
        # Single image type (legacy compatibility)
        metadata = {
            "name": f"MM-SafetyBench: {scenario} ({image_type})",
            "description": f"Evaluate MLLM safety on {scenario} scenario with {image_type} image-based jailbreaks",
            "scenario": scenario,
            "image_type": image_type,
            "reference": "ECCV 2024 - https://arxiv.org/abs/2311.17600"
        }

        dataset = load_mm_safety_data(
            scenario=scenario,
            image_type=image_type,
            limit=limit
        )
    else:
        # All 3 image types (default, matches original paper)
        metadata = {
            "name": f"MM-SafetyBench: {scenario} (All Types)",
            "description": f"Evaluate MLLM safety on {scenario} scenario with all 3 image types (SD, SD_TYPO, TYPO)",
            "scenario": scenario,
            "image_types": ["SD", "SD_TYPO", "TYPO"],
            "reference": "ECCV 2024 - https://arxiv.org/abs/2311.17600"
        }

        dataset = load_mm_safety_data_all_types(
            scenario=scenario,
            limit=limit
        )

    # Use custom multimodal solver
    solver = [
        mm_safety_solver()
    ]

    # Use GPT-4 judge as scorer
    scorer = mm_safety_judge()

    return Task(
        dataset=dataset,
        solver=solver,
        scorer=scorer,
        metadata=metadata
    )


# Future extensions: other scenarios
@task
def mm_safety_bench_hate_speech(
    scenario: str = "02-HateSpeech",
    image_type: str = "SD",
    limit: int = None
) -> Task:
    """
    MM-SafetyBench: Hate speech scenario evaluation
    """
    metadata = {
        "name": f"MM-SafetyBench: {scenario}",
        "description": f"Evaluate MLLM safety on {scenario} scenario with image-based jailbreaks",
        "scenario": scenario,
        "image_type": image_type,
        "reference": "ECCV 2024 - https://arxiv.org/abs/2311.17600"
    }

    dataset = load_mm_safety_data(scenario, image_type, limit)
    solver = [generate()]
    scorer = mm_safety_judge()

    return Task(dataset, solver, scorer, metadata)


# Future extensions: all 13 scenario IDs
AVAILABLE_SCENARIOS = [
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
    "13-Gov_Decision",
]
