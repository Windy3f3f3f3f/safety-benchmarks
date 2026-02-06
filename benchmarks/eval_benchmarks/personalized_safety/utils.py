"""Utility functions for PersonalizedSafety benchmark."""

from typing import Any


def build_background_description(record: dict[str, Any]) -> str:
    """Build full background description for judge evaluation.

    Converts HuggingFace dataset fields (snake_case) to the format expected
    by the judge prompt.

    Args:
        record: Dataset record with user profile information

    Returns:
        Formatted background description string
    """
    # Map HuggingFace field names to display format
    age = record.get("age", "unknown")
    gender = record.get("gender", "unknown")
    marital_status = record.get("marital_status", "unknown")
    profession = record.get("profession", "unknown")
    economic_status = record.get("economic_status", "unknown")
    health_status = record.get("health_status", "unknown")
    education_level = record.get("education_level", "unknown")
    mental_health_status = record.get("mental_health_status", "unknown")
    emotional_state = record.get("emotional_state", "unknown")

    # Build description in the format expected by the judge
    # Note: Some fields like "Past Self-Harm History" may not be in all records
    parts = [
        f"Age: {age}",
        f"Gender: {gender}",
        f"Marital Status: {marital_status}",
        f"Profession: {profession}",
        f"Economic Status: {economic_status}",
        f"Health Status: {health_status}",
        f"Education Level: {education_level}",
        f"Mental Health Status: {mental_health_status}",
        f"Emotional State: {emotional_state}",
    ]

    return ", ".join(parts) + "."
