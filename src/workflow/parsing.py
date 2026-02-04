"""Parsing utilities for LLM response content.

Pure functions extracted from WorkflowNodes for reuse across
different agent architectures.
"""

import json
import logging
import re
from typing import Any, Optional

from ..skills.registry import SkillRegistry
from .state import PlanStep

logger = logging.getLogger(__name__)


def extract_json_payload(content: str) -> Optional[dict[str, Any]]:
    """Extract a JSON object from LLM output.

    Supports fenced code blocks (```json ... ```), otherwise falls back to
    searching for the first balanced brace section.
    """
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})```", content, re.DOTALL)
    raw_payload: Optional[str] = None
    if fence_match:
        raw_payload = fence_match.group(1)
    else:
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            raw_payload = content[json_start:json_end]

    if not raw_payload:
        return None

    try:
        return json.loads(raw_payload)
    except json.JSONDecodeError:
        logger.debug("JSON decoding failed for payload: %s", raw_payload[:200])
        return None


def parse_plan(
    content: str, skill_registry: SkillRegistry
) -> tuple[list[PlanStep], Optional[str]]:
    """Parse plan from LLM response.

    Returns:
        Tuple of (plan_steps, selected_skill_name).
        selected_skill_name is None if no skill was selected.
    """
    selected_skill = None
    plan_data = extract_json_payload(content)

    if plan_data:
        skill_name = plan_data.get("selected_skill")
        if skill_name and str(skill_name).lower() not in ("none", "null", ""):
            if skill_registry.get(skill_name):
                selected_skill = skill_name

        steps: list[PlanStep] = []
        for step in plan_data.get("steps", []):
            steps.append(
                PlanStep(
                    step_number=step.get("step_number", len(steps) + 1),
                    description=step.get("description", ""),
                    status="pending",
                    result=None,
                )
            )
        if steps:
            return steps, selected_skill

    logger.warning("Failed to parse plan JSON, using fallback")

    return [
        PlanStep(
            step_number=1,
            description=content[:500],
            status="pending",
            result=None,
        )
    ], None


def parse_review(content: str) -> dict[str, Any]:
    """Parse review from LLM response."""
    review_data = extract_json_payload(content)
    if review_data is not None:
        return review_data

    logger.warning("Failed to parse review JSON, using fallback")

    return {
        "is_complete": True,
        "final_answer": content,
    }
