"""Data types for multi-turn conversation management."""

from typing import Optional, TypedDict


class ConversationTurn(TypedDict):
    """Summary of a completed conversation turn for multi-turn context."""

    turn_number: int
    user_request: str
    final_answer: str
    plan_summary: list[str]  # step descriptions
    plan_results: list[str]  # step results (populated only for recent turns)
    key_facts: list[str]  # reviewer-extracted key facts
    skill_used: Optional[str]
