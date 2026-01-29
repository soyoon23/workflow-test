"""Agent state definition for Plan-Act workflow."""

from typing import Annotated, Optional, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..conversation.types import ConversationTurn


class PlanStep(TypedDict):
    """A single step in the plan."""

    step_number: int
    description: str
    status: str  # "pending", "in_progress", "completed", "failed"
    result: Optional[str]


class AgentState(TypedDict):
    """State for the Plan-Act workflow agent."""

    # Message history with automatic message merging
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # User's original request
    user_request: str

    # Generated plan
    plan: list[PlanStep]

    # Current step index being executed
    current_step_index: int

    # Whether the workflow is complete
    is_complete: bool

    # Final answer/result
    final_answer: Optional[str]

    # Number of iterations (to prevent infinite loops)
    iteration_count: int

    # Error message if any
    error: Optional[str]

    # Active skill name (selected by router or use_skill tool)
    active_skill_name: Optional[str]

    # Whether to auto-select skill (set by user)
    auto_select_skill: bool

    # Multi-turn conversation history from previous turns
    conversation_history: list[ConversationTurn]

    # Key facts extracted by reviewer (used to build ConversationTurn)
    review_key_facts: list[str]
