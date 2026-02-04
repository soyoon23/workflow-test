"""LangGraph workflow definition for Plan-Act workflow."""

import logging
from typing import Any, Generator, List, Literal, Optional, Union

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, StateGraph

from ..skills.registry import Skill
from .components import WorkflowComponents
from .nodes import ActNode, PlanNode, ReviewNode
from .state import AgentState, ConversationTurn

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> Literal["act", "review", "plan", "end"]:
    """Determine the next step in the workflow."""
    if state.get("error"):
        return "end"

    if state.get("is_complete"):
        return "end"

    if state.get("iteration_count", 0) >= 10:
        return "end"

    if not state.get("plan"):
        return "plan"

    plan = state["plan"]
    current_index = state.get("current_step_index", 0)

    if current_index >= len(plan):
        return "review"

    return "act"


def after_review(state: AgentState) -> Literal["plan", "end"]:
    """Determine what to do after review."""
    if state.get("is_complete"):
        return "end"

    return "plan"


def _build_initial_state(
    user_request: str,
    skill: Optional[Skill],
    auto_select_skill: bool,
    conversation_history: Optional[list[ConversationTurn]],
) -> AgentState:
    """Build the initial AgentState for a workflow invocation."""
    return {
        "messages": [],
        "user_request": user_request,
        "plan": [],
        "current_step_index": 0,
        "is_complete": False,
        "final_answer": None,
        "iteration_count": 0,
        "error": None,
        "active_skill_name": skill.name if skill else None,
        "auto_select_skill": auto_select_skill,
        "conversation_history": conversation_history or [],
        "review_key_facts": [],
    }


def create_workflow(components: WorkflowComponents) -> StateGraph:
    """Create the Plan-Act workflow graph.

    Args:
        components: Bundle of shared dependencies for all nodes.
    """
    plan = PlanNode(components)
    act = ActNode(components)
    review = ReviewNode(components)

    workflow = StateGraph(AgentState)

    workflow.add_node("plan", plan)
    workflow.add_node("act", act)
    workflow.add_node("review", review)

    workflow.set_entry_point("plan")

    workflow.add_conditional_edges(
        "plan",
        should_continue,
        {
            "act": "act",
            "review": "review",
            "plan": "plan",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "act",
        should_continue,
        {
            "act": "act",
            "review": "review",
            "plan": "plan",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "review",
        after_review,
        {
            "plan": "plan",
            "end": END,
        },
    )

    return workflow.compile()


def _execute_workflow(
    user_request: str,
    components: WorkflowComponents,
    *,
    auto_select_skill: bool = False,
    conversation_history: Optional[list[ConversationTurn]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
    stream: bool = False,
) -> Union[dict, Generator]:
    """Unified workflow execution (invoke or stream).

    Args:
        user_request: The user's input text.
        components: Bundle of shared dependencies.
        auto_select_skill: Whether to let the LLM auto-select a skill.
        conversation_history: Previous conversation turns for context.
        callbacks: LangGraph callback handlers (e.g., Langfuse).
        stream: If True, yield events; if False, return final state.
    """
    workflow = create_workflow(components)

    initial_state = _build_initial_state(
        user_request,
        components.initial_skill,
        auto_select_skill,
        conversation_history,
    )
    logger.info(
        "%s workflow: request=%s, history_turns=%d, callbacks=%s",
        "Streaming" if stream else "Running",
        user_request[:80],
        len(initial_state["conversation_history"]),
        "enabled" if callbacks else "disabled",
    )

    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks

    if stream:
        return workflow.stream(initial_state, config=config if config else None)
    else:
        return workflow.invoke(initial_state, config=config if config else None)


def run_workflow(
    user_request: str,
    components: WorkflowComponents,
    *,
    auto_select_skill: bool = False,
    conversation_history: Optional[list[ConversationTurn]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> dict:
    """Run the workflow with a user request.

    Args:
        user_request: The user's input text.
        components: Bundle of shared dependencies.
        auto_select_skill: Whether to let the LLM auto-select a skill.
        conversation_history: Previous conversation turns for context.
        callbacks: LangGraph callback handlers (e.g., Langfuse).
    """
    return _execute_workflow(
        user_request,
        components,
        auto_select_skill=auto_select_skill,
        conversation_history=conversation_history,
        callbacks=callbacks,
        stream=False,
    )


def stream_workflow(
    user_request: str,
    components: WorkflowComponents,
    *,
    auto_select_skill: bool = False,
    conversation_history: Optional[list[ConversationTurn]] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
):
    """Stream the workflow execution.

    Args:
        user_request: The user's input text.
        components: Bundle of shared dependencies.
        auto_select_skill: Whether to let the LLM auto-select a skill.
        conversation_history: Previous conversation turns for context.
        callbacks: LangGraph callback handlers (e.g., Langfuse).
    """
    yield from _execute_workflow(
        user_request,
        components,
        auto_select_skill=auto_select_skill,
        conversation_history=conversation_history,
        callbacks=callbacks,
        stream=True,
    )
