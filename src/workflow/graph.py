"""LangGraph workflow definition for Plan-Act workflow."""

import logging
from typing import Any, List, Literal, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import END, StateGraph

from ..llm.client import LLMClient
from ..prompts.registry import PromptRegistry
from ..skills.registry import Skill, SkillRegistry
from ..tools.registry import ToolRegistry
from .nodes import WorkflowNodes
from .state import AgentState, ConversationTurn
from .streaming import StreamCallback

logger = logging.getLogger(__name__)


def should_continue(state: AgentState) -> Literal["act", "review", "plan", "end"]:
    """Determine the next step in the workflow."""
    # Check for errors
    if state.get("error"):
        return "end"

    # Check if complete
    if state.get("is_complete"):
        return "end"

    # Check iteration limit
    if state.get("iteration_count", 0) >= 10:
        return "end"

    # If no plan, go to plan
    if not state.get("plan"):
        return "plan"

    # If we have a plan, check step progress
    plan = state["plan"]
    current_index = state.get("current_step_index", 0)

    # All steps completed -> review
    if current_index >= len(plan):
        return "review"

    # More steps to execute -> act
    return "act"


def after_review(state: AgentState) -> Literal["plan", "end"]:
    """Determine what to do after review."""
    if state.get("is_complete"):
        return "end"

    # Need to replan
    return "plan"


def _build_initial_state(
    user_request: str,
    skill: Optional[Skill],
    auto_select_skill: bool,
    conversation_history: Optional[list[ConversationTurn]],
) -> AgentState:
    """Build the initial AgentState for a workflow invocation.

    Centralizes state initialization to avoid duplication between
    run_workflow() and stream_workflow().
    """
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


def create_workflow(
    llm_client: LLMClient,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    initial_skill: Optional[Skill] = None,
    auto_select_skill: bool = False,
    stream_callback: Optional[StreamCallback] = None,
) -> StateGraph:
    """Create the Plan-Act workflow graph."""

    # Initialize nodes
    nodes = WorkflowNodes(
        llm_client,
        prompt_registry,
        tool_registry,
        skill_registry,
        initial_skill,
        stream_callback=stream_callback,
    )

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("plan", nodes.plan_node)
    workflow.add_node("act", nodes.act_node)
    workflow.add_node("review", nodes.review_node)

    # Always start from plan (plan handles skill selection internally)
    workflow.set_entry_point("plan")

    # Add edges
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


def run_workflow(
    user_request: str,
    llm_client: LLMClient,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    skill: Optional[Skill] = None,
    auto_select_skill: bool = False,
    conversation_history: Optional[list[ConversationTurn]] = None,
    stream_callback: Optional[StreamCallback] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
) -> dict:
    """Run the workflow with a user request.

    Args:
        callbacks: Optional list of callback handlers (e.g., LangfuseCallbackHandler)
                   for observability and tracing.
    """
    workflow = create_workflow(
        llm_client,
        prompt_registry,
        tool_registry,
        skill_registry,
        skill,
        auto_select_skill,
        stream_callback=stream_callback,
    )

    initial_state = _build_initial_state(
        user_request, skill, auto_select_skill, conversation_history
    )
    logger.info(
        "Running workflow: request=%s, history_turns=%d, callbacks=%s",
        user_request[:80],
        len(initial_state["conversation_history"]),
        "enabled" if callbacks else "disabled",
    )

    # Build config for LangGraph with callbacks
    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks

    final_state = workflow.invoke(initial_state, config=config if config else None)
    return final_state


def stream_workflow(
    user_request: str,
    llm_client: LLMClient,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    skill: Optional[Skill] = None,
    auto_select_skill: bool = False,
    conversation_history: Optional[list[ConversationTurn]] = None,
    stream_callback: Optional[StreamCallback] = None,
    callbacks: Optional[List[BaseCallbackHandler]] = None,
):
    """Stream the workflow execution.

    Args:
        callbacks: Optional list of callback handlers (e.g., LangfuseCallbackHandler)
                   for observability and tracing.
    """
    workflow = create_workflow(
        llm_client,
        prompt_registry,
        tool_registry,
        skill_registry,
        skill,
        auto_select_skill,
        stream_callback=stream_callback,
    )

    initial_state = _build_initial_state(
        user_request, skill, auto_select_skill, conversation_history
    )
    logger.info(
        "Streaming workflow: request=%s, history_turns=%d, callbacks=%s",
        user_request[:80],
        len(initial_state["conversation_history"]),
        "enabled" if callbacks else "disabled",
    )

    # Build config for LangGraph with callbacks
    config: dict[str, Any] = {}
    if callbacks:
        config["callbacks"] = callbacks

    for event in workflow.stream(initial_state, config=config if config else None):
        yield event
