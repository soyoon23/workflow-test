"""LangGraph workflow definition for Plan-Act workflow."""

from typing import Literal, Optional
from langgraph.graph import StateGraph, END

from .state import AgentState
from .nodes import WorkflowNodes
from ..llm.client import LLMClient
from ..prompts.registry import PromptRegistry
from ..tools.registry import ToolRegistry
from ..skills.registry import Skill, SkillRegistry


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


def create_workflow(
    llm_client: LLMClient,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    initial_skill: Optional[Skill] = None,
    auto_select_skill: bool = False,
) -> StateGraph:
    """Create the Plan-Act workflow graph."""

    # Initialize nodes
    nodes = WorkflowNodes(
        llm_client, prompt_registry, tool_registry, skill_registry, initial_skill
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
) -> dict:
    """Run the workflow with a user request."""
    workflow = create_workflow(
        llm_client, prompt_registry, tool_registry, skill_registry, skill, auto_select_skill
    )

    initial_state: AgentState = {
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
    }

    # Run the workflow
    final_state = workflow.invoke(initial_state)

    return final_state


def stream_workflow(
    user_request: str,
    llm_client: LLMClient,
    prompt_registry: PromptRegistry,
    tool_registry: ToolRegistry,
    skill_registry: SkillRegistry,
    skill: Optional[Skill] = None,
    auto_select_skill: bool = False,
):
    """Stream the workflow execution."""
    workflow = create_workflow(
        llm_client, prompt_registry, tool_registry, skill_registry, skill, auto_select_skill
    )

    initial_state: AgentState = {
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
    }

    # Stream the workflow
    for event in workflow.stream(initial_state):
        yield event
