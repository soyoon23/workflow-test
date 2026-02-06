"""Traced workflow wrapper for DeepEval agentic metrics.

Creates execution traces required by DeepEval's agentic metrics
(TaskCompletion, PlanQuality, PlanAdherence, StepEfficiency) by wrapping
plan, act, and review nodes with tracing calls.

This file is evaluation-only — production code in src/ is not modified.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from deepeval.test_case import ToolCall
from deepeval.tracing import update_current_span, update_current_trace

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from src.observability.base import TracingContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _build_components(config: dict) -> dict:
    """Initialise all workflow components from config."""
    from src.llm.client import LLMClient
    from src.prompts.registry import PromptRegistry
    from src.skills.registry import SkillRegistry
    from src.tools.registry import ToolRegistry
    from src.workflow.components import WorkflowComponents
    from src.workflow.nodes import ActNode, PlanNode, ReviewNode

    llm_cfg = config["llm"]
    components = WorkflowComponents(
        llm=LLMClient(
            base_url=llm_cfg["base_url"],
            model=llm_cfg["model"],
            api_key=llm_cfg["api_key"],
            temperature=llm_cfg.get("temperature", 0.7),
            max_tokens=llm_cfg.get("max_tokens", 4096),
        ),
        prompts=PromptRegistry(templates_dir=str(PROJECT_ROOT / "src" / "prompts" / "templates")),
        tools=ToolRegistry(),
        skills=SkillRegistry(templates_dir=str(PROJECT_ROOT / "src" / "skills" / "templates")),
    )

    return {
        "components": components,
        "plan_node": PlanNode(components),
        "act_node": ActNode(components),
        "review_node": ReviewNode(components),
    }


def _initial_state(user_request: str, skill=None, auto_select: bool = False) -> dict:
    """Build initial AgentState dict."""
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
        "auto_select_skill": auto_select,
        "conversation_history": [],
        "review_key_facts": [],
    }


def _merge_state(state: dict, update: dict) -> dict:
    """Apply a node return dict to the current state (simplified merge)."""
    merged = {**state}
    for key, value in update.items():
        if key == "messages":
            merged["messages"] = list(merged.get("messages", [])) + list(value)
        else:
            merged[key] = value
    return merged


def _make_runnable_config(obs_callback=None) -> dict:
    """Build a RunnableConfig dict with optional observability callback."""
    if obs_callback:
        return {"callbacks": [obs_callback]}
    return {}


def traced_workflow(user_request: str, config: dict, obs_callback=None) -> str:
    """Run the full Plan-Act-Review workflow with DeepEval tracing.

    This creates a complete execution trace that agentic metrics
    (TaskCompletion, PlanQuality, PlanAdherence, StepEfficiency) can
    evaluate.

    Args:
        obs_callback: Optional LangChain callback handler (e.g. Langfuse)
            so that every LLM call inside nodes is captured.
    """
    built = _build_components(config)
    plan_node = built["plan_node"]
    act_node = built["act_node"]
    review_node = built["review_node"]
    components = built["components"]

    runnable_config = _make_runnable_config(obs_callback)

    # Detect skill trigger
    skill, actual_request = components.skills.parse_input(user_request)
    state = _initial_state(actual_request, skill=skill, auto_select=(skill is None))
    all_tools_called: list[ToolCall] = []
    max_iterations = 10

    for _ in range(max_iterations):
        # ---- Plan ----
        plan_result = _traced_plan(plan_node, state, config=runnable_config)
        state = _merge_state(state, plan_result)

        # ---- Act (execute each step) ----
        while state["current_step_index"] < len(state.get("plan", [])):
            act_result, step_tools = _traced_act(act_node, state, config=runnable_config)
            all_tools_called.extend(step_tools)
            state = _merge_state(state, act_result)

        # ---- Review ----
        review_result = _traced_review(review_node, state, config=runnable_config)
        state = _merge_state(state, review_result)

        if state.get("is_complete") or state.get("error"):
            break

    final_answer = state.get("final_answer", "") or ""
    update_current_trace(
        input=user_request,
        output=final_answer,
        tools_called=all_tools_called,
    )
    return final_answer


def _traced_plan(
    node,
    state: dict,
    tracing_ctx: Optional["TracingContext"] = None,
    config: Optional["RunnableConfig"] = None,
) -> dict:
    """Traced wrapper around PlanNode with observability integration."""
    runnable_config = config or {}
    result = node(state, runnable_config)
    plan_steps = result.get("plan", [])
    plan_text = "\n".join(f"{s['step_number']}. {s['description']}" for s in plan_steps)

    if tracing_ctx:
        span = tracing_ctx.create_span(
            name="plan_node",
            metadata={"role": "planner"},
        )
        tracing_ctx.end_span(
            span,
            input=state["user_request"],
            output=plan_text,
            metadata={
                "plan_step_count": len(plan_steps),
                "active_skill": result.get("active_skill_name"),
            },
        )

    update_current_span(input=state["user_request"], output=plan_text)
    return result


def _traced_act(
    node,
    state: dict,
    tracing_ctx: Optional["TracingContext"] = None,
    config: Optional["RunnableConfig"] = None,
) -> tuple[dict, list[ToolCall]]:
    """Traced wrapper around ActNode. Returns (result, tools_called)."""
    step_idx = state["current_step_index"]
    step = state["plan"][step_idx]

    span = None
    if tracing_ctx:
        span = tracing_ctx.create_span(
            name="act_node",
            metadata={
                "role": "actor",
                "step_number": step.get("step_number"),
                "active_skill": state.get("active_skill_name"),
            },
        )

    runnable_config = config or {}
    result = node(state, runnable_config)

    # Collect tool calls from the messages returned by act_node
    tools_called: list[ToolCall] = []
    for msg in result.get("messages", []):
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tools_called.append(ToolCall(name=tc["name"]))

    updated_plan = result.get("plan") or state.get("plan", [])
    updated_step = updated_plan[step_idx] if step_idx < len(updated_plan) else step
    step_result_text = updated_step.get("result", "") or ""

    if tracing_ctx and span:
        tracing_ctx.end_span(
            span,
            input=step["description"],
            output=step_result_text,
            metadata={
                "tools_called": [tc.name for tc in tools_called],
            },
        )

    update_current_span(
        input=step["description"],
        output=step_result_text,
        tools_called=tools_called,
    )
    return result, tools_called


def _traced_review(
    node,
    state: dict,
    tracing_ctx: Optional["TracingContext"] = None,
    config: Optional["RunnableConfig"] = None,
) -> dict:
    """Traced wrapper around ReviewNode."""
    span = None
    if tracing_ctx:
        span = tracing_ctx.create_span(
            name="review_node",
            metadata={
                "role": "reviewer",
            },
        )

    runnable_config = config or {}
    result = node(state, runnable_config)
    final_answer = result.get("final_answer", "") or ""

    if tracing_ctx and span:
        tracing_ctx.end_span(
            span,
            input=state["user_request"],
            output=final_answer,
            metadata={
                "key_facts_count": len(result.get("review_key_facts", []) or []),
                "is_complete": result.get("is_complete"),
            },
        )

    update_current_span(
        input=state["user_request"],
        output=final_answer,
    )
    return result
