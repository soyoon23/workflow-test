"""Plan node for the Plan-Act-Review workflow."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..components import WorkflowComponents
from ..parsing import parse_plan
from ..state import AgentState, PlanStep
from .base import NodeMixin

logger = logging.getLogger(__name__)


class PlanNode(NodeMixin):
    """Creates or updates the execution plan.

    Analyzes the user request (and any prior execution results) to produce
    a structured list of PlanSteps. Optionally auto-selects a skill.
    """

    def __init__(self, components: WorkflowComponents) -> None:
        self.components = components

    def __call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Create or update the execution plan."""
        c = self.components
        c.callback.phase_start("plan")

        system_prompt = self._get_prompt("planner", state)
        user_request = state["user_request"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a plan for: {user_request}"),
        ]

        # Include previous results for replanning
        if state.get("plan") and any(
            step["status"] in ["completed", "failed"] for step in state["plan"]
        ):
            results_summary = self._summarize_results(state["plan"])
            messages.append(
                HumanMessage(
                    content=(
                        f"Previous execution results:\n{results_summary}\n\n"
                        "Please update the plan if needed."
                    )
                )
            )

        logger.debug("plan_node: sending %d messages to LLM", len(messages))
        response = c.llm.chat(messages, config=config)

        plan, selected_skill = parse_plan(response.content, c.skills)
        logger.info("plan_node: generated %d steps", len(plan))

        c.callback.plan_ready(
            plan=[{"step_number": s["step_number"], "description": s["description"]} for s in plan],
            skill=selected_skill,
        )
        c.callback.phase_end("plan", step_count=len(plan))

        result: dict[str, Any] = {
            "messages": [response],
            "plan": plan,
            "current_step_index": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

        if selected_skill and state.get("auto_select_skill") and not state.get("active_skill_name"):
            result["active_skill_name"] = selected_skill
            logger.info("plan_node: auto-selected skill=%s", selected_skill)

        return result

    @staticmethod
    def _summarize_results(plan: list[PlanStep]) -> str:
        """Summarize execution results for replanning."""
        lines = []
        for step in plan:
            status_emoji = {
                "completed": "[OK]",
                "failed": "[FAIL]",
                "pending": "[...]",
                "in_progress": "[>>]",
            }.get(step["status"], "[?]")

            lines.append(f"{status_emoji} Step {step['step_number']}: {step['description']}")
            if step.get("result"):
                lines.append(f"   Result: {step['result'][:200]}")

        return "\n".join(lines)
