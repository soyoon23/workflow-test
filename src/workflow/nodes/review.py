"""Review node for the Plan-Act-Review workflow."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..components import WorkflowComponents
from ..parsing import parse_review
from ..state import AgentState, PlanStep
from .base import NodeMixin

logger = logging.getLogger(__name__)


class ReviewNode(NodeMixin):
    """Reviews execution results and determines next steps.

    Evaluates whether the plan execution fulfilled the original request.
    Extracts key_facts for multi-turn conversation context.
    """

    def __init__(self, components: WorkflowComponents) -> None:
        self.components = components

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Review execution results and determine next steps."""
        c = self.components
        c.callback.phase_start("review")

        system_prompt = self._get_prompt("reviewer", state)
        user_request = state["user_request"]
        plan = state["plan"]

        plan_summary = self._format_plan_for_review(plan)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Original request: {user_request}\n\n"
                    f"Plan and execution results:\n{plan_summary}\n\n"
                    "Please review and determine next steps."
                )
            ),
        ]

        logger.debug("review_node: sending review request to LLM")
        response = c.llm.chat(messages)

        review = parse_review(response.content)
        logger.info(
            "review_node: is_complete=%s, has_key_facts=%d",
            review.get("is_complete"),
            len(review.get("key_facts", [])),
        )

        c.callback.review_ready(
            is_complete=review.get("is_complete", False),
            final_answer=review.get("final_answer"),
        )
        c.callback.phase_end("review")

        return {
            "messages": [response],
            "is_complete": review.get("is_complete", False),
            "final_answer": review.get("final_answer"),
            "error": review.get("error"),
            "review_key_facts": review.get("key_facts", []),
        }

    @staticmethod
    def _format_plan_for_review(plan: list[PlanStep]) -> str:
        """Format plan for review."""
        lines = []
        for step in plan:
            lines.append(f"Step {step['step_number']}: {step['description']}")
            lines.append(f"  Status: {step['status']}")
            if step.get("result"):
                lines.append(f"  Result: {step['result']}")
            lines.append("")

        return "\n".join(lines)
