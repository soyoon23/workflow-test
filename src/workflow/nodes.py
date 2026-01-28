"""Workflow nodes for Plan-Act workflow."""

import json
from typing import Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

from .state import AgentState, PlanStep
from ..llm.client import LLMClient
from ..prompts.registry import PromptRegistry
from ..tools.registry import ToolRegistry


class WorkflowNodes:
    """Nodes for the Plan-Act workflow."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_registry: PromptRegistry,
        tool_registry: ToolRegistry,
    ):
        self.llm = llm_client
        self.prompts = prompt_registry
        self.tools = tool_registry

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        """Create or update the execution plan."""
        system_prompt = self.prompts.get("planner")
        user_request = state["user_request"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Create a plan for: {user_request}"),
        ]

        # If we have previous results, include them for replanning
        if state.get("plan") and any(
            step["status"] in ["completed", "failed"] for step in state["plan"]
        ):
            results_summary = self._summarize_results(state["plan"])
            messages.append(
                HumanMessage(
                    content=f"Previous execution results:\n{results_summary}\n\n"
                    "Please update the plan if needed."
                )
            )

        response = self.llm.chat(messages)

        # Parse the plan from response
        plan = self._parse_plan(response.content)

        return {
            "messages": [response],
            "plan": plan,
            "current_step_index": 0,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

    def act_node(self, state: AgentState) -> dict[str, Any]:
        """Execute the current step in the plan."""
        system_prompt = self.prompts.get("actor")
        plan = state["plan"]
        current_index = state["current_step_index"]

        if current_index >= len(plan):
            return {"error": "No more steps to execute"}

        current_step = plan[current_index]

        # Mark step as in progress
        plan[current_index]["status"] = "in_progress"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Execute this step:\n"
                f"Step {current_step['step_number']}: {current_step['description']}"
            ),
        ]

        # Get available tools
        tools = self.tools.get_all_definitions()

        # Call LLM with tools
        response = self.llm.chat(messages, tools=tools)

        result_messages = [response]
        step_result = response.content

        # Handle tool calls
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = self.tools.execute_tool_call(tool_call)
                tool_message = ToolMessage(
                    content=json.dumps(tool_result),
                    tool_call_id=tool_call["id"],
                )
                result_messages.append(tool_message)
                step_result += f"\nTool result: {json.dumps(tool_result)}"

            # Get final response after tool execution
            messages.extend(result_messages)
            final_response = self.llm.chat(messages)
            result_messages.append(final_response)
            step_result = final_response.content

        # Update step with result
        plan[current_index]["status"] = "completed"
        plan[current_index]["result"] = step_result

        return {
            "messages": result_messages,
            "plan": plan,
            "current_step_index": current_index + 1,
        }

    def review_node(self, state: AgentState) -> dict[str, Any]:
        """Review execution results and determine next steps."""
        system_prompt = self.prompts.get("reviewer")
        user_request = state["user_request"]
        plan = state["plan"]

        # Build review context
        plan_summary = self._format_plan_for_review(plan)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"Original request: {user_request}\n\n"
                f"Plan and execution results:\n{plan_summary}\n\n"
                "Please review and determine next steps."
            ),
        ]

        response = self.llm.chat(messages)

        # Parse review result
        review = self._parse_review(response.content)

        return {
            "messages": [response],
            "is_complete": review.get("is_complete", False),
            "final_answer": review.get("final_answer"),
            "error": review.get("error"),
        }

    def _parse_plan(self, content: str) -> list[PlanStep]:
        """Parse plan from LLM response."""
        try:
            # Try to extract JSON from response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                plan_data = json.loads(json_str)

                steps = []
                for step in plan_data.get("steps", []):
                    steps.append(
                        PlanStep(
                            step_number=step.get("step_number", len(steps) + 1),
                            description=step.get("description", ""),
                            status="pending",
                            result=None,
                        )
                    )
                return steps
        except json.JSONDecodeError:
            pass

        # Fallback: create a single step from the content
        return [
            PlanStep(
                step_number=1,
                description=content[:500],
                status="pending",
                result=None,
            )
        ]

    def _parse_review(self, content: str) -> dict[str, Any]:
        """Parse review from LLM response."""
        try:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # Fallback: assume complete if we can't parse
        return {
            "is_complete": True,
            "final_answer": content,
        }

    def _summarize_results(self, plan: list[PlanStep]) -> str:
        """Summarize execution results for replanning."""
        lines = []
        for step in plan:
            status_emoji = {
                "completed": "[OK]",
                "failed": "[FAIL]",
                "pending": "[...]",
                "in_progress": "[>>]",
            }.get(step["status"], "[?]")

            lines.append(
                f"{status_emoji} Step {step['step_number']}: {step['description']}"
            )
            if step.get("result"):
                lines.append(f"   Result: {step['result'][:200]}")

        return "\n".join(lines)

    def _format_plan_for_review(self, plan: list[PlanStep]) -> str:
        """Format plan for review."""
        lines = []
        for step in plan:
            lines.append(f"Step {step['step_number']}: {step['description']}")
            lines.append(f"  Status: {step['status']}")
            if step.get("result"):
                lines.append(f"  Result: {step['result']}")
            lines.append("")

        return "\n".join(lines)
