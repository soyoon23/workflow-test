"""Act node for the Plan-Act-Review workflow."""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..components import WorkflowComponents
from ..state import AgentState
from ..tool_executor import execute_tool_calls
from .base import NodeMixin

logger = logging.getLogger(__name__)


class ActNode(NodeMixin):
    """Executes the current step in the plan.

    Calls the LLM with available tools, handles tool execution,
    and records the step result.
    """

    def __init__(self, components: WorkflowComponents) -> None:
        self.components = components

    def __call__(self, state: AgentState) -> dict[str, Any]:
        """Execute the current step in the plan."""
        c = self.components
        system_prompt = self._get_prompt("actor", state)
        plan = state["plan"]
        current_index = state["current_step_index"]

        if current_index >= len(plan):
            return {"error": "No more steps to execute"}

        current_step = plan[current_index]

        c.callback.phase_start("act")
        c.callback.step_start(current_step["step_number"], current_step["description"])

        # Mark step as in progress
        plan[current_index]["status"] = "in_progress"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=(
                    f"Execute this step:\n"
                    f"Step {current_step['step_number']}: {current_step['description']}"
                )
            ),
        ]

        tools = self._get_tools(state)

        logger.debug(
            "act_node: executing step %d/%d: %s",
            current_step["step_number"],
            len(plan),
            current_step["description"][:80],
        )
        if c.callback.is_active:
            response = c.llm.stream_with_callback(
                messages,
                on_token=lambda t: c.callback.token("act", t),
                tools=tools,
            )
        else:
            response = c.llm.chat(messages, tools=tools)

        result_messages = [response]
        step_result = response.content

        # Handle tool calls
        new_skill_name = None
        if response.tool_calls:
            tool_result = execute_tool_calls(
                response,
                tool_registry=c.tools,
                callback=c.callback,
            )
            result_messages.extend(tool_result.messages)
            step_result += tool_result.combined_result_text
            new_skill_name = tool_result.new_skill_name

            # Get final response after tool execution
            messages.extend(result_messages)
            if c.callback.is_active:
                final_response = c.llm.stream_with_callback(
                    messages,
                    on_token=lambda t: c.callback.token("act", t),
                )
            else:
                final_response = c.llm.chat(messages)
            result_messages.append(final_response)
            step_result = final_response.content

        # Update step with result
        plan[current_index]["status"] = "completed"
        plan[current_index]["result"] = step_result
        logger.info("act_node: completed step %d", current_step["step_number"])

        c.callback.step_result(current_step["step_number"], step_result)
        c.callback.phase_end("act")

        result: dict[str, Any] = {
            "messages": result_messages,
            "plan": plan,
            "current_step_index": current_index + 1,
        }

        if new_skill_name:
            result["active_skill_name"] = new_skill_name

        return result
