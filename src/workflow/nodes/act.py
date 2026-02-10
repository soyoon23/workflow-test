"""Act node for the Plan-Act-Review workflow."""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..components import WorkflowComponents
from ..state import AgentState
from ..tool_executor import execute_tool_calls
from .base import NodeMixin

logger = logging.getLogger(__name__)


class ActNode(NodeMixin):
    """Executes the current step in the plan.

    Calls the LLM with available tools, handles tool execution
    (including multi-round tool call loops), and records the step result.
    """

    MAX_TOOL_ROUNDS = 3

    def __init__(self, components: WorkflowComponents) -> None:
        self.components = components

    @staticmethod
    def _build_prior_context(plan: list, current_index: int) -> str:
        """Build context from completed steps prior to current_index."""
        lines = []
        for step in plan[:current_index]:
            if step["status"] == "completed" and step.get("result"):
                lines.append(
                    f"Step {step['step_number']}: {step['description']}\n  Result: {step['result']}"
                )
        return "\n".join(lines)

    def _call_llm(
        self,
        messages: list,
        config: RunnableConfig,
        tools: list[dict] | None = None,
    ) -> AIMessage:
        """Invoke the LLM, streaming tokens to the callback when active."""
        c = self.components
        if c.callback.is_active:
            return c.llm.stream_with_callback(
                messages,
                on_token=lambda t: c.callback.token("act", t),
                tools=tools,
                config=config,
            )
        return c.llm.chat(messages, tools=tools, config=config)

    def __call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
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
        plan[current_index]["status"] = "in_progress"

        # Build step instruction with user_request and prior context
        user_request = state["user_request"]
        prior_context = self._build_prior_context(plan, current_index)

        parts = [f"Original request: {user_request}"]
        if prior_context:
            parts.append(f"Previous step results:\n{prior_context}")
        parts.append(
            f"Execute this step:\nStep {current_step['step_number']}: {current_step['description']}"
        )
        step_instruction = "\n\n".join(parts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=step_instruction),
        ]
        tools = self._get_tools(state)

        # Initialize before try for safe access in except
        result_messages: list = []
        new_skill_name = None

        try:
            logger.debug(
                "act_node: executing step %d/%d: %s",
                current_step["step_number"],
                len(plan),
                current_step["description"][:80],
            )

            response = self._call_llm(messages, config, tools=tools)
            result_messages = [response]

            # Multi-round tool call loop
            tool_round = 0
            while response.tool_calls and tool_round < self.MAX_TOOL_ROUNDS:
                tool_result = execute_tool_calls(
                    response,
                    tool_registry=c.tools,
                    callback=c.callback,
                )
                result_messages.extend(tool_result.messages)
                if tool_result.new_skill_name:
                    new_skill_name = tool_result.new_skill_name

                messages.append(response)
                messages.extend(tool_result.messages)
                tool_round += 1

                # Last round: omit tools to force text response
                offer_tools = tools if tool_round < self.MAX_TOOL_ROUNDS else None
                response = self._call_llm(messages, config, tools=offer_tools)
                result_messages.append(response)

            step_result = response.content

            plan[current_index]["status"] = "completed"
            plan[current_index]["result"] = step_result
            logger.info("act_node: completed step %d", current_step["step_number"])
            c.callback.step_result(current_step["step_number"], step_result)

        except Exception as e:
            error_msg = f"Step {current_step['step_number']} failed: {e}"
            logger.error("act_node: %s", error_msg, exc_info=True)
            plan[current_index]["status"] = "failed"
            plan[current_index]["result"] = error_msg
            c.callback.error("act", error_msg)
            c.callback.phase_end("act")
            return {
                "messages": result_messages,
                "plan": plan,
                "current_step_index": current_index + 1,
                "error": error_msg,
            }

        c.callback.phase_end("act")

        result: dict[str, Any] = {
            "messages": result_messages,
            "plan": plan,
            "current_step_index": current_index + 1,
        }
        if new_skill_name:
            result["active_skill_name"] = new_skill_name
        return result
