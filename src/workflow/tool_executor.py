"""Tool execution loop extracted from act_node for reuse.

Handles iterating over LLM tool calls, dispatching to the tool registry,
processing the use_skill special case, and emitting streaming events.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import AIMessage, ToolMessage

from ..tools.registry import ToolRegistry
from .streaming import StreamCallback

logger = logging.getLogger(__name__)


@dataclass
class ToolExecutionResult:
    """Result of executing tool calls from an LLM response."""

    messages: list[ToolMessage] = field(default_factory=list)
    combined_result_text: str = ""
    new_skill_name: Optional[str] = None


def execute_tool_calls(
    response: AIMessage,
    *,
    tool_registry: ToolRegistry,
    callback: StreamCallback,
) -> ToolExecutionResult:
    """Execute tool calls from an LLM response.

    Handles the use_skill special case for dynamic skill switching.
    Emits tool_call / tool_result streaming events.

    Args:
        response: AIMessage containing tool_calls.
        tool_registry: Registry to dispatch tool execution.
        callback: StreamCallback for emitting events.

    Returns:
        ToolExecutionResult with messages, combined text, and optional new skill.
    """
    result = ToolExecutionResult()

    for tool_call in response.tool_calls:
        callback.tool_call(tool_call["name"], tool_call.get("args", {}))

        if tool_call["name"] == "use_skill":
            result.new_skill_name = tool_call["args"].get("skill_name")
            reason = tool_call["args"].get("reason", "")
            tool_result = {
                "success": True,
                "message": f"Switched to skill: {result.new_skill_name}",
                "reason": reason,
            }
        else:
            tool_result = tool_registry.execute_tool_call(tool_call)

        callback.tool_result(tool_call["name"], tool_result)

        tool_message = ToolMessage(
            content=json.dumps(tool_result),
            tool_call_id=tool_call["id"],
        )
        result.messages.append(tool_message)
        result.combined_result_text += f"\nTool result: {json.dumps(tool_result)}"

    return result
