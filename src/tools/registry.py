"""Tool registry for managing available tools."""

from typing import Any, Callable, Optional
from .base import TOOL_DEFINITIONS, TOOL_FUNCTIONS


class ToolRegistry:
    """Registry for managing tools and their definitions."""

    def __init__(self):
        self._tools: dict[str, Callable] = {}
        self._definitions: dict[str, dict] = {}
        self._load_default_tools()

    def _load_default_tools(self) -> None:
        """Load default tools."""
        for name, func in TOOL_FUNCTIONS.items():
            self._tools[name] = func
            self._definitions[name] = TOOL_DEFINITIONS[name]

    def register(
        self,
        name: str,
        func: Callable,
        definition: dict,
    ) -> None:
        """Register a new tool."""
        self._tools[name] = func
        self._definitions[name] = definition

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)
        self._definitions.pop(name, None)

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a tool function by name."""
        return self._tools.get(name)

    def get_definition(self, name: str) -> Optional[dict]:
        """Get a tool definition by name."""
        return self._definitions.get(name)

    def get_all_definitions(self) -> list[dict]:
        """Get all tool definitions in OpenAI format."""
        return list(self._definitions.values())

    def get_available_tools(self) -> list[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name with given arguments."""
        tool = self._tools.get(name)
        if tool is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return tool(**kwargs)
        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def execute_tool_call(self, tool_call: dict) -> Any:
        """Execute a tool call from LLM response."""
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        return self.execute(name, **args)
