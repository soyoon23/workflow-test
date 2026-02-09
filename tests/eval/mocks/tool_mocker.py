"""Mock tool registry for evaluation tests.

Provides MockToolRegistry that extends ToolRegistry to inject predefined responses
during testing, simulating tool execution without calling real APIs.
"""

from typing import Any, Callable, Optional

from src.tools.registry import ToolRegistry


class MockToolRegistry(ToolRegistry):
    """Test-only tool registry with mock response injection capability.

    Extends ToolRegistry to allow registering mock responses for tools.
    When a tool is executed, if a mock response is registered, it returns
    the mock response; otherwise, it falls back to the real tool execution.

    Example:
        >>> registry = MockToolRegistry()
        >>> registry.register_mock("web_search", [
        ...     {"success": True, "results": ["result1", "result2"]}
        ... ])
        >>> result = registry.execute("web_search", query="test")
        >>> assert result["success"] is True
    """

    def __init__(self):
        """Initialize MockToolRegistry with empty mock response storage."""
        super().__init__()
        self._mock_responses: dict[str, list[dict]] = {}
        self._mock_call_counts: dict[str, int] = {}

    def register_mock(
        self,
        tool_name: str,
        responses: list[dict] | dict,
    ) -> None:
        """Register mock response(s) for a tool.

        Mock responses are returned sequentially on each tool execution.
        If responses is a dict, it's converted to a single-item list.
        If all responses are exhausted, the last response is repeated.

        Args:
            tool_name: Name of the tool to mock.
            responses: Mock response(s) to return.
                - If dict: single response (converted to list)
                - If list: responses returned sequentially

        Example:
            >>> registry = MockToolRegistry()
            >>> # Single response
            >>> registry.register_mock("calculator", {"result": 42})
            >>> # Multiple responses
            >>> registry.register_mock("web_search", [
            ...     {"results": ["first_call"]},
            ...     {"results": ["second_call"]}
            ... ])
        """
        if isinstance(responses, dict):
            responses = [responses]
        self._mock_responses[tool_name] = responses
        self._mock_call_counts[tool_name] = 0

    def unregister_mock(self, tool_name: str) -> None:
        """Unregister mock response for a tool.

        After unregistering, the tool will fall back to real execution.

        Args:
            tool_name: Name of the tool to unmock.
        """
        self._mock_responses.pop(tool_name, None)
        self._mock_call_counts.pop(tool_name, None)

    def clear_mocks(self) -> None:
        """Clear all registered mock responses."""
        self._mock_responses.clear()
        self._mock_call_counts.clear()

    def has_mock(self, tool_name: str) -> bool:
        """Check if a tool has a registered mock.

        Args:
            tool_name: Name of the tool.

        Returns:
            True if mock is registered, False otherwise.
        """
        return tool_name in self._mock_responses

    def execute(self, name: str, **kwargs) -> Any:
        """Execute a tool by name, using mock if registered.

        If a mock response is registered for the tool, returns the mock response.
        Otherwise, falls back to the parent class implementation (real execution).

        Args:
            name: Name of the tool to execute.
            **kwargs: Tool arguments.

        Returns:
            Mock response if registered, otherwise real tool execution result.
        """
        if name in self._mock_responses:
            responses = self._mock_responses[name]
            call_count = self._mock_call_counts[name]

            # Cycle through responses or repeat last one
            response_idx = min(call_count, len(responses) - 1)
            response = responses[response_idx]

            # Increment call count for next execution
            self._mock_call_counts[name] += 1

            # Mark as mock response
            result = {"__mock__": True, **response}
            return result

        # Fallback to real tool execution
        return super().execute(name, **kwargs)

    def execute_tool_call(self, tool_call: dict) -> Any:
        """Execute a tool call from LLM response, using mock if registered.

        Args:
            tool_call: Tool call dict with 'name' and 'args' keys.

        Returns:
            Mock response if registered, otherwise real tool execution result.
        """
        name = tool_call.get("name")
        args = tool_call.get("args", {})
        return self.execute(name, **args)

    def get_call_count(self, tool_name: str) -> int:
        """Get the number of times a mock tool has been called.

        Args:
            tool_name: Name of the tool.

        Returns:
            Number of calls to the mock tool, or 0 if no mock is registered.
        """
        return self._mock_call_counts.get(tool_name, 0)

    def reset_call_counts(self) -> None:
        """Reset all mock call counters to 0.

        Useful for reusing the same MockToolRegistry across multiple tests.
        """
        for tool_name in self._mock_responses:
            self._mock_call_counts[tool_name] = 0
