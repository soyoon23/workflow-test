"""Tool mocking infrastructure for evaluation tests.

This module provides utilities for mocking tool execution during evaluation,
allowing tests to simulate tool responses without calling actual external APIs.
"""

from .tool_mocker import MockToolRegistry

__all__ = ["MockToolRegistry"]
