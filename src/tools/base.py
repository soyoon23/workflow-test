"""Sample tools for Plan-Act workflow."""

import math
from typing import Any


def calculator(expression: str) -> dict[str, Any]:
    """
    Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)")

    Returns:
        Dictionary with result or error
    """
    try:
        # Safe evaluation with limited functions
        allowed_names = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

        # Evaluate the expression
        result = eval(expression, {"__builtins__": {}}, allowed_names)

        return {
            "success": True,
            "expression": expression,
            "result": result,
        }
    except Exception as e:
        return {
            "success": False,
            "expression": expression,
            "error": str(e),
        }


def web_search(query: str, num_results: int = 3) -> dict[str, Any]:
    """
    Search the web for information (simulated).

    Args:
        query: The search query
        num_results: Number of results to return

    Returns:
        Dictionary with search results
    """
    # Simulated search results for demonstration
    # In production, this would call an actual search API via LiteLLM
    simulated_results = [
        {
            "title": f"Result {i+1} for: {query}",
            "snippet": f"This is a simulated search result for '{query}'. "
                       f"In production, this would be real search data.",
            "url": f"https://example.com/result{i+1}",
        }
        for i in range(num_results)
    ]

    return {
        "success": True,
        "query": query,
        "num_results": len(simulated_results),
        "results": simulated_results,
    }


# Tool definitions for LiteLLM/OpenAI format
TOOL_DEFINITIONS = {
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports basic arithmetic, "
                          "trigonometry (sin, cos, tan), logarithms (log, log10), "
                          "sqrt, pow, and constants (pi, e).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns relevant search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3)",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
}

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "calculator": calculator,
    "web_search": web_search,
}
