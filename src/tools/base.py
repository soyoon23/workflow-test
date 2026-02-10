"""Tools for Plan-Act workflow."""

import logging
import math
from pathlib import Path
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)


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


def _load_search_config() -> dict:
    """Load search-related config from config.yaml."""
    config = {}
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    llm_cfg = config.get("llm", {})
    search_cfg = config.get("search", {})

    return {
        "base_url": llm_cfg.get("base_url", "http://localhost:4000").rstrip("/"),
        "api_key": llm_cfg.get("api_key", "sk-1234"),
        "tool_name": search_cfg.get("tool_name", "tavily-search"),
        "default_depth": search_cfg.get("default_depth", "basic"),
        "max_results": search_cfg.get("max_results", 5),
        "timeout": search_cfg.get("timeout", 10),
    }


def web_search(query: str, num_results: int = 3, search_depth: str = "basic") -> dict[str, Any]:
    """Search the web via Tavily through the LiteLLM proxy.

    Args:
        query: The search query.
        num_results: Number of results to return.
        search_depth: "basic" (fast) or "advanced" (thorough).

    Returns:
        Dictionary with search results.
    """
    cfg = _load_search_config()
    url = f"{cfg['base_url']}/v1/search/{cfg['tool_name']}"
    headers = {
        "Authorization": f"Bearer {cfg['api_key']}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": query,
        "max_results": num_results,
        "search_depth": search_depth,
    }

    try:
        with httpx.Client(timeout=cfg["timeout"]) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        logger.warning("Tavily search timed out for query: %s", query)
        return {
            "success": False,
            "query": query,
            "num_results": 0,
            "results": [],
            "error": "Search request timed out. Please try again.",
        }
    except httpx.HTTPStatusError as e:
        logger.error("Tavily search HTTP error %d: %s", e.response.status_code, e.response.text)
        return {
            "success": False,
            "query": query,
            "num_results": 0,
            "results": [],
            "error": f"Search service returned HTTP {e.response.status_code}.",
        }
    except httpx.ConnectError:
        logger.error("Cannot connect to LiteLLM proxy at %s", cfg["base_url"])
        return {
            "success": False,
            "query": query,
            "num_results": 0,
            "results": [],
            "error": "Cannot connect to search service. Is the LiteLLM proxy running?",
        }
    except Exception as e:
        logger.error("Unexpected search error: %s", e, exc_info=True)
        return {
            "success": False,
            "query": query,
            "num_results": 0,
            "results": [],
            "error": f"Search failed: {str(e)}",
        }

    results = []
    for item in data.get("results", []):
        results.append(
            {
                "title": item.get("title", ""),
                "snippet": item.get("content", item.get("snippet", "")),
                "url": item.get("url", ""),
            }
        )

    return {
        "success": True,
        "query": query,
        "num_results": len(results),
        "results": results,
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
                        "description": (
                            "The mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
                        ),
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
            "description": (
                "Search the web for current information using Tavily. "
                "Returns real search results with titles, snippets, and URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default: 3, max: 10)",
                        "default": 3,
                    },
                    "search_depth": {
                        "type": "string",
                        "description": (
                            "Search depth: 'basic' for fast results, "
                            "'advanced' for thorough research"
                        ),
                        "enum": ["basic", "advanced"],
                        "default": "basic",
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
