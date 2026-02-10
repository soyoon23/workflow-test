"""Unit tests for ToolRegistry."""

from unittest.mock import MagicMock, patch

import httpx

from src.tools.registry import ToolRegistry


class TestToolRegistry:
    """Tests for ToolRegistry functionality."""

    def test_default_tools_loaded(self, tool_registry):
        names = tool_registry.get_available_tools()
        assert "calculator" in names
        assert "web_search" in names

    def test_calculator_execution(self, tool_registry):
        result = tool_registry.execute("calculator", expression="2 + 3")
        assert result["success"] is True
        assert result["result"] == 5

    def test_calculator_sqrt(self, tool_registry):
        result = tool_registry.execute("calculator", expression="sqrt(16)")
        assert result["success"] is True
        assert result["result"] == 4.0

    def test_calculator_invalid_expression(self, tool_registry):
        result = tool_registry.execute("calculator", expression="invalid")
        assert result["success"] is False
        assert "error" in result

    @patch("src.tools.base.httpx.Client")
    def test_web_search_execution(self, mock_client_cls, tool_registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": "Result 1", "content": "Snippet 1", "url": "https://example.com/1"},
                {"title": "Result 2", "content": "Snippet 2", "url": "https://example.com/2"},
                {"title": "Result 3", "content": "Snippet 3", "url": "https://example.com/3"},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = tool_registry.execute("web_search", query="test query")
        assert result["success"] is True
        assert result["query"] == "test query"
        assert len(result["results"]) == 3
        assert result["results"][0]["title"] == "Result 1"
        assert result["results"][0]["snippet"] == "Snippet 1"

    @patch("src.tools.base.httpx.Client")
    def test_web_search_custom_num_results(self, mock_client_cls, tool_registry):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"title": f"R{i}", "content": f"S{i}", "url": f"https://example.com/{i}"}
                for i in range(5)
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = tool_registry.execute("web_search", query="test", num_results=5)
        assert result["success"] is True
        assert len(result["results"]) == 5

    @patch("src.tools.base.httpx.Client")
    def test_web_search_timeout(self, mock_client_cls, tool_registry):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = tool_registry.execute("web_search", query="test")
        assert result["success"] is False
        assert "timed out" in result["error"]

    @patch("src.tools.base.httpx.Client")
    def test_web_search_connection_error(self, mock_client_cls, tool_registry):
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("refused")
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

        result = tool_registry.execute("web_search", query="test")
        assert result["success"] is False
        assert "Cannot connect" in result["error"]

    def test_unknown_tool_returns_error(self, tool_registry):
        result = tool_registry.execute("nonexistent_tool")
        assert "error" in result

    def test_get_definition_returns_openai_format(self, tool_registry):
        defn = tool_registry.get_definition("calculator")
        assert defn is not None
        assert defn["type"] == "function"
        assert defn["function"]["name"] == "calculator"
        assert "parameters" in defn["function"]

    def test_get_definition_unknown_returns_none(self, tool_registry):
        assert tool_registry.get_definition("nonexistent") is None

    def test_get_all_definitions(self, tool_registry):
        definitions = tool_registry.get_all_definitions()
        assert len(definitions) >= 2
        names = [d["function"]["name"] for d in definitions]
        assert "calculator" in names
        assert "web_search" in names

    def test_register_and_unregister(self):
        registry = ToolRegistry()

        def dummy_tool(x: str) -> dict:
            return {"result": x}

        defn = {
            "type": "function",
            "function": {
                "name": "dummy",
                "description": "A dummy tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        registry.register("dummy", dummy_tool, defn)
        assert "dummy" in registry.get_available_tools()
        assert registry.execute("dummy", x="hello") == {"result": "hello"}

        registry.unregister("dummy")
        assert "dummy" not in registry.get_available_tools()

    def test_execute_tool_call(self, tool_registry):
        tool_call = {"name": "calculator", "args": {"expression": "10 * 2"}}
        result = tool_registry.execute_tool_call(tool_call)
        assert result["success"] is True
        assert result["result"] == 20
