"""Unit tests for ToolRegistry."""

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

    def test_web_search_execution(self, tool_registry):
        result = tool_registry.execute("web_search", query="test query")
        assert result["success"] is True
        assert result["query"] == "test query"
        assert len(result["results"]) == 3

    def test_web_search_custom_num_results(self, tool_registry):
        result = tool_registry.execute("web_search", query="test", num_results=5)
        assert len(result["results"]) == 5

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
