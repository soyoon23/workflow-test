"""Evaluation tests for the Act node."""

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from .conftest import load_goldens
from .metrics.step_execution import StepRelevancyGEval


@pytest.mark.eval
class TestActNodeEval:
    """Evaluate Act node execution quality using DeepEval metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, workflow_nodes, eval_model):
        self.nodes = workflow_nodes
        self.eval_model = eval_model
        self.goldens = load_goldens("act_goldens")

    def _make_state_with_step(self, step_description: str, skill_name=None) -> dict:
        """Create a state with a single pending step for act_node to execute."""
        return {
            "messages": [],
            "user_request": step_description,
            "plan": [
                {
                    "step_number": 1,
                    "description": step_description,
                    "status": "pending",
                    "result": None,
                },
            ],
            "current_step_index": 0,
            "is_complete": False,
            "final_answer": None,
            "iteration_count": 1,
            "error": None,
            "active_skill_name": skill_name,
            "auto_select_skill": False,
            "conversation_history": [],
            "review_key_facts": [],
        }

    @pytest.mark.parametrize("golden_idx", range(4))
    def test_tool_correctness(self, golden_idx):
        """Act node should use the correct tools for the step."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        metadata = golden.get("metadata", {})
        step_desc = metadata.get("step_description", golden["input"])
        expected_tools_names = metadata.get("expected_tools", [])

        state = self._make_state_with_step(step_desc)
        result = self.nodes.act_node(state)

        # Collect actual tool calls from messages
        actual_tools: list[ToolCall] = []
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] != "use_skill":
                        actual_tools.append(ToolCall(name=tc["name"]))

        expected_tools = [ToolCall(name=n) for n in expected_tools_names]

        step_result = result.get("plan", [{}])[0].get("result", "")
        test_case = LLMTestCase(
            input=step_desc,
            actual_output=str(step_result),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        metric = ToolCorrectnessMetric(threshold=0.5, model=self.eval_model)
        assert_test(test_case, [metric])

    @pytest.mark.parametrize("golden_idx", range(4))
    def test_step_relevancy(self, golden_idx):
        """Act node result should be relevant to the step description."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        metadata = golden.get("metadata", {})
        step_desc = metadata.get("step_description", golden["input"])

        state = self._make_state_with_step(step_desc)
        result = self.nodes.act_node(state)

        step_result = result.get("plan", [{}])[0].get("result", "")
        test_case = LLMTestCase(
            input=step_desc,
            actual_output=str(step_result),
        )

        assert_test(test_case, [StepRelevancyGEval])
