"""Evaluation tests for the Act node."""

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from .conftest import load_goldens
from .metrics.step_execution import StepRelevancyGEval
from .observability_helpers import assert_metrics_with_tracing
from .traced_workflow import _make_runnable_config, _traced_act


@pytest.mark.eval
class TestActNodeEval:
    """Evaluate Act node execution quality using DeepEval metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, workflow_components, eval_model, obs_test_trace):
        from src.workflow.nodes import ActNode

        self.act_node = ActNode(workflow_components)
        self.eval_model = eval_model
        self.goldens = load_goldens("act_goldens")
        # Unpack (callback, tracing_ctx) tuple from obs_test_trace
        obs_callback, tracing_ctx = obs_test_trace
        self.tracing_ctx = tracing_ctx
        self.runnable_config = _make_runnable_config(obs_callback)

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

    def _make_multistep_state(self, golden: dict) -> dict:
        """Create a state with multi-step plan where earlier steps are completed."""
        steps_data = golden["metadata"]["steps"]
        plan = [
            {
                "step_number": s["step_number"],
                "description": s["description"],
                "status": s["status"],
                "result": s.get("result"),
            }
            for s in steps_data
        ]
        current_index = next(i for i, s in enumerate(steps_data) if s["status"] == "pending")
        return {
            "messages": [],
            "user_request": golden["input"],
            "plan": plan,
            "current_step_index": current_index,
            "is_complete": False,
            "final_answer": None,
            "iteration_count": 1,
            "error": None,
            "active_skill_name": None,
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
        result, tool_calls = _traced_act(
            self.act_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        # Collect actual tool calls from messages
        actual_tools: list[ToolCall] = [tc for tc in tool_calls if tc.name != "use_skill"]

        expected_tools = [ToolCall(name=n) for n in expected_tools_names]

        step_result = result.get("plan", [{}])[0].get("result", "")
        test_case = LLMTestCase(
            input=step_desc,
            actual_output=str(step_result),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        metric = ToolCorrectnessMetric(threshold=0.5, model=self.eval_model)
        assert_metrics_with_tracing(
            test_case,
            [metric],
            self.tracing_ctx,
            span_name="tool_correctness_metric",
            metadata={
                "golden_idx": golden_idx,
                "expected_tools": expected_tools_names,
                "actual_tools": [tc.name for tc in actual_tools],
            },
        )

    @pytest.mark.parametrize("golden_idx", range(4))
    def test_step_relevancy(self, golden_idx):
        """Act node result should be relevant to the step description."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        metadata = golden.get("metadata", {})
        step_desc = metadata.get("step_description", golden["input"])

        state = self._make_state_with_step(step_desc)
        result, _ = _traced_act(
            self.act_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        step_result = result.get("plan", [{}])[0].get("result", "")
        test_case = LLMTestCase(
            input=step_desc,
            actual_output=str(step_result),
        )

        assert_metrics_with_tracing(
            test_case,
            [StepRelevancyGEval],
            self.tracing_ctx,
            span_name="step_relevancy_geval",
            metadata={
                "golden_idx": golden_idx,
                "step_description": step_desc,
            },
        )

    @pytest.mark.parametrize("golden_idx", range(2))
    def test_multistep_context_passing(self, golden_idx):
        """Act node should use prior step results when executing subsequent steps."""
        multistep_goldens = [g for g in self.goldens if g.get("metadata", {}).get("is_multistep")]
        if golden_idx >= len(multistep_goldens):
            pytest.skip("Golden index out of range")

        golden = multistep_goldens[golden_idx]
        state = self._make_multistep_state(golden)

        current_index = state["current_step_index"]
        current_step = state["plan"][current_index]

        result, _ = _traced_act(
            self.act_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        step_result = result.get("plan", [{}])[current_index].get("result", "")

        test_case = LLMTestCase(
            input=current_step["description"],
            actual_output=str(step_result),
        )

        assert_metrics_with_tracing(
            test_case,
            [StepRelevancyGEval],
            self.tracing_ctx,
            span_name="multistep_context_relevancy",
            metadata={
                "golden_idx": golden_idx,
                "current_step": current_index + 1,
                "prior_step_result": state["plan"][0].get("result"),
            },
        )

    @pytest.mark.parametrize("golden_idx", range(2))
    def test_multistep_tool_correctness(self, golden_idx):
        """Act node should use correct tools in multi-step context."""
        multistep_goldens = [g for g in self.goldens if g.get("metadata", {}).get("is_multistep")]
        if golden_idx >= len(multistep_goldens):
            pytest.skip("Golden index out of range")

        golden = multistep_goldens[golden_idx]
        state = self._make_multistep_state(golden)

        current_index = state["current_step_index"]
        current_step_meta = golden["metadata"]["steps"][current_index]
        expected_tools_names = current_step_meta.get("expected_tools", [])

        result, tool_calls = _traced_act(
            self.act_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        actual_tools: list[ToolCall] = [tc for tc in tool_calls if tc.name != "use_skill"]
        expected_tools = [ToolCall(name=n) for n in expected_tools_names]
        step_result = result.get("plan", [{}])[current_index].get("result", "")

        test_case = LLMTestCase(
            input=current_step_meta["description"],
            actual_output=str(step_result),
            tools_called=actual_tools,
            expected_tools=expected_tools,
        )

        metric = ToolCorrectnessMetric(threshold=0.5, model=self.eval_model)
        assert_metrics_with_tracing(
            test_case,
            [metric],
            self.tracing_ctx,
            span_name="multistep_tool_correctness",
            metadata={
                "golden_idx": golden_idx,
                "expected_tools": expected_tools_names,
                "actual_tools": [tc.name for tc in actual_tools],
            },
        )
