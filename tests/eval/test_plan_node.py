"""Evaluation tests for the Plan node."""

import pytest
from deepeval.test_case import LLMTestCase

from .conftest import load_goldens
from .metrics.plan_quality import (
    PlanCompletenessGEval,
    PlanDecompositionGEval,
    SkillSelectionMetric,
)
from .observability_helpers import assert_metrics_with_tracing
from .traced_workflow import _make_runnable_config, _traced_plan


@pytest.mark.eval
class TestPlanNodeEval:
    """Evaluate Plan node output quality using DeepEval metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, workflow_components, request, obs_test_trace):
        from src.workflow.nodes import PlanNode

        self.plan_node = PlanNode(workflow_components)
        self.goldens = load_goldens("plan_goldens")
        self.tracing_ctx = getattr(request, "obs_tracing_context", None)
        self.runnable_config = _make_runnable_config(getattr(request, "obs_callback", None))

    def _make_state(self, user_request: str, auto_select: bool = True) -> dict:
        return {
            "messages": [],
            "user_request": user_request,
            "plan": [],
            "current_step_index": 0,
            "is_complete": False,
            "final_answer": None,
            "iteration_count": 0,
            "error": None,
            "active_skill_name": None,
            "auto_select_skill": auto_select,
            "conversation_history": [],
            "review_key_facts": [],
        }

    @pytest.mark.parametrize("golden_idx", range(5))
    def test_plan_decomposition(self, golden_idx):
        """Plan should logically decompose the user request into steps."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        state = self._make_state(golden["input"])
        result = _traced_plan(
            self.plan_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        plan_steps = result.get("plan", [])
        plan_text = "\n".join(f"{s['step_number']}. {s['description']}" for s in plan_steps)

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=plan_text,
        )

        assert_metrics_with_tracing(
            test_case,
            [PlanDecompositionGEval],
            self.tracing_ctx,
            span_name="plan_decomposition_geval",
            metadata={"golden_idx": golden_idx},
        )

    @pytest.mark.parametrize("golden_idx", range(5))
    def test_plan_completeness(self, golden_idx):
        """Plan should include all steps needed to fulfill the request."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        state = self._make_state(golden["input"])
        result = _traced_plan(
            self.plan_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        plan_steps = result.get("plan", [])
        plan_text = "\n".join(f"{s['step_number']}. {s['description']}" for s in plan_steps)

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=plan_text,
        )

        assert_metrics_with_tracing(
            test_case,
            [PlanCompletenessGEval],
            self.tracing_ctx,
            span_name="plan_completeness_geval",
            metadata={"golden_idx": golden_idx},
        )

    @pytest.mark.parametrize("golden_idx", range(5))
    def test_skill_selection(self, golden_idx):
        """Plan node should select the appropriate skill for the request."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        expected_skill = golden.get("metadata", {}).get("expected_skill")

        state = self._make_state(golden["input"], auto_select=True)
        result = _traced_plan(
            self.plan_node, state, tracing_ctx=self.tracing_ctx, config=self.runnable_config
        )

        actual_skill = result.get("active_skill_name")
        plan_text = "\n".join(
            f"{s['step_number']}. {s['description']}" for s in result.get("plan", [])
        )

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=plan_text,
            additional_metadata={
                "expected_skill": expected_skill,
                "actual_skill": actual_skill,
            },
        )

        metric = SkillSelectionMetric()
        assert_metrics_with_tracing(
            test_case,
            [metric],
            self.tracing_ctx,
            span_name="skill_selection_metric",
            metadata={
                "golden_idx": golden_idx,
                "expected_skill": expected_skill,
                "actual_skill": actual_skill,
            },
        )
