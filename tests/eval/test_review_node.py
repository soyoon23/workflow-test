"""Evaluation tests for the Review node."""

import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from .conftest import load_goldens
from .metrics.review_quality import CompletionDecisionMetric, KeyFactsQualityGEval
from .traced_workflow import _traced_review


@pytest.mark.eval
class TestReviewNodeEval:
    """Evaluate Review node quality using DeepEval metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self, workflow_components, eval_model, request):
        from src.workflow.nodes import ReviewNode

        self.review_node = ReviewNode(workflow_components)
        self.eval_model = eval_model
        self.goldens = load_goldens("review_goldens")
        self.tracing_ctx = getattr(request, "obs_tracing_context", None)

    def _make_state_with_completed_plan(self, golden: dict) -> dict:
        """Build a state with completed plan steps from golden metadata."""
        metadata = golden.get("metadata", {})
        plan_results = metadata.get("plan_results", [])

        plan = [
            {
                "step_number": pr["step"],
                "description": pr["description"],
                "status": "completed",
                "result": pr["result"],
            }
            for pr in plan_results
        ]

        return {
            "messages": [],
            "user_request": golden["input"],
            "plan": plan,
            "current_step_index": len(plan),
            "is_complete": False,
            "final_answer": None,
            "iteration_count": 1,
            "error": None,
            "active_skill_name": None,
            "auto_select_skill": False,
            "conversation_history": [],
            "review_key_facts": [],
        }

    @pytest.mark.parametrize("golden_idx", range(3))
    def test_answer_relevancy(self, golden_idx):
        """Review final_answer should be relevant to the original request."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        state = self._make_state_with_completed_plan(golden)
        result = _traced_review(self.review_node, state, tracing_ctx=self.tracing_ctx)

        final_answer = result.get("final_answer", "") or ""
        plan_context = [step["result"] for step in state["plan"] if step.get("result")]

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=final_answer,
            retrieval_context=plan_context,
        )

        metric = AnswerRelevancyMetric(threshold=0.7, model=self.eval_model)
        assert_test(test_case, [metric])

    @pytest.mark.parametrize("golden_idx", range(3))
    def test_faithfulness(self, golden_idx):
        """Review final_answer should faithfully reflect plan results."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        state = self._make_state_with_completed_plan(golden)
        result = _traced_review(self.review_node, state, tracing_ctx=self.tracing_ctx)

        final_answer = result.get("final_answer", "") or ""
        plan_context = [step["result"] for step in state["plan"] if step.get("result")]

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=final_answer,
            retrieval_context=plan_context,
        )

        metric = FaithfulnessMetric(threshold=0.7, model=self.eval_model)
        assert_test(test_case, [metric])

    @pytest.mark.parametrize("golden_idx", range(3))
    def test_key_facts_quality(self, golden_idx):
        """Reviewer-extracted key_facts should be useful for multi-turn context."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        state = self._make_state_with_completed_plan(golden)
        result = _traced_review(self.review_node, state, tracing_ctx=self.tracing_ctx)

        key_facts = result.get("review_key_facts", [])
        key_facts_text = "\n".join(f"- {f}" for f in key_facts) if key_facts else "None"
        plan_context = [step["result"] for step in state["plan"] if step.get("result")]

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=key_facts_text,
            context=plan_context,
        )

        assert_test(test_case, [KeyFactsQualityGEval])

    @pytest.mark.parametrize("golden_idx", range(3))
    def test_completion_decision(self, golden_idx):
        """Review node should make the correct is_complete decision."""
        if golden_idx >= len(self.goldens):
            pytest.skip("Golden index out of range")

        golden = self.goldens[golden_idx]
        expected_complete = golden.get("metadata", {}).get("expected_complete")
        state = self._make_state_with_completed_plan(golden)
        result = _traced_review(self.review_node, state, tracing_ctx=self.tracing_ctx)

        actual_complete = result.get("is_complete", False)

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=result.get("final_answer", "") or "",
            additional_metadata={
                "expected_complete": expected_complete,
                "actual_complete": actual_complete,
            },
        )

        metric = CompletionDecisionMetric()
        assert_test(test_case, [metric])
