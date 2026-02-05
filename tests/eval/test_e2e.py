"""End-to-end evaluation tests using DeepEval agentic metrics."""

import pytest
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    PlanAdherenceMetric,
    PlanQualityMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
)
from deepeval.test_case import LLMTestCase

from .conftest import load_goldens, run_workflow_for_eval
from .observability_helpers import assert_metrics_with_tracing
from .traced_workflow import traced_workflow


@pytest.mark.eval
class TestE2EAgenticMetrics:
    """E2E evaluation using DeepEval's trace-based agentic metrics.

    These tests use the @observe-traced workflow wrapper so that
    TaskCompletion, PlanQuality, PlanAdherence, and StepEfficiency
    metrics can analyse the full execution trace.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, eval_config, eval_model, eval_threshold, obs_test_trace):
        self.config = eval_config
        self.eval_model = eval_model
        self.threshold = eval_threshold
        self.goldens_data = load_goldens("e2e_goldens")
        # Unpack (callback, tracing_ctx) tuple from obs_test_trace
        obs_callback, tracing_ctx = obs_test_trace
        self.tracing_ctx = tracing_ctx
        self.obs_callback = obs_callback

    def test_task_completion(self):
        """Agent should complete the task successfully."""
        goldens = [Golden(input=g["input"]) for g in self.goldens_data[:4]]
        dataset = EvaluationDataset(goldens=goldens)
        metric = TaskCompletionMetric(
            threshold=self.threshold,
            model=self.eval_model,
        )

        for golden in dataset.evals_iterator(metrics=[metric]):
            traced_workflow(golden.input, self.config, obs_callback=self.obs_callback)

    def test_plan_quality(self):
        """Agent's plan should align well with the task."""
        goldens = [Golden(input=g["input"]) for g in self.goldens_data[:4]]
        dataset = EvaluationDataset(goldens=goldens)
        metric = PlanQualityMetric(
            threshold=self.threshold,
            model=self.eval_model,
        )

        for golden in dataset.evals_iterator(metrics=[metric]):
            traced_workflow(golden.input, self.config, obs_callback=self.obs_callback)

    def test_plan_adherence(self):
        """Agent should follow its plan during execution."""
        goldens = [Golden(input=g["input"]) for g in self.goldens_data[:4]]
        dataset = EvaluationDataset(goldens=goldens)
        metric = PlanAdherenceMetric(
            threshold=self.threshold,
            model=self.eval_model,
        )

        for golden in dataset.evals_iterator(metrics=[metric]):
            traced_workflow(golden.input, self.config, obs_callback=self.obs_callback)

    def test_step_efficiency(self):
        """Agent should execute steps efficiently without redundancy."""
        goldens = [Golden(input=g["input"]) for g in self.goldens_data[:4]]
        dataset = EvaluationDataset(goldens=goldens)
        metric = StepEfficiencyMetric(
            threshold=0.5,  # Lower threshold: efficiency is harder to score high
            model=self.eval_model,
        )

        for golden in dataset.evals_iterator(metrics=[metric]):
            traced_workflow(golden.input, self.config, obs_callback=self.obs_callback)


@pytest.mark.eval
class TestE2EAnswerQuality:
    """E2E evaluation using LLMTestCase-based metrics for answer quality.

    These tests run the workflow and evaluate the final answer directly.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, eval_config, eval_model, eval_threshold, obs_test_trace):
        self.config = eval_config
        self.eval_model = eval_model
        self.threshold = eval_threshold
        self.goldens_data = load_goldens("e2e_goldens")
        # Unpack (callback, tracing_ctx) tuple from obs_test_trace
        obs_callback, tracing_ctx = obs_test_trace
        self.tracing_ctx = tracing_ctx
        self.obs_callback = obs_callback

    @pytest.mark.parametrize("golden_idx", range(6))
    def test_answer_relevancy(self, golden_idx):
        """Final answer should be relevant to the user's question."""
        if golden_idx >= len(self.goldens_data):
            pytest.skip("Golden index out of range")

        golden = self.goldens_data[golden_idx]
        state = run_workflow_for_eval(golden["input"], self.config, obs_callback=self.obs_callback)
        final_answer = state.get("final_answer", "") or ""
        plan_context = [step["result"] for step in state.get("plan", []) if step.get("result")]

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=final_answer,
            retrieval_context=plan_context,
        )

        metric = AnswerRelevancyMetric(threshold=self.threshold, model=self.eval_model)
        assert_metrics_with_tracing(
            test_case,
            [metric],
            self.tracing_ctx,
            span_name="e2e_answer_relevancy_metric",
            metadata={
                "golden_idx": golden_idx,
            },
        )

    @pytest.mark.parametrize("golden_idx", range(6))
    def test_faithfulness(self, golden_idx):
        """Final answer should faithfully reflect execution results."""
        if golden_idx >= len(self.goldens_data):
            pytest.skip("Golden index out of range")

        golden = self.goldens_data[golden_idx]
        state = run_workflow_for_eval(golden["input"], self.config, obs_callback=self.obs_callback)
        final_answer = state.get("final_answer", "") or ""
        plan_context = [step["result"] for step in state.get("plan", []) if step.get("result")]

        test_case = LLMTestCase(
            input=golden["input"],
            actual_output=final_answer,
            retrieval_context=plan_context,
        )

        metric = FaithfulnessMetric(threshold=self.threshold, model=self.eval_model)
        assert_metrics_with_tracing(
            test_case,
            [metric],
            self.tracing_ctx,
            span_name="e2e_faithfulness_metric",
            metadata={
                "golden_idx": golden_idx,
            },
        )
