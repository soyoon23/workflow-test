"""Utilities for piping DeepEval metric runs into observability traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_run import MetricData

from src.observability.base import TracingContext


def _extract_metric_data_details(metric_data: MetricData) -> dict[str, Any]:
    """Extract evaluation details from MetricData object returned by evaluate()."""
    details: dict[str, Any] = {
        "name": metric_data.name,
        "score": metric_data.score,
        "threshold": metric_data.threshold,
        "success": metric_data.success,
        "reason": metric_data.reason,
        "evaluation_model": metric_data.evaluation_model,
        "error": metric_data.error,
    }

    # Include verbose logs if available
    if metric_data.verbose_logs:
        details["verbose_logs"] = metric_data.verbose_logs

    return details


def _format_metric_output(details: dict[str, Any]) -> str:
    """Format metric evaluation results into a human-readable string."""
    lines = []

    score = details.get("score")
    threshold = details.get("threshold")
    success = details.get("success")

    if score is not None:
        lines.append(f"Score: {score}")
    if threshold is not None:
        lines.append(f"Threshold: {threshold}")
    if success is not None:
        lines.append(f"Pass: {'YES' if success else 'NO'}")

    reason = details.get("reason")
    if reason:
        lines.append(f"\nReason:\n{reason}")

    error = details.get("error")
    if error:
        lines.append(f"\nError:\n{error}")

    return "\n".join(lines)


def assert_metrics_with_tracing(
    test_case: LLMTestCase,
    metrics: Sequence[BaseMetric],
    tracing_ctx: Optional[TracingContext],
    *,
    span_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Run DeepEval evaluation and emit per-metric Langfuse spans.

    Uses evaluate() instead of assert_test() to get structured results
    (score, reason, etc.) that can be recorded in Langfuse.

    For each metric, a dedicated span is created containing:
    - input: the test case input
    - output: human-readable summary (score, threshold, pass/fail, reason)
    - metadata: structured evaluation details (score, reason, criteria, etc.)

    A summary span groups all metric results for quick overview.
    """

    for metric in metrics:
        metric.measure(test_case=test_case)
        test_input = getattr(test_case, "input", None)
        
        if tracing_ctx and metric:
            details=_extract_metric_data_details(metric)
            metric_span=tracing_ctx.create_span(
                name=f"eval:{details['name']}",
                metadata={"metric_type": type(metric).__name__}
            )

            tracing_ctx.end_span(
                metric_span,
                input=test_input,
                output=_format_metric_output(details),
                metadata={**details}
            )
