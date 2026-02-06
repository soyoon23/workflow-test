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
        
        if tracing_ctx and metric:
            details=_extract_metric_data_details(metric)
            metric_span=tracing_ctx.create_span(
                name=f"eval:{details['name']}",
                metadata={"metric_type": type(metric).__name__}
            )

    
    # Record to Langfuse if tracing context is available
    if tracing_ctx and test_result.metrics_data:
        metric_summaries = []

        # Create a dedicated span per metric
        for metric_data in test_result.metrics_data:
            details = _extract_metric_data_details(metric_data)
            metric_summaries.append(details)

            span_meta = {**details}
            # Include test case actual_output in metadata for full context
            if test_actual_output:
                span_meta["actual_output"] = test_actual_output
            if metadata:
                span_meta.update(metadata)

            metric_span = tracing_ctx.create_span(
                name=f"eval:{details['name']}",
                metadata={"metric_type": type(metric_data).__name__},
            )
            tracing_ctx.end_span(
                metric_span,
                input=test_input,
                output=_format_metric_output(details),
                metadata=span_meta,
            )

        # Summary span for grouping
        summary_span = tracing_ctx.create_span(
            name=span_name or "deepeval_metrics",
            metadata={"metric_names": [d["name"] for d in metric_summaries]},
        )

        summary_lines = []
        for d in metric_summaries:
            status = "PASS" if d.get("success") else "FAIL"
            score = d.get("score")
            score_str = f"{score}" if score is not None else "N/A"
            summary_lines.append(f"[{status}] {d['name']}: {score_str}")

        summary_meta: dict[str, Any] = {"metrics": metric_summaries}
        # Include test case actual_output in summary metadata
        if test_actual_output:
            summary_meta["actual_output"] = test_actual_output
        if metadata:
            summary_meta.update(metadata)

        tracing_ctx.end_span(
            summary_span,
            input=test_input,
            output="\n".join(summary_lines),
            metadata=summary_meta,
        )

    # Raise AssertionError if any metric failed (mimic assert_test behavior)
    if not test_result.success:
        failed_metrics = [
            m for m in test_result.metrics_data if not m.success or m.error
        ]
        failed_str = ", ".join(
            f"{m.name} (score: {m.score}, threshold: {m.threshold}, reason: {m.reason}, error: {m.error})"
            for m in failed_metrics
        )
        raise AssertionError(f"Metrics failed: {failed_str}")
