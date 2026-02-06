"""Utilities for piping DeepEval metric runs into observability traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from src.observability.base import TracingContext


def _extract_base_metric_details(metric: BaseMetric) -> dict[str, Any]:
    """Extract evaluation details from a BaseMetric instance after measure() is called.

    Accesses metric attributes directly after evaluation, including score,
    threshold, success status, and optional reason/error information.
    """
    metric_name = getattr(metric, "__name__", None) or metric.__class__.__name__

    details: dict[str, Any] = {
        "name": metric_name,
        "score": getattr(metric, "score", None),
        "threshold": getattr(metric, "threshold", None),
        "success": getattr(metric, "success", None),
        "reason": getattr(metric, "reason", None),
        "evaluation_model": getattr(metric, "evaluation_model", None),
        "error": getattr(metric, "error", None),
    }

    verbose_logs = getattr(metric, "verbose_logs", None)
    if verbose_logs:
        details["verbose_logs"] = verbose_logs

    evaluation_cost = getattr(metric, "evaluation_cost", None)
    if evaluation_cost is not None:
        details["evaluation_cost"] = evaluation_cost

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


def _format_summary_output(results: list[dict[str, Any]]) -> str:
    """Format a summary of all metric results into a human-readable string."""
    if not results:
        return "No metrics evaluated."

    lines = [f"Evaluated {len(results)} metric(s):"]
    for result in results:
        name = result.get("name", "Unknown")
        score = result.get("score")
        success = result.get("success")
        status = "PASS" if success else "FAIL"
        score_str = f"{score:.2f}" if score is not None else "N/A"
        lines.append(f"  - {name}: {score_str} ({status})")

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

    Evaluates each metric using measure() and records results in Langfuse.

    Span Structure:
        [Summary Span: span_name or "eval:metrics"]
          |-- [eval:MetricName1] - individual metric result
          |-- [eval:MetricName2] - individual metric result

    Args:
        test_case: The DeepEval test case to evaluate.
        metrics: Sequence of BaseMetric instances to run.
        tracing_ctx: Optional TracingContext for Langfuse span creation.
        span_name: Name for the summary span grouping all metrics.
                   Defaults to "eval:metrics" if not provided.
        metadata: Additional metadata to include in all spans (e.g., golden_idx).
                  Merged with metric-specific metadata.
    """
    test_input = getattr(test_case, "input", None)
    base_metadata = metadata or {}

    all_results: list[dict[str, Any]] = []
    all_passed = True

    # Create summary span
    summary_span = None
    if tracing_ctx:
        summary_span = tracing_ctx.create_span(
            name=span_name or "eval:metrics",
            metadata={**base_metadata, "metric_count": len(metrics)},
        )

    for metric in metrics:
        metric.measure(test_case=test_case)
        details = _extract_base_metric_details(metric)
        all_results.append(details)

        if not details.get("success", False):
            all_passed = False

        # Create per-metric span
        if tracing_ctx:
            metric_span = tracing_ctx.create_span(
                name=f"eval:{details['name']}",
                metadata={**base_metadata, "metric_type": type(metric).__name__},
            )
            tracing_ctx.end_span(
                metric_span,
                input=test_input,
                output=_format_metric_output(details),
                metadata={**base_metadata, **details},
            )

    # End summary span with aggregated results
    if tracing_ctx and summary_span:
        tracing_ctx.end_span(
            summary_span,
            input=test_input,
            output=_format_summary_output(all_results),
            metadata={
                **base_metadata,
                "all_passed": all_passed,
                "metrics_evaluated": [r["name"] for r in all_results],
                "scores": {r["name"]: r.get("score") for r in all_results},
            },
        )
