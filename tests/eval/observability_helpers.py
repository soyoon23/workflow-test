"""Utilities for piping DeepEval metric runs into observability traces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional

from deepeval import assert_test
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from src.observability.base import TracingContext


def _metric_label(metric: BaseMetric) -> str:
    return getattr(metric, "name", None) or getattr(metric, "__name__", metric.__class__.__name__)


def _metric_success(metric: BaseMetric) -> Optional[bool]:
    success_attr = getattr(metric, "success", None)
    if isinstance(success_attr, bool):
        return success_attr
    is_successful = getattr(metric, "is_successful", None)
    if callable(is_successful):
        try:
            return bool(is_successful())
        except Exception:  # pragma: no cover - defensive
            return None
    return None


def _extract_metric_details(metric: BaseMetric) -> dict[str, Any]:
    """Extract all available evaluation details from a metric."""
    details: dict[str, Any] = {
        "name": _metric_label(metric),
        "score": getattr(metric, "score", None),
        "threshold": getattr(metric, "threshold", None),
        "success": _metric_success(metric),
        "reason": getattr(metric, "reason", None),
    }

    # GEval-specific fields
    evaluation_steps = getattr(metric, "evaluation_steps", None)
    if evaluation_steps:
        details["evaluation_steps"] = evaluation_steps

    evaluation_model = getattr(metric, "evaluation_model", None) or getattr(metric, "model", None)
    if evaluation_model:
        details["evaluation_model"] = str(evaluation_model)

    criteria = getattr(metric, "criteria", None)
    if criteria:
        details["criteria"] = criteria

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

    evaluation_steps = details.get("evaluation_steps")
    if evaluation_steps:
        lines.append("\nEvaluation Steps:")
        for i, step in enumerate(evaluation_steps, 1):
            lines.append(f"  {i}. {step}")

    return "\n".join(lines)


def assert_metrics_with_tracing(
    test_case: LLMTestCase,
    metrics: Sequence[BaseMetric],
    tracing_ctx: Optional[TracingContext],
    *,
    span_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Call deepeval.assert_test while emitting per-metric Langfuse spans.

    For each metric, a dedicated span is created containing:
    - input: the test case input
    - output: human-readable summary (score, threshold, pass/fail, reason)
    - metadata: structured evaluation details (score, reason, criteria, etc.)

    A summary span groups all metric results for quick overview.
    """

    try:
        assert_test(test_case, list(metrics))
    finally:
        if not tracing_ctx:
            return

        test_input = getattr(test_case, "input", None)
        metric_summaries = []

        # Create a dedicated span per metric
        for metric in metrics:
            details = _extract_metric_details(metric)
            metric_summaries.append(details)

            span_meta = {**details}
            if metadata:
                span_meta.update(metadata)

            metric_span = tracing_ctx.create_span(
                name=f"eval:{details['name']}",
                metadata={"metric_type": metric.__class__.__name__},
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
        if metadata:
            summary_meta.update(metadata)

        tracing_ctx.end_span(
            summary_span,
            input=test_input,
            output="\n".join(summary_lines),
            metadata=summary_meta,
        )
