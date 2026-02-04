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


def assert_metrics_with_tracing(
    test_case: LLMTestCase,
    metrics: Sequence[BaseMetric],
    tracing_ctx: Optional[TracingContext],
    *,
    span_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Call deepeval.assert_test while emitting Langfuse spans for each metric run."""

    span = None
    metric_names = [_metric_label(metric) for metric in metrics]
    span_metadata = {"metric_names": metric_names}
    if metadata:
        span_metadata.update(metadata)

    if tracing_ctx:
        span = tracing_ctx.create_span(
            name=span_name or "deepeval_metric",
            metadata=span_metadata,
        )

    try:
        assert_test(test_case, list(metrics))
    finally:
        if tracing_ctx and span:
            metric_payload = []
            for metric in metrics:
                metric_payload.append(
                    {
                        "name": _metric_label(metric),
                        "score": getattr(metric, "score", None),
                        "threshold": getattr(metric, "threshold", None),
                        "success": _metric_success(metric),
                        "reason": getattr(metric, "reason", None),
                    }
                )

            trace_metadata = {"metrics": metric_payload}
            if metadata:
                trace_metadata.update(metadata)

            tracing_ctx.end_span(
                span,
                input=getattr(test_case, "input", None),
                output=getattr(test_case, "actual_output", None),
                metadata=trace_metadata,
            )
