"""Observability integration for workflow tracing and monitoring.

Provides a provider-agnostic factory for creating observability callbacks.
Supports multiple backends (Langfuse, OpenTelemetry, MLflow, etc.) via
the ObservabilityProvider interface and ObservabilityRegistry.
"""

from typing import Any, Optional

from .base import ObservabilityProvider, SpanHandle, TracingContext
from .registry import ObservabilityRegistry

__all__ = [
    "ObservabilityProvider",
    "ObservabilityRegistry",
    "TracingContext",
    "SpanHandle",
    "create_callback",
]

_default_registry = ObservabilityRegistry()


def create_callback(
    config: dict,
    trace_name: str = "workflow",
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[Any, Optional[TracingContext]]:
    """Create an observability callback handler from config.

    Resolves the active provider from config and creates a callback handler
    along with a TracingContext for manual span creation.

    Args:
        config: Project config dict with observability/langfuse sections.
        trace_name: Name for the trace.
        metadata: Additional metadata to attach.

    Returns:
        (callback_handler, tracing_context) tuple.
        Both are None if no provider is enabled/available.
    """
    provider = _default_registry.resolve_provider(config)
    if provider is None:
        return None, None

    if not provider.configure(config):
        return None, None

    handler = provider.create_callback(trace_name=trace_name, metadata=metadata)
    if handler is None:
        return None, None

    context = provider.get_tracing_context()
    return handler, context
