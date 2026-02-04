"""Observability integration for workflow tracing and monitoring.

Provides a provider-agnostic factory for creating observability callbacks.
Currently supports Langfuse (v3 SDK) via the ObservabilityProvider interface.
"""

import logging
from typing import Any, Optional

from .base import ObservabilityProvider, SpanHandle, TracingContext
from .registry import ObservabilityRegistry

__all__ = [
    "ObservabilityProvider",
    "ObservabilityRegistry",
    "TracingContext",
    "SpanHandle",
    "create_callback",
    "flush",
]

logger = logging.getLogger(__name__)

_default_registry = ObservabilityRegistry()
_active_provider: Optional[ObservabilityProvider] = None


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
    global _active_provider

    provider = _default_registry.resolve_provider(config)
    if provider is None:
        return None, None

    if not provider.configure(config):
        return None, None

    handler = provider.create_callback(trace_name=trace_name, metadata=metadata)
    if handler is None:
        return None, None

    _active_provider = provider
    context = provider.get_tracing_context()
    return handler, context


def flush() -> None:
    """Flush pending traces and close the active provider's propagation context.

    Safe to call even when no provider is active.
    """
    global _active_provider

    if _active_provider is not None:
        _active_provider.flush()
        _active_provider = None
