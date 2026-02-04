"""Base abstractions for observability providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class SpanHandle:
    """Opaque handle to a provider-specific span.

    Providers store their internal span reference internally.
    Consumers should use TracingContext.end_span() instead of accessing internals.
    """

    def __init__(self, raw: Any = None):
        self._raw = raw


class TracingContext(ABC):
    """Provider-agnostic interface for creating manual spans.

    Abstracts away provider-specific span APIs (e.g. Langfuse's trace.span())
    so that test code can create manual spans without knowing which provider is active.
    """

    @abstractmethod
    def create_span(
        self,
        name: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpanHandle:
        """Create a new child span under the current trace."""

    @abstractmethod
    def end_span(
        self,
        span: SpanHandle,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """End a span with input/output data."""


class ObservabilityProvider(ABC):
    """Abstract base for observability providers.

    Each provider must:
    1. Create a BaseCallbackHandler for LangGraph integration
    2. Expose a TracingContext for manual span creation
    3. Support flush() for async-sending providers
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'langfuse', 'otel', 'mlflow')."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider SDK is installed and credentials are configured."""

    @abstractmethod
    def create_callback(
        self,
        trace_name: str = "workflow",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Create a LangChain callback handler for LangGraph.

        Returns a BaseCallbackHandler instance, or None if not available.
        """

    @abstractmethod
    def get_tracing_context(self) -> Optional[TracingContext]:
        """Get the tracing context for manual span creation.

        Must be called AFTER create_callback(). Returns the TracingContext
        associated with the most recently created callback handler.
        """

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending trace data to the backend."""

    def configure(self, config: dict) -> bool:
        """Configure the provider from config dict.

        Returns True if configuration succeeded (credentials found, etc.).
        Default implementation returns True (no config needed).
        """
        return True
