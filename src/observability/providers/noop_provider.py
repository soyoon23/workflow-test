"""No-op observability provider for when tracing is disabled."""

from typing import Any, Optional

from langchain_core.callbacks import BaseCallbackHandler

from ..base import ObservabilityProvider, SpanHandle, TracingContext


class NoopTracingContext(TracingContext):
    """TracingContext that silently discards all operations."""

    def create_span(self, name: str, *, metadata: Optional[dict[str, Any]] = None) -> SpanHandle:
        return SpanHandle(raw=None)

    def end_span(
        self,
        span: SpanHandle,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        pass


class NoopCallbackHandler(BaseCallbackHandler):
    """Callback handler that silently discards all events."""

    pass


class NoopProvider(ObservabilityProvider):
    """Provider that discards all tracing data. Always available."""

    @property
    def name(self) -> str:
        return "noop"

    def is_available(self) -> bool:
        return True

    def create_callback(
        self,
        trace_name: str = "workflow",
        metadata: Optional[dict[str, Any]] = None,
    ) -> BaseCallbackHandler:
        return NoopCallbackHandler()

    def get_tracing_context(self) -> TracingContext:
        return NoopTracingContext()

    def flush(self) -> None:
        pass
