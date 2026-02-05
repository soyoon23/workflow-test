"""Langfuse observability provider (v3 SDK).

Uses the official ``langfuse.langchain.CallbackHandler`` for automatic
LangGraph tracing.  Custom configuration (trace name, metadata, session/user
IDs) is applied through ``propagate_attributes`` so that all spans created
during a workflow run are tagged consistently.
"""

import logging
import os
from typing import Any, Optional

from ..base import ObservabilityProvider, SpanHandle, TracingContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TracingContext — manual span creation via Langfuse client
# ---------------------------------------------------------------------------


class LangfuseTracingContext(TracingContext):
    """TracingContext for creating child spans under the current trace.

    In Langfuse v3, spans created via start_span() automatically attach
    to the current trace/span context set by start_as_current_span().
    """

    def create_span(
        self,
        name: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpanHandle:
        from langfuse import get_client

        # In v3, start_span() auto-attaches to current trace context
        span = get_client().start_span(name=name, metadata=metadata or {})
        return SpanHandle(raw=span)

    def end_span(
        self,
        span: SpanHandle,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        raw = span._raw
        if input is not None:
            raw.update(input=input)
        if output is not None:
            raw.update(output=output)
        if metadata is not None:
            raw.update(metadata=metadata)
        raw.end()

    def update_parent_span(
        self,
        *,
        input: Any = None,
        output: Any = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update the parent span's input/output/metadata."""
        from langfuse import get_client

        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata

        if kwargs:
            get_client().update_current_span(**kwargs)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LangfuseProvider(ObservabilityProvider):
    """Langfuse observability provider using the official v3 CallbackHandler.

    Lifecycle (per workflow run)::

        provider.configure(config)          # set env-var credentials
        handler = provider.create_callback(  # creates parent span + handler
            trace_name="workflow_20240101",
            metadata={...},
            session_id="session-abc",
            user_id="user-xyz",
            tags=["eval"],
        )
        stream_workflow(..., callbacks=[handler])
        provider.flush()                    # flush + end parent span
    """

    def __init__(self):
        self._current_handler: Any = None
        self._span_context: Any = None  # Context manager from start_as_current_span
        self._current_span: Any = None  # The actual span object

    # -- identity / availability -------------------------------------------

    @property
    def name(self) -> str:
        return "langfuse"

    def is_available(self) -> bool:
        try:
            import langfuse  # noqa: F401

            return True
        except ImportError:
            return False

    # -- configuration -----------------------------------------------------

    def configure(self, config: dict) -> bool:
        """Resolve Langfuse credentials from *config* / env-vars.

        Looks up keys from ``config['langfuse']`` and
        ``config['observability']['langfuse']`` (the latter wins on conflict).
        Environment variables (``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``,
        ``LANGFUSE_HOST``) always take precedence.

        Returns ``True`` when valid credentials were found.
        """
        langfuse_cfg = config.get("langfuse", {})
        obs_cfg = config.get("observability", {})
        merged = {**langfuse_cfg, **obs_cfg.get("langfuse", {})}

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or merged.get("public_key")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") or merged.get("secret_key")
        host = os.getenv("LANGFUSE_HOST") or merged.get("host", "http://localhost:3000")

        if not public_key or not secret_key:
            logger.warning("Langfuse credentials not found — tracing disabled")
            return False

        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = host
        return True

    # -- callback creation -------------------------------------------------

    def create_callback(
        self,
        trace_name: str = "workflow",
        metadata: Optional[dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> Any:
        """Create the official v3 ``CallbackHandler`` with a parent span.

        Uses start_as_current_span() to create a parent span. All LLM calls
        via CallbackHandler and manual spans via get_tracing_context() will
        automatically attach as children of this span.
        """
        if not self.is_available():
            return None

        from langfuse import get_client
        from langfuse.langchain import CallbackHandler

        client = get_client()

        # ---- build metadata including session/user/tags ----
        span_metadata: dict[str, Any] = metadata.copy() if metadata else {}
        if session_id:
            span_metadata["session_id"] = session_id
        if user_id:
            span_metadata["user_id"] = user_id
        if tags:
            span_metadata["tags"] = tags

        # ---- create parent span that will contain all children ----
        self._span_context = client.start_as_current_span(
            name=trace_name,
            metadata=span_metadata if span_metadata else None,
        )
        self._current_span = self._span_context.__enter__()

        # ---- create handler (auto-attaches to current span context) ----
        handler = CallbackHandler()
        self._current_handler = handler

        logger.info(
            "Langfuse tracing started — trace_name=%s",
            trace_name,
        )
        return handler

    # -- tracing context (manual spans) ------------------------------------

    def get_tracing_context(self) -> Optional[LangfuseTracingContext]:
        """Return a ``TracingContext`` for manual span creation.

        Must be called *after* :meth:`create_callback`.
        Returns a context that creates spans as children of the current span.
        """
        if self._current_span is not None:
            return LangfuseTracingContext()
        return None

    # -- flush / teardown --------------------------------------------------

    def flush(self) -> None:
        """Flush pending traces and end the parent span.

        Safe to call multiple times or when no handler is active.
        """
        try:
            # End the parent span context
            if self._span_context is not None:
                self._span_context.__exit__(None, None, None)
        except Exception:
            logger.debug("Langfuse span exit failed", exc_info=True)

        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)
        finally:
            self._current_handler = None
            self._span_context = None
            self._current_span = None
