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
    """TracingContext backed by the global Langfuse v3 client.

    Spans created here are attached to the currently-active trace (set by
    ``propagate_attributes``).  If no active trace exists the spans become
    top-level traces — still visible in the Langfuse UI but not nested.
    """

    def create_span(
        self,
        name: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpanHandle:
        from langfuse import get_client

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
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if kwargs:
            raw.update(**kwargs)
        raw.end()


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class LangfuseProvider(ObservabilityProvider):
    """Langfuse observability provider using the official v3 CallbackHandler.

    Lifecycle (per workflow run)::

        provider.configure(config)          # set env-var credentials
        handler = provider.create_callback(  # opens propagate_attributes ctx
            trace_name="workflow_20240101",
            metadata={...},
            session_id="session-abc",
            user_id="user-xyz",
            tags=["eval"],
        )
        stream_workflow(..., callbacks=[handler])
        provider.flush()                    # flush + close propagate ctx
    """

    def __init__(self):
        self._current_handler: Any = None
        self._attr_context: Any = None

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
        """Create the official v3 ``CallbackHandler``.

        Custom attributes (trace name, metadata, session/user IDs, tags) are
        propagated via ``langfuse.propagate_attributes`` so that **every** span
        the handler creates is automatically tagged.

        The propagation context is kept open until :meth:`flush` is called.
        """
        if not self.is_available():
            return None

        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        # ---- propagate custom attributes to all child spans ----
        propagate_kwargs: dict[str, Any] = {"trace_name": trace_name}
        if metadata:
            propagate_kwargs["metadata"] = metadata
        if session_id:
            propagate_kwargs["session_id"] = session_id
        if user_id:
            propagate_kwargs["user_id"] = user_id
        if tags:
            propagate_kwargs["tags"] = tags

        self._attr_context = propagate_attributes(**propagate_kwargs)
        self._attr_context.__enter__()

        # ---- create the official handler ----
        handler = CallbackHandler()
        self._current_handler = handler

        logger.info(
            "Langfuse tracing started — trace_name=%s, session=%s, user=%s",
            trace_name,
            session_id,
            user_id,
        )
        return handler

    # -- tracing context (manual spans) ------------------------------------

    def get_tracing_context(self) -> Optional[LangfuseTracingContext]:
        """Return a ``TracingContext`` for manual span creation.

        Must be called *after* :meth:`create_callback`.
        """
        if self._current_handler is not None:
            return LangfuseTracingContext()
        return None

    # -- flush / teardown --------------------------------------------------

    def flush(self) -> None:
        """Flush pending traces and close the propagation context.

        Safe to call multiple times or when no handler is active.
        """
        try:
            from langfuse import get_client

            get_client().flush()
        except Exception:
            logger.debug("Langfuse flush failed", exc_info=True)
        finally:
            if self._attr_context:
                try:
                    self._attr_context.__exit__(None, None, None)
                except Exception:
                    pass
                self._attr_context = None
            self._current_handler = None
