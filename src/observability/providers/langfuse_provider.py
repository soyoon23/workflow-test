"""Langfuse observability provider.

Integrates with LangGraph's callback system to automatically trace all node
executions, LLM calls, and tool invocations to Langfuse.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from ..base import ObservabilityProvider, SpanHandle, TracingContext

logger = logging.getLogger(__name__)


class LangfuseTracingContext(TracingContext):
    """TracingContext backed by a Langfuse trace object."""

    def __init__(self, trace: Any):
        self._trace = trace

    def create_span(
        self,
        name: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SpanHandle:
        span = self._trace.span(name=name, metadata=metadata or {})
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
        raw.end(**kwargs)


class LangfuseCallbackHandler(BaseCallbackHandler):
    """Langfuse callback handler for LangGraph workflows.

    Automatically captures:
    - Node executions (plan, act, review)
    - LLM calls with prompts and responses
    - Tool invocations
    - Workflow metadata
    """

    def __init__(self, trace_name: str = "workflow", metadata: Optional[Dict] = None):
        from langfuse import Langfuse

        self.langfuse = Langfuse()
        self.trace = self.langfuse.trace(name=trace_name, metadata=metadata or {})
        self.current_span = None
        self.node_spans: Dict[str, Any] = {}

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        run_id = kwargs.get("run_id")

        tags = kwargs.get("tags", [])
        node_name = next((tag for tag in tags if tag.startswith("seq:")), None)
        if node_name:
            node_name = node_name.replace("seq:", "").strip()
        else:
            node_name = serialized.get("name", "unknown_node")

        if "plan" in node_name.lower():
            span_name = "plan_node"
            metadata = {"role": "planner"}
        elif "act" in node_name.lower():
            span_name = "act_node"
            metadata = {"role": "actor"}
        elif "review" in node_name.lower():
            span_name = "review_node"
            metadata = {"role": "reviewer"}
        else:
            span_name = node_name
            metadata = {}

        self.node_spans[str(run_id)] = {
            "span_name": span_name,
            "metadata": metadata,
            "input": inputs,
        }

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        span_info = self.node_spans.get(str(run_id))

        if span_info:
            span = self.trace.span(
                name=span_info["span_name"],
                metadata=span_info["metadata"],
            )
            span.end(
                input=span_info["input"],
                output=outputs,
            )
            del self.node_spans[str(run_id)]

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")

        self.node_spans[str(run_id)] = {
            "type": "llm",
            "prompts": prompts,
            "model": serialized.get("name", "unknown"),
        }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        llm_info = self.node_spans.get(str(run_id))

        if llm_info and llm_info.get("type") == "llm":
            outputs = []
            for generations in response.generations:
                for gen in generations:
                    outputs.append(gen.text)

            generation = self.trace.generation(
                name="llm_call",
                model=llm_info["model"],
                input=llm_info["prompts"],
                output=outputs[0] if outputs else "",
                metadata={
                    "token_usage": response.llm_output.get("token_usage")
                    if response.llm_output
                    else None
                },
            )
            generation.end()
            del self.node_spans[str(run_id)]

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")

        self.node_spans[str(run_id)] = {
            "type": "tool",
            "name": serialized.get("name", "unknown"),
            "input": input_str,
        }

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        run_id = kwargs.get("run_id")
        tool_info = self.node_spans.get(str(run_id))

        if tool_info and tool_info.get("type") == "tool":
            span = self.trace.span(
                name=f"tool_{tool_info['name']}",
                metadata={"tool_type": "tool"},
            )
            span.end(
                input=tool_info["input"],
                output=output,
            )
            del self.node_spans[str(run_id)]

    def flush(self):
        self.langfuse.flush()


class LangfuseProvider(ObservabilityProvider):
    """Langfuse observability provider."""

    def __init__(self):
        self._current_handler: Optional[LangfuseCallbackHandler] = None
        self._current_context: Optional[LangfuseTracingContext] = None

    @property
    def name(self) -> str:
        return "langfuse"

    def is_available(self) -> bool:
        try:
            import langfuse  # noqa: F401

            return True
        except ImportError:
            return False

    def configure(self, config: dict) -> bool:
        """Set environment variables from config. Returns True if credentials found."""
        langfuse_cfg = config.get("langfuse", {})
        obs_cfg = config.get("observability", {})
        merged = {**langfuse_cfg, **obs_cfg.get("langfuse", {})}

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or merged.get("public_key")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY") or merged.get("secret_key")
        host = os.getenv("LANGFUSE_HOST") or merged.get("host", "http://localhost:3000")

        if not public_key or not secret_key:
            logger.warning("Langfuse credentials not found, skipping tracing")
            return False

        os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
        os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        os.environ["LANGFUSE_HOST"] = host
        return True

    def create_callback(
        self,
        trace_name: str = "workflow",
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[LangfuseCallbackHandler]:
        if not self.is_available():
            return None

        handler = LangfuseCallbackHandler(trace_name=trace_name, metadata=metadata)
        self._current_handler = handler
        self._current_context = LangfuseTracingContext(handler.trace)
        return handler

    def get_tracing_context(self) -> Optional[LangfuseTracingContext]:
        return self._current_context

    def flush(self) -> None:
        if self._current_handler:
            self._current_handler.flush()
