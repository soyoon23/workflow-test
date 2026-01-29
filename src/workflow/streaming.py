"""Streaming callback for bridging workflow nodes to UI display."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class StreamEventType(Enum):
    """Types of streaming events emitted by workflow nodes."""

    PHASE_START = "phase_start"
    PHASE_END = "phase_end"
    TOKEN = "token"
    PLAN_READY = "plan_ready"
    STEP_START = "step_start"
    STEP_RESULT = "step_result"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REVIEW_READY = "review_ready"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A single streaming event from the workflow."""

    event_type: StreamEventType
    node_name: str
    data: dict[str, Any] = field(default_factory=dict)


class StreamCallback:
    """Callback interface for streaming workflow events to the UI.

    Passed into WorkflowNodes so each node can emit fine-grained events.
    The UI registers a handler function that receives StreamEvent objects.
    """

    def __init__(self) -> None:
        self._handler: Optional[Callable[[StreamEvent], None]] = None

    @property
    def is_active(self) -> bool:
        return self._handler is not None

    def set_handler(self, handler: Callable[[StreamEvent], None]) -> None:
        self._handler = handler

    def emit(self, event: StreamEvent) -> None:
        if self._handler:
            self._handler(event)

    def phase_start(self, node_name: str, **data: Any) -> None:
        self.emit(StreamEvent(StreamEventType.PHASE_START, node_name, data))

    def phase_end(self, node_name: str, **data: Any) -> None:
        self.emit(StreamEvent(StreamEventType.PHASE_END, node_name, data))

    def token(self, node_name: str, text: str) -> None:
        self.emit(StreamEvent(StreamEventType.TOKEN, node_name, {"text": text}))

    def plan_ready(self, plan: list[dict], skill: Optional[str] = None) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.PLAN_READY,
                "plan",
                {"plan": plan, "selected_skill": skill},
            )
        )

    def step_start(self, step_number: int, description: str) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.STEP_START,
                "act",
                {"step_number": step_number, "description": description},
            )
        )

    def step_result(self, step_number: int, result: str) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.STEP_RESULT,
                "act",
                {"step_number": step_number, "result": result},
            )
        )

    def tool_call(self, tool_name: str, args: dict) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.TOOL_CALL,
                "act",
                {"tool_name": tool_name, "args": args},
            )
        )

    def tool_result(self, tool_name: str, result: Any) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.TOOL_RESULT,
                "act",
                {"tool_name": tool_name, "result": result},
            )
        )

    def review_ready(self, is_complete: bool, final_answer: Optional[str] = None) -> None:
        self.emit(
            StreamEvent(
                StreamEventType.REVIEW_READY,
                "review",
                {"is_complete": is_complete, "final_answer": final_answer},
            )
        )

    def error(self, node_name: str, message: str) -> None:
        self.emit(StreamEvent(StreamEventType.ERROR, node_name, {"message": message}))
