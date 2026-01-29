"""Conversation history manager for multi-turn workflows.

Handles turn extraction from workflow state, sliding window management,
and history lifecycle (add, clear, trim).
"""

import logging
from typing import Optional

from .types import ConversationTurn

logger = logging.getLogger(__name__)


class ConversationHistoryManager:
    """Manages conversation turn history with sliding window.

    Maintains a bounded list of ConversationTurn summaries. Recent turns
    retain full plan results for detailed follow-up, while older turns
    are compacted to save tokens.

    Args:
        max_turns: Maximum number of turns to retain.
        detail_window: Number of most recent turns that keep full plan_results.
            Older turns have plan_results cleared to reduce token usage.
    """

    def __init__(self, max_turns: int = 20, detail_window: int = 1) -> None:
        self._history: list[ConversationTurn] = []
        self._max_turns = max_turns
        self._detail_window = detail_window
        self._turn_counter = 0

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def history(self) -> list[ConversationTurn]:
        """Current conversation history (read-only view)."""
        return list(self._history)

    def add_turn(self, state: dict) -> ConversationTurn:
        """Extract a ConversationTurn from completed workflow state and add to history.

        Automatically applies sliding window compaction and max_turns trimming.

        Args:
            state: Completed workflow state dict containing plan, final_answer, etc.

        Returns:
            The newly created ConversationTurn.
        """
        self._turn_counter += 1
        turn = self._extract_turn(state, self._turn_counter)

        # Compact older turns before adding the new one
        self._apply_sliding_window()

        self._history.append(turn)

        # Trim to max_turns
        if len(self._history) > self._max_turns:
            removed = len(self._history) - self._max_turns
            self._history = self._history[removed:]
            logger.debug("Trimmed %d oldest turns (max_turns=%d)", removed, self._max_turns)

        logger.info(
            "Added turn %d to history (total=%d): request=%s",
            turn["turn_number"],
            len(self._history),
            turn["user_request"][:80],
        )
        return turn

    def clear(self) -> None:
        """Clear all conversation history and reset turn counter."""
        count = len(self._history)
        self._history.clear()
        self._turn_counter = 0
        logger.info("Cleared conversation history (%d turns removed)", count)

    def _extract_turn(self, state: dict, turn_number: int) -> ConversationTurn:
        """Extract a ConversationTurn from a completed workflow state."""
        plan = state.get("plan", [])

        # Use reviewer-extracted key_facts if available, otherwise derive from answer
        key_facts = state.get("review_key_facts", [])
        if not key_facts:
            key_facts = self._derive_key_facts(state)

        return ConversationTurn(
            turn_number=turn_number,
            user_request=state.get("user_request", ""),
            final_answer=state.get("final_answer") or "",
            plan_summary=[step["description"] for step in plan],
            plan_results=[step.get("result") or "" for step in plan],
            key_facts=key_facts,
            skill_used=state.get("active_skill_name"),
        )

    def _apply_sliding_window(self) -> None:
        """Clear plan_results from turns outside the detail window.

        Only the most recent `detail_window` turns retain full plan_results.
        Older turns keep only the summary (descriptions) to save tokens.
        """
        window_start = max(0, len(self._history) - self._detail_window)
        for i in range(window_start):
            if self._history[i]["plan_results"]:
                logger.debug(
                    "Compacting turn %d plan_results (sliding window)",
                    self._history[i]["turn_number"],
                )
                self._history[i]["plan_results"] = []

    @staticmethod
    def _derive_key_facts(state: dict) -> list[str]:
        """Derive key facts from state when reviewer doesn't provide them.

        This is a fallback heuristic. Prefer reviewer-extracted key_facts.
        """
        facts: list[str] = []
        answer: Optional[str] = state.get("final_answer")
        if answer:
            truncated = answer[:300] + "..." if len(answer) > 300 else answer
            facts.append(truncated)
        return facts
