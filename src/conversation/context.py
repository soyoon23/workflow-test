"""Conversation context builder for multi-turn workflows.

Builds role-specific context strings from conversation history.
Each role (planner, actor, reviewer) receives different levels of detail
to optimize token usage and reduce noise.

Role context levels:
    - planner: Full detail (request, answer, recent plan structure, skill)
    - actor: Minimal (key_facts only)
    - reviewer: Medium (request, answer, key_facts)
"""

import logging
from abc import ABC, abstractmethod

from .types import ConversationTurn

logger = logging.getLogger(__name__)


class RoleContextFormatter(ABC):
    """Base class for role-specific context formatting.

    Subclass this to define how conversation turns are rendered
    for a specific workflow role.
    """

    @property
    @abstractmethod
    def role_name(self) -> str:
        """Role identifier used for logging and registration."""

    @abstractmethod
    def format_turn(self, turn: ConversationTurn, *, is_most_recent: bool) -> list[str]:
        """Format a single conversation turn into context lines.

        Args:
            turn: The conversation turn to format.
            is_most_recent: Whether this is the most recent turn in history.
                Most recent turns may include additional detail (e.g., plan structure).

        Returns:
            List of formatted text lines for this turn.
        """


class PlannerContextFormatter(RoleContextFormatter):
    """Planner sees the most detail for informed plan creation.

    Includes: request, answer, plan structure (recent only), skill used.
    """

    @property
    def role_name(self) -> str:
        return "planner"

    def format_turn(self, turn: ConversationTurn, *, is_most_recent: bool) -> list[str]:
        lines = [
            f"User: {turn['user_request']}",
            f"Answer: {turn['final_answer']}",
        ]
        if is_most_recent and turn.get("plan_summary"):
            lines.append("Plan used:")
            for i, desc in enumerate(turn["plan_summary"], 1):
                lines.append(f"  {i}. {desc}")
                # Include step result for most recent turn's plan detail
                results = turn.get("plan_results", [])
                if i <= len(results) and results[i - 1]:
                    result_text = results[i - 1]
                    if len(result_text) > 200:
                        result_text = result_text[:200] + "..."
                    lines.append(f"     Result: {result_text}")
        if turn.get("skill_used"):
            lines.append(f"Skill used: {turn['skill_used']}")
        return lines


class ActorContextFormatter(RoleContextFormatter):
    """Actor sees minimal context to avoid noise during step execution.

    Includes: key_facts only (numbers, entities for reference).
    """

    @property
    def role_name(self) -> str:
        return "actor"

    def format_turn(self, turn: ConversationTurn, *, is_most_recent: bool) -> list[str]:
        lines = []
        if turn.get("key_facts"):
            lines.append("Key facts: " + "; ".join(turn["key_facts"]))
        else:
            # Fallback: truncated answer so actor has some reference
            answer = turn.get("final_answer", "")
            if answer:
                if len(answer) > 150:
                    answer = answer[:150] + "..."
                lines.append(f"Answer: {answer}")
        return lines


class ReviewerContextFormatter(RoleContextFormatter):
    """Reviewer sees medium detail for consistency judgment.

    Includes: request, answer, key_facts.
    """

    @property
    def role_name(self) -> str:
        return "reviewer"

    def format_turn(self, turn: ConversationTurn, *, is_most_recent: bool) -> list[str]:
        lines = [
            f"User: {turn['user_request']}",
            f"Answer: {turn['final_answer']}",
        ]
        if turn.get("key_facts"):
            lines.append("Key facts: " + "; ".join(turn["key_facts"]))
        return lines


# Default formatters for each role
_DEFAULT_FORMATTERS: dict[str, type[RoleContextFormatter]] = {
    "planner": PlannerContextFormatter,
    "actor": ActorContextFormatter,
    "reviewer": ReviewerContextFormatter,
}


class ConversationContextBuilder:
    """Builds role-specific conversation context strings from history.

    Uses registered RoleContextFormatter instances to produce different
    context detail levels per role.

    Usage:
        builder = ConversationContextBuilder()
        context = builder.build(history, role="planner")
        system_prompt = f"{base_prompt}\\n\\n{context}"
    """

    def __init__(self) -> None:
        self._formatters: dict[str, RoleContextFormatter] = {}
        for role, formatter_cls in _DEFAULT_FORMATTERS.items():
            self._formatters[role] = formatter_cls()

    def register_formatter(self, formatter: RoleContextFormatter) -> None:
        """Register a custom formatter for a role.

        Use this to override default behavior or add formatters for new roles.
        """
        self._formatters[formatter.role_name] = formatter
        logger.debug("Registered context formatter for role=%s", formatter.role_name)

    def build(self, history: list[ConversationTurn], role: str) -> str:
        """Build conversation context string for the given role.

        Args:
            history: List of previous conversation turns.
            role: Workflow role ("planner", "actor", "reviewer").

        Returns:
            Formatted context string, or empty string if no history.
        """
        if not history:
            return ""

        formatter = self._formatters.get(role)
        if formatter is None:
            logger.warning("No context formatter for role=%s, skipping context", role)
            return ""

        lines = ["## Previous Conversation"]

        for i, turn in enumerate(history):
            is_most_recent = i == len(history) - 1
            lines.append(f"\n### Turn {turn['turn_number']}")
            turn_lines = formatter.format_turn(turn, is_most_recent=is_most_recent)
            lines.extend(turn_lines)

        context = "\n".join(lines)
        logger.debug(
            "Built conversation context for role=%s: %d turns, %d chars",
            role,
            len(history),
            len(context),
        )
        return context
