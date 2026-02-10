"""Base protocol and mixin for workflow nodes.

BaseNode defines the structural interface that all nodes must satisfy.
NodeMixin provides shared helper methods for prompt/skill/tool resolution.
"""

import logging
from typing import Any, Optional, Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

from ...skills.registry import Skill
from ..components import WorkflowComponents
from ..state import AgentState

logger = logging.getLogger(__name__)


@runtime_checkable
class BaseNode(Protocol):
    """Protocol that all workflow nodes must satisfy.

    Any callable with signature (AgentState, RunnableConfig) -> dict[str, Any]
    satisfies this protocol. LangGraph automatically passes the RunnableConfig
    (containing callback handlers, etc.) to each node.

    Example:
        class MyCustomNode:
            def __call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
                ...

        node: BaseNode = MyCustomNode()  # type-checks OK
    """

    def __call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]: ...


class NodeMixin:
    """Shared helper methods for workflow nodes.

    Provides prompt resolution, skill lookup, and tool filtering
    that are common across Plan, Act, and Review nodes.

    Subclasses must set ``self.components`` to a ``WorkflowComponents``
    instance (typically in ``__init__``).
    """

    components: WorkflowComponents

    def _get_skill(self, state: AgentState) -> Optional[Skill]:
        """Get active skill from state or initial skill."""
        skill_name = state.get("active_skill_name")
        if skill_name:
            return self.components.skills.get(skill_name)
        return self.components.initial_skill

    def _get_prompt(self, role: str, state: AgentState) -> str:
        """Get prompt for a role, considering active skill and conversation context."""
        c = self.components
        skill = self._get_skill(state)

        prompt = c.prompts.get(role)

        if role == "planner":
            skill_selection_block = self._build_skill_selection_block(state)
            prompt = prompt.replace("{skill_selection_block}", skill_selection_block)

        if skill and skill.prompt:
            prompt = f"{prompt}\n\n## Skill Context\n{skill.prompt}"

        history = state.get("conversation_history", [])
        if history:
            conv_context = c.context_builder.build(history, role)
            if conv_context:
                prompt = f"{prompt}\n\n{conv_context}"
                logger.debug(
                    "Injected conversation context into %s prompt (%d chars)",
                    role,
                    len(conv_context),
                )

        return prompt

    def _build_skill_selection_block(self, state: AgentState) -> str:
        """Build skill selection instruction block for planner prompt.

        Returns instruction text when auto_select_skill is True and no skill
        is currently active. Returns empty string otherwise.
        """
        if not state.get("auto_select_skill"):
            return ""

        if state.get("active_skill_name"):
            return ""

        available = self.components.skills.get_available()
        if not available:
            return ""

        skills_desc = "\n".join([f"- {s.name} ({s.trigger}): {s.description}" for s in available])

        return (
            "## Skill Selection\n"
            "You also need to select the most appropriate skill for this request.\n"
            "Available skills:\n"
            f"{skills_desc}\n\n"
            'Analyze the request and set "selected_skill" in your output '
            "to the best skill name, or null if no specific skill is needed."
        )

    def _get_tools(self, state: AgentState) -> list[dict]:
        """Get tools, filtered by skill if active."""
        skill = self._get_skill(state)
        use_skill_tool = self._get_use_skill_tool()

        if skill and skill.tools:
            skill_tools = [
                self.components.tools.get_definition(name)
                for name in skill.tools
                if self.components.tools.get_definition(name)
            ]
            return skill_tools + [use_skill_tool]

        return self.components.tools.get_all_definitions() + [use_skill_tool]

    def _get_use_skill_tool(self) -> dict:
        """Get the use_skill tool definition."""
        available = self.components.skills.get_available()
        skill_names = [s.name for s in available]

        return {
            "type": "function",
            "function": {
                "name": "use_skill",
                "description": (
                    "Switch to a specialized skill for better task handling. "
                    f"Available skills: {', '.join(skill_names)}"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {
                            "type": "string",
                            "description": (
                                f"Name of the skill to activate. Options: {', '.join(skill_names)}"
                            ),
                            "enum": skill_names,
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for switching to this skill",
                        },
                    },
                    "required": ["skill_name"],
                },
            },
        }
