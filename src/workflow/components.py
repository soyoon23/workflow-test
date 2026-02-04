"""Shared workflow component dependencies."""

from dataclasses import dataclass, field
from typing import Optional

from ..conversation.context import ConversationContextBuilder
from ..llm.client import LLMClient
from ..prompts.registry import PromptRegistry
from ..skills.registry import Skill, SkillRegistry
from ..tools.registry import ToolRegistry
from .streaming import StreamCallback


@dataclass
class WorkflowComponents:
    """Bundle of shared dependencies injected into workflow nodes.

    Groups all registries and clients that nodes need, replacing
    the previous pattern of passing 5-7 individual parameters.
    """

    llm: LLMClient
    prompts: PromptRegistry
    tools: ToolRegistry
    skills: SkillRegistry
    initial_skill: Optional[Skill] = None
    callback: StreamCallback = field(default_factory=StreamCallback)
    context_builder: ConversationContextBuilder = field(
        default_factory=ConversationContextBuilder
    )
