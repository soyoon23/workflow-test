"""Skill registry for managing context/instruction injection."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class Skill:
    """A skill injects context and instructions into the workflow."""

    name: str
    description: str
    trigger: str  # e.g., "/research", "/curate"

    # Context/instructions to inject into prompts
    prompt: str = ""

    # Tools enabled for this skill (empty = all tools available)
    tools: list[str] = field(default_factory=list)

    # Whether this skill is enabled
    enabled: bool = True


class SkillRegistry:
    """Registry for managing and loading skills."""

    def __init__(self, templates_dir: str = "src/skills/templates"):
        self.templates_dir = Path(templates_dir)
        self._skills: dict[str, Skill] = {}
        self._load_all_skills()

    def _load_all_skills(self) -> None:
        """Load all skill definitions from templates directory."""
        if not self.templates_dir.exists():
            return

        # Load all yaml files in templates directory
        for file_path in self.templates_dir.glob("*.yaml"):
            self._load_skill_file(file_path)

    def _load_skill_file(self, file_path: Path) -> None:
        """Load a single skill file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        skill = Skill(
            name=data.get("name", file_path.stem),
            description=data.get("description", ""),
            trigger=data.get("trigger", f"/{file_path.stem}"),
            prompt=data.get("prompt", ""),
            tools=data.get("tools", []),
            enabled=data.get("enabled", True),
        )

        self._skills[skill.name] = skill

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_by_trigger(self, trigger: str) -> Optional[Skill]:
        """Get a skill by its trigger command."""
        for skill in self._skills.values():
            if skill.trigger == trigger:
                return skill
        return None

    def get_all(self) -> list[Skill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def get_available(self) -> list[Skill]:
        """Get all enabled skills."""
        return [s for s in self._skills.values() if s.enabled]

    def register(self, skill: Skill) -> None:
        """Register a skill programmatically."""
        self._skills[skill.name] = skill

    def unregister(self, name: str) -> None:
        """Unregister a skill."""
        self._skills.pop(name, None)

    def parse_input(self, user_input: str) -> tuple[Optional[Skill], str]:
        """
        Parse user input to detect skill triggers.
        Returns (skill, remaining_input) or (None, original_input).
        """
        stripped = user_input.strip()

        for skill in self._skills.values():
            if stripped.startswith(skill.trigger):
                remaining = stripped[len(skill.trigger):].strip()
                return skill, remaining

        return None, user_input

    def reload(self) -> None:
        """Reload all skills from disk."""
        self._skills = {}
        self._load_all_skills()
