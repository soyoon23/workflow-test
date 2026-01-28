"""Prompt registry for easy prompt management and testing."""

import os
from pathlib import Path
from typing import Optional
import yaml


class PromptRegistry:
    """Registry for managing and switching between prompt versions."""

    def __init__(self, templates_dir: str = "src/prompts/templates"):
        self.templates_dir = Path(templates_dir)
        self._prompts: dict[str, dict[str, str]] = {}  # {role: {version: prompt}}
        self._active_versions: dict[str, str] = {}  # {role: version}
        self._load_all_prompts()

    def _load_all_prompts(self) -> None:
        """Load all prompt templates from the templates directory."""
        if not self.templates_dir.exists():
            return

        for file_path in self.templates_dir.glob("*.yaml"):
            self._load_prompt_file(file_path)

    def _load_prompt_file(self, file_path: Path) -> None:
        """Load a single prompt file."""
        # Parse filename: {role}_{version}.yaml
        filename = file_path.stem  # e.g., "planner_v1"
        parts = filename.rsplit("_", 1)

        if len(parts) != 2:
            return

        role, version = parts

        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if role not in self._prompts:
            self._prompts[role] = {}

        self._prompts[role][version] = data.get("system_prompt", "")

        # Set first loaded version as active if not set
        if role not in self._active_versions:
            self._active_versions[role] = version

    def get_available_roles(self) -> list[str]:
        """Get list of available prompt roles."""
        return list(self._prompts.keys())

    def get_available_versions(self, role: str) -> list[str]:
        """Get available versions for a role."""
        return list(self._prompts.get(role, {}).keys())

    def set_active_version(self, role: str, version: str) -> None:
        """Set the active version for a role."""
        if role not in self._prompts:
            raise ValueError(f"Unknown role: {role}")
        if version not in self._prompts[role]:
            raise ValueError(f"Unknown version '{version}' for role '{role}'")
        self._active_versions[role] = version

    def get_active_version(self, role: str) -> Optional[str]:
        """Get the active version for a role."""
        return self._active_versions.get(role)

    def get(self, role: str, version: Optional[str] = None) -> str:
        """Get prompt for a role (optionally specific version)."""
        if role not in self._prompts:
            raise ValueError(f"Unknown role: {role}")

        if version is None:
            version = self._active_versions.get(role)

        if version not in self._prompts[role]:
            raise ValueError(f"Unknown version '{version}' for role '{role}'")

        return self._prompts[role][version]

    def register_prompt(self, role: str, version: str, prompt: str) -> None:
        """Register a prompt programmatically (useful for testing)."""
        if role not in self._prompts:
            self._prompts[role] = {}
        self._prompts[role][version] = prompt

        if role not in self._active_versions:
            self._active_versions[role] = version

    def get_all_prompts_info(self) -> dict[str, dict[str, str]]:
        """Get info about all prompts for UI display."""
        info = {}
        for role in self._prompts:
            info[role] = {
                "versions": list(self._prompts[role].keys()),
                "active": self._active_versions.get(role, ""),
            }
        return info

    def reload(self) -> None:
        """Reload all prompts from disk."""
        self._prompts = {}
        self._active_versions = {}
        self._load_all_prompts()
