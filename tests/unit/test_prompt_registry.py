"""Unit tests for PromptRegistry."""

import pytest

from src.prompts.registry import PromptRegistry


class TestPromptRegistry:
    """Tests for PromptRegistry functionality."""

    def test_loads_templates(self, prompt_registry):
        roles = prompt_registry.get_available_roles()
        assert len(roles) > 0
        assert "planner" in roles
        assert "actor" in roles
        assert "reviewer" in roles

    def test_get_prompt_returns_string(self, prompt_registry):
        prompt = prompt_registry.get("planner")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_prompt_with_version(self, prompt_registry):
        versions = prompt_registry.get_available_versions("planner")
        assert "v1" in versions
        prompt = prompt_registry.get("planner", version="v1")
        assert isinstance(prompt, str)

    def test_unknown_role_raises(self, prompt_registry):
        with pytest.raises(ValueError, match="Unknown role"):
            prompt_registry.get("nonexistent_role")

    def test_unknown_version_raises(self, prompt_registry):
        with pytest.raises(ValueError, match="Unknown version"):
            prompt_registry.get("planner", version="v999")

    def test_set_active_version(self):
        registry = PromptRegistry()
        registry.register_prompt("test_role", "v1", "prompt v1")
        registry.register_prompt("test_role", "v2", "prompt v2")

        registry.set_active_version("test_role", "v2")
        assert registry.get_active_version("test_role") == "v2"
        assert registry.get("test_role") == "prompt v2"

    def test_set_active_version_invalid_role_raises(self):
        registry = PromptRegistry()
        with pytest.raises(ValueError):
            registry.set_active_version("nonexistent", "v1")

    def test_set_active_version_invalid_version_raises(self):
        registry = PromptRegistry()
        registry.register_prompt("test_role", "v1", "prompt v1")
        with pytest.raises(ValueError):
            registry.set_active_version("test_role", "v99")

    def test_register_prompt(self):
        registry = PromptRegistry()
        registry.register_prompt("custom", "v1", "custom prompt")
        assert registry.get("custom") == "custom prompt"
        assert "custom" in registry.get_available_roles()

    def test_get_all_prompts_info(self, prompt_registry):
        info = prompt_registry.get_all_prompts_info()
        assert "planner" in info
        assert "versions" in info["planner"]
        assert "active" in info["planner"]
