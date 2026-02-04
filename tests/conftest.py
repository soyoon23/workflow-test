"""Shared test fixtures for unit tests and evaluation tests."""

import sys
from pathlib import Path

import pytest
import yaml

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def config():
    """Load config.yaml."""
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def prompt_registry():
    """Create PromptRegistry with default templates."""
    from src.prompts.registry import PromptRegistry

    return PromptRegistry(templates_dir=str(PROJECT_ROOT / "src" / "prompts" / "templates"))


@pytest.fixture(scope="session")
def tool_registry():
    """Create ToolRegistry with default tools."""
    from src.tools.registry import ToolRegistry

    return ToolRegistry()


@pytest.fixture(scope="session")
def skill_registry():
    """Create SkillRegistry with default skills."""
    from src.skills.registry import SkillRegistry

    return SkillRegistry(templates_dir=str(PROJECT_ROOT / "src" / "skills" / "templates"))


@pytest.fixture(scope="session")
def llm_client(config):
    """Create LLMClient from config (requires LiteLLM proxy running)."""
    from src.llm.client import LLMClient

    llm_config = config["llm"]
    return LLMClient(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key=llm_config["api_key"],
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
    )


@pytest.fixture
def workflow_components(llm_client, prompt_registry, tool_registry, skill_registry):
    """Create WorkflowComponents instance."""
    from src.workflow.components import WorkflowComponents

    return WorkflowComponents(
        llm=llm_client,
        prompts=prompt_registry,
        tools=tool_registry,
        skills=skill_registry,
    )


@pytest.fixture
def empty_state():
    """Create a minimal empty AgentState for testing."""
    return {
        "messages": [],
        "user_request": "",
        "plan": [],
        "current_step_index": 0,
        "is_complete": False,
        "final_answer": None,
        "iteration_count": 0,
        "error": None,
        "active_skill_name": None,
        "auto_select_skill": False,
        "conversation_history": [],
        "review_key_facts": [],
    }
