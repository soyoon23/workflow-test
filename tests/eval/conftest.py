"""Evaluation-specific fixtures and helpers for DeepEval tests."""

import json
from pathlib import Path

import pytest

from .litellm_eval import get_eval_model, load_project_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "tests" / "datasets"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def eval_config():
    """Load config.yaml with eval section."""
    return load_project_config()


@pytest.fixture(scope="session")
def eval_model():
    """Create the evaluator LLM instance."""
    return get_eval_model()


@pytest.fixture(scope="session")
def eval_threshold(eval_config):
    """Default threshold for eval metrics."""
    return eval_config.get("eval", {}).get("threshold", 0.7)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------


def load_goldens(dataset_name: str) -> list[dict]:
    """Load golden dataset from tests/datasets/{dataset_name}.json."""
    path = DATASETS_DIR / f"{dataset_name}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("goldens", [])


@pytest.fixture(scope="session")
def e2e_goldens():
    """Load E2E golden dataset."""
    return load_goldens("e2e_goldens")


@pytest.fixture(scope="session")
def plan_goldens():
    """Load plan golden dataset."""
    return load_goldens("plan_goldens")


@pytest.fixture(scope="session")
def act_goldens():
    """Load act golden dataset."""
    return load_goldens("act_goldens")


@pytest.fixture(scope="session")
def review_goldens():
    """Load review golden dataset."""
    return load_goldens("review_goldens")


# ---------------------------------------------------------------------------
# Workflow execution helpers
# ---------------------------------------------------------------------------


def run_workflow_for_eval(
    user_request: str,
    config: dict,
) -> dict:
    """Run the full workflow and return the final state for evaluation.

    This is a convenience wrapper that initialises all components from config
    and invokes run_workflow().
    """
    from src.llm.client import LLMClient
    from src.prompts.registry import PromptRegistry
    from src.skills.registry import SkillRegistry
    from src.tools.registry import ToolRegistry
    from src.workflow.graph import run_workflow

    llm_cfg = config["llm"]
    llm_client = LLMClient(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        api_key=llm_cfg["api_key"],
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 4096),
    )
    prompt_registry = PromptRegistry(
        templates_dir=str(PROJECT_ROOT / "src" / "prompts" / "templates")
    )
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry(templates_dir=str(PROJECT_ROOT / "src" / "skills" / "templates"))

    # Detect skill trigger
    skill, actual_request = skill_registry.parse_input(user_request)

    final_state = run_workflow(
        user_request=actual_request,
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        skill=skill,
        auto_select_skill=(skill is None),
    )
    return final_state
