"""Evaluation-specific fixtures and helpers for DeepEval tests."""

import json
import os
from pathlib import Path

import pytest

from .litellm_eval import get_eval_model, load_project_config


# ---------------------------------------------------------------------------
# Monkey-patch: DeepEval JSON 저장 시 ensure_ascii=False 적용
# (한글 등 non-ASCII 문자가 \uXXXX로 이스케이프되지 않도록)
# ---------------------------------------------------------------------------
def _patch_deepeval_json_encoding():
    from deepeval.test_run.test_run import TestRun, TestRunEncoder, TestRunManager

    def _save_utf8(self, f):
        try:
            body = self.model_dump(by_alias=True, exclude_none=True)
        except AttributeError:
            body = self.dict(by_alias=True, exclude_none=True)
        json.dump(body, f, cls=TestRunEncoder, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
        return self

    TestRun.save = _save_utf8

    def _save_test_run_utf8(self, path, save_under_key=None):
        import portalocker

        if portalocker and self.save_to_disk:
            try:
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with portalocker.Lock(path, mode="w", encoding="utf-8") as file:
                    if save_under_key:
                        try:
                            test_run_data = self.test_run.model_dump(
                                by_alias=True, exclude_none=True
                            )
                        except AttributeError:
                            test_run_data = self.test_run.dict(by_alias=True, exclude_none=True)
                        wrapper_data = {save_under_key: test_run_data}
                        json.dump(wrapper_data, file, cls=TestRunEncoder, ensure_ascii=False)
                    else:
                        self.test_run.save(file)
                    file.flush()
                    os.fsync(file.fileno())
            except portalocker.exceptions.LockException:
                pass

    TestRunManager.save_test_run = _save_test_run_utf8


_patch_deepeval_json_encoding()

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
    obs_callback=None,
) -> dict:
    """Run the full workflow and return the final state for evaluation.

    This is a convenience wrapper that initialises all components from config
    and invokes run_workflow().

    Args:
        obs_callback: Optional observability callback handler for tracing.
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

    # Prepare callbacks
    callbacks = [obs_callback] if obs_callback else None

    final_state = run_workflow(
        user_request=actual_request,
        llm_client=llm_client,
        prompt_registry=prompt_registry,
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        skill=skill,
        auto_select_skill=(skill is None),
        callbacks=callbacks,
    )

    # Flush observability traces
    if obs_callback and hasattr(obs_callback, "flush"):
        obs_callback.flush()

    return final_state


# ---------------------------------------------------------------------------
# Observability Integration for Test Tracing
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def obs_registry():
    """Create ObservabilityRegistry for the test session."""
    from src.observability import ObservabilityRegistry

    return ObservabilityRegistry()


@pytest.fixture(autouse=True)
def obs_test_trace(request, eval_config, obs_registry):
    """Create an observability callback handler for each test with proper context."""
    provider = obs_registry.resolve_provider(eval_config)
    if not provider:
        request.obs_callback = None
        request.obs_tracing_context = None
        yield None
        return

    if not provider.configure(eval_config):
        request.obs_callback = None
        request.obs_tracing_context = None
        yield None
        return

    test_name = request.node.name
    test_file = request.node.fspath.basename

    callback = provider.create_callback(
        trace_name=f"{test_file}::{test_name}",
        metadata={
            "test_file": test_file,
            "test_name": test_name,
            "test_id": request.node.nodeid,
        },
    )
    tracing_ctx = provider.get_tracing_context()

    request.obs_callback = callback
    request.obs_tracing_context = tracing_ctx

    yield callback

    provider.flush()
