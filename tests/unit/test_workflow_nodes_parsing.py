"""Tests for WorkflowNodes parsing helpers."""

from unittest.mock import MagicMock

import pytest

from src.workflow.nodes import WorkflowNodes


@pytest.fixture()
def workflow_nodes_stub() -> WorkflowNodes:
    """Create a WorkflowNodes instance with mocked dependencies."""

    return WorkflowNodes(
        llm_client=MagicMock(),
        prompt_registry=MagicMock(),
        tool_registry=MagicMock(),
        skill_registry=MagicMock(),
    )


def test_parse_plan_extracts_json_from_code_fence(workflow_nodes_stub: WorkflowNodes):
    workflow_nodes_stub.skill_registry.get.return_value = object()

    content = (
        "Plan follows.\n"
        "```json\n"
        "{\n"
        '  "selected_skill": "research",\n'
        '  "steps": [\n'
        '    {"step_number": 1, "description": "Collect data"},\n'
        '    {"step_number": 2, "description": "Summarize"}\n'
        "  ]\n"
        "}\n"
        "```"
    )

    plan, selected_skill = workflow_nodes_stub._parse_plan(content)

    assert selected_skill == "research"
    assert len(plan) == 2
    assert plan[0]["description"] == "Collect data"
    assert plan[1]["description"] == "Summarize"


def test_parse_plan_falls_back_when_json_missing(workflow_nodes_stub: WorkflowNodes):
    content = "No structured output provided"

    plan, selected_skill = workflow_nodes_stub._parse_plan(content)

    assert selected_skill is None
    assert len(plan) == 1
    assert plan[0]["description"].startswith("No structured output")


def test_parse_review_extracts_payload(workflow_nodes_stub: WorkflowNodes):
    content = """
    ```json
    {
      "is_complete": false,
      "final_answer": "Need more info",
      "key_facts": ["fact A"],
      "error": null
    }
    ```
    """

    review = workflow_nodes_stub._parse_review(content)

    assert review["is_complete"] is False
    assert review["final_answer"] == "Need more info"
    assert review["key_facts"] == ["fact A"]


def test_parse_review_fallback_on_plain_text(workflow_nodes_stub: WorkflowNodes):
    content = "Unable to parse structured review"

    review = workflow_nodes_stub._parse_review(content)

    assert review["is_complete"] is True
    assert review["final_answer"] == content
