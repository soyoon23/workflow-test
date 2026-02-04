"""Tests for parsing utilities (extract_json_payload, parse_plan, parse_review)."""

from unittest.mock import MagicMock

from src.workflow.parsing import extract_json_payload, parse_plan, parse_review


def test_parse_plan_extracts_json_from_code_fence():
    skill_registry = MagicMock()
    skill_registry.get.return_value = object()

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

    plan, selected_skill = parse_plan(content, skill_registry)

    assert selected_skill == "research"
    assert len(plan) == 2
    assert plan[0]["description"] == "Collect data"
    assert plan[1]["description"] == "Summarize"


def test_parse_plan_falls_back_when_json_missing():
    skill_registry = MagicMock()
    content = "No structured output provided"

    plan, selected_skill = parse_plan(content, skill_registry)

    assert selected_skill is None
    assert len(plan) == 1
    assert plan[0]["description"].startswith("No structured output")


def test_parse_review_extracts_payload():
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

    review = parse_review(content)

    assert review["is_complete"] is False
    assert review["final_answer"] == "Need more info"
    assert review["key_facts"] == ["fact A"]


def test_parse_review_fallback_on_plain_text():
    content = "Unable to parse structured review"

    review = parse_review(content)

    assert review["is_complete"] is True
    assert review["final_answer"] == content


def test_extract_json_payload_from_code_fence():
    content = '```json\n{"key": "value"}\n```'
    result = extract_json_payload(content)
    assert result == {"key": "value"}


def test_extract_json_payload_from_bare_json():
    content = 'Some text {"key": "value"} more text'
    result = extract_json_payload(content)
    assert result == {"key": "value"}


def test_extract_json_payload_returns_none_for_no_json():
    content = "No JSON here"
    result = extract_json_payload(content)
    assert result is None
