"""Custom metrics for evaluating Plan node quality."""

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ..litellm_eval import get_eval_model

_EVAL_MODEL = get_eval_model()

# ---------------------------------------------------------------------------
# G-Eval metrics for plan quality
# ---------------------------------------------------------------------------

PlanDecompositionGEval = GEval(
    name="Plan Decomposition",
    criteria=(
        "주어진 사용자 요청(input)에 대해 생성된 실행 계획(actual_output)이 "
        "요청을 논리적이고 실행 가능한 독립적 단계들로 적절히 분해했는지 평가하시오. "
        "각 단계가 명확하고 구체적인 행동을 기술하고 있는지, "
        "단계 간 의존성이 올바르게 반영되었는지 확인하시오."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_EVAL_MODEL,
)

PlanCompletenessGEval = GEval(
    name="Plan Completeness",
    criteria=(
        "주어진 사용자 요청(input)을 완전히 달성하기 위해 필요한 모든 단계가 "
        "실행 계획(actual_output)에 포함되었는지 평가하시오. "
        "누락된 핵심 단계가 없는지, 불필요하게 중복된 단계가 없는지 확인하시오."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_EVAL_MODEL,
)


# ---------------------------------------------------------------------------
# Custom metric for skill selection accuracy
# ---------------------------------------------------------------------------


class SkillSelectionMetric(BaseMetric):
    """Evaluates whether the correct skill was selected for the request.

    Compares the expected_skill from golden metadata against the actual
    skill selected by the planner.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        # Expected skill from golden metadata (stored in additional_metadata)
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        expected_skill = metadata.get("expected_skill")
        actual_skill = metadata.get("actual_skill")

        if expected_skill is None:
            # No skill expected — pass if no skill was selected
            if actual_skill is None or actual_skill == "":
                self.score = 1.0
                self.reason = "No skill expected, none selected. Correct."
            else:
                self.score = 0.0
                self.reason = f"No skill expected but '{actual_skill}' was selected."
        else:
            if actual_skill == expected_skill:
                self.score = 1.0
                self.reason = f"Correct skill selected: {actual_skill}"
            else:
                self.score = 0.0
                self.reason = f"Expected skill '{expected_skill}' but got '{actual_skill}'"

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Skill Selection"
