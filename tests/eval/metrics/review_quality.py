"""Custom metrics for evaluating Review node quality."""

from deepeval.metrics import BaseMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from ..litellm_eval import get_eval_model

_EVAL_MODEL = get_eval_model()

# ---------------------------------------------------------------------------
# G-Eval metric for key facts quality
# ---------------------------------------------------------------------------

KeyFactsQualityGEval = GEval(
    name="Key Facts Quality",
    criteria=(
        "리뷰어가 추출한 key_facts(actual_output)가 "
        "원래 사용자 요청(input)과 실행 결과(context)를 기반으로, "
        "후속 대화에서 유용한 맥락 정보를 제공하는지 평가하시오. "
        "key_facts가 핵심 정보를 간결하게 요약하고 있으며, "
        "다음 대화에서 참고할 만한 가치가 있는지 확인하시오."
    ),
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.CONTEXT,
    ],
    threshold=0.7,
    model=_EVAL_MODEL,
)


# ---------------------------------------------------------------------------
# Custom metric for completion decision accuracy
# ---------------------------------------------------------------------------


class CompletionDecisionMetric(BaseMetric):
    """Evaluates whether the review node made the correct is_complete decision.

    Compares the expected completion status from golden metadata against
    the actual is_complete value from the review result.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase) -> float:
        metadata = getattr(test_case, "additional_metadata", {}) or {}
        expected_complete = metadata.get("expected_complete")
        actual_complete = metadata.get("actual_complete")

        if expected_complete is None:
            self.score = 1.0
            self.reason = "No expected completion status specified, skipping."
        elif expected_complete == actual_complete:
            self.score = 1.0
            self.reason = f"Correct decision: is_complete={actual_complete}"
        else:
            self.score = 0.0
            self.reason = (
                f"Expected is_complete={expected_complete} but got is_complete={actual_complete}"
            )

        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Completion Decision"
