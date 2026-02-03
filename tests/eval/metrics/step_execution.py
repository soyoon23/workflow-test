"""Custom metrics for evaluating Act node (step execution) quality."""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from ..litellm_eval import get_eval_model

_EVAL_MODEL = get_eval_model()

# ---------------------------------------------------------------------------
# G-Eval metric for step relevancy
# ---------------------------------------------------------------------------

StepRelevancyGEval = GEval(
    name="Step Relevancy",
    criteria=(
        "실행 단계의 설명(input)이 요구하는 작업에 대해, "
        "실제 실행 결과(actual_output)가 해당 작업을 정확하게 수행했는지 평가하시오. "
        "결과가 단계 설명과 직접적으로 관련되고, "
        "요구된 작업을 완전히 이행했는지 확인하시오."
    ),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_EVAL_MODEL,
)
