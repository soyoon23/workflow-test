"""Enhanced plan quality metrics with rubric-based evaluation.

Version 2 of plan quality metrics that use structured rubrics and exemplars
to provide more accurate and consistent evaluation scores.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

from .exemplars import (
    PLAN_DECOMPOSITION_EXEMPLARS,
    PLAN_COMPLETENESS_EXEMPLARS,
)
from .rubrics import PLAN_COMPLETENESS_RUBRIC, PLAN_DECOMPOSITION_RUBRIC
from ..litellm_eval import get_eval_model

_EVAL_MODEL = get_eval_model()


# ---------------------------------------------------------------------------
# Plan Decomposition V2: Rubric + Exemplar Based
# ---------------------------------------------------------------------------

def _build_plan_decomposition_criteria():
    """Build evaluation criteria for plan decomposition with rubric and exemplars."""
    good_ex1 = PLAN_DECOMPOSITION_EXEMPLARS["good_example_1"]
    good_ex2 = PLAN_DECOMPOSITION_EXEMPLARS["good_example_2"]
    bad_ex1 = PLAN_DECOMPOSITION_EXEMPLARS["bad_example_1"]

    return f"""{PLAN_DECOMPOSITION_RUBRIC}

## 평가 예시

### 좋은 예시 1 (점수: {good_ex1['score']})
**입력:**
{good_ex1['input']}

**계획:**
{good_ex1['plan']}

**평가 사유:**
{good_ex1['reason']}

---

### 좋은 예시 2 (점수: {good_ex2['score']})
**입력:**
{good_ex2['input']}

**계획:**
{good_ex2['plan']}

**평가 사유:**
{good_ex2['reason']}

---

### 나쁜 예시 1 (점수: {bad_ex1['score']})
**입력:**
{bad_ex1['input']}

**계획:**
{bad_ex1['plan']}

**평가 사유:**
{bad_ex1['reason']}

---

## 평가 지침

위의 루브릭에 따라 다음 계획을 4개 항목(독립성, 구체성, 논리순서, 도구선택)별로 평가한 후,
총점을 계산하여 0.0-1.0 범위로 정규화된 점수를 제시하시오.

점수를 제시할 때는 다음 형식으로:
- 각 항목별 점수 (예: "독립성: 3/3")
- 총점 (예: "총점: 10/10 = 1.0")
- 최종 평가 점수 (0.0-1.0)

## 평가 대상

**사용자 요청:** {{input}}

**생성된 계획:** {{actual_output}}

위의 루브릭과 예시를 참고하여 평가하시오.
"""


PlanDecompositionGEvalV2 = GEval(
    name="Plan Decomposition V2",
    criteria=_build_plan_decomposition_criteria(),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_EVAL_MODEL,
)


# ---------------------------------------------------------------------------
# Plan Completeness V2: Rubric + Exemplar Based
# ---------------------------------------------------------------------------

def _build_plan_completeness_criteria():
    """Build evaluation criteria for plan completeness with rubric and exemplars."""
    complete = PLAN_COMPLETENESS_EXEMPLARS["complete_plan"]
    incomplete = PLAN_COMPLETENESS_EXEMPLARS["incomplete_plan"]

    return f"""{PLAN_COMPLETENESS_RUBRIC}

## 평가 예시

### 완전한 계획 (점수: {complete['score']})
**입력:**
{complete['input']}

**계획:**
{complete['plan']}

**평가 사유:**
{complete['reason']}

---

### 불완전한 계획 (점수: {incomplete['score']})
**입력:**
{incomplete['input']}

**계획:**
{incomplete['plan']}

**평가 사유:**
{incomplete['reason']}

---

## 평가 지침

위의 루브릭에 따라 계획의 완전성을 3개 항목(핵심단계, 중복제거, 분기점처리)별로 평가한 후,
총점을 7로 정규화하여 0.0-1.0 범위로 변환하시오.

점수를 제시할 때는 다음 형식으로:
- 각 항목별 점수 (예: "핵심단계: 3/3")
- 총점 (예: "총점: 7/7 = 1.0")
- 정규화된 최종 평가 점수 (0.0-1.0)

## 평가 대상

**사용자 요청:** {{input}}

**생성된 계획:** {{actual_output}}

위의 루브릭과 예시를 참고하여 평가하시오.
"""


PlanCompletenessGEvalV2 = GEval(
    name="Plan Completeness V2",
    criteria=_build_plan_completeness_criteria(),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model=_EVAL_MODEL,
)


if __name__ == "__main__":
    # Display the criteria
    print("=" * 80)
    print("PLAN DECOMPOSITION V2 CRITERIA")
    print("=" * 80)
    print(_build_plan_decomposition_criteria())
    print("\n" + "=" * 80)
    print("PLAN COMPLETENESS V2 CRITERIA")
    print("=" * 80)
    print(_build_plan_completeness_criteria())
