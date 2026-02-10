## Plan-Act 워크플로 개요

이 문서는 LangGraph 기반 Plan-Act-Review 아키텍처의 현재 설계를 설명한다. 각 노드는 LangGraph `StateGraph` 위에서 돌아가며, 상태(`AgentState`)는 LangChain 메시지 스택, 플랜 스텝, 스킬, 히스토리 등을 추적한다.

### Plan 노드 (planner)

- **주요 역할**: 사용자 요청을 분석해 다단계 실행 계획(`PlanStep` 목록)을 만든다. 필요 시 이전 실행 결과를 참고해 재계획한다.
- **프롬프트 구성**: `src/workflow/nodes.py`의 `_get_prompt("planner")`가 기본 템플릿(`PromptRegistry`)에 다음 요소를 주입한다. 1) 자동 스킬 선택이 켜져 있고 활성 스킬이 없다면, 등록된 스킬 목록과 선택 지침(`_build_skill_selection_block`). 2) 활성 스킬의 추가 컨텍스트. 3) `ConversationContextBuilder`가 만든 히스토리(Planner는 요청·응답·최신 계획 세부 사항·사용 스킬까지 본다).
- **계획 생성**: LLM에 “Create a plan for …” 메시지를 보낸 뒤 JSON 계획을 파싱한다(`_parse_plan`). 스텝에는 번호, 설명, 초기 상태가 포함된다. JSON이 없으면 단일 스텝으로 폴백한다.
- **스킬 자동 선택**: 플래그(`auto_select_skill`)가 켠 상태에서 LLM 응답이 `selected_skill`을 지정하면, 상태의 `active_skill_name`이 갱신되어 이후 액터/툴 필터링에 반영된다.
- **상태 영향**: 새 메시지, 플랜, 스텝 인덱스 초기화, 반복 카운터 증가를 반환한다.

### Act 노드 (actor)

- **주요 역할**: 현재 플랜 스텝을 실행한다. LangGraph는 `should_continue`를 통해 미완료 스텝이 있을 때만 Act 노드로 분기한다.
- **프롬프트 구성**: `_get_prompt("actor")`가 기본 액터 프롬프트에 히스토리와 스킬 지침을 병합한다. 액터는 `ConversationContextBuilder` 기준 최소한의 정보(Review 노드가 추출한 `key_facts` 또는 답변 요약)만 받는다.
- **툴 접근**: `_get_tools`가 현재 활성 스킬이 허용한 툴만 노출한다. 항상 `use_skill` 함수 툴이 추가되어 실행 중 스킬 전환이 가능하다. 스킬이 없으면 전체 툴 레지스트리 정의를 쓴다.
- **실행 흐름**:
  1. 현재 스텝 상태를 `in_progress`로 마킹한다. `_build_prior_context`로 이전 완료 스텝 결과를 수집하고, 원본 `user_request`와 함께 HumanMessage를 구성한다.
  2. `_call_llm` 헬퍼를 통해 LLM 호출(스트리밍/비스트리밍 분기를 내부 처리). 응답이 툴 호출을 포함하면 multi-round 루프(`MAX_TOOL_ROUNDS=3`)를 돌며 `execute_tool_calls`로 툴을 실행한다. 마지막 라운드에서는 `tools=None`으로 호출해 텍스트 응답을 강제한다. `use_skill` 호출 시 `active_skill_name`이 업데이트된다.
  3. 전체 LLM/툴 실행이 try/except로 감싸져 있다. 성공 시 스텝 상태를 `completed`로, 결과 문자열을 저장한다. 실패 시 `status="failed"`, 에러 메시지를 기록하고 `callback.error()`를 호출한 뒤 `error` 필드를 반환해 `should_continue`가 워크플로를 종료한다.

### Review 노드 (reviewer)

- **주요 역할**: 플랜 전체의 실행 로그를 검토해 완료 여부(`is_complete`), 최종 답변, 오류, `key_facts`를 결정한다. 리뷰는 LangGraph에서 모든 스텝이 끝났거나 `should_continue`가 “review”를 리턴했을 때 호출된다.
- **프롬프트 구성**: `_get_prompt("reviewer")`가 기본 템플릿에 히스토리(Review는 사용자 요청·응답·key facts)를 붙인다. 활성 스킬 컨텍스트도 포함된다.
- **입력 데이터**: `review_node`는 요청과 `_format_plan_for_review`가 만든 스텝별 요약(설명/상태/결과)을 전달한다.
- **결과 파싱**: `_parse_review`가 JSON에서 `is_complete`, `final_answer`, `error`, `key_facts`를 추출한다. 실패 시 완료로 가정하고 응답 전체를 최종 답변으로 사용한다.
- **상태 영향**: Review 노드가 반환한 값은 `AgentState`의 완료 플래그, 최종 답, 오류 메시지, `review_key_facts` 필드를 갱신한다. key facts는 다음 턴의 히스토리 빌드에 사용된다.

### Edge / 분기 로직

- **should_continue** (`src/workflow/graph.py`)
  - 에러 발생(`state.error`) 또는 완료(`is_complete=True`) → `end`로 전환.
  - 반복 제한(`iteration_count >= 10`) → 강제 종료.
  - 플랜이 없으면 `plan`으로 이동해 새 전략을 생성.
- 플랜은 있으나 `current_step_index`가 스텝 수 미만이면 `act`, 모든 스텝 완료 시 `review`로 이동.
- **after_review**: 리뷰 후 `is_complete=True`면 종료, 아니면 항상 `plan`으로 돌아가 재계획한다. 리뷰 결과가 재계획 필요를 명시하지 않아도, 완료 플래그가 False면 새 플랜을 강제한다.
- **초기 진입**: 그래프 엔트리 포인트는 항상 `plan`이다. Plan 노드가 내부적으로 스킬 선택을 처리하기 때문에 별도 라우터 필요가 없다.

## 에이전트 컨텍스트와 히스토리

### ConversationHistoryManager & ContextBuilder

- `ConversationHistoryManager`(Streamlit 세션 상태)는 최근 대화 턴을 저장한다. 최신 턴은 플랜 세부 사항을 포함하고, 오래된 턴은 요약만 남겨 토큰 사용을 줄인다. 설정(`config.yaml`)의 `max_history_turns`, `include_plan_detail_turns`가 슬라이딩 윈도우를 제어한다.
- `ConversationContextBuilder`는 역할별 포매터를 사용해 히스토리를 문자열로 조합한다.
  - **Planner**: 사용자 요청, 모델 답변, 최신 플랜 구조(스텝 설명 + 결과), 사용된 스킬을 본다. 다음 플랜을 만들 때 과거 전략과 실행 맥락을 최대한 활용하도록 설계됐다.
  - **Actor**: 최소 정보만 본다. 기본적으로 Review가 추출한 `key_facts`를 전달하고, 없으면 최종 답변 요약으로 대체한다. 실행 단계에서 노이즈를 줄이는 목적이다.
  - **Reviewer**: 사용자 요청, 모델 답변, key facts를 본다. 충분한 맥락을 주되 Planner만큼 자세한 플랜 구조는 제외해 리뷰 비용을 줄인다.

### 스킬과 툴 컨텍스트

- **SkillRegistry**: 사용자 지정 YAML 기반 스킬 정의를 로드한다. 각 스킬은 이름, 설명, 트리거 프리픽스, 추가 프롬프트(`skill.prompt`), 허용 툴 목록을 가진다. Planner는 자동 선택 기능이 켜졌을 때 스킬 목록과 트리거를 참고해 `selected_skill`을 설정한다.
- **Skill 적용 시 프롬프트 변화**: `_get_prompt`는 활성 스킬의 추가 지침을 모든 역할 프롬프트에 붙인다. 이렇게 하면 특정 도메인/언어/작업 스타일을 강제할 수 있다.
- **툴 제한**: 활성 스킬과 연결된 툴만 Act 노드에서 노출된다. 스킬이 지정되지 않으면 툴 레지스트리 전체가 사용 가능하다. `use_skill` 툴은 항상 접근 가능하며 실행 중 스킬 변경을 허용한다.

### 히스토리 저장 항목

- 각 턴(`ConversationTurn`)은 사용자 요청, 모델 최종 답변, 사용된 스킬, 플랜 요약, 스텝별 결과, 리뷰 key facts 등을 담는다. Review 노드에서 추출한 key facts는 다음 턴의 Actor/Reviewer 컨텍스트에 직접 쓰이며, Planner는 최신 턴에서 플랜 구조와 결과를 상세히 확인한다.

## 요약

이 Plan-Act-Review 설계는 LangGraph 상태 머신을 기반으로 플랜 생성 → 단계별 실행 → 결과 검토 루프를 명확히 분리한다. 분기 조건과 반복 제한을 통해 무한 루프를 방지하고, 스킬/툴 레지스트리가 역할별 프롬프트와 실행 권한을 제어한다. 또한 히스토리 컨텍스트를 역할별로 다르게 제공해 Planner는 최대 정보를, Actor는 최소 정보를 받도록 설계해 비용과 품질을 균형 있게 맞춘다.
