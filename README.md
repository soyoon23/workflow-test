# Plan-Act Workflow

LangGraph 기반 Plan-Act-Review 워크플로우 (멀티턴 대화 지원)

## 설치

```bash
uv sync
```

## 실행

```bash
# LiteLLM proxy 실행
litellm --model gpt-4 --port 4000

# Streamlit 앱 실행
uv run streamlit run app.py
```

## 주요 기능

- **Plan-Act-Review 루프**: LLM이 계획 수립 → 단계별 실행 → 결과 검토를 자동으로 반복
- **멀티턴 대화**: 이전 턴의 결과를 참조하여 후속 질문/수정 요청 처리
- **Skill 시스템**: `/research`, `/calc` 등 트리거로 특화된 워크플로우 활성화
- **도구 호출**: calculator, web_search 등 도구를 실행 중 자동 호출
- **프롬프트 버전 관리**: 역할별(planner, actor, reviewer) 프롬프트를 YAML로 관리, 런타임 교체 가능
- **확장 가능한 노드 구조**: `BaseNode` Protocol 기반으로 새 에이전트 아키텍처 추가 용이

## 아키텍처

### 프로젝트 구조

```
src/
├── workflow/                  # 워크플로우 코어
│   ├── state.py               # AgentState TypedDict
│   ├── streaming.py           # 스트리밍 이벤트 콜백
│   ├── components.py          # WorkflowComponents (의존성 번들)
│   ├── parsing.py             # JSON/plan/review 파싱 유틸리티
│   ├── tool_executor.py       # 도구 실행 루프
│   ├── graph.py               # LangGraph StateGraph 정의 및 실행
│   └── nodes/                 # 노드 모듈
│       ├── base.py            # BaseNode Protocol + NodeMixin
│       ├── plan.py            # PlanNode
│       ├── act.py             # ActNode
│       └── review.py          # ReviewNode
├── conversation/              # 멀티턴 대화 관리
│   ├── history.py             # ConversationHistoryManager
│   └── context.py             # ConversationContextBuilder
├── llm/                       # LLM 클라이언트 (LiteLLM 래퍼)
├── prompts/                   # 프롬프트 레지스트리 + YAML 템플릿
├── skills/                    # 스킬 레지스트리 + YAML 템플릿
├── tools/                     # 도구 레지스트리 + 구현
└── observability/             # 관측성 (트레이싱)
```

### 워크플로우 노드 구조

모든 워크플로우 노드는 `BaseNode` Protocol을 충족하며, `NodeMixin`을 통해 공통 헬퍼를 공유합니다.

- **`BaseNode` (Protocol)** — `__call__(state: AgentState) -> dict[str, Any]` 시그니처를 정의. 구조적 서브타이핑으로 어떤 클래스든 이 시그니처만 맞으면 노드로 사용 가능.
- **`NodeMixin`** — 프롬프트 조회, 스킬 조회, 도구 필터링 등 공통 로직을 제공.
- **`WorkflowComponents`** — 모든 노드가 필요로 하는 의존성(LLM, 프롬프트, 도구, 스킬 등)을 하나의 dataclass로 번들링.

```python
# 새 노드 추가 예시
class ReactNode(NodeMixin):
    def __init__(self, components: WorkflowComponents):
        self.components = components

    def __call__(self, state: AgentState) -> dict[str, Any]:
        prompt = self._get_prompt("actor", state)  # NodeMixin 헬퍼 재사용
        tools = self._get_tools(state)              # NodeMixin 헬퍼 재사용
        ...
```

### 워크플로우 루프

LangGraph StateGraph로 정의된 상태 머신:

1. **Plan** (`PlanNode`) — LLM이 사용자 요청을 분석하여 구조화된 실행 계획 생성
2. **Act** (`ActNode`) — 한 단계씩 실행, 필요 시 도구 호출 또는 스킬 전환
3. **Review** (`ReviewNode`) — 결과 평가 후 완료 → END, 또는 재계획 → Plan으로 복귀. `key_facts` 추출

노드 간 라우팅은 `should_continue()`, `after_review()` 조건부 엣지 함수가 처리합니다.

## 멀티턴 대화

채팅 UI에서 연속적인 대화가 가능합니다. 이전 턴의 결과를 컨텍스트로 활용합니다.

- 각 턴 완료 시 reviewer가 `key_facts`(핵심 수치/엔티티)를 추출하여 저장
- 최근 턴은 상세 계획 결과 포함, 이전 턴은 요약만 유지 (sliding window)
- 각 노드(planner/actor/reviewer)가 역할에 맞는 수준의 이력을 주입받음
- `config.yaml`의 `conversation` 섹션에서 이력 보관 설정 조정 가능

## 확장 가이드

| 확장 항목 | 방법 |
|-----------|------|
| 프롬프트 버전 | `src/prompts/templates/`에 `{role}_v{N}.yaml` 추가 |
| 도구 | `src/tools/base.py`에 구현 후 `src/tools/registry.py`에 등록 |
| 스킬 | `src/skills/templates/`에 YAML 파일 추가 |
| 워크플로우 노드 | `NodeMixin`을 상속한 새 클래스 작성 후 `graph.py`에 등록 |
| 컨텍스트 포매터 | `RoleContextFormatter` 서브클래스 작성 후 `ConversationContextBuilder.register_formatter()`로 등록 |

## 설정

`config.yaml`에서 LLM 연결, 프롬프트 버전, 워크플로우 제한, 대화 이력 설정을 관리합니다.
사이드바에서도 런타임 오버라이드가 가능합니다.
