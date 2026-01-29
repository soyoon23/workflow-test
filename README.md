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

## 멀티턴 대화

채팅 UI에서 연속적인 대화가 가능합니다. 이전 턴의 결과를 컨텍스트로 활용합니다.

- 각 턴 완료 시 reviewer가 `key_facts`(핵심 수치/엔티티)를 추출하여 저장
- 최근 턴은 상세 계획 결과 포함, 이전 턴은 요약만 유지 (sliding window)
- 각 노드(planner/actor/reviewer)가 역할에 맞는 수준의 이력을 주입받음
- `config.yaml`의 `conversation` 섹션에서 이력 보관 설정 조정 가능

## 프롬프트 교체

`src/prompts/templates/` 디렉토리에 `{role}_v{N}.yaml` 형식으로 YAML 파일 추가

## 설정

`config.yaml`에서 LLM 연결, 프롬프트 버전, 워크플로우 제한, 대화 이력 설정을 관리합니다.
사이드바에서도 런타임 오버라이드가 가능합니다.
