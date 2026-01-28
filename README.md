# Plan-Act Workflow

LangGraph 기반 Plan and Act 워크플로우

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

## 프롬프트 교체

`src/prompts/templates/` 디렉토리에 YAML 파일 추가
