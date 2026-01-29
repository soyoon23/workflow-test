# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the app (requires LiteLLM proxy running first)
litellm --model gpt-4 --port 4000    # Terminal 1: start LLM proxy
uv run streamlit run app.py           # Terminal 2: start Streamlit UI

# Lint and format
uv run ruff check .
uv run ruff format .

# Tests
uv run pytest

# Generate workflow graph visualization
python visualize_graph.py
```

## Architecture

This is a **Plan-Act-Review** AI workflow built on LangGraph and LiteLLM with a Streamlit UI.

### Workflow Loop

The core loop is a LangGraph state machine defined in `src/workflow/graph.py`:

1. **Plan** (`plan_node`) — LLM analyzes the user request and produces a structured execution plan (list of `PlanStep`)
2. **Act** (`act_node`) — Executes one step at a time, optionally calling tools (calculator, web_search) or switching skills mid-execution
3. **Review** (`review_node`) — LLM evaluates results and decides: complete → END, or needs replan → back to Plan

Routing between nodes is handled by `should_continue()` and `after_review()` conditional edge functions. The workflow state (`AgentState` TypedDict in `src/workflow/state.py`) carries messages, plan steps, iteration count, and active skill context through the graph.

### Registry Pattern

Three registries manage extensible components:

- **PromptRegistry** (`src/prompts/registry.py`) — Loads versioned YAML prompt templates from `src/prompts/templates/`. Each role (planner, actor, reviewer) can have multiple versions (v1, v2). Active version is switchable at runtime.
- **ToolRegistry** (`src/tools/registry.py`) — Registers tool definitions (OpenAI function format) and their Python implementations. Tools are defined in `src/tools/base.py`.
- **SkillRegistry** (`src/skills/registry.py`) — Loads skill YAML files from `src/skills/templates/`. Skills are triggered via user input prefixes (e.g., `/research`, `/calc`) or auto-selected by the LLM. Skills inject additional prompt context and can restrict available tools.

### LLM Client

`src/llm/client.py` wraps LiteLLM for OpenAI-compatible API calls, converting between LangChain message types and OpenAI format. Configured via `config.yaml` to point at a LiteLLM proxy (default `localhost:4000`).

### Adding New Components

- **Prompt version**: Add a YAML file to `src/prompts/templates/` named `{role}_v{N}.yaml`
- **Tool**: Add implementation to `src/tools/base.py` and register in `src/tools/registry.py`
- **Skill**: Add a YAML file to `src/skills/templates/` with name, description, trigger, prompt, and tools fields

## Configuration

`config.yaml` holds LLM connection settings (base_url, model, api_key), default prompt version, and workflow limits (max_iterations). The Streamlit sidebar also allows runtime overrides.

## Code Style

- Python 3.11+
- Ruff with line-length 100, rules: E, F, I, W
- UI text is in Korean; code and variable names are in English
