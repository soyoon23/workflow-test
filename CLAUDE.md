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

## Recent Changes (2026-02-04)

- LLM stack now uses `langchain-openai`'s `ChatOpenAI` (via LiteLLM proxy). Always pass the `RunnableConfig` from LangGraph nodes down to `LLMClient` so Langfuse callbacks capture every call.
- Added official Langfuse v3 integration. Use `create_callback()` to get both the callback handler and tracing context, and call `src/observability.flush()` once a workflow run completes to ensure traces are persisted.
- `python-dotenv` is loaded at `app.py` startup so secrets can live in `.env`. `config.yaml` is gitignored—copy from `config.yaml.example` when bootstrapping local envs.
- Workflow metadata sent to observability is now normalized (`skill="none"`, `auto_select` as lowercase string) to keep Langfuse dashboards clean.
- Dependencies updated (`langchain`, `langchain-openai`, `python-dotenv`), so run `uv sync` after pulling.

## Architecture

This is a **Plan-Act-Review** AI workflow built on LangGraph and LiteLLM with a Streamlit chat UI supporting **multi-turn conversations**.

### Workflow Node Architecture

Nodes are independent classes satisfying the `BaseNode` protocol (`src/workflow/nodes/base.py`). Each node receives a `WorkflowComponents` dataclass that bundles all shared dependencies (LLM client, registries, callback).

```
src/workflow/
├── components.py        # WorkflowComponents dataclass
├── parsing.py           # JSON/plan/review parsing utilities
├── tool_executor.py     # Reusable tool execution loop
├── graph.py             # LangGraph workflow definition & execution
├── state.py             # AgentState TypedDict
├── streaming.py         # StreamCallback for UI events
└── nodes/
    ├── base.py          # BaseNode Protocol + NodeMixin (shared helpers)
    ├── plan.py          # PlanNode — creates execution plans
    ├── act.py           # ActNode — executes plan steps with tools
    └── review.py        # ReviewNode — evaluates results
```

**Key types:**
- **`BaseNode`** (Protocol) — Any callable `(AgentState, RunnableConfig) -> dict[str, Any]` satisfies this. LangGraph auto-passes `RunnableConfig` (with callback handlers) to each node.
- **`NodeMixin`** — Provides shared helpers (`_get_prompt`, `_get_skill`, `_get_tools`) used by all built-in nodes.
- **`WorkflowComponents`** — Bundles `LLMClient`, `PromptRegistry`, `ToolRegistry`, `SkillRegistry`, `StreamCallback`, and `ConversationContextBuilder`.

### Workflow Loop

The core loop is a LangGraph state machine defined in `src/workflow/graph.py`:

1. **Plan** (`PlanNode`) — LLM analyzes the user request and produces a structured execution plan (list of `PlanStep`)
2. **Act** (`ActNode`) — Executes one step at a time, calling tools via `execute_tool_calls()` from `tool_executor.py`
3. **Review** (`ReviewNode`) — LLM evaluates results and decides: complete → END, or needs replan → back to Plan. Extracts `key_facts` for multi-turn context.

Routing between nodes is handled by `should_continue()` and `after_review()` conditional edge functions. The workflow state (`AgentState` TypedDict in `src/workflow/state.py`) carries messages, plan steps, iteration count, active skill context, and conversation history through the graph.

### Multi-Turn Conversation

`src/conversation/` manages cross-turn context with a hybrid sliding window strategy:

- **`ConversationHistoryManager`** (`src/conversation/history.py`) — Tracks `ConversationTurn` summaries across workflow invocations. Applies a sliding window: recent turns retain full plan results, older turns keep only summaries. Configurable via `config.yaml` (`max_history_turns`, `include_plan_detail_turns`).
- **`ConversationContextBuilder`** (`src/conversation/context.py`) — Builds role-specific context strings from history. Uses `RoleContextFormatter` subclasses for each role:
  - **Planner**: Sees request, answer, recent plan structure, skill used (most detail)
  - **Actor**: Sees only `key_facts` (minimal, avoids noise)
  - **Reviewer**: Sees request, answer, key_facts (medium detail)
- Custom formatters can be registered via `ConversationContextBuilder.register_formatter()`.

Context is injected into each node's system prompt via `NodeMixin._get_prompt()`. The `ConversationHistoryManager` lives in Streamlit session state and persists across reruns.

### Registry Pattern

Three registries manage extensible components:

- **PromptRegistry** (`src/prompts/registry.py`) — Loads versioned YAML prompt templates from `src/prompts/templates/`. Each role (planner, actor, reviewer) can have multiple versions (v1, v2). Active version is switchable at runtime.
- **ToolRegistry** (`src/tools/registry.py`) — Registers tool definitions (OpenAI function format) and their Python implementations. Tools are defined in `src/tools/base.py`.
- **SkillRegistry** (`src/skills/registry.py`) — Loads skill YAML files from `src/skills/templates/`. Skills are triggered via user input prefixes (e.g., `/research`, `/calc`) or auto-selected by the LLM. Skills inject additional prompt context and can restrict available tools.

### LLM Client

`src/llm/client.py` wraps `langchain-openai`'s `ChatOpenAI` pointed at a LiteLLM proxy. All LLM calls are proper LangChain Runnable invocations, so `RunnableConfig` (containing Langfuse callback handlers) automatically captures full input/output. Configured via `config.yaml` to point at a LiteLLM proxy (default `localhost:4000`).

### Observability

`src/observability/` provides Langfuse v3 integration via the official `langfuse.langchain.CallbackHandler`. Custom attributes (trace name, metadata, session/user IDs) are propagated via `langfuse.propagate_attributes`. The module-level `flush()` function handles trace flushing and propagation context cleanup.

### Adding New Components

- **Workflow node**: Implement `__call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]` (satisfies `BaseNode` protocol). Pass `config` to LLM calls for callback propagation. Use `NodeMixin` for shared helpers. Initialize with `WorkflowComponents`.
- **Prompt version**: Add a YAML file to `src/prompts/templates/` named `{role}_v{N}.yaml`
- **Tool**: Add implementation to `src/tools/base.py` and register in `src/tools/registry.py`
- **Skill**: Add a YAML file to `src/skills/templates/` with name, description, trigger, prompt, and tools fields
- **Context formatter**: Subclass `RoleContextFormatter` in `src/conversation/context.py` and register via `ConversationContextBuilder.register_formatter()`

## Configuration

Copy `config.yaml.example` to `config.yaml` and edit. It holds LLM connection settings (base_url, model, api_key), default prompt version, workflow limits (max_iterations), conversation history settings (max_history_turns, include_plan_detail_turns), and observability settings. `config.yaml` is gitignored to protect secrets.

Environment variables can also be loaded from a `.env` file (via `python-dotenv`). Langfuse credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) can be set via env vars or `config.yaml`. The Streamlit sidebar also allows runtime overrides.

## Code Style

- Python 3.11+
- Ruff with line-length 100, rules: E, F, I, W
- UI text is in Korean; code and variable names are in English
