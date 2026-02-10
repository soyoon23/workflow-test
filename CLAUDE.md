# CLAUDE.md

Plan-Act-Review AI workflow built on LangGraph + LiteLLM with a Streamlit chat UI, supporting multi-turn conversations with skill-based routing.

## Critical Invariants

These rules apply to ALL changes. Violating them breaks observability, CI, or the UI.

1. **RunnableConfig propagation**: Always pass the `RunnableConfig` from LangGraph nodes down through every `LLMClient` call. This is how Langfuse callback handlers capture LLM I/O.
2. **Langfuse flush**: Call `src/observability.flush()` after every workflow run completes — in `app.py`, CLI scripts, and tests. Traces are lost without it.
3. **Korean UI copy**: All user-facing strings (Streamlit UI, error messages shown to users) must be in Korean unless a feature explicitly targets another locale. Code, logs, and variable names stay in English.
4. **Observability metadata normalization**: Set `skill="none"` (not `null`/empty) when no skill is active; store `auto_select` as lowercase string `"true"`/`"false"`.
5. **WorkflowComponents bundle**: Pass `WorkflowComponents` to nodes and extensions — never leak individual dependencies (LLMClient, registries) as separate arguments.

## Commands

```bash
# Install dependencies (append --extra dev in CI parity, --extra eval for evaluation suites)
uv sync

# Run the app (requires LiteLLM proxy running first)
litellm --model gpt-4 --port 4000    # Terminal 1: start LLM proxy
uv run streamlit run app.py           # Terminal 2: start Streamlit UI

# Lint and format
uv run ruff check .
uv run ruff format .                  # CI runs: uv run ruff format --check .

# Tests
uv run pytest                         # all tests
uv run pytest tests/unit -v           # unit only
uv run pytest tests/eval -v --tb=short  # eval only (requires LLM connection)
uv run pytest tests/unit/test_routing.py::test_should_continue_plan -vv  # single test
uv run pytest -k "plan_node and not eval"  # pattern filtering

# Generate workflow graph visualization
python visualize_graph.py
```

## Repository & Environment Setup

- Python 3.11+; dependency graph managed with `uv` (see `pyproject.toml` and `uv.lock`).
- Use `uv add <package>` / `uv remove <package>` instead of pip to keep `uv.lock` canonical.
- Config lives in `config.yaml`; copy from `config.yaml.example` and keep secrets in `.env` (auto-loaded via `python-dotenv` at `app.py` startup).
- LiteLLM proxy must be running before the Streamlit app or eval tests.
- Avoid committing `config.yaml`, `.env`, or secrets; they are gitignored.

## Architecture

> Detailed architecture documentation: see `agent_structure.md`

Plan-Act-Review state machine defined in `src/workflow/graph.py`. Nodes satisfy the `BaseNode` protocol (`src/workflow/nodes/base.py`), share dependencies via `WorkflowComponents` (`src/workflow/components.py`), and use `NodeMixin` for shared helpers. Three registries (Prompts, Tools, Skills) manage extensible components via YAML files. Multi-turn context is handled by `ConversationHistoryManager` and role-specific `ConversationContextBuilder` formatters in `src/conversation/`.

## Adding New Components

- **Workflow node**: Implement `__call__(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]`. Use `NodeMixin` for shared helpers. Initialize with `WorkflowComponents`.
- **Prompt version**: Add `src/prompts/templates/{role}_v{N}.yaml`.
- **Tool**: Implement in `src/tools/base.py`, register in `src/tools/registry.py` with OpenAI function-format schema.
- **Skill**: Add YAML to `src/skills/templates/` with `name`, `description`, `trigger` (must start with `/`), optional `prompt`, `tools` list. Tool names must exist in ToolRegistry.
- **Context formatter**: Subclass `RoleContextFormatter` in `src/conversation/context.py`, register via `ConversationContextBuilder.register_formatter()`.
- After modifying node wiring, run `python visualize_graph.py` and update `agent_structure.md`.

## Configuration

- Copy `config.yaml.example` to `config.yaml` and edit. It holds LLM connection, prompt version, workflow limits, conversation history settings, and observability settings.
- Langfuse credentials (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`) can be set via env vars or `config.yaml`. The Streamlit sidebar also allows runtime overrides.
- Keep Langfuse keys out of history and logs; never hardcode keys.

## Testing

- Unit tests (`tests/unit/`) do not require external LLMs and have no pytest markers — use directory-based filtering.
- Eval tests (`tests/eval/`) require LiteLLM + DeepEval and are decorated with `@pytest.mark.eval`. Export `DEEPEVAL_API_KEY` via `.env`.
- CI (`.github/workflows/test.yml`): runs Ruff lint+format check, then `pytest tests/unit/ -v`. Eval suite runs only on `main` push or manual dispatch.
- Golden datasets live in `tests/datasets/*.json`; keep them deterministic and human-readable. Multi-step goldens use `metadata.is_multistep: true` with a `steps` array.
- Fixtures: `tests/conftest.py` (shared), `tests/eval/conftest.py` (eval-scoped). Keep them deterministic.
- Prefer asserting on structured data (plans, tool outputs) over raw strings.

## Code Style

- Ruff: line-length 100, rules `E, F, I, W`. Run `uv run ruff format .` before committing.
- Every module starts with a triple-double-quoted docstring.
- Use `@dataclass` for structured data (`Skill`, `WorkflowComponents`, `StreamEvent`).
- UI logic stays in `app.py`; workflow logic in `src/workflow/`.
- Module-level `logger = logging.getLogger(__name__)`; never instantiate loggers inside functions.
- No mypy yet, but add type hints consistent with existing code. Keep Protocols and mixins lightweight.
- Workflow state keys (`plan`, `current_step_index`, `review_key_facts`) must stay consistent — reuse existing keys when extending state.
- Keep prompt placeholders descriptive (e.g., `{skill_selection_block}`) and document in `NodeMixin`.
- Tool results must be JSON-serializable with `success`/`error` fields. Propagate errors via `state["error"]` so `should_continue` can terminate gracefully.
- Emit `StreamCallback` events (`plan_ready`, `step_start`, `tool_call`, `review_ready`, `error`) in new nodes/flows so the Streamlit UI stays reactive.
- Truncate large tool results to 500 chars before rendering in UI. Keep Streamlit settings in sidebar, persist in `st.session_state`.

## Gotchas

- **Forgetting `config` in LLM calls**: If you call `self.components.llm` without passing the `RunnableConfig`, Langfuse traces silently lose that call. The call still works, so this bug is invisible until you check dashboards.
- **LiteLLM proxy must be running**: Both the Streamlit app and eval tests require the proxy at `localhost:4000`. Unit tests do not.
- **`use_skill` tool always included**: `_get_tools()` appends the `use_skill` function definition to every tool list. If you add a new tool-exposing path, include it too.
- **Max tool rounds = 3**: `ActNode` loops up to `MAX_TOOL_ROUNDS=3` tool calls per step. On the final round, tools are omitted to force a text response.
- **Eval tests cost real API calls**: Tests under `tests/eval/` invoke LLMs via LiteLLM. Never remove the `@pytest.mark.eval` marker or move eval tests into `tests/unit/`.
- **Observability metadata trimming**: Trim `user_input` to ~100 chars in metadata sent to Langfuse (see `app.py`). Long inputs bloat the dashboard.
- **Registry `.reload()`**: After modifying YAML prompt/skill files at runtime, call `.reload()` on the registry. The Streamlit UI already has buttons for this.

## Git & Workflow Hygiene

- Never force-reset or discard user changes; commit only what the task requires.
- Match formatting of existing files instead of restyling entire modules.
- Use `uv sync` to update `pyproject.toml` and `uv.lock` when adding dependencies.
- Document notable workflow changes in README when appropriate.
- Prefer smaller commits with clear messages; CI expects green lint and unit tests before merging.

## Pre-Ship Checklist

- `uv run ruff check . && uv run ruff format .` — must pass.
- Run the relevant pytest subset (`tests/unit` at minimum; `tests/eval` if you touched workflow logic).
- If you modified node wiring: `python visualize_graph.py` and verify `workflow_graph.png`.
- Smoke-test LiteLLM proxy + Streamlit if you touched workflow or UI code.
- Update this file if you add commands, skills, prompt templates, or conventions.
