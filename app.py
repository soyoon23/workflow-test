"""Streamlit UI for Plan-Act Workflow."""

import streamlit as st
import yaml
from pathlib import Path

from src.llm.client import LLMClient
from src.prompts.registry import PromptRegistry
from src.tools.registry import ToolRegistry
from src.workflow.graph import stream_workflow


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "llm": {
            "base_url": "http://localhost:4000",
            "model": "openai/qwen:7b",
            "api_key": "sk-1234",
            "temperature": 0.7,
            "max_tokens": 4096,
        },
        "prompts": {
            "templates_dir": "src/prompts/templates",
        },
    }


def initialize_components(config: dict):
    """Initialize all components."""
    llm_config = config.get("llm", {})
    llm_client = LLMClient(
        base_url=llm_config.get("base_url", "http://localhost:4000"),
        model=llm_config.get("model", "openai/qwen:7b"),
        api_key=llm_config.get("api_key", "sk-1234"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
    )

    prompts_config = config.get("prompts", {})
    prompt_registry = PromptRegistry(
        templates_dir=prompts_config.get("templates_dir", "src/prompts/templates")
    )

    tool_registry = ToolRegistry()

    return llm_client, prompt_registry, tool_registry


def main():
    st.set_page_config(
        page_title="Plan-Act Workflow",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Plan-Act Workflow")
    st.markdown("LangGraph 기반 Plan and Act 워크플로우 테스트")

    # Load config
    config = load_config()

    # Initialize components
    if "llm_client" not in st.session_state:
        llm_client, prompt_registry, tool_registry = initialize_components(config)
        st.session_state.llm_client = llm_client
        st.session_state.prompt_registry = prompt_registry
        st.session_state.tool_registry = tool_registry

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")

        # LLM Settings
        st.subheader("LLM Configuration")
        base_url = st.text_input(
            "LiteLLM Base URL",
            value=config["llm"]["base_url"],
            key="base_url",
        )
        model = st.text_input(
            "Model",
            value=config["llm"]["model"],
            key="model",
        )
        api_key = st.text_input(
            "API Key",
            value=config["llm"]["api_key"],
            key="api_key",
            type="password",
        )
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=config["llm"]["temperature"],
            step=0.1,
            key="temperature",
        )

        # Update LLM client if settings changed
        if st.button("Apply LLM Settings"):
            st.session_state.llm_client = LLMClient(
                base_url=base_url,
                model=model,
                api_key=api_key,
                temperature=temperature,
            )
            st.success("LLM settings applied!")

        st.divider()

        # Prompt Settings
        st.subheader("Prompt Versions")
        prompt_registry = st.session_state.prompt_registry

        for role in prompt_registry.get_available_roles():
            versions = prompt_registry.get_available_versions(role)
            current = prompt_registry.get_active_version(role)

            selected = st.selectbox(
                f"{role.capitalize()} Prompt",
                options=versions,
                index=versions.index(current) if current in versions else 0,
                key=f"prompt_{role}",
            )

            if selected != current:
                prompt_registry.set_active_version(role, selected)

        if st.button("Reload Prompts"):
            prompt_registry.reload()
            st.success("Prompts reloaded!")

        st.divider()

        # Tools info
        st.subheader("Available Tools")
        tool_registry = st.session_state.tool_registry
        for tool_name in tool_registry.get_available_tools():
            st.write(f"• {tool_name}")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📝 User Request")
        user_request = st.text_area(
            "Enter your request:",
            height=100,
            placeholder="예: Calculate the square root of 144 and search for information about it",
        )

        run_button = st.button("🚀 Run Workflow", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 Current Prompts")
        with st.expander("View Active Prompts"):
            for role in prompt_registry.get_available_roles():
                st.markdown(f"**{role.capitalize()}:**")
                prompt = prompt_registry.get(role)
                st.code(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # Workflow execution
    if run_button and user_request:
        st.divider()
        st.subheader("🔄 Workflow Execution")

        # Progress display
        progress_container = st.container()
        result_container = st.container()

        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

            plan_expander = st.expander("📋 Plan", expanded=True)
            execution_expander = st.expander("⚡ Execution", expanded=True)

        try:
            step_count = 0
            max_steps = 10

            for event in stream_workflow(
                user_request,
                st.session_state.llm_client,
                st.session_state.prompt_registry,
                st.session_state.tool_registry,
            ):
                step_count += 1
                progress = min(step_count / max_steps, 1.0)
                progress_bar.progress(progress)

                # Get node name and state
                for node_name, state in event.items():
                    status_text.markdown(f"**Current Step:** {node_name}")

                    if node_name == "plan":
                        with plan_expander:
                            if state.get("plan"):
                                st.markdown("### Generated Plan")
                                for step in state["plan"]:
                                    status_icon = {
                                        "pending": "⏳",
                                        "in_progress": "🔄",
                                        "completed": "✅",
                                        "failed": "❌",
                                    }.get(step["status"], "❓")
                                    st.markdown(
                                        f"{status_icon} **Step {step['step_number']}:** {step['description']}"
                                    )

                    elif node_name == "act":
                        with execution_expander:
                            if state.get("plan"):
                                current_idx = state.get("current_step_index", 1) - 1
                                if current_idx >= 0 and current_idx < len(state["plan"]):
                                    step = state["plan"][current_idx]
                                    st.markdown(f"### Executed Step {step['step_number']}")
                                    st.markdown(f"**Description:** {step['description']}")
                                    if step.get("result"):
                                        st.markdown("**Result:**")
                                        st.info(step["result"])

                    elif node_name == "review":
                        with execution_expander:
                            st.markdown("### Review")
                            if state.get("is_complete"):
                                st.success("Workflow completed!")
                            if state.get("final_answer"):
                                st.markdown("**Final Answer:**")
                                st.success(state["final_answer"])

            progress_bar.progress(1.0)
            status_text.markdown("**Status:** ✅ Complete")

            # Final result
            with result_container:
                st.divider()
                st.subheader("📊 Final Result")

                # Get final state from last event
                if event:
                    for node_name, final_state in event.items():
                        if final_state.get("final_answer"):
                            st.success(final_state["final_answer"])
                        elif final_state.get("error"):
                            st.error(final_state["error"])
                        else:
                            # Show completed steps
                            st.markdown("**Completed Steps:**")
                            if final_state.get("plan"):
                                for step in final_state["plan"]:
                                    if step["status"] == "completed":
                                        st.markdown(f"✅ {step['description']}")
                                        if step.get("result"):
                                            st.caption(step["result"][:200])

        except Exception as e:
            st.error(f"Workflow execution failed: {str(e)}")
            st.exception(e)

    # Footer
    st.divider()
    st.caption(
        "💡 Tip: Configure LiteLLM proxy with `litellm --model gpt-4 --port 4000`"
    )


if __name__ == "__main__":
    main()
