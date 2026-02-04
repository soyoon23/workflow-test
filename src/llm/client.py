"""LangChain ChatOpenAI wrapper for Plan-Act workflow.

Uses ChatOpenAI pointed at the LiteLLM proxy so that every LLM call is
a proper LangChain Runnable invocation.  When LangGraph passes a
RunnableConfig (containing Langfuse callback handlers) through to these
methods, the callbacks automatically capture full input/output — including
system prompts and user prompts.
"""

from typing import Callable, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI


class LLMClient:
    """Client for LLM calls via LangChain ChatOpenAI + LiteLLM proxy."""

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        model: str = "openai/qwen:7b",
        api_key: str = "sk-1234",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        # ChatOpenAI expects an OpenAI-compatible /v1 base URL
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized += "/v1"

        self._model = ChatOpenAI(
            base_url=normalized,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AIMessage:
        """Send chat completion request."""
        llm = self._model
        if tools:
            bind_kwargs: dict = {"tools": tools}
            if tool_choice:
                bind_kwargs["tool_choice"] = tool_choice
            llm = llm.bind(**bind_kwargs)

        return llm.invoke(messages, config=config)

    async def achat(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AIMessage:
        """Async chat completion request."""
        llm = self._model
        if tools:
            bind_kwargs: dict = {"tools": tools}
            if tool_choice:
                bind_kwargs["tool_choice"] = tool_choice
            llm = llm.bind(**bind_kwargs)

        return await llm.ainvoke(messages, config=config)

    def stream(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
        config: Optional[RunnableConfig] = None,
    ):
        """Stream chat completion response."""
        llm = self._model
        if tools:
            llm = llm.bind(tools=tools)

        for chunk in llm.stream(messages, config=config):
            if chunk.content:
                yield chunk.content

    def stream_with_callback(
        self,
        messages: list,
        on_token: Callable[[str], None],
        tools: Optional[list[dict]] = None,
        config: Optional[RunnableConfig] = None,
    ) -> AIMessage:
        """Stream LLM response, calling on_token for each content chunk.

        Returns the complete AIMessage (with tool_calls if any) after
        streaming completes. This allows callers to both stream tokens
        to the UI AND get the full structured response for further processing.
        """
        llm = self._model
        if tools:
            llm = llm.bind(tools=tools)

        full = None
        for chunk in llm.stream(messages, config=config):
            if chunk.content:
                on_token(chunk.content)
            full = chunk if full is None else full + chunk

        if full is None:
            return AIMessage(content="")

        return AIMessage(
            content=full.content or "",
            tool_calls=full.tool_calls if full.tool_calls else [],
        )
