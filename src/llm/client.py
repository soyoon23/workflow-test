"""LiteLLM client wrapper for Plan-Act workflow."""

import json
from typing import Callable, Optional

import litellm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


class LLMClient:
    """Client for LLM calls via LiteLLM proxy."""

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        model: str = "openai/qwen:7b",
        api_key: str = "sk-1234",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure LiteLLM to use proxy
        litellm.api_base = base_url
        litellm.api_key = api_key

    def _convert_messages(self, messages: list) -> list[dict]:
        """Convert LangChain messages to OpenAI format."""
        converted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                converted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                converted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                message_dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    message_dict["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"]),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                converted.append(message_dict)
            elif isinstance(msg, ToolMessage):
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content,
                    }
                )
        return converted

    def chat(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> AIMessage:
        """Send chat completion request."""
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.base_url,
        }

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response = litellm.completion(**kwargs)

        # Extract response
        choice = response.choices[0]
        message = choice.message

        # Build AIMessage
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                )

        if tool_calls:
            return AIMessage(content=message.content or "", tool_calls=tool_calls)
        return AIMessage(content=message.content or "")

    async def achat(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[str] = None,
    ) -> AIMessage:
        """Async chat completion request."""
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.base_url,
        }

        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice

        response = await litellm.acompletion(**kwargs)

        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "args": json.loads(tc.function.arguments),
                    }
                )

        if tool_calls:
            return AIMessage(content=message.content or "", tool_calls=tool_calls)
        return AIMessage(content=message.content or "")

    def stream(
        self,
        messages: list,
        tools: Optional[list[dict]] = None,
    ):
        """Stream chat completion response."""
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.base_url,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        response = litellm.completion(**kwargs)

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def stream_with_callback(
        self,
        messages: list,
        on_token: Callable[[str], None],
        tools: Optional[list[dict]] = None,
    ) -> AIMessage:
        """Stream LLM response, calling on_token for each content chunk.

        Returns the complete AIMessage (with tool_calls if any) after
        streaming completes. This allows callers to both stream tokens
        to the UI AND get the full structured response for further processing.
        """
        converted_messages = self._convert_messages(messages)

        kwargs = {
            "model": self.model,
            "messages": converted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_base": self.base_url,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        response = litellm.completion(**kwargs)

        full_content = ""
        tool_calls_accumulator: dict[int, dict] = {}

        for chunk in response:
            delta = chunk.choices[0].delta

            if delta.content:
                full_content += delta.content
                on_token(delta.content)

            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_accumulator:
                        tool_calls_accumulator[idx] = {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc_delta.id:
                        tool_calls_accumulator[idx]["id"] = tc_delta.id
                    if hasattr(tc_delta, "function") and tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_accumulator[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_accumulator[idx]["arguments"] += tc_delta.function.arguments

        tool_calls = []
        for idx in sorted(tool_calls_accumulator.keys()):
            tc = tool_calls_accumulator[idx]
            try:
                args = json.loads(tc["arguments"]) if tc["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                {
                    "id": tc["id"],
                    "name": tc["name"],
                    "args": args,
                }
            )

        if tool_calls:
            return AIMessage(content=full_content, tool_calls=tool_calls)
        return AIMessage(content=full_content)
