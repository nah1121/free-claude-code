"""Message and tool format converters."""

import json
from typing import Any


def get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
    """Get attribute from object or dict."""
    if hasattr(block, attr):
        return getattr(block, attr)
    if isinstance(block, dict):
        return block.get(attr, default)
    return default


def get_block_type(block: Any) -> str | None:
    """Get block type from object or dict."""
    return get_block_attr(block, "type")


class AnthropicToOpenAIConverter:
    """Converts Anthropic message format to OpenAI format."""

    @staticmethod
    def convert_messages(
        messages: list[Any],
        *,
        include_reasoning_for_openrouter: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert a list of Anthropic messages to OpenAI format.

        When include_reasoning_for_openrouter is True, assistant messages with
        thinking blocks get reasoning_content added for OpenRouter multi-turn
        reasoning continuation.
        """
        result = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                if role == "assistant":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_assistant_message(
                            content,
                            include_reasoning_for_openrouter=include_reasoning_for_openrouter,
                        )
                    )
                elif role == "user":
                    result.extend(
                        AnthropicToOpenAIConverter._convert_user_message(content)
                    )
            else:
                result.append({"role": role, "content": str(content)})

        return result

    @staticmethod
    def _convert_assistant_message(
        content: list[Any],
        *,
        include_reasoning_for_openrouter: bool = False,
    ) -> list[dict[str, Any]]:
        """Convert assistant message blocks, preserving interleaved thinking+text order."""
        content_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                content_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "thinking":
                thinking = get_block_attr(block, "thinking", "")
                content_parts.append(f"<think>\n{thinking}\n</think>")
                if include_reasoning_for_openrouter:
                    thinking_parts.append(thinking)
            elif block_type == "tool_use":
                tool_input = get_block_attr(block, "input", {})
                tool_calls.append(
                    {
                        "id": get_block_attr(block, "id"),
                        "type": "function",
                        "function": {
                            "name": get_block_attr(block, "name"),
                            "arguments": json.dumps(tool_input)
                            if isinstance(tool_input, dict)
                            else str(tool_input),
                        },
                    }
                )

        content_str = "\n\n".join(content_parts)

        # Ensure content is never an empty string for assistant messages
        # NIM (especially Mistral models) requires non-empty content if there are no tool calls
        if not content_str and not tool_calls:
            content_str = " "

        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content_str,
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if include_reasoning_for_openrouter and thinking_parts:
            msg["reasoning_content"] = "\n".join(thinking_parts)

        return [msg]

    @staticmethod
    def _convert_user_message(content: list[Any]) -> list[dict[str, Any]]:
        """Convert user message blocks (including tool results), preserving order."""
        result: list[dict[str, Any]] = []
        text_parts: list[str] = []

        def flush_text() -> None:
            if text_parts:
                result.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts.clear()

        for block in content:
            block_type = get_block_type(block)

            if block_type == "text":
                text_parts.append(get_block_attr(block, "text", ""))
            elif block_type == "tool_result":
                flush_text()
                tool_content = get_block_attr(block, "content", "")
                if isinstance(tool_content, list):
                    tool_content = "\n".join(
                        item.get("text", str(item))
                        if isinstance(item, dict)
                        else str(item)
                        for item in tool_content
                    )
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": get_block_attr(block, "tool_use_id"),
                        "content": str(tool_content) if tool_content else "",
                    }
                )

        flush_text()
        return result

    @staticmethod
    def convert_tools(
        tools: list[Any], *, sanitize_name: bool = False
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools to OpenAI format.

        Args:
            tools: List of Anthropic tool objects
            sanitize_name: If True, replace hyphens with underscores in tool names.
                          Required for some providers (e.g. NVIDIA NIM) that don't
                          accept hyphens in function names.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name.replace("-", "_") if sanitize_name else tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,
                },
            }
            for tool in tools
        ]

    @staticmethod
    def convert_tool_choice(
        tool_choice: dict[str, Any] | str | None,
        *,
        sanitize_name: bool = False,
    ) -> dict[str, Any] | str | None:
        """Convert Anthropic tool_choice to OpenAI format.

        Anthropic format:
        - "auto" (string)
        - "any" (string)
        - {"type": "tool", "name": "tool-name"}

        OpenAI format:
        - "auto" (string)
        - "required" (for "any")
        - {"type": "function", "function": {"name": "tool_name"}}

        Args:
            tool_choice: Anthropic tool_choice specification
            sanitize_name: If True, replace hyphens with underscores in tool names

        Returns:
            OpenAI-format tool_choice or None
        """
        if tool_choice is None:
            return None

        # String values: "auto" stays "auto", "any" becomes "required"
        if isinstance(tool_choice, str):
            if tool_choice == "any":
                return "required"
            return tool_choice

        # Dict format: convert from Anthropic to OpenAI structure
        if isinstance(tool_choice, dict):
            # If it's already in OpenAI format (has "function" key), just sanitize if needed
            if "function" in tool_choice:
                if sanitize_name:
                    fn = tool_choice.get("function", {})
                    if isinstance(fn, dict) and "name" in fn:
                        return {
                            "type": "function",
                            "function": {
                                **fn,
                                "name": fn["name"].replace("-", "_"),
                            },
                        }
                return tool_choice

            # Convert from Anthropic format: {"type": "tool", "name": "..."}
            tool_name = tool_choice.get("name", "")
            if sanitize_name and isinstance(tool_name, str):
                tool_name = tool_name.replace("-", "_")

            return {
                "type": "function",
                "function": {"name": tool_name},
            }

        return tool_choice

    @staticmethod
    def sanitize_tool_choice(
        tool_choice: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Sanitize tool_choice by replacing hyphens with underscores in names.

        Handles both Anthropic-style {"type": "tool", "name": "..."} and
        OpenAI-style {"type": "function", "function": {"name": "..."}} payloads.

        DEPRECATED: Use convert_tool_choice with sanitize_name=True instead.
        """
        if tool_choice is None:
            return None

        # Create a shallow copy to avoid modifying the original
        sanitized = tool_choice.copy()

        # If tool_choice has a 'name' field, sanitize it
        if "name" in sanitized and isinstance(sanitized["name"], str):
            sanitized["name"] = sanitized["name"].replace("-", "_")

        fn = sanitized.get("function")
        if isinstance(fn, dict):
            fn_copy = fn.copy()
            if "name" in fn_copy and isinstance(fn_copy["name"], str):
                fn_copy["name"] = fn_copy["name"].replace("-", "_")
            sanitized["function"] = fn_copy

        return sanitized

    @staticmethod
    def convert_system_prompt(system: Any) -> dict[str, str] | None:
        """Convert Anthropic system prompt to OpenAI format."""
        if isinstance(system, str):
            return {"role": "system", "content": system}
        elif isinstance(system, list):
            text_parts = [
                get_block_attr(block, "text", "")
                for block in system
                if get_block_type(block) == "text"
            ]
            if text_parts:
                return {"role": "system", "content": "\n\n".join(text_parts).strip()}
        return None


def build_base_request_body(
    request_data: Any,
    *,
    default_max_tokens: int | None = None,
    include_reasoning_for_openrouter: bool = False,
    sanitize_tool_names: bool = False,
) -> dict[str, Any]:
    """Build the common parts of an OpenAI-format request body.

    Handles message conversion, system prompt, max_tokens, temperature,
    top_p, stop sequences, tools, and tool_choice. Provider-specific
    parameters (extra_body, penalties, NIM settings) are added by callers.

    Args:
        request_data: The Anthropic-format request data
        default_max_tokens: Default max_tokens if not set in request
        include_reasoning_for_openrouter: Include reasoning_content for OpenRouter
        sanitize_tool_names: Replace hyphens with underscores in tool names
    """
    from providers.common.utils import set_if_not_none

    messages = AnthropicToOpenAIConverter.convert_messages(
        request_data.messages,
        include_reasoning_for_openrouter=include_reasoning_for_openrouter,
    )

    system = getattr(request_data, "system", None)
    if system:
        system_msg = AnthropicToOpenAIConverter.convert_system_prompt(system)
        if system_msg:
            messages.insert(0, system_msg)

    body: dict[str, Any] = {"model": request_data.model, "messages": messages}

    max_tokens = getattr(request_data, "max_tokens", None)
    set_if_not_none(body, "max_tokens", max_tokens or default_max_tokens)
    set_if_not_none(body, "temperature", getattr(request_data, "temperature", None))
    set_if_not_none(body, "top_p", getattr(request_data, "top_p", None))

    stop_sequences = getattr(request_data, "stop_sequences", None)
    if stop_sequences:
        body["stop"] = stop_sequences

    tools = getattr(request_data, "tools", None)
    if tools:
        body["tools"] = AnthropicToOpenAIConverter.convert_tools(
            tools, sanitize_name=sanitize_tool_names
        )
    tool_choice = getattr(request_data, "tool_choice", None)
    if tool_choice:
        body["tool_choice"] = AnthropicToOpenAIConverter.convert_tool_choice(
            tool_choice, sanitize_name=sanitize_tool_names
        )

    return body
