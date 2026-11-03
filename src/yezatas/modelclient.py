"""Meta's Model API (OpenAI-compatible Chat Completions) behind the agent's `ModelClient` protocol.

Thin on purpose: one call, the reply mapped to `ModelReply`, errors mapped to `ModelError`.
The API key is only handed to the `openai` client; it is never logged or put in an error message.
"""

from __future__ import annotations

from typing import Any

import openai

from . import agent
from .agent import ModelError, ModelReply, ToolCall

AUTH_MESSAGE = "The Meta API key was refused. Check it in Configuration."


class MetaModelClient:
    def __init__(self, api_key: str, model: str, base_url: str = agent.MODEL_BASE_URL,
                 timeout: float = 120, local: bool = False) -> None:
        self.model = model
        self.local = local  # Muse Glimmer through Ollama on this computer: the messages say so
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelReply:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        try:
            response = self._client.chat.completions.create(**kwargs)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ModelError("auth", AUTH_MESSAGE) from exc
        except openai.RateLimitError as exc:
            raise ModelError("rate_limit",
                             "The Meta API is limiting requests. Try again in a minute.") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            if self.local:
                raise ModelError("network", "Could not reach Muse Glimmer on this computer. "
                                            "Start Ollama with that model and try again.") from exc
            raise ModelError("network", "Could not reach the Meta API. Check your "
                                        "internet connection and try again.") from exc
        except openai.APIError as exc:
            status = getattr(exc, "status_code", None)
            detail = " ".join(str(getattr(exc, "message", "") or "").split())[:200]
            text = ("Muse Glimmer (Ollama) returned an error" if self.local
                    else "The Meta API returned an error") + (f" (HTTP {status})" if status else "")
            raise ModelError("api", f"{text}: {detail}" if detail else f"{text}.") from exc
        return _to_reply(response)


def _to_reply(response: Any) -> ModelReply:
    if not getattr(response, "choices", None):
        raise ModelError("api", "The Meta API returned no answer.")
    message = response.choices[0].message
    content = message.content
    calls = tuple(ToolCall(c.id, c.function.name, c.function.arguments or "")
                  for c in (message.tool_calls or ()))
    wire: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        wire["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in calls]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return ModelReply(
        content=content,
        tool_calls=calls,
        message=wire,
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        cached_tokens=int(getattr(details, "cached_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
    )
