from pydantic import BaseModel


class Usage(BaseModel):
    """Token counts reported by the model for one call."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCall(BaseModel):
    """A tool call emitted by the model, captured verbatim."""

    id: str = ""
    name: str = ""
    arguments: str = ""


def _envelope_json(resp: object) -> str:
    dump = getattr(resp, "model_dump_json", None)
    if callable(dump):
        try:
            return dump()
        except Exception:  # noqa: BLE001 - capture must not fail on an odd envelope
            return ""
    return ""


class Response(BaseModel):
    """The full, normalized output of one model call.

    Pass it as ``schema=`` to store the complete output instead of a parsed
    object: ``llm.complete(col, schema=dc.llm.Response)``. ``content`` is the
    assistant text before any parsing and ``raw`` holds the original provider
    envelope as JSON, so nothing is lost. Recover a typed object later with
    ``llm.parse(col, schema=Model)`` and no further model call.
    """

    content: str = ""
    model: str = ""
    finish_reason: str = ""
    usage: Usage = Usage()
    tool_calls: list[ToolCall] = []
    system_fingerprint: str = ""
    id: str = ""
    created: int = 0
    raw: str = ""

    @classmethod
    def from_litellm(cls, resp: object) -> "Response":
        choices = getattr(resp, "choices", None) or []
        first = choices[0] if choices else None
        message = getattr(first, "message", None)
        usage = getattr(resp, "usage", None)
        content = ""
        tool_calls: list[ToolCall] = []
        if message is not None:
            content = getattr(message, "content", "") or ""
            for tc in getattr(message, "tool_calls", None) or []:
                fn = getattr(tc, "function", None)
                tool_calls.append(
                    ToolCall(
                        id=getattr(tc, "id", "") or "",
                        name=getattr(fn, "name", "") or "",
                        arguments=getattr(fn, "arguments", "") or "",
                    )
                )
        return cls(
            content=content,
            model=getattr(resp, "model", "") or "",
            finish_reason=getattr(first, "finish_reason", "") or "",
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            tool_calls=tool_calls,
            system_fingerprint=getattr(resp, "system_fingerprint", "") or "",
            id=getattr(resp, "id", "") or "",
            created=getattr(resp, "created", 0) or 0,
            raw=_envelope_json(resp),
        )
