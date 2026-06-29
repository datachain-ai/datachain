import base64
import mimetypes
from typing import Any

from pydantic import BaseModel

from datachain.lib.file import File, ImageFile, TextFile, VideoFrame

# Content parts are the OpenAI-style chat message parts LiteLLM forwards to providers.
TextPart = dict[str, Any]
ContentParts = list[TextPart]

_IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff"}


def _data_uri(data: bytes, mime: str) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _image_part(data: bytes, mime: str) -> TextPart:
    return {"type": "image_url", "image_url": {"url": _data_uri(data, mime)}}


def _text_part(text: str) -> TextPart:
    return {"type": "text", "text": text}


def _file_mime(file: File, default: str) -> str:
    mime, _ = mimetypes.guess_type(file.path)
    return mime or default


def serialize_value(value: Any) -> str:
    """Render a value as plain text (used for context and text-only models)."""
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, TextFile):
        return value.read_text()
    if isinstance(value, File):
        return value.read_text()
    return str(value)


def value_to_parts(value: Any) -> ContentParts:  # noqa: PLR0911
    """Encode a single column value into one or more message content parts."""
    if isinstance(value, ImageFile):
        return [_image_part(value.read_bytes(), _file_mime(value, "image/jpeg"))]
    if isinstance(value, VideoFrame):
        return [_image_part(value.read_bytes(format="jpg"), "image/jpeg")]
    if isinstance(value, TextFile):
        return [_text_part(value.read_text())]
    if isinstance(value, File):
        ext = value.get_file_ext().lower()
        if ext in _IMAGE_EXTS:
            return [_image_part(value.read_bytes(), _file_mime(value, "image/jpeg"))]
        return [_text_part(value.read_text())]
    if isinstance(value, BaseModel):
        return [_text_part(value.model_dump_json())]
    return [_text_part(str(value))]


def build_messages(
    prompt: str | None,
    value: Any,
    context: Any = None,
) -> list[dict[str, Any]]:
    """Build a single-user-message chat payload from a prompt, a value, and context.

    Collapses to a plain string content when nothing multimodal is present so that
    text-only providers receive the simplest possible payload.
    """
    parts: ContentParts = []
    if prompt:
        parts.append(_text_part(prompt))
    parts.extend(value_to_parts(value))
    if context is not None:
        parts.append(_text_part(f"Context:\n{serialize_value(context)}"))

    if all(p["type"] == "text" for p in parts):
        content: Any = "\n\n".join(p["text"] for p in parts)
    else:
        content = parts
    return [{"role": "user", "content": content}]
