import base64
import mimetypes
from io import BytesIO
from typing import Any

from pydantic import BaseModel

from datachain.lib.file import File, ImageFile, TextFile, VideoFrame
from datachain.llm.engine import LLMError

ContentPart = dict[str, Any]
ContentParts = list[ContentPart]


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _image_part(data: bytes, mime: str) -> ContentPart:
    return {"type": "image_url", "image_url": {"url": _data_uri(data, mime)}}


def _text_part(text: str) -> ContentPart:
    return {"type": "text", "text": text}


def _mime(file: File) -> str | None:
    return mimetypes.guess_type(file.path)[0]


def _image_bytes_mime(data: bytes) -> str | None:
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(BytesIO(data)) as img:
            fmt = img.format
    except (UnidentifiedImageError, OSError):
        return None
    return f"image/{fmt.lower()}" if fmt else None


def _read_text(file: File) -> str:
    try:
        return file.read_text()
    except UnicodeDecodeError as e:
        raise LLMError(
            f"cannot read '{file.path}' as text; convert it first "
            "(extract video frames, OCR a document, or pass an image column)"
        ) from e


def serialize_value(value: Any) -> str:
    """Render a value as plain text (for context and text embeddings)."""
    if isinstance(value, File):
        return _read_text(value)
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return str(value)


def value_to_parts(value: Any) -> ContentParts:  # noqa: PLR0911
    """Encode a column value as message content parts (text or inline image)."""
    if isinstance(value, ImageFile):
        return [_image_part(value.read_bytes(), _mime(value) or "image/jpeg")]
    if isinstance(value, VideoFrame):
        return [_image_part(value.read_bytes(format="jpg"), "image/jpeg")]
    if isinstance(value, TextFile):  # explicit text type: never MIME-routed to image
        return [_text_part(_read_text(value))]
    if isinstance(value, File):  # ambiguous type: dispatch by MIME
        mime = _mime(value)
        if mime and mime.startswith("image/"):
            return [_image_part(value.read_bytes(), mime)]
        return [_text_part(_read_text(value))]
    if isinstance(value, BaseModel):
        return [_text_part(value.model_dump_json())]
    if isinstance(value, bytes):
        mime = _image_bytes_mime(value)
        if mime:
            return [_image_part(value, mime)]
        raise LLMError(
            "raw bytes are only supported when they are a known image format"
        )
    return [_text_part(str(value))]


def build_messages(
    prompt: str | None,
    value: Any,
    context: Any = None,
) -> list[dict[str, Any]]:
    """Build a single-user-message chat payload, collapsing to plain text when
    nothing multimodal is present."""
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
