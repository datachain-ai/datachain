import base64
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal, get_args

from pydantic import BaseModel

from datachain.lib.file import AudioFile, File, ImageFile, VideoFile, VideoFrame
from datachain.llm.engine import LLMError

Media = Literal["text", "image", "document"]
MEDIA_VALUES = get_args(Media)

ContentPart = dict[str, Any]


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@dataclass(frozen=True)
class Text:
    text: str

    def as_part(self) -> ContentPart:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class Image:
    data: bytes
    mime: str

    def as_part(self) -> ContentPart:
        return {
            "type": "image_url",
            "image_url": {"url": _data_uri(self.data, self.mime)},
        }


@dataclass(frozen=True)
class Document:
    data: bytes
    mime: str

    def as_part(self) -> ContentPart:
        uri = _data_uri(self.data, self.mime)
        return {"type": "file", "file": {"file_data": uri, "format": self.mime}}


Content = Text | Image | Document


def _path_image_mime(file: File) -> str | None:
    mime = mimetypes.guess_type(file.path)[0]
    return mime if mime and mime.startswith("image/") else None


def _sniff_image_mime(data: bytes) -> str | None:
    from PIL import Image as PILImage

    try:
        with PILImage.open(BytesIO(data)) as img:
            fmt = img.format
    except Exception:  # noqa: BLE001 - any open failure just means "not an image"
        return None
    return f"image/{fmt.lower()}" if fmt else None


def _binary_error(what: str, suggest_type: bool) -> LLMError:
    fix = (
        "set type='image'/'document', or decode it to text first"
        if suggest_type
        else "decode it to text first"
    )
    return LLMError(f"cannot read {what} as text (it looks binary); {fix}")


def _read_text(file: File, suggest_type: bool = True) -> str:
    try:
        return file.read_text()
    except UnicodeDecodeError as e:
        raise _binary_error(f"'{file.path}'", suggest_type) from e


def _decode(data: bytes, suggest_type: bool = True) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        raise _binary_error("the bytes", suggest_type) from e


def _as_text(value: Any, suggest_type: bool = True) -> Text:
    if isinstance(value, File):
        return Text(_read_text(value, suggest_type))
    if isinstance(value, BaseModel):
        return Text(value.model_dump_json())
    if isinstance(value, bytes):
        return Text(_decode(value, suggest_type))
    return Text(str(value))


def _as_image(value: Any) -> Image:
    if isinstance(value, VideoFrame):
        return Image(value.read_bytes(format="jpg"), "image/jpeg")
    if isinstance(value, File):
        data = value.read_bytes()
    elif isinstance(value, bytes):
        data = value
    else:
        raise LLMError(f"cannot send {type(value).__name__} as an image")
    mime = _path_image_mime(value) if isinstance(value, File) else None
    mime = mime or _sniff_image_mime(data)
    if mime is None:
        raise LLMError("the value is not a recognized image format")
    return Image(data, mime)


def _as_document(value: Any) -> Document:
    if isinstance(value, File):
        data = value.read_bytes()
    elif isinstance(value, bytes):
        data = value
    else:
        raise LLMError(f"cannot send {type(value).__name__} as a document")
    if not data.startswith(b"%PDF-"):
        raise LLMError(
            "type='document' expects a PDF, but the value is not a PDF "
            "(convert it to PDF, or extract its text first)"
        )
    return Document(data, "application/pdf")


def resolve(value: Any, type_: Media | None = None) -> Content:
    """Resolve a column value to the modality the model receives.

    With no ``type_``, the value's type decides: image types become images and
    everything else becomes text. ``type_`` overrides the modality for raw
    ``bytes`` or an untyped ``File``.
    """
    if type_ == "image":
        return _as_image(value)
    if type_ == "document":
        return _as_document(value)
    if type_ == "text":
        return _as_text(value)
    if isinstance(value, (ImageFile, VideoFrame)):
        return _as_image(value)
    if isinstance(value, (AudioFile, VideoFile)):
        raise LLMError(
            f"{type(value).__name__} cannot be sent to the model directly; decode it "
            "first (e.g. extract video frames or an audio transcript) and pass that "
            "column"
        )
    return _as_text(value)


def to_text(value: Any) -> str:
    return _as_text(value, suggest_type=False).text


def build_messages(
    prompt: str | None,
    value: Any,
    type_: Media | None = None,
    context: Any = None,
) -> list[dict[str, Any]]:
    """Build a single-user-message chat payload, collapsing to plain text when
    nothing multimodal is present."""
    parts: list[ContentPart] = []
    if prompt:
        parts.append(Text(prompt).as_part())
    parts.append(resolve(value, type_).as_part())
    if context is not None and (ctx := to_text(context)):
        parts.append(Text(f"Context:\n{ctx}").as_part())

    if all(p["type"] == "text" for p in parts):
        content: Any = "\n\n".join(p["text"] for p in parts)
        if not content.strip():
            raise LLMError(
                "cannot send an empty message; the input column rendered to "
                "empty text and there is no prompt"
            )
    else:
        content = parts
    return [{"role": "user", "content": content}]
