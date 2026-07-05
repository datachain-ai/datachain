from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WriteConfig:
    """Normalized, backend-agnostic metadata for object writes.

    Each :class:`~datachain.client.fsspec.Client` maps these fields to its
    backend's native parameters (S3 ``s3_additional_kwargs``, GCS
    ``fixed_key_metadata``, Azure ``ContentSettings``). Fields left as ``None``
    are not sent. ``extra`` is an escape hatch of raw, backend-native kwargs for
    keys this vocabulary doesn't cover (e.g. S3 ``ACL``); it is forwarded on S3
    only — GCS and Azure raise ``NotImplementedError`` because their fsspec
    backends have no raw write-kwargs passthrough. The local filesystem has no
    notion of any of these and ignores all fields.
    """

    content_type: str | None = None
    content_disposition: str | None = None
    cache_control: str | None = None
    content_encoding: str | None = None
    metadata: Mapping[str, str] | None = None
    extra: Mapping[str, Any] | None = None

    def is_empty(self) -> bool:
        return not (
            self.content_type
            or self.content_disposition
            or self.cache_control
            or self.content_encoding
            or self.metadata
            or self.extra
        )

    def has_content_settings(self) -> bool:
        return bool(
            self.content_type
            or self.content_disposition
            or self.cache_control
            or self.content_encoding
        )
