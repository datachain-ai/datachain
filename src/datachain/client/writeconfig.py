from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WriteConfig:
    """Normalized, backend-agnostic metadata for object writes.

    The first fields are portable content settings, mapped by each
    :class:`~datachain.client.fsspec.Client` to its backend's native parameters
    (S3 ``s3_additional_kwargs``, GCS ``fixed_key_metadata``, Azure
    ``ContentSettings``); ``None`` fields are not sent.

    - ``metadata`` is *custom* user-defined key/value object metadata (S3 user
      metadata / ``x-amz-meta-*``, GCS custom metadata, Azure blob metadata).
    - ``write_options`` is a raw, backend-native escape hatch for keys this
      vocabulary doesn't cover (e.g. S3 ``ACL``, ``Tagging``). It is forwarded on
      S3 only (into ``s3_additional_kwargs``); GCS and Azure raise
      ``NotImplementedError`` because their fsspec backends have no raw
      write-kwargs passthrough.

    The local filesystem has no notion of any of these and ignores all fields.
    """

    content_type: str | None = None
    content_disposition: str | None = None
    cache_control: str | None = None
    content_encoding: str | None = None
    metadata: Mapping[str, str] | None = None
    write_options: Mapping[str, Any] | None = None

    def is_empty(self) -> bool:
        return not (
            self.content_type
            or self.content_disposition
            or self.cache_control
            or self.content_encoding
            or self.metadata
            or self.write_options
        )

    def has_content_settings(self) -> bool:
        return bool(
            self.content_type
            or self.content_disposition
            or self.cache_control
            or self.content_encoding
        )
