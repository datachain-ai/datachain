import builtins
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, TypeVar

from datachain.error import InvalidNamespaceNameError

N = TypeVar("N", bound="Namespace")
NAMESPACE_NAME_RESERVED_CHARS = [".", "@"]


def parse_name(name: str) -> tuple[str, str | None]:
    """
    Parses namespace name into namespace and optional project name.
    If both namespace and project are defined in name, they need to be split by dot
    e.g dev.my-project
    Valid inputs:
        - dev.my-project
        - dev
    """
    parts = name.split(".")
    if len(parts) == 1:
        return name, None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise InvalidNamespaceNameError(
        f"Invalid namespace format: {name}. Expected 'namespace' or 'ns1.ns2'."
    )


@dataclass(frozen=True)
class Namespace:
    id: int
    uuid: str
    name: str
    descr: str | None
    created_at: datetime

    @staticmethod
    def validate_name(name: str) -> None:
        """Throws exception if name is invalid, otherwise returns None"""
        if not name:
            raise InvalidNamespaceNameError("Namespace name cannot be empty")

        for c in NAMESPACE_NAME_RESERVED_CHARS:
            if c in name:
                raise InvalidNamespaceNameError(
                    f"Character {c} is reserved and not allowed in namespace name"
                )

        if name in [Namespace.default(), Namespace.system()]:
            raise InvalidNamespaceNameError(
                f"Namespace name {name} is reserved and cannot be used."
            )

    @staticmethod
    def default() -> str:
        """Name of default namespace"""
        return "local"

    @staticmethod
    def system() -> str:
        """Name of the system namespace"""
        return "system"

    @property
    def is_system(self):
        return self.name == Namespace.system()

    @classmethod
    def parse(
        cls: builtins.type[N],
        id: int,
        uuid: str,
        name: str,
        descr: str | None,
        created_at: datetime,
    ) -> "Namespace":
        return cls(id, uuid, name, descr, created_at)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Namespace":
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs)
