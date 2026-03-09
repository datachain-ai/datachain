import builtins
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any, TypeVar

from datachain.error import InvalidProjectNameError
from datachain.namespace import Namespace

P = TypeVar("P", bound="Project")
PROJECT_NAME_RESERVED_CHARS = [".", "@"]


@dataclass(frozen=True)
class Project:
    id: int
    uuid: str
    name: str
    descr: str | None
    created_at: datetime
    namespace: Namespace

    @staticmethod
    def validate_name(name: str) -> None:
        """Throws exception if name is invalid, otherwise returns None"""
        if not name:
            raise InvalidProjectNameError("Project name cannot be empty")

        for c in PROJECT_NAME_RESERVED_CHARS:
            if c in name:
                raise InvalidProjectNameError(
                    f"Character {c} is reserved and not allowed in project name."
                )

        if name in [Project.default(), Project.listing()]:
            raise InvalidProjectNameError(
                f"Project name {name} is reserved and cannot be used."
            )

    @staticmethod
    def default() -> str:
        """Name of default project"""
        return "local"

    @staticmethod
    def listing() -> str:
        """Name of listing project where all listing datasets will be saved"""
        return "listing"

    @classmethod
    def parse(
        cls: builtins.type[P],
        namespace_id: int,
        namespace_uuid: str,
        namespace_name: str,
        namespace_descr: str | None,
        namespace_created_at: datetime,
        project_id: int,
        uuid: str,
        name: str,
        descr: str | None,
        created_at: datetime,
        project_namespace_id: int,
    ) -> "Project":
        namespace = Namespace.parse(
            namespace_id,
            namespace_uuid,
            namespace_name,
            namespace_descr,
            namespace_created_at,
        )

        return cls(project_id, uuid, name, descr, created_at, namespace)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Project":
        namespace = Namespace.from_dict(d.pop("namespace"))
        kwargs = {f.name: d[f.name] for f in fields(cls) if f.name in d}
        return cls(**kwargs, namespace=namespace)
