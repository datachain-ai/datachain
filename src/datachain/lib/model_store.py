import inspect
from typing import Any, ClassVar

from pydantic import BaseModel


class ModelStore:
    store: ClassVar[dict[str, dict[int, type[BaseModel]]]] = {}

    @staticmethod
    def _base_name(model: type[BaseModel]) -> str:
        # Some models are generated/restored with a versioned Python class name
        # (e.g. `MyType_v1`) so that multiple versions can coexist in-process without
        # `__name__` collisions.
        #
        # `_modelstore_base_name` preserves the original/logical name (e.g. `MyType`)
        # after we alter the class name, so `ModelStore.get_name(model)` still
        # returns `"MyType@v{model._version}"` and schema serialization stays stable.
        return getattr(model, "_modelstore_base_name", model.__name__)

    @classmethod
    def get_version(cls, model: type[BaseModel]) -> int:
        if not hasattr(model, "_version"):
            return 0
        return model._version  # type: ignore[attr-defined]

    @classmethod
    def get_name(cls, model) -> str:
        base_name = cls._base_name(model)
        if (version := cls.get_version(model)) > 0:
            return f"{base_name}@v{version}"
        return base_name

    @classmethod
    def register(cls, fr: type):
        """Register a class as a data model for deserialization."""
        if (model := ModelStore.to_pydantic(fr)) is None:
            return

        base_name = cls._base_name(model)
        unique_name = model.__name__
        version = ModelStore.get_version(model)

        # Register under both:
        # - `base_name` (logical/original name, from `_modelstore_base_name` when set)
        #   so callers can resolve by original name + version, e.g.
        #   `ModelStore.get("Foo", 1)` after deserialization/restore.
        # - `unique_name` (Python class `__name__`, e.g. `Foo_v1`) so callers can also
        #   resolve by the runtime class name.
        for name in (base_name, unique_name):
            cls.store.setdefault(name, {})[version] = model

        for f_info in model.model_fields.values():
            if (anno := ModelStore.to_pydantic(f_info.annotation)) is not None:
                cls.register(anno)

    @classmethod
    def get(cls, name: str, version: int | None = None) -> type | None:
        class_dict = cls.store.get(name, None)
        if class_dict is None:
            return None
        if version is None:
            max_ver = max(class_dict.keys(), default=None)
            if max_ver is None:
                return None
            return class_dict[max_ver]
        return class_dict.get(version, None)

    @classmethod
    def parse_name_version(cls, fullname: str) -> tuple[str, int]:
        name = fullname
        version = 0

        if "@" in fullname:
            name, version_str = fullname.split("@")
            if version_str.strip() != "":
                version = int(version_str[1:])

        return name, version

    @classmethod
    def remove(cls, fr: type) -> None:
        version = fr._version  # type: ignore[attr-defined]
        base_name = cls._base_name(fr)
        unique_name = fr.__name__

        for name in (base_name, unique_name):
            if name in cls.store and version in cls.store[name]:
                del cls.store[name][version]

    @staticmethod
    def is_pydantic(val: Any) -> bool:
        return (
            not hasattr(val, "__origin__")
            and inspect.isclass(val)
            and issubclass(val, BaseModel)
        )

    @staticmethod
    def to_pydantic(val) -> type[BaseModel] | None:
        if val is None or not ModelStore.is_pydantic(val):
            return None
        return val

    @staticmethod
    def is_partial(parent_type) -> bool:
        return (
            parent_type
            and ModelStore.is_pydantic(parent_type)
            and "@" in ModelStore.get_name(parent_type)
        )

    @classmethod
    def rebuild_all(cls) -> None:
        """Ensure pydantic schemas are (re)built for all registered models.

        Uses ``force=True`` to avoid subtle cases where a deserialized class
        (e.g. from by-value cloudpickle in workers) reports built state but
        nested model field schemas aren't fully resolved yet.
        """
        visited: set[type[BaseModel]] = set()
        visiting: set[type[BaseModel]] = set()

        def visit(model: type[BaseModel]) -> None:
            if model in visited or model in visiting:
                return
            visiting.add(model)
            for field in model.model_fields.values():
                child = cls.to_pydantic(field.annotation)
                if child is not None:
                    visit(child)
            visiting.remove(model)
            model.model_rebuild(force=True)
            visited.add(model)

        for versions in cls.store.values():
            for model in versions.values():
                visit(model)
