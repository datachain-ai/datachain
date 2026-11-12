import inspect
from importlib import import_module
from typing import Any, ClassVar

from pydantic import BaseModel

# Models defined in datachain register on import; `get` imports them by name.
BUILTIN_MODELS = {
    "ArrowRow": "datachain.lib.file",
    "Audio": "datachain.lib.file",
    "AudioFile": "datachain.lib.file",
    "AudioFragment": "datachain.lib.file",
    "BBox": "datachain.model.bbox",
    "DatasetInfo": "datachain.lib.dataset_info",
    "File": "datachain.lib.file",
    "HFAudio": "datachain.lib.hf",
    "HFClassLabel": "datachain.lib.hf",
    "HFImage": "datachain.lib.hf",
    "Image": "datachain.lib.file",
    "ImageFile": "datachain.lib.file",
    "Laion": "datachain.lib.webdataset_laion",
    "ListingInfo": "datachain.lib.listing_info",
    "McapFile": "datachain.lib.mcap",
    "McapMessage": "datachain.lib.mcap",
    "McapSlice": "datachain.lib.mcap",
    "McapSummary": "datachain.lib.mcap",
    "McapTopic": "datachain.lib.mcap",
    "OBBox": "datachain.model.bbox",
    "Pose": "datachain.model.pose",
    "Pose3D": "datachain.model.pose",
    "Segment": "datachain.model.segment",
    "Sys": "datachain.lib.dc.utils",
    "TextFile": "datachain.lib.file",
    "Video": "datachain.lib.file",
    "VideoFile": "datachain.lib.file",
    "VideoFragment": "datachain.lib.file",
    "VideoFrame": "datachain.lib.file",
    "WDSAllFile": "datachain.lib.webdataset",
    "WDSBasic": "datachain.lib.webdataset",
    "WDSLaion": "datachain.lib.webdataset_laion",
    "WDSReadableSubclass": "datachain.lib.webdataset",
    "YoloBBox": "datachain.model.ultralytics.bbox",
    "YoloBBoxes": "datachain.model.ultralytics.bbox",
    "YoloOBBox": "datachain.model.ultralytics.bbox",
    "YoloOBBoxes": "datachain.model.ultralytics.bbox",
    "YoloPose": "datachain.model.ultralytics.pose",
    "YoloPoses": "datachain.model.ultralytics.pose",
    "YoloSegment": "datachain.model.ultralytics.segment",
    "YoloSegments": "datachain.model.ultralytics.segment",
    "ZarrArray": "datachain.lib.zarr",
    "ZarrInfo": "datachain.lib.zarr",
    "ZarrSelection": "datachain.lib.zarr",
    "ZarrStore": "datachain.lib.zarr",
}


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
    def get(
        cls, name: str, version: int | None = None, *, module: str | None = None
    ) -> type | None:
        class_dict = cls.store.get(name, {})
        model = class_dict.get(
            max(class_dict, default=0) if version is None else version
        )
        builtin_module = BUILTIN_MODELS.get(name)
        if model is not None and (
            module is None
            or model.__module__ == module
            or ("_modelstore_base_name" in model.__dict__ and module != builtin_module)
        ):
            return model
        if builtin_module and (module is None or module == builtin_module):
            try:
                model = getattr(import_module(builtin_module), name)
            except ImportError:  # an optional dependency is not installed
                return None
            cls.register(model)
            if version is None or cls.get_version(model) == version:
                return model
        return None

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
