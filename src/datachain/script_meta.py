import re
from dataclasses import dataclass
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    # tomllib is in standard library from python 3.11 so for earlier versions
    # we need tomli
    import tomli as tomllib  # type: ignore[no-redef]


class ScriptConfigParsingError(Exception):
    def __init__(self, message):
        super().__init__(message)


@dataclass
class ScriptConfig:
    """
    Class that is parsing inline script metadata to get some basic information for
    running datachain script like python version, dependencies, attachments etc.
    Inline script metadata must follow the format described in https://packaging.python.org/en/latest/specifications/inline-script-metadata/#inline-script-metadata.
    Example of script with inline metadata:
        # /// script
        # requires-python = ">=3.12"
        #
        # dependencies = [
        #   "pandas < 2.1.0",
        #   "numpy == 1.26.4"
        # ]
        #
        # [tools.datachain.workers]
        # num_workers = 3
        #
        # [tools.datachain.attachments]
        # image1 = "s3://ldb-public/image1.jpg"
        # file1 = "s3://ldb-public/file.pdf"
        #
        # [tools.datachain.params]
        # min_length_sec = 1
        # cache = false
        #
        # [tools.datachain.inputs]
        # threshold = 0.5
        # start_ds_name = "ds://start"
        #
        # [tools.datachain.outputs]
        # result_dataset = "ds://res"
        # result_dir = "/temp"
        #
        # ///

        import sys
        import pandas as pd

        print(f"Python version: {sys.version_info}")
        print(f"Pandas version: {pd.__version__}")

    """

    python_version: str | None
    dependencies: list[str]
    attachments: dict[str, str]
    params: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    num_workers: int | None = None

    def __init__(
        self,
        python_version: str | None = None,
        dependencies: list[str] | None = None,
        attachments: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        num_workers: int | None = None,
    ):
        self.python_version = python_version
        self.dependencies = dependencies or []
        self.attachments = attachments or {}
        self.params = params or {}
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.num_workers = num_workers

    def get_param(self, name: str, default: Any) -> Any:
        return self.params.get(name, default)

    def get_input(self, name: str, default: Any) -> Any:
        return self.inputs.get(name, default)

    def get_output(self, name: str, default: Any) -> Any:
        return self.outputs.get(name, default)

    def get_attachment(self, name: str, default: Any) -> Any:
        return self.attachments.get(name, default)

    @staticmethod
    def read(script: str) -> dict | None:
        """Converts inline script metadata to dict with all found data"""
        regex = (
            r"(?m)^# \/\/\/ (?P<type>[a-zA-Z0-9-]+)[ \t]*$[\r\n|\r|\n]"
            "(?P<content>(?:^#(?:| .*)$[\r\n|\r|\n])+)^# \\/\\/\\/[ \t]*$"
        )
        name = "script"
        matches = list(
            filter(lambda m: m.group("type") == name, re.finditer(regex, script))
        )
        if len(matches) > 1:
            raise ValueError(f"Multiple {name} blocks found")
        if len(matches) == 1:
            content = "".join(
                line[2:] if line.startswith("# ") else line[1:]
                for line in matches[0].group("content").splitlines(keepends=True)
            )
            return tomllib.loads(content)
        return None

    @staticmethod
    def parse(script: str) -> "ScriptConfig | None":
        """
        Method that is parsing inline script metadata from datachain script and
        instantiating ScriptConfig class with found data. If no inline metadata is
        found, it returns None
        """
        try:
            meta = ScriptConfig.read(script)
            if not meta:
                return None
            custom = meta.get("tools", {}).get("datachain", {})
            return ScriptConfig(
                python_version=meta.get("requires-python"),
                dependencies=meta.get("dependencies"),
                num_workers=custom.get("workers", {}).get("num_workers"),
                attachments=custom.get("attachments"),
                params={k: str(v) for k, v in custom.get("params").items()}
                if custom.get("params")
                else None,
                inputs=custom.get("inputs"),
                outputs=custom.get("outputs"),
            )
        except Exception as e:
            raise ScriptConfigParsingError(
                f"Error when parsing script meta: {e}"
            ) from e
