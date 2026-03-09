import sys
from enum import Enum
from typing import Any

import pytest

from datachain.lib.meta_formats import (
    gen_datamodel_code,
    pick_datamodel_target_python_version,
)


@pytest.mark.parametrize(
    "major, minor, python_version_names, expected_enum_name",
    [
        (3, 9, ["PY_39", "PY_310"], "PY_39"),
        (3, 9, ["PY_310", "PY_311", "PY_312"], "PY_310"),
        (3, 11, ["PY_38", "PY_310"], "PY_310"),
        (3, 11, ["PY_38", "PY_312", "PY_313"], "PY_38"),
        (3, 14, ["PY_39", "PY_310", "PY_311"], "PY_311"),
        (3, 11, ["PY_27", "PY_280"], "PY_280"),
    ],
)
def test_pick_datamodel_target_python_version(
    major: int,
    minor: int,
    python_version_names: list[str],
    expected_enum_name: str,
):
    def _make_python_version_enum(names: list[str]) -> Any:
        return Enum("PythonVersion", {name: name for name in names})

    python_version_enum = _make_python_version_enum(python_version_names)
    chosen = pick_datamodel_target_python_version(python_version_enum, major, minor)
    assert chosen.name == expected_enum_name


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_gen_datamodel_code_smoke_real_generator(monkeypatch, tmp_path):
    import datamodel_code_generator

    from datachain.lib import meta_formats

    calls: list[tuple[object, int, int]] = []
    original_picker = meta_formats.pick_datamodel_target_python_version

    def _spy_picker(python_version_enum, major: int, minor: int):
        calls.append((python_version_enum, major, minor))
        return original_picker(python_version_enum, major, minor)

    monkeypatch.setattr(
        meta_formats, "pick_datamodel_target_python_version", _spy_picker
    )

    input_path = tmp_path / "input.json"
    input_path.write_text('{"a": 1}', encoding="utf-8")
    code = gen_datamodel_code(input_path, format="json")

    assert calls == [
        (
            datamodel_code_generator.PythonVersion,
            sys.version_info.major,
            sys.version_info.minor,
        )
    ]
    assert "DataModel.register" in code
    assert "spec =" in code
