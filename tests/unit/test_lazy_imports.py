import ast
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

import datachain
from datachain.lazy import lazy_exports
from datachain.lib.model_store import BUILTIN_MODELS

SRC = Path(datachain.__file__).parent

# Standalone scripts that import their siblings by bare name.
NOT_MODULES = ("datachain.__main__", "datachain.skill.knowledge.scripts.")

LAZY_PACKAGES = (
    "datachain",
    "datachain.cli",
    "datachain.data_storage",
    "datachain.query",
)

RESOLVE_NAMES = """
import importlib, sys
package_name, style = sys.argv[1:]
package = importlib.import_module(package_name)
for name in package._LAZY:
    if style == "from":
        scope = {}
        exec(f"from {package_name} import {name} as value", scope)
        value = scope["value"]
    else:
        value = getattr(package, name)
    module_name, _, attr = package._LAZY[name].partition(":")
    module = importlib.import_module(module_name)
    assert value is (getattr(module, attr) if attr else module), name
    assert vars(package)[name] is value, name
"""


def _run(code: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _check(code: str) -> str:
    proc = _run(code)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _modules() -> list[str]:
    names = []
    for path in sorted(SRC.rglob("*.py")):
        parts = path.relative_to(SRC.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        name = ".".join(parts)
        if not name.startswith(NOT_MODULES):
            names.append(name)
    return names


def _type_checking_names(package_name: str) -> set[str]:
    path = importlib.util.find_spec(package_name).origin  # type: ignore[union-attr]
    tree = ast.parse(Path(path).read_text())  # type: ignore[arg-type]
    names = set()
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "TYPE_CHECKING":
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom):
                    names |= {alias.asname or alias.name for alias in stmt.names}
                elif isinstance(stmt, ast.AnnAssign):
                    names.add(ast.unparse(stmt.target))
    return names


@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
def test_lazy_map_matches_type_checking_imports_and_all(package_name):
    package = importlib.import_module(package_name)
    assert _type_checking_names(package_name) == set(package._LAZY)
    exported = getattr(package, "__all__", None)
    if exported is not None:
        assert sorted(exported) == sorted(set(package._LAZY) - {"__version__"})
    assert set(package._LAZY) <= set(dir(package))


@pytest.mark.parametrize("package_name", LAZY_PACKAGES)
@pytest.mark.parametrize("style", ["from", "attribute"])
def test_lazy_names_resolve_in_fresh_process(package_name, style):
    proc = _run(RESOLVE_NAMES, package_name, style)
    assert proc.returncode == 0, proc.stderr


def test_cli_command_names_are_functions_after_their_module_is_imported():
    _check(
        "import inspect\n"
        "import datachain.cli.commands.du, datachain.cli.commands.index\n"
        "import datachain.cli.commands.ls, datachain.cli.commands.show\n"
        "from datachain.cli import du, index, ls, show\n"
        "assert all(inspect.isfunction(f) for f in (du, index, ls, show))\n"
    )


def test_cli_main_after_command_module_is_imported(tmp_path):
    code = (
        "import sys, datachain.cli.commands.ls\n"
        "from datachain.cli import main\n"
        "sys.exit(main(['ls', sys.argv[1]]))\n"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={**os.environ, "DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")},
    )
    assert proc.returncode == 0, proc.stderr


def test_star_import_and_version_in_fresh_process():
    out = _check(
        "from datachain import *\n"
        "import datachain\n"
        "assert all(n in globals() for n in datachain.__all__)\n"
        "print(datachain.__version__)\n"
    )
    assert out.strip()


def test_import_datachain_loads_only_stdlib():
    out = _check(
        "import json, sys\n"
        "before = set(sys.modules)\n"
        "import datachain\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    loaded = json.loads(out)
    not_stdlib = [m for m in loaded if m.split(".")[0] not in sys.stdlib_module_names]
    assert sorted(not_stdlib) == ["datachain", "datachain.lazy"]
    assert "importlib.metadata" not in loaded


@pytest.mark.parametrize(
    "entry,forbidden",
    [
        (
            "datachain.client.listing_worker",
            [
                "datachain.catalog",
                "datachain.func",
                "datachain.lib.dc",
                "datachain.query.dataset",
            ],
        ),
        (
            "datachain.cli",
            [
                "datachain.catalog",
                "datachain.lib.dc",
                "datachain.studio",
                "pydantic",
                "sqlalchemy",
            ],
        ),
        ("datachain.lib.model_store", ["datachain.lib.file", "sqlalchemy"]),
    ],
)
def test_entry_point_does_not_import(entry, forbidden):
    out = _check(
        f"import json, sys\nimport {entry}\nprint(json.dumps(sorted(sys.modules)))\n"
    )
    assert sorted(set(json.loads(out)) & set(forbidden)) == []


MODELS_DEFINED_HERE = """
import json, sys
from datachain.lib.model_store import ModelStore
module = sys.argv[1]
print(json.dumps(sorted(
    name
    for name, versions in ModelStore.store.items()
    for model in versions.values()
    if model.__module__ == module and model.__name__ == name
)))
"""


@pytest.mark.parametrize("module", _modules())
def test_module_imports_first_in_fresh_process(module):
    proc = _run(f"import {module}\n{MODELS_DEFINED_HERE}", module)
    if proc.returncode and "ModuleNotFoundError: No module named" in proc.stderr:
        missing = proc.stderr.rsplit("No module named", 1)[1].split("'")[1]
        if not missing.startswith("datachain"):
            pytest.skip(f"optional dependency {missing} is not installed")
    assert proc.returncode == 0, proc.stderr
    models = json.loads(proc.stdout.splitlines()[-1])
    assert {name: BUILTIN_MODELS.get(name) for name in models} == dict.fromkeys(
        models, module
    )


@pytest.mark.parametrize("module", sorted(set(BUILTIN_MODELS.values())))
def test_builtin_models_resolve_before_their_module_is_imported(module):
    names = sorted(name for name, target in BUILTIN_MODELS.items() if target == module)
    _check(
        "import importlib, sys\n"
        "from datachain.lib.model_store import ModelStore\n"
        f"assert {module!r} not in sys.modules\n"
        f"models = {{name: ModelStore.get(name) for name in {names!r}}}\n"
        f"module = importlib.import_module({module!r})\n"
        "for name, model in models.items():\n"
        "    assert model is getattr(module, name), (name, model)\n"
    )


def test_model_store_miss_imports_no_builtin_module():
    _check(
        "import sys\n"
        "from datachain.lib.model_store import BUILTIN_MODELS, ModelStore\n"
        "assert ModelStore.get('NoSuchModel') is None\n"
        "assert ModelStore.get('InfoPartial_abc', 1) is None\n"
        "loaded = set(BUILTIN_MODELS.values()) & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )


def test_builtin_model_resolves_after_store_is_cleared(monkeypatch):
    from datachain.lib.mcap import McapFile
    from datachain.lib.model_store import ModelStore

    monkeypatch.setattr(ModelStore, "store", {})
    assert ModelStore.get("McapFile", 1) is McapFile
    assert ModelStore.get("NoSuchModel") is None


def test_saved_builtin_model_restores_its_class_in_fresh_process(tmp_path):
    code = (
        "import sys\n"
        "import datachain as dc\n"
        "if sys.argv[1] == 'save':\n"
        "    f = dc.McapFile(path='a.mcap', source='file:///tmp')\n"
        "    dc.read_values(f=[f]).save('mcap_files')\n"
        "else:\n"
        "    file_type = dc.read_dataset('mcap_files').signals_schema.values['f']\n"
        "    from datachain.lib.mcap import McapFile\n"
        "    assert file_type is McapFile, file_type\n"
    )
    env = {"DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")}
    for step in ("save", "load"):
        proc = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code, step],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
            env={**os.environ, **env},
        )
        assert proc.returncode == 0, proc.stderr


def _fake_package(targets: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType("fake_lazy_package")
    module.__getattr__, module.__dir__ = lazy_exports(  # type: ignore[method-assign]
        module.__name__, vars(module), targets
    )
    return module


def test_lazy_exports_resolves_and_caches():
    package = _fake_package({"dumps": "json:dumps", "jsonmod": "json"})
    assert package.dumps is json.dumps
    assert vars(package)["dumps"] is json.dumps
    assert package.jsonmod is json
    assert {"dumps", "jsonmod"} <= set(dir(package))


def test_lazy_exports_unknown_name_is_attribute_error():
    package = _fake_package({})
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        package.missing  # noqa: B018
    assert not hasattr(package, "missing")


def test_lazy_exports_missing_module_raises_module_not_found():
    package = _fake_package({"thing": "datachain.no_such_module:thing"})
    with pytest.raises(ModuleNotFoundError, match=r"datachain\.no_such_module"):
        package.thing  # noqa: B018


def test_lazy_exports_missing_attribute_raises_import_error(monkeypatch):
    package = _fake_package({"thing": "json:no_such_name"})
    monkeypatch.setitem(sys.modules, package.__name__, package)
    with pytest.raises(ImportError, match="no_such_name") as exc_info:
        from fake_lazy_package import thing  # noqa: F401
    assert isinstance(exc_info.value.__cause__, AttributeError)
