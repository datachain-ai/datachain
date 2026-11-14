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
from datachain.lib.model_store import BUILTIN_MODELS, ModelStore

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


def _run(
    code: str, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        cwd=cwd,
        env={**os.environ, **env} if env is not None else None,
    )


def _check(
    code: str, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> str:
    proc = _run(code, *args, cwd=cwd, env=env)
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


def test_star_import_and_version_in_fresh_process():
    out = _check(
        "from datachain import *\n"
        "import datachain\n"
        "assert all(n in globals() for n in datachain.__all__)\n"
        "print(datachain.__version__)\n"
    )
    assert out.strip()


@pytest.mark.parametrize(
    "entries",
    [
        ("datachain:read_values", "datachain:func"),
        ("datachain.query.dataset", "datachain:func"),
        ("datachain:File", "datachain:bucket_status"),
        ("datachain.data_storage:AbstractMetastore", "datachain.query:Session"),
        ("datachain.query.dataset", "datachain.query:Session"),
        ("datachain:Mapper", "datachain:func"),
        ("datachain.query.dispatch", "datachain.query:Session"),
        ("datachain:C", "datachain.cli:describe_dataset_cmd"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="Recent CPython import-lock behavior"
)
def test_concurrent_first_imports(entries, reverse):
    _check(
        "import importlib, sys\n"
        "import datachain\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "from threading import Barrier\n"
        "gate = Barrier(2)\n"
        "def first_use(target):\n"
        "    gate.wait(timeout=10)\n"
        "    module_name, _, name = target.partition(':')\n"
        "    module = importlib.import_module(module_name)\n"
        "    return getattr(module, name) if name else module\n"
        "with ThreadPoolExecutor(2) as pool:\n"
        "    list(pool.map(first_use, sys.argv[1:]))\n",
        *(reversed(entries) if reverse else entries),
    )


def test_direct_session_import_composes_with_lazy_access():
    _check(
        """
import builtins, faulthandler, importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Event, current_thread
import datachain, datachain.lazy as lazy
direct_started, lazy_started = Event(), Event()
original_import, original_load = builtins.__import__, lazy.import_module
def importing(name, *args, **kwargs):
    if name == "datachain.data_storage" and current_thread().name.startswith("direct"):
        direct_started.set()
        assert lazy_started.wait(timeout=10)
    return original_import(name, *args, **kwargs)
def load(name):
    if name == "datachain.query.session":
        lazy_started.set()
    return original_load(name)
builtins.__import__, lazy.import_module = importing, load
faulthandler.dump_traceback_later(20, exit=True)
with ThreadPoolExecutor(1, thread_name_prefix="direct") as direct:
    with ThreadPoolExecutor(1) as indirect:
        raw = direct.submit(importlib.import_module, "datachain.query.session")
        assert direct_started.wait(timeout=10)
        exported = indirect.submit(lambda: datachain.Session)
        assert raw.result(timeout=20).Session is exported.result(timeout=20)
faulthandler.cancel_dump_traceback_later()
"""
    )


@pytest.mark.parametrize(
    "package_name,target,name",
    [
        ("datachain", "datachain.lib.file", "File"),
        ("datachain.query", "datachain.query.params", "param"),
        ("datachain.cli", "datachain.cli.commands.du", "du"),
        ("datachain.data_storage", "datachain.data_storage.job", "JobStatus"),
    ],
)
def test_reloaded_package_refreshes_cached_exports(package_name, target, name):
    _check(
        "import importlib, sys\n"
        "package_name, target, name = sys.argv[1:]\n"
        "package = importlib.import_module(package_name)\n"
        "previous = getattr(package, name)\n"
        "implementation = importlib.import_module(target)\n"
        "importlib.reload(implementation)\n"
        "importlib.reload(package)\n"
        "assert getattr(package, name) is getattr(implementation, name)\n"
        "assert getattr(package, name) is not previous\n",
        package_name,
        target,
        name,
    )


@pytest.mark.parametrize("name", ["Image", "File", "BBox"])
def test_cold_read_preserves_registered_user_class(tmp_path, name):
    model = (
        "import datachain as dc\n"
        f"class {name}(dc.DataModel):\n"
        "    score: float\n"
        "    def label(self):\n"
        "        return 'user-model'\n"
    )
    env = {"DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")}
    _check(
        model + f"dc.read_values(value=[{name}(score=0.9)]).save('custom')\n",
        cwd=tmp_path,
        env=env,
    )
    _check(
        model
        + "value = dc.read_dataset('custom').to_values('value')[0]\n"
        + f"assert isinstance(value, {name})\n"
        + "assert value.score == 0.9\n"
        + "assert value.label() == 'user-model'\n",
        cwd=tmp_path,
        env=env,
    )


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


def test_model_store_miss_imports_no_builtin_module():
    _check(
        "import sys\n"
        "from datachain.lib.model_store import BUILTIN_MODELS, ModelStore\n"
        "assert ModelStore.get('NoSuchModel') is None\n"
        "assert ModelStore.get('InfoPartial_abc', 1) is None\n"
        "loaded = set(BUILTIN_MODELS.values()) & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )


def _fake_package(targets: dict[str, str]) -> types.ModuleType:
    module = types.ModuleType("fake_lazy_package")
    lazy_exports(vars(module), targets)
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


def test_version_is_the_installed_distribution_version():
    from importlib.metadata import version

    assert datachain.__version__ == version("datachain")


def test_cli_main_after_command_module_is_imported(tmp_path):
    _check(
        "import sys, datachain.cli.commands.ls\n"
        "from datachain.cli import main\n"
        "sys.exit(main(['ls', sys.argv[1]]))\n",
        str(tmp_path),
        cwd=tmp_path,
        env={"DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")},
    )


def test_modules_import_with_a_clean_datachain_cache(record_property):
    result = json.loads(
        _check(
            """
import importlib, json, sys
models, skipped = {}, {}
for name in json.loads(sys.argv[1]):
    for key in list(sys.modules):
        if key == "datachain" or key.startswith("datachain."):
            del sys.modules[key]
    try:
        importlib.import_module(name)
    except ImportError as exc:
        missing = exc.__cause__ or exc
        if not isinstance(missing, ModuleNotFoundError):
            raise
        if missing.name and missing.name.startswith("datachain"):
            raise
        skipped[name] = str(missing)
        continue
    from datachain.lib.model_store import ModelStore
    models[name] = sorted({
        key for key, versions in ModelStore.store.items()
        for model in versions.values()
        if model.__module__ == name and model.__name__ == key
    })
print(json.dumps({"models": models, "skipped": skipped}))
""",
            json.dumps(_modules()),
        )
    )
    for module, names in result["models"].items():
        assert {name: BUILTIN_MODELS.get(name) for name in names} == dict.fromkeys(
            names, module
        )
    record_property("missing_optional_dependencies", json.dumps(result["skipped"]))


@pytest.mark.usefixtures("empty_model_store")
@pytest.mark.parametrize("name,module", sorted(BUILTIN_MODELS.items()))
def test_builtin_models_resolve_from_an_empty_store(name, module):
    try:
        expected = getattr(importlib.import_module(module), name)
    except ImportError as exc:
        missing = exc.__cause__ or exc
        if not isinstance(missing, ModuleNotFoundError):
            raise
        if missing.name and missing.name.startswith("datachain"):
            raise
        pytest.skip(f"optional dependency is not installed: {missing}")
    ModelStore.store.clear()
    assert ModelStore.get(name) is expected


def test_builtin_model_resolves_before_its_module_is_imported():
    _check(
        "import sys\n"
        "from datachain.lib.model_store import ModelStore\n"
        "assert 'datachain.lib.mcap' not in sys.modules\n"
        "model = ModelStore.get('McapFile', 1)\n"
        "from datachain.lib.mcap import McapFile\n"
        "assert model is McapFile\n"
    )


def test_saved_models_restore_in_a_fresh_process(tmp_path):
    env = {"DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")}
    _check(
        "import datachain as dc\n"
        "class BBox(dc.DataModel):\n"
        "    score: float\n"
        "dc.read_values(b=[BBox(score=0.9)]).save('boxes')\n"
        "dc.read_values(f=[dc.McapFile(path='a.mcap', source='file:///tmp')])"
        ".save('mcap_files')\n",
        cwd=tmp_path,
        env=env,
    )
    _check(
        "import datachain as dc\n"
        "boxes = dc.read_dataset('boxes')\n"
        "assert boxes.to_values('b.score') == [0.9]\n"
        "assert set(boxes.signals_schema.values['b'].model_fields) == {'score'}\n"
        "files = dc.read_dataset('mcap_files')\n"
        "from datachain.lib.mcap import McapFile\n"
        "assert files.signals_schema.values['f'] is McapFile\n"
        "assert files.to_values('f')[0].path == 'a.mcap'\n",
        cwd=tmp_path,
        env=env,
    )


@pytest.mark.parametrize(
    "argv,name",
    [
        (["ls", "s3://bucket"], "ls"),
        (["show", "ds"], "show"),
        (["du", "s3://bucket"], "du"),
        (["index", "s3://bucket"], "index"),
        (["skill", "list"], "list_skills"),
        (["skill", "install"], "install_skills"),
        (["completion"], "completion"),
        (["clear-cache"], "clear_cache"),
        (["gc"], "garbage_collect"),
        (["auth", "team"], "process_auth_cli_args"),
        (["job", "ls"], "process_jobs_args"),
        (["pipeline", "list"], "process_pipeline_args"),
    ],
)
def test_cli_dispatch_resolves_only_the_selected_command(monkeypatch, argv, name):
    from datachain import cli, lazy

    args = cli.get_parser().parse_args(argv)
    target, _, attr = cli._LAZY[name].partition(":")
    imported, calls = [], []
    original_import = lazy.import_module

    def command(*args, **kwargs):
        calls.append((args, kwargs))
        return 0

    def load(module):
        imported.append(module)
        if module == target:
            return types.SimpleNamespace(**{attr: command})
        return original_import(module)

    monkeypatch.setitem(vars(cli), name, None)
    monkeypatch.delitem(vars(cli), name)
    monkeypatch.setattr(lazy, "import_module", load)
    assert cli.handle_command(args, "catalog", {}) == 0
    assert imported == [target]
    assert len(calls) == 1


def test_offline_cli_main_does_not_load_catalog_or_studio(tmp_path):
    _check(
        "import sys\n"
        "from datachain.cli import main\n"
        "assert main(['skill', 'list']) == 0\n"
        "assert 'datachain.catalog' not in sys.modules\n"
        "assert 'datachain.studio' not in sys.modules\n",
        cwd=tmp_path,
        env={"DATACHAIN_ROOT_DIR": str(tmp_path / ".datachain")},
    )
