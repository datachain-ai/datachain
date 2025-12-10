import dataclasses
import io
import math
import os
import posixpath
import tarfile
from collections.abc import Callable
from string import printable
from tarfile import DIRTYPE, TarInfo
from time import sleep, time
from typing import Any

import pytest
import sqlalchemy as sa
from PIL import Image

import datachain as dc
from datachain.catalog.catalog import Catalog
from datachain.dataset import DatasetDependency, DatasetRecord
from datachain.lib.tar import process_tar
from datachain.query import C

DEFAULT_TREE: dict[str, Any] = {
    "description": "Cats and Dogs",
    "cats": {
        "cat1": "meow",
        "cat2": "mrow",
    },
    "dogs": {
        "dog1": "woof",
        "dog2": "arf",
        "dog3": "bark",
        "others": {"dog4": "ruff"},
    },
}

# Need to run in a distributed mode to at least have a decent amount of tasks
# Has the same structure as the DEFAULT_TREE - cats and dogs
LARGE_TREE: dict[str, Any] = {
    "description": "Cats and Dogs",
    "cats": {f"cat{i}": "a" * i for i in range(1, 128)},
    "dogs": {
        **{f"dogs{i}": "a" * i for i in range(1, 64)},
        "others": {f"dogs{i}": "a" * i for i in range(64, 98)},
    },
}

NUM_TREE = {f"{i:06d}": f"{i}" for i in range(1024)}


def instantiate_tree(path, tree):
    for key, value in tree.items():
        if isinstance(value, str):
            (path / key).write_text(value)
        elif isinstance(value, bytes):
            (path / key).write_bytes(value)
        elif isinstance(value, dict):
            (path / key).mkdir()
            instantiate_tree(path / key, value)
        else:
            raise TypeError(f"{value=}")


def tree_from_path(path, binary=False):
    tree = {}
    for child in path.iterdir():
        if child.is_dir():
            tree[child.name] = tree_from_path(child, binary)
        else:  # noqa: PLR5501
            if binary:
                tree[child.name] = child.read_bytes()
            else:
                tree[child.name] = child.read_text()
    return tree


def make_tar(tree) -> bytes:
    with io.BytesIO() as tmp:
        with tarfile.open(fileobj=tmp, mode="w") as archive:
            write_tar(tree, archive)
        return tmp.getvalue()


def write_tar(tree, archive, curr_dir=""):
    for key, value in tree.items():
        name = f"{curr_dir}/{key}" if curr_dir else key
        if isinstance(value, str):
            value = value.encode("utf-8")
        if isinstance(value, bytes):
            info = TarInfo(name)
            info.size = len(value)
            f = io.BytesIO(value)
            archive.addfile(info, f)
        elif isinstance(value, dict):
            info = TarInfo(name)
            info.type = DIRTYPE
            archive.addfile(info, io.BytesIO())
            write_tar(value, archive, name)


TARRED_TREE: dict[str, Any] = {"animals.tar": make_tar(DEFAULT_TREE)}


def create_tar_dataset(session, uri: str, ds_name: str) -> dc.DataChain:
    """
    Create a dataset from a storage location containing tar archives and other files.
    The resulting dataset contains both the original files (as regular objects)
    and the tar members (as v-objects).
    """
    chain = dc.read_storage(uri, session=session)
    tar_entries = chain.filter(C("file.path").glob("*.tar")).gen(file=process_tar)
    return chain.union(tar_entries).save(ds_name)


skip_if_not_sqlite = pytest.mark.skipif(
    os.environ.get("DATACHAIN_METASTORE") is not None
    or os.environ.get("DATACHAIN_WAREHOUSE") is not None,
    reason="This test is not supported on other data storages",
)


def text_embedding(text: str) -> list[float]:
    """
    Compute a simple text embedding based on character counts.

    These aren't the most meaningful, but will produce a 100-element
    vector of floats between 0 and 1 where texts with similar
    character counts will have similar embeddings. Useful for writing
    unit tests without loading a heavy ML model.
    """
    emb = dict.fromkeys(printable, 0.01)
    for c in text:
        try:
            emb[c] += 1.0
        except KeyError:
            pass
    # sqeeze values between 0 and 1 with an adjusted sigmoid function
    return [2.0 / (1.0 + math.e ** (-x)) - 1.0 for x in emb.values()]


def dataset_dependency_asdict(
    dep: DatasetDependency | None,
) -> dict[str, Any] | None:
    """
    Converting to dict with making sure we don't have any additional fields
    that could've been added with subclasses
    """
    if not dep:
        return None
    parsed = {
        k: dataclasses.asdict(dep)[k] for k in DatasetDependency.__dataclass_fields__
    }

    if dep.dependencies:
        parsed["dependencies"] = [
            dataset_dependency_asdict(d) for d in dep.dependencies
        ]

    return parsed


def wait_for_condition(
    callback: Callable,
    message: str,
    check_interval: float = 0.01,
    timeout: float = 1.0,
) -> Any:
    start_time = time()
    while time() - start_time < timeout:
        result = callback()
        if result:
            return result
        sleep(check_interval)
    raise TimeoutError(f"Timeout expired while waiting for: {message}")


def assert_row_names(
    catalog: Catalog, dataset: DatasetRecord, version: str, expected_names: set
) -> None:
    dataset_rows = catalog.ls_dataset_rows(dataset, version, limit=20)
    assert dataset_rows
    preview = dataset.get_version(version).preview
    assert preview

    assert {r["file__path"] for r in dataset_rows} == {
        r.get("file__path") for r in preview
    }
    assert {posixpath.basename(r["file__path"]) for r in dataset_rows} == expected_names


def images_equal(img1: Image.Image, img2: Image.Image):
    """Checks if two image objects have exactly the same data"""
    return list(img1.getdata()) == list(img2.getdata())


def sorted_dicts(list_of_dicts, *keys):
    return sorted(list_of_dicts, key=lambda x: tuple(x[k] for k in keys))


class ANY_VALUE:  # noqa: N801, PLW1641
    """A helper object that compares equal to any value from the list."""

    def __init__(self, *args):
        self.values = args

    def __eq__(self, other) -> bool:
        return other in self.values

    def __ne__(self, other) -> bool:
        return other not in self.values

    def __repr__(self) -> str:
        return f"<ANY_VALUE: {', '.join(repr(val) for val in self.values)}>"


def sort_df(df):
    """Sorts dataframe by all columns"""
    return df.sort_values(by=df.columns.tolist()).reset_index(drop=True)


def df_equal(df1, df2) -> bool:
    """Helper function to check if two dataframes are equal regardless of ordering"""
    return sort_df(df1).equals(sort_df(df2))


def table_row_count(db, table_name) -> int | None:
    if not db.has_table(table_name):
        return None
    query = sa.select(sa.func.count()).select_from(sa.table(table_name))
    return next(db.execute(query), (None,))[0]


def reset_session_job_state():
    """
    Reset job state for testing purposes.
    Useful for simulating multiple script runs.
    """
    import atexit

    from datachain.query.session import Session

    # Unregister atexit hook if registered
    if Session._JOB_FINALIZE_HOOK is not None:
        try:
            atexit.unregister(Session._JOB_FINALIZE_HOOK)
        except ValueError:
            # Hook was already unregistered
            pass
        Session._JOB_FINALIZE_HOOK = None

    # Clear job state (class-level attributes)
    Session._CURRENT_JOB = None
    Session._JOB_STATUS = None
    Session._OWNS_JOB = None
    Session._JOB_HOOKS_REGISTERED = False

    # Clear DATACHAIN_JOB_ID env var to allow new job creation on next run
    os.environ.pop("DATACHAIN_JOB_ID", None)
