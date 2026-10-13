import subprocess
import sys

import numpy as np
import pytest

import datachain as dc
from datachain.lib.hdf5 import Hdf5Dataset, Hdf5File, Hdf5Selection

h5py = pytest.importorskip("h5py")


def _make_file(path, who, n=4):
    with h5py.File(str(path), "w") as f:
        f.attrs["who"] = who
        robot = f.create_group("robot")
        robot.attrs["units"] = "radians"
        frames = robot.create_dataset(
            "frames", shape=(n, 8, 8), chunks=(1, 8, 8), dtype="uint8"
        )
        frames.attrs["camera"] = "workspace"
        frames[:] = np.arange(n * 8 * 8).reshape(n, 8, 8).astype("uint8")
        labels = f.create_group("labels")
        y = labels.create_dataset("y", shape=(n,), dtype="int64")
        y[:] = np.arange(n)


def _single_file(tmp_dir, test_session, name="s"):
    _make_file(tmp_dir / f"{name}.h5", name)
    chain = dc.read_storage(
        (tmp_dir / f"{name}.h5").as_uri(), type="hdf5", session=test_session
    )
    (file,) = next(iter(chain.to_iter("file")))
    return file


def test_read_storage_hdf5_yields_hdf5_file(tmp_dir, test_session):
    for name in ("scan001", "scan002", "scan003"):
        _make_file(tmp_dir / f"{name}.h5", name)

    chain = dc.read_storage(tmp_dir.as_uri(), type="hdf5", session=test_session)
    files = [f for (f,) in chain.order_by("file.path").to_iter("file")]

    assert [type(f).__name__ for f in files] == ["Hdf5File"] * 3
    assert [f.path for f in files] == ["scan001.h5", "scan002.h5", "scan003.h5"]


def test_as_hdf5_file_converts_and_keeps_stream(tmp_dir, test_session):
    _make_file(tmp_dir / "s.h5", "s")

    chain = dc.read_storage((tmp_dir / "s.h5").as_uri(), session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    hdf5_file = file.as_hdf5_file()

    assert type(hdf5_file).__name__ == "Hdf5File"
    assert hdf5_file.as_hdf5_file() is hdf5_file
    assert hdf5_file.get_info().attrs == {"who": "s"}


def test_hdf5_file_get_info(tmp_dir, test_session):
    info = _single_file(tmp_dir, test_session).get_info()

    assert info.attrs == {"who": "s"}
    assert sorted(info.datasets) == ["/labels/y", "/robot/frames"]
    assert sorted(info.groups) == ["/labels", "/robot"]


def test_hdf5_file_get_datasets(tmp_dir, test_session):
    datasets = {d.path: d for d in _single_file(tmp_dir, test_session).get_datasets()}

    assert sorted(datasets) == ["/labels/y", "/robot/frames"]
    assert datasets["/robot/frames"].shape == [4, 8, 8]
    assert datasets["/robot/frames"].chunks == [1, 8, 8]
    assert datasets["/robot/frames"].dtype == "uint8"
    assert datasets["/labels/y"].shape == [4]
    assert datasets["/labels/y"].chunks is None


def test_hdf5_file_get_datasets_scoped_to_group(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    assert [d.path for d in file.get_datasets("/robot")] == ["/robot/frames"]
    assert [d.path for d in file.get_datasets("/robot/frames")] == ["/robot/frames"]


def test_hdf5_file_get_dataset(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    assert file.get_dataset("/robot/frames").shape == [4, 8, 8]
    assert file.get_dataset("robot/frames").path == "/robot/frames"


def test_hdf5_file_get_dataset_not_a_dataset(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    with pytest.raises(ValueError, match="not an HDF5 dataset"):
        file.get_dataset("/robot")


def test_hdf5_file_get_dataset_missing_path(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    with pytest.raises(KeyError):
        file.get_dataset("/nope")


def test_hdf5_dataset_read(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    data = file.get_dataset("/robot/frames").read()
    assert data.shape == (4, 8, 8)
    assert int(data.flat[0]) == 0

    one = file.get_dataset("/robot/frames").read(np.s_[0])
    assert one.shape == (8, 8)

    assert file.get_dataset("/labels/y").read().tolist() == [0, 1, 2, 3]


def test_hdf5_dataset_attrs_are_json_serializable(tmp_dir, test_session):
    path = tmp_dir / "attrs.h5"
    with h5py.File(str(path), "w") as f:
        f.attrs["text"] = "plain"
        f.attrs["count"] = 5
        f.attrs["ratio"] = np.float32(1.5)
        f.attrs["array"] = np.arange(3)
        f.attrs["raw"] = np.bytes_(b"encoded")
        f.attrs["matrix"] = np.arange(4).reshape(2, 2)
        d = f.create_dataset("x", data=np.zeros(2))
        d.attrs["shift"] = np.int64(7)

    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    info = file.get_info()

    assert info.attrs == {
        "text": "plain",
        "count": 5,
        "ratio": 1.5,
        "array": [0, 1, 2],
        "raw": "encoded",
        "matrix": [[0, 1], [2, 3]],
    }
    assert dc.json.dumps(info.attrs)
    assert file.get_dataset("/x").attrs == {"shift": 7}


def test_hdf5_dataset_degenerate_shapes_and_dtypes(tmp_dir, test_session):
    path = tmp_dir / "edge.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("scalar", data=42)
        f.create_dataset("empty", shape=(0,), dtype="int64")
        f.create_dataset(
            "events", data=np.zeros(3, dtype=np.dtype([("t", "f8"), ("v", "i4", (3,))]))
        )
        f.create_dataset(
            "text",
            data=np.array(["a", "bb"], dtype=object),
            dtype=h5py.string_dtype(),
        )

    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    datasets = {d.path: d for d in file.get_datasets()}

    # A scalar dataset has an empty shape; a zero-length 1-D one has [0].
    assert datasets["/scalar"].shape == []
    assert datasets["/empty"].shape == [0]
    assert file.get_dataset("/scalar").read().tolist() == 42
    assert file.get_dataset("/empty").read().tolist() == []

    assert datasets["/events"].dtype == "[('t', '<f8'), ('v', '<i4', (3,))]"
    assert datasets["/text"].dtype == "object"


def test_hdf5_file_without_attrs_or_datasets(tmp_dir, test_session):
    path = tmp_dir / "bare.h5"
    with h5py.File(str(path), "w"):
        pass

    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    info = file.get_info()

    assert info.attrs == {}
    assert info.datasets == []
    assert info.groups == []
    assert list(file.get_datasets()) == []


def test_hdf5_file_walks_hard_linked_group_once(tmp_dir, test_session):
    path = tmp_dir / "linked.h5"
    with h5py.File(str(path), "w") as f:
        g = f.create_group("real")
        g.create_dataset("x", data=np.zeros(2))
        f["alias"] = g

    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    datasets = list(file.get_datasets())

    # A hard-linked object has several equally valid names, so the reported one
    # depends on traversal order; what must hold is that it is reported once.
    assert len(datasets) == 1
    assert datasets[0].path in ("/real/x", "/alias/x")
    assert datasets[0].read().tolist() == [0.0, 0.0]


def test_hdf5_selection_read(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    selection = file.get_dataset("/robot/frames").select(0)
    assert selection.index == [0]
    assert selection.read().shape == (8, 8)

    block = file.get_dataset("/robot/frames").select([1])
    assert block.read().shape == (8, 8)


def test_hdf5_selection_read_bytes_image(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    content = file.get_dataset("/robot/frames").select(0, media="image").read_bytes()

    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_hdf5_selection_read_bytes_rejects_non_image(tmp_dir, test_session):
    file = _single_file(tmp_dir, test_session)

    selection = file.get_dataset("/robot/frames").select(0, media="audio")

    with pytest.raises(ValueError, match="supports image media"):
        selection.read_bytes()


def test_hdf5_selection_read_bytes_converts_non_uint8(tmp_dir, test_session):
    path = tmp_dir / "f.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("img", data=np.zeros((2, 4, 4), dtype="float32"))

    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))
    content = file.get_dataset("/img").select(0, media="image").read_bytes()

    assert content[:8] == b"\x89PNG\r\n\x1a\n"


def test_hdf5_reads_only_the_requested_region(tmp_dir, test_session):
    """A slice must not pull the whole file — the point of a native HDF5 type."""
    path = tmp_dir / "big.h5"
    n = 60
    with h5py.File(str(path), "w") as f:
        f.create_dataset(
            "frames",
            data=np.zeros((n, 256, 256, 3), dtype="uint8"),
            chunks=(1, 256, 256, 3),
        )

    total = path.stat().st_size
    chain = dc.read_storage(path.as_uri(), type="hdf5", session=test_session)
    (file,) = next(iter(chain.to_iter("file")))

    read_bytes = 0
    original_readinto = None

    def counting_readinto(self, b):
        nonlocal read_bytes
        result = original_readinto(self, b)
        read_bytes += result or 0
        return result

    from datachain.client.fileslice import FileWrapper

    original_readinto = FileWrapper.readinto
    FileWrapper.readinto = counting_readinto
    try:
        frame = file.get_dataset("/frames").select(n // 2).read()
    finally:
        FileWrapper.readinto = original_readinto

    assert frame.shape == (256, 256, 3)
    assert total > 10 * 1024 * 1024
    assert read_bytes < total // 10


def test_hdf5_missing_dependency_error(tmp_dir, test_session, monkeypatch):
    from datachain.lib import hdf5

    monkeypatch.setattr(hdf5, "h5py", None)
    file = _single_file(tmp_dir, test_session)

    with pytest.raises(ImportError, match=r"datachain\[hdf5\]"):
        file.get_info()


def _selection_chain(tmp_dir, test_session, names=("a", "b")):
    """A chain of one Hdf5Selection per file."""
    for name in names:
        _make_file(tmp_dir / f"{name}.h5", name)

    return dc.read_storage(tmp_dir.as_uri(), type="hdf5", session=test_session).map(
        sel=lambda file: file.get_dataset("/robot/frames").select(0, media="image"),
        output=Hdf5Selection,
    )


def test_hdf5_selection_survives_save_and_reload(tmp_dir, test_session):
    _selection_chain(tmp_dir, test_session).save("h5sel")

    reloaded = dc.read_dataset("h5sel", session=test_session)

    assert reloaded.schema["sel"] is Hdf5Selection
    selections = sorted(
        (s for (s,) in reloaded.to_iter("sel")), key=lambda s: s.dataset.file.path
    )
    assert [s.dataset.file.path for s in selections] == ["a.h5", "b.h5"]
    assert [s.dataset.path for s in selections] == ["/robot/frames"] * 2
    assert [s.dataset.shape for s in selections] == [[4, 8, 8]] * 2
    assert [s.index for s in selections] == [[0], [0]]
    assert [s.media for s in selections] == ["image", "image"]
    # The nested file keeps its stream through the round-trip, so it still reads.
    assert selections[0].read().shape == (8, 8)


def test_hdf5_selection_as_only_signal(tmp_dir, test_session):
    chain = _selection_chain(tmp_dir, test_session).select("sel")

    assert chain.schema == {"sel": Hdf5Selection}
    assert len(list(chain.to_iter("sel"))) == 2


def test_hdf5_selection_filter_and_order_by_nested_leaf(tmp_dir, test_session):
    chain = _selection_chain(tmp_dir, test_session, names=("a", "b", "c"))

    paths = [
        p
        for (p,) in chain.filter(dc.C("sel.dataset.file.path") != "b.h5")
        .order_by("sel.dataset.file.path", descending=True)
        .to_iter("sel.dataset.file.path")
    ]

    assert paths == ["c.h5", "a.h5"]


def test_hdf5_selection_nested_leaf_inside_composed_func(tmp_dir, test_session):
    from datachain import func

    chain = _selection_chain(tmp_dir, test_session, names=("a",))
    mutated = chain.mutate(
        name_len=func.string.length(dc.C("sel.dataset.file.path")) + 1
    )

    assert [n for (n,) in mutated.to_iter("name_len")] == [len("a.h5") + 1]


def test_hdf5_selection_distinct_group_by_and_window(tmp_dir, test_session):
    from datachain import func

    chain = _selection_chain(tmp_dir, test_session, names=("a", "b"))

    distinct = list(chain.distinct("sel.dataset.path").to_iter("sel.dataset.path"))
    assert distinct == [("/robot/frames",)]

    grouped = chain.group_by(total=func.count(), partition_by="sel.dataset.path")
    assert list(grouped.to_iter("sel.dataset.path", "total")) == [("/robot/frames", 2)]

    window = func.window(
        partition_by="sel.dataset.path", order_by="sel.dataset.file.path"
    )
    windowed = chain.mutate(
        first_file=func.first(dc.C("sel.dataset.file.path")).over(window)
    )
    assert {f for (f,) in windowed.to_iter("first_file")} == {"a.h5"}


def test_hdf5_selection_merge_carries_nested_signal(tmp_dir, test_session):
    chain = _selection_chain(tmp_dir, test_session, names=("a", "b"))
    labels = dc.read_values(
        key=["a.h5", "b.h5"], label=["left", "right"], session=test_session
    )

    merged = chain.merge(labels, on="sel.dataset.file.path", right_on="key")

    assert sorted(merged.to_iter("sel.dataset.file.path", "label")) == [
        ("a.h5", "left"),
        ("b.h5", "right"),
    ]
    assert merged.schema["sel"] is Hdf5Selection


def test_hdf5_selection_union_in_both_arm_orders(tmp_dir, test_session):
    chain = _selection_chain(tmp_dir, test_session, names=("a", "b"))
    left = chain.filter(dc.C("sel.dataset.file.path") == "a.h5")
    right = chain.filter(dc.C("sel.dataset.file.path") == "b.h5")

    for first, second in ((left, right), (right, left)):
        combined = first.union(second)

        assert combined.schema["sel"] is Hdf5Selection
        assert sorted(p for (p,) in combined.to_iter("sel.dataset.file.path")) == [
            "a.h5",
            "b.h5",
        ]


def test_hdf5_attrs_declared_hidden_for_display(tmp_dir, test_session):
    from datachain.lib.signal_schema import SignalSchema

    _selection_chain(tmp_dir, test_session, names=("a",)).save("h5hidden")
    saved = dc.read_dataset("h5hidden", session=test_session)
    schema = SignalSchema({"sel": Hdf5Selection}).serialize()

    assert "attrs" in Hdf5Dataset.hidden_fields()
    assert "sel__dataset__attrs" in SignalSchema.get_flatten_hidden_fields(schema)
    # attrs is a display hint, not an export filter, so it still round-trips.
    assert saved.to_records()[0]["sel__dataset__attrs"] == {"camera": "workspace"}


def test_hdf5_selection_export_paths_keep_unique_flat_names(tmp_dir, test_session):
    import pyarrow.parquet as pq

    _selection_chain(tmp_dir, test_session, names=("a",)).save("h5exp")
    saved = dc.read_dataset("h5exp", session=test_session)

    for names in (
        [str(c) for c in saved.to_pandas().columns],
        list(saved.to_records()[0]),
    ):
        assert len(names) == len(set(names))

    parquet = tmp_dir / "out.parquet"
    saved.to_parquet(str(parquet))
    parquet_names = pq.read_schema(str(parquet)).names
    assert len(parquet_names) == len(set(parquet_names))
    assert "sel.dataset.file.path" in parquet_names

    csv_path = tmp_dir / "out.csv"
    saved.to_csv(str(csv_path))
    header = csv_path.read_text().splitlines()[0].split(",")
    assert len(header) == len(set(header))

    json_path = tmp_dir / "out.json"
    saved.to_json(str(json_path))
    assert '"sel"' in json_path.read_text()


@pytest.mark.xfail(
    reason="empty dict signals cannot be written to Parquet (general limitation, "
    "not HDF5-specific: any DataModel dict field with no keys hits the same "
    "ArrowNotImplementedError)",
    raises=Exception,
    strict=True,
)
def test_hdf5_dataset_without_attrs_exports_to_parquet(tmp_dir, test_session):
    path = tmp_dir / "noattrs.h5"
    with h5py.File(str(path), "w") as f:
        f.create_dataset("x", data=np.zeros(2))

    dc.read_storage(path.as_uri(), type="hdf5", session=test_session).map(
        dataset=lambda file: file.get_dataset("/x")
    ).save("h5noattrs")

    dc.read_dataset("h5noattrs", session=test_session).to_parquet(
        str(tmp_dir / "noattrs.parquet")
    )


def test_hdf5_selection_through_generator_udf(tmp_dir, test_session):
    from collections.abc import Iterator

    _make_file(tmp_dir / "a.h5", "a")

    def frames(file: Hdf5File) -> Iterator[Hdf5Selection]:
        dataset = file.get_dataset("/robot/frames")
        for i in range(dataset.shape[0]):
            yield dataset.select(i, media="image")

    chain = dc.read_storage(tmp_dir.as_uri(), type="hdf5", session=test_session).gen(
        sel=frames
    )

    selections = sorted((s for (s,) in chain.to_iter("sel")), key=lambda s: s.index)
    assert [s.index for s in selections] == [[0], [1], [2], [3]]
    assert selections[2].read().shape == (8, 8)


def test_hdf5_dataset_optional_signal_with_null_rows(tmp_dir, test_session):
    _make_file(tmp_dir / "a.h5", "a")
    _make_file(tmp_dir / "b.h5", "b")

    def maybe(file: Hdf5File) -> Hdf5Dataset | None:
        return file.get_dataset("/robot/frames") if file.path == "a.h5" else None

    dc.read_storage(tmp_dir.as_uri(), type="hdf5", session=test_session).map(
        maybe_dataset=maybe
    ).save("h5opt")

    reloaded = dc.read_dataset("h5opt", session=test_session)
    got = dict(reloaded.to_iter("file.path", "maybe_dataset"))

    assert got["a.h5"].path == "/robot/frames"
    assert got["b.h5"] is None


def test_importing_datachain_does_not_import_h5py():
    """h5py stays optional: `import datachain` must not load it."""
    code = "import sys; import datachain; print('h5py' in sys.modules)"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip() == "False"


def test_read_storage_hdf5_works_without_explicit_model_import(tmp_dir, test_session):
    """`read_storage(type="hdf5")` pulls the models in on its own."""
    _make_file(tmp_dir / "s.h5", "s")
    code = (
        "import sys, datachain as dc; "
        f"chain = dc.read_storage({str((tmp_dir / 's.h5').as_uri())!r}, type='hdf5'); "
        "(f,) = next(iter(chain.to_iter('file'))); "
        "print(type(f).__name__, f.get_info().attrs['who'], 'h5py' in sys.modules)"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert out.stdout.strip().endswith("Hdf5File s True")


def test_schema_deserialization_requires_the_module_to_be_imported():
    """Being lazy has a cost: a consumer process must import the models itself.

    Without the import the signal is silently dropped rather than raising, so
    anything deserializing a stored schema (e.g. Studio) has to import them.
    """
    probe = (
        "import warnings, datachain; "
        "warnings.simplefilter('ignore'); "
        "{extra}"
        "from datachain.lib.signal_schema import SignalSchema; "
        "print(list(SignalSchema.deserialize({{'sel': 'Hdf5Selection@v1'}}).values))"
    )

    without = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe.format(extra="")],
        capture_output=True,
        text=True,
        check=True,
    )
    with_import = subprocess.run(  # noqa: S603
        [sys.executable, "-c", probe.format(extra="import datachain.lib.hdf5; ")],
        capture_output=True,
        text=True,
        check=True,
    )

    assert without.stdout.strip() == "[]"
    assert with_import.stdout.strip() == "['sel']"
