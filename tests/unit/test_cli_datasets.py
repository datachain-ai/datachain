import json

import pytest

import datachain as dc
from datachain.cli.commands.datasets import describe_dataset_cmd, group_dataset_versions
from datachain.client import Client
from datachain.data_storage import JobStatus
from datachain.dataset import DatasetDependency, DatasetStatus
from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing_info import ListingInfo


def test_group_dataset_versions_latest_only():
    """Test grouping datasets with latest_only=True returns only highest version."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "1.5.0"),
        ("dataset2", "0.1.0"),
        ("dataset2", "0.2.0"),
    ]

    result = group_dataset_versions(datasets, latest_only=True)

    assert result == {
        "dataset1": "2.0.0",
        "dataset2": "0.2.0",
    }


def test_group_dataset_versions_all_versions():
    """Test grouping datasets with latest_only=False returns all versions."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "1.5.0"),
        ("dataset2", "0.1.0"),
        ("dataset2", "0.2.0"),
    ]

    result = group_dataset_versions(datasets, latest_only=False)

    assert result == {
        "dataset1": ["1.0.0", "1.5.0", "2.0.0"],
        "dataset2": ["0.1.0", "0.2.0"],
    }


def test_group_dataset_versions_empty_input():
    """Test grouping empty dataset list."""
    result = group_dataset_versions([], latest_only=True)
    assert result == {}


def test_group_dataset_versions_single_dataset():
    """Test grouping single dataset with single version."""
    datasets = [("dataset1", "1.0.0")]

    result = group_dataset_versions(datasets, latest_only=True)
    assert result == {"dataset1": "1.0.0"}


def test_group_dataset_versions_semver_parsing():
    """Test that semver parsing is used correctly."""
    datasets = [
        ("dataset1", "1.0.0"),
        ("dataset1", "10.0.0"),
        ("dataset1", "2.0.0"),
        ("dataset1", "9.9.9"),
    ]

    result = group_dataset_versions(datasets, latest_only=True)
    assert result == {"dataset1": "10.0.0"}


@pytest.fixture
def describe(capsys):
    def run(catalog, name, version=None):
        describe_dataset_cmd(catalog, name, version, json_output=True)
        return json.loads(capsys.readouterr().out)

    return run


@pytest.fixture
def prefix(test_session):
    metastore = test_session.catalog.metastore
    return f"{metastore.default_namespace_name}.{metastore.default_project_name}"


@pytest.fixture
def lineage(test_session):
    dc.read_values(value=[1, 2], session=test_session).save(
        "source", version="1.0.0", query_script="MODEL = 'v1'\n"
    )
    dc.read_dataset("source", version="1.0.0", session=test_session).save("derived")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    metastore = test_session.catalog.metastore
    source = metastore.get_dataset("source", versions=None)
    metastore.create_dataset_version(source, "3.0.0", status=DatasetStatus.CREATED)
    derived, _ = metastore.create_dataset_version(
        metastore.get_dataset("derived", versions=None),
        "2.0.0",
        status=DatasetStatus.FAILED,
    )
    metastore.add_dataset_dependency(derived, "2.0.0", source, "1.0.0")
    return test_session.catalog


@pytest.mark.parametrize(
    "name,version,qualified",
    [
        ("source@1.0.0", None, False),
        ("source", "1.0.0", False),
        ("source@1.0.0", None, True),
    ],
)
def test_describe_selects_version(lineage, describe, prefix, name, version, qualified):
    if qualified:
        name = f"{prefix}.{name}"
    info = describe(lineage, name, version)
    assert (info["ref"], info["num_objects"]) == (f"{prefix}.source@1.0.0", 2)
    assert info["schema"]["value"] == "int"
    assert info["query_script"] == "MODEL = 'v1'\n"
    assert [(v["ref"], v["num_objects"]) for v in info["versions"]] == [
        (f"{prefix}.source@1.0.0", 2),
        (f"{prefix}.source@2.0.0", 1),
    ]
    assert [(d["ref"], d["num_objects"]) for d in info["dependents"]] == [
        (f"{prefix}.derived@1.0.0", 2)
    ]


def test_describe_only_complete_versions(lineage, describe, prefix):
    info = describe(lineage, "source")
    assert (info["ref"], info["num_objects"], info["dependents"]) == (
        f"{prefix}.source@2.0.0",
        1,
        [],
    )
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset_cmd(lineage, "source", "3.0.0")
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset_cmd(lineage, "source@99.0.0")
    with pytest.raises(DatasetNotFoundError):
        describe_dataset_cmd(lineage, "missing")


def test_describe_text(lineage, capsys, prefix):
    describe_dataset_cmd(lineage, "derived")
    output = capsys.readouterr().out
    assert f"{prefix}.derived@1.0.0\nRows: 2\n" in output
    assert "Versions: 1.0.0\nSchema:\n  value: int\n" in output
    assert (
        "Sources: none recorded\n"
        f"Depends on: {prefix}.source@1.0.0\n"
        "Lineage:\n"
        f"  {prefix}.source@1.0.0\n"
        "Dependents: none recorded\n"
    ) in output

    describe_dataset_cmd(lineage, "source@1.0.0")
    output = capsys.readouterr().out
    assert (
        "Depends on: none recorded\n"
        "Lineage: none recorded\n"
        "Dependents:\n"
        f"  {prefix}.derived@1.0.0\n"
    ) in output
    assert "Script:\nMODEL = 'v1'\n" in output


def test_describe_storage_lineage_reads_metadata_only(
    test_session, tmp_path, monkeypatch, capsys, describe, prefix
):
    path = tmp_path / "source with spaces"
    path.mkdir()
    (path / "sample.txt").write_text("hello")
    dc.read_storage(str(path), session=test_session).save("inventory")
    dc.read_dataset("inventory", session=test_session).save("report")

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must only read recorded metadata")

    monkeypatch.setattr(Client, "get_implementation", forbidden)
    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    uri = path_to_fsspec_uri(str(path)) + "/"
    listing = test_session.catalog.get_dataset_dependencies("inventory", "1.0.0")[0]
    assert listing is not None
    listing_ref = f"{uri}@{listing.version} (all versions)"
    typed = dc.describe_dataset("report", session=test_session)
    storage = typed.lineage[1]
    assert isinstance(storage.dataset, ListingInfo)
    assert storage.dataset.uri == uri
    assert storage.dataset.version == listing.version
    assert storage.ref == listing_ref
    assert storage.num_objects == 1
    info = describe(test_session.catalog, "report")
    assert info["sources"] == [listing_ref]
    assert info["depends_on"] == [f"{prefix}.inventory@1.0.0"]
    assert [
        (e["ref"], e["type"], e["num_objects"], e["removed"], e["depends_on"])
        for e in info["lineage"]
    ] == [
        (f"{prefix}.inventory@1.0.0", "dataset", 1, False, [listing_ref]),
        (listing_ref, "storage", 1, False, []),
    ]

    describe_dataset_cmd(test_session.catalog, "report")
    output = capsys.readouterr().out
    assert (
        f"  {prefix}.inventory@1.0.0\n"
        f"    depends on: {uri}@1.0.0 (all versions)\n"
        f"  {uri}@1.0.0 (all versions)\n"
    ) in output
    assert "Schema:\n  file: File@v1\n    source: str\n" in output
    assert "lst__" not in output


def test_describe_lineage_lists_each_version_once(test_session, describe, prefix):
    def read(name, version=None):
        return dc.read_dataset(name, version=version, session=test_session)

    dc.read_values(value=[1, 2], session=test_session).save("source", version="1.0.0")
    read("source", "1.0.0").save("first")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    read("first").union(read("source", "2.0.0")).save("second")
    read("first").union(read("second")).save("top")

    info = describe(test_session.catalog, "top")
    assert sorted(info["depends_on"]) == [
        f"{prefix}.first@1.0.0",
        f"{prefix}.second@1.0.0",
    ]
    lineage = {
        e["ref"]: (e["num_objects"], sorted(e["depends_on"])) for e in info["lineage"]
    }
    assert len(lineage) == len(info["lineage"])
    assert lineage == {
        f"{prefix}.first@1.0.0": (2, [f"{prefix}.source@1.0.0"]),
        f"{prefix}.second@1.0.0": (
            3,
            [f"{prefix}.first@1.0.0", f"{prefix}.source@2.0.0"],
        ),
        f"{prefix}.source@1.0.0": (2, []),
        f"{prefix}.source@2.0.0": (1, []),
    }


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_describe_removed_dependency(
    test_session, capsys, describe, prefix, keep_metadata
):
    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save("source")
    dc.read_dataset("source", session=test_session).save("derived", query_script="")
    catalog.remove_dataset("source", version="1.0.0", keep_metadata=keep_metadata)

    info = describe(catalog, "derived")
    if keep_metadata:
        assert info["depends_on"] == [f"{prefix}.source@1.0.0"]
        assert [(e["num_objects"], e["removed"]) for e in info["lineage"]] == [
            (2, True)
        ]
    else:
        assert info["depends_on"] == [None]
        assert info["lineage"] == []

    describe_dataset_cmd(catalog, "derived")
    output = capsys.readouterr().out
    expected = (
        f"Lineage:\n  {prefix}.source@1.0.0 (removed)\n"
        if keep_metadata
        else "Depends on: unavailable dependency\n"
    )
    assert expected in output
    assert "Script:\nNo script recorded.\n" in output


def test_describe_text_includes_description(test_session, capsys):
    dc.read_values(value=[1], session=test_session).save(
        "source", description="Recorded labels"
    )
    describe_dataset_cmd(test_session.catalog, "source")
    assert "Description: Recorded labels\n" in capsys.readouterr().out


def test_describe_shared_lineage_populates_each_edge_once(
    test_session, describe, monkeypatch
):
    metastore = test_session.catalog.metastore
    previous = []
    edge_count = 0
    for layer in range(18):
        current = []
        for side in range(2):
            record = metastore.create_dataset(name=f"layer_{layer}_{side}")
            record, _ = metastore.create_dataset_version(
                record, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=1
            )
            for parent in previous:
                metastore.add_dataset_dependency(record, "1.0.0", parent, "1.0.0")
                edge_count += 1
            current.append(record)
        previous = current

    original = DatasetDependency.__setattr__
    writes = 0

    def bounded(self, name, value):
        nonlocal writes
        if name == "dependencies":
            writes += 1
            assert writes <= 4 * edge_count
        original(self, name, value)

    monkeypatch.setattr(DatasetDependency, "__setattr__", bounded)
    info = describe(test_session.catalog, "layer_17_0")
    assert len(info["lineage"]) == 34


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/dir/",
        "gs://bucket/dir/",
        "az://container/dir/",
        "file:///data/photos/",
    ],
)
def test_describe_keeps_listing_kinds_apart(test_session, describe, capsys, uri):
    metastore = test_session.catalog.metastore

    def dataset(name, rows, project_id=None):
        record = metastore.create_dataset(name=name, project_id=project_id)
        return metastore.create_dataset_version(
            record, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=rows
        )[0]

    full = dataset(f"lst__{uri}", 10, metastore.listing_project.id)
    current = dataset(f"lst__current__{uri}", 3, metastore.listing_project.id)
    top = dataset("top", 13)
    metastore.add_dataset_dependency(top, "1.0.0", full, "1.0.0")
    metastore.add_dataset_dependency(top, "1.0.0", current, "1.0.0")

    metastore.create_dataset_version(
        full, "2.0.0", status=DatasetStatus.COMPLETE, num_objects=99
    )
    info = describe(test_session.catalog, "top")
    assert len(set(info["depends_on"])) == 2
    assert set(info["depends_on"]) == {e["ref"] for e in info["lineage"]}
    assert sorted((e["ref"], e["num_objects"]) for e in info["lineage"]) == [
        (f"{uri}@1.0.0 (all versions)", 10),
        (f"{uri}@1.0.0 (current only)", 3),
    ]
    describe_dataset_cmd(test_session.catalog, "top")
    output = capsys.readouterr().out
    assert output.count(f"{uri}@1.0.0 (all versions)") == 3
    assert output.count(f"{uri}@1.0.0 (current only)") == 3


@pytest.mark.parametrize("explicit_scope", [False, True])
def test_describe_python_api(lineage, test_session, prefix, capsys, explicit_scope):
    metastore = lineage.metastore
    scope = (
        {
            "namespace": metastore.default_namespace_name,
            "project": metastore.default_project_name,
        }
        if explicit_scope
        else {}
    )
    with test_session:
        info = dc.describe_dataset("source@1.0.0", **scope)
    assert isinstance(info, dc.DatasetDescription)
    assert isinstance(info.selected_version, DatasetInfo)
    assert info.ref == f"{prefix}.source@1.0.0"
    assert info.selected_version.version == "1.0.0"
    assert info.selected_version.num_objects == 2
    assert info.selected_version.query_script == "MODEL = 'v1'\n"
    assert info.available_versions == ["1.0.0", "2.0.0"]
    assert [v.num_objects for v in info.versions] == [2, 1]
    assert info.schema.serialize()["value"] == "int"
    assert info.dependents[0].ref == f"{prefix}.derived@1.0.0"
    assert capsys.readouterr().out == ""
    payload = info.to_dict()
    assert "name" not in payload and "version" not in payload
    for item in [*payload["versions"], *payload["lineage"], *payload["dependents"]]:
        assert "ref" in item
        assert "name" not in item and "version" not in item
    assert json.loads(json.dumps(payload)) == payload


def test_describe_all_available_versions(test_session, capsys):
    metastore = test_session.catalog.metastore
    record = metastore.create_dataset(name="history")
    for major in range(1, 56):
        record, _ = metastore.create_dataset_version(
            record, f"{major}.0.0", status=DatasetStatus.COMPLETE, num_objects=major
        )
    info = dc.describe_dataset("history", session=test_session)
    assert info.selected_version.version == "55.0.0"
    assert info.available_versions == [f"{major}.0.0" for major in range(1, 56)]
    older = dc.describe_dataset("history@2.0.0", session=test_session)
    assert older.selected_version.num_objects == 2
    assert older.available_versions == info.available_versions
    describe_dataset_cmd(test_session.catalog, "history")
    assert (
        "Versions: " + ", ".join(info.available_versions) + "\n"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("job_state", ["recorded", "missing", "none"])
def test_describe_recorded_jobs(test_session, capsys, monkeypatch, job_state):
    from datetime import UTC, datetime

    from datachain.job import Job

    metastore = test_session.catalog.metastore
    parent_id = metastore.create_job("parent", "")
    job_id = metastore.create_job(
        "produce_inventory",
        "process()",
        status=JobStatus.RUNNING,
        workers=4,
        python_version="3.12",
        params={"model": "v1"},
        parent_job_id=parent_id,
        rerun_from_job_id=parent_id,
        run_group_id=parent_id,
    )
    finished_at = datetime(2026, 11, 3, 12, 0, tzinfo=UTC)
    metastore.update_job(
        job_id,
        status=JobStatus.COMPLETE,
        finished_at=finished_at,
        metrics={"score": 0.75},
    )
    saved_job_id = (
        job_id
        if job_state == "recorded"
        else (
            "00000000-0000-0000-0000-000000000000" if job_state == "missing" else None
        )
    )
    record = metastore.create_dataset(name="job_output")
    for version in ["1.0.0", "2.0.0"]:
        record, _ = metastore.create_dataset_version(
            record,
            version,
            status=DatasetStatus.COMPLETE,
            num_objects=1,
            job_id=saved_job_id,
        )
    original = metastore.list_jobs_by_ids
    calls = []

    def load(ids):
        calls.append(ids)
        return original(ids)

    monkeypatch.setattr(metastore, "list_jobs_by_ids", load)
    info = dc.describe_dataset("job_output", session=test_session)
    assert calls == ([[saved_job_id]] if saved_job_id else [])
    assert info.selected_version.job_id == saved_job_id
    payload = info.to_dict()
    assert payload["job_id"] == saved_job_id
    if job_state == "recorded":
        assert type(info.job) is Job
        assert info.job.id == job_id
        assert info.job.finished_at == finished_at
        assert info.job.parent_job_id == parent_id
        assert info.job.rerun_from_job_id == parent_id
        assert info.job.run_group_id == parent_id
        assert info.selected_version.params == {"model": "v1"}
        assert info.selected_version.metrics == {"score": 0.75}
        assert [j["id"] for j in payload["jobs"]] == [job_id]
        assert payload["jobs"][0]["finished_at"] == finished_at.isoformat()
    else:
        assert info.job is None
        assert payload["jobs"] == []
    json.dumps(payload)
    describe_dataset_cmd(test_session.catalog, "job_output")
    output = capsys.readouterr().out
    if job_state == "recorded":
        assert f"Job: {job_id}\nJob status: COMPLETE\n" in output
        assert (
            f"Job finished: {finished_at.isoformat()}\nWorkers: 4\nPython: 3.12\n"
            in output
        )
    elif job_state == "missing":
        assert f"Job: {saved_job_id} (unavailable)\n" in output
    else:
        assert "Job:" not in output


@pytest.mark.parametrize("json_output", [False, True])
def test_describe_cli_dispatch(lineage, capsys, prefix, json_output):
    from datachain.cli import get_parser, handle_dataset_command

    command = ["dataset", "describe", "source@1.0.0"]
    if json_output:
        command.append("--json")
    args = get_parser().parse_args(command)
    handle_dataset_command(args, lineage)
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["ref"] == f"{prefix}.source@1.0.0"
    else:
        assert output.startswith(f"{prefix}.source@1.0.0\n")
        assert "Versions: 1.0.0, 2.0.0\n" in output


def test_describe_nondefault_namespace_and_project(test_session):
    metastore = test_session.catalog.metastore
    metastore.create_project("research", "analysis")
    dc.read_values(value=[1, 2], session=test_session).save(
        "research.analysis.scoped", version="2.0.0"
    )
    for info in [
        dc.describe_dataset("research.analysis.scoped", session=test_session),
        dc.describe_dataset(
            "scoped", namespace="research", project="analysis", session=test_session
        ),
        dc.describe_dataset("scoped", "research", "analysis", "2.0.0", test_session),
    ]:
        assert info.ref == "research.analysis.scoped@2.0.0"
        assert info.selected_version.num_objects == 2
    with pytest.raises(DatasetNotFoundError):
        dc.describe_dataset("scoped", session=test_session)


def test_describe_preserves_unresolved_schema(test_session, describe):
    metastore = test_session.catalog.metastore
    record = metastore.create_dataset(name="legacy")
    schema = {
        "id": "int",
        "embedding": "UnavailableEmbedding2134@v1",
        "optional": "Optional[UnavailableModel2134@v1]",
    }
    metastore.create_dataset_version(
        record,
        "1.0.0",
        status=DatasetStatus.COMPLETE,
        feature_schema=schema,
        num_objects=5,
    )
    info = dc.describe_dataset("legacy", session=test_session)
    assert info.feature_schema == schema
    assert info.to_dict()["schema"] == schema
    assert describe(test_session.catalog, "legacy")["schema"] == schema


def test_describe_jobs_include_lineage_and_dependents(test_session):
    metastore = test_session.catalog.metastore

    def record(name, job_id):
        dataset = metastore.create_dataset(name=name)
        return metastore.create_dataset_version(
            dataset,
            "1.0.0",
            status=DatasetStatus.COMPLETE,
            job_id=job_id,
            num_objects=1,
        )[0]

    source_job = metastore.create_job("source", "")
    derived_job = metastore.create_job("derived", "")
    report_job = metastore.create_job("report", "")
    source = record("source", source_job)
    derived = record("derived", derived_job)
    report = record("report", report_job)
    metastore.add_dataset_dependency(derived, "1.0.0", source, "1.0.0")
    metastore.add_dataset_dependency(report, "1.0.0", derived, "1.0.0")
    info = dc.describe_dataset("derived", session=test_session)
    assert info.job.id == derived_job
    assert info.lineage[0].dataset.job_id == source_job
    assert info.dependents[0].dataset.job_id == report_job
    assert set(info.jobs) == {source_job, derived_job, report_job}
    payload = info.to_dict()
    assert [job["id"] for job in payload["jobs"]] == sorted(info.jobs)
