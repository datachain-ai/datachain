import json
import time
from datetime import UTC, datetime, timedelta, timezone

import pytest

import datachain as dc
from datachain.cli.commands.datasets import describe_dataset_cmd, group_dataset_versions
from datachain.client import Client
from datachain.data_storage import JobStatus
from datachain.dataset import DatasetStatus
from datachain.error import DatasetNotFoundError, DatasetVersionNotFoundError
from datachain.fs.utils import path_to_fsspec_uri
from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing_info import ListingInfo


def test_describe_command_uses_supplied_catalog_with_an_active_session(
    test_session, catalog_tmpfile, capsys
):
    dc.read_values(id=[1], session=catalog_tmpfile.session).save("other_catalog")
    describe_dataset_cmd(catalog_tmpfile, "other_catalog")
    assert "other_catalog" in capsys.readouterr().out


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
        describe_dataset_cmd(
            catalog, f"{name}@{version}" if version else name, json_output=True
        )
        return json.loads(capsys.readouterr().out)

    return run


@pytest.fixture
def prefix(test_session):
    metastore = test_session.catalog.metastore
    return f"{metastore.default_namespace_name}.{metastore.default_project_name}"


@pytest.fixture
def display_prefix(prefix):
    return "" if prefix == "local.local" else prefix + "."


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
    assert (info["name"], info["num_objects"]) == (f"{prefix}.source@1.0.0", 2)
    assert info["schema"]["value"] == "int"
    assert info["query_script"] == "MODEL = 'v1'\n"
    assert [(v["name"], v["num_objects"]) for v in info["versions"]] == [
        (f"{prefix}.source@1.0.0", 2),
        (f"{prefix}.source@2.0.0", 1),
    ]
    assert [(d["name"], d["num_objects"]) for d in info["dependents"]] == [
        (f"{prefix}.derived@1.0.0", 2)
    ]


def test_describe_only_complete_versions(lineage, describe, prefix):
    info = describe(lineage, "source")
    assert (info["name"], info["num_objects"], info["dependents"]) == (
        f"{prefix}.source@2.0.0",
        1,
        [],
    )
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset_cmd(lineage, "source@3.0.0")
    with pytest.raises(DatasetVersionNotFoundError):
        describe_dataset_cmd(lineage, "source@99.0.0")
    with pytest.raises(DatasetNotFoundError):
        describe_dataset_cmd(lineage, "missing")


def test_describe_text(lineage, capsys, display_prefix):
    describe_dataset_cmd(lineage, "derived")
    output = capsys.readouterr().out
    assert f"Name: {display_prefix}derived@1.0.0\nRows: 2\n" in output
    assert "Versions: 1.0.0\nSchema:\n  value: int\n" in output
    assert (
        f"Dependencies:\n  {display_prefix}source@1.0.0\nDependents: none recorded\n"
    ) in output
    assert "Sources:" not in output and "Lineage:" not in output
    assert "local.local." not in output

    describe_dataset_cmd(lineage, "source@1.0.0")
    output = capsys.readouterr().out
    assert (
        f"Dependencies: none recorded\nDependents:\n  {display_prefix}derived@1.0.0\n"
    ) in output
    assert "Script:\nMODEL = 'v1'\n" in output


def test_describe_storage_dependencies_read_metadata_only(
    test_session, tmp_path, monkeypatch, capsys, describe, prefix
):
    path = tmp_path / "source with spaces"
    path.mkdir()
    (path / "sample.txt").write_text("hello")
    dc.read_storage(str(path), session=test_session).save("inventory")
    dc.read_dataset("inventory", session=test_session).save("report")

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must only read direct recorded metadata")

    monkeypatch.setattr(Client, "get_implementation", forbidden)
    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    monkeypatch.setattr(
        test_session.catalog, "get_dataset_dependencies_by_ids", forbidden
    )
    uri = path_to_fsspec_uri(str(path)) + "/"
    listing = test_session.catalog.get_dataset_dependencies("inventory", "1.0.0")[0]
    assert listing is not None
    listing_ref = (
        f"{listing.namespace}.{listing.project}.{listing.name}@{listing.version}"
    )
    typed = dc.describe("inventory", session=test_session)
    storage = typed.dependencies[0]
    assert isinstance(storage.dataset, ListingInfo)
    assert storage.dataset.uri == uri
    assert storage.dataset.version == listing.version
    assert storage.name == listing_ref
    assert storage.num_objects == 1
    info = describe(test_session.catalog, "inventory")
    assert [(e["name"], e["type"]) for e in info["dependencies"]] == [
        (listing_ref, "storage")
    ]
    assert not {"sources", "lineage", "depends_on"} & info.keys()
    assert "depends_on" not in info["dependencies"][0]
    report = describe(test_session.catalog, "report")
    assert [e["name"] for e in report["dependencies"]] == [f"{prefix}.inventory@1.0.0"]

    describe_dataset_cmd(test_session.catalog, "inventory")
    output = capsys.readouterr().out
    assert f"Dependencies:\n  {uri}@1.0.0\n" in output
    assert "Schema:\n  file: File@v1\n    source: str\n" in output
    assert "lst__" not in output


def test_describe_dependencies_keep_exact_versions(test_session, describe, prefix):
    dc.read_values(value=[1, 2], session=test_session).save("source", version="1.0.0")
    dc.read_values(value=[3], session=test_session).save("source", version="2.0.0")
    dc.read_dataset("source@1.0.0", session=test_session).union(
        dc.read_dataset("source@2.0.0", session=test_session)
    ).save("top")

    info = describe(test_session.catalog, "top")
    assert [(e["name"], e["num_objects"]) for e in info["dependencies"]] == [
        (f"{prefix}.source@1.0.0", 2),
        (f"{prefix}.source@2.0.0", 1),
    ]


@pytest.mark.parametrize("keep_metadata", [False, True])
def test_describe_removed_dependency(
    test_session, capsys, describe, prefix, display_prefix, keep_metadata
):
    catalog = test_session.catalog
    dc.read_values(value=[1, 2], session=test_session).save("source")
    dc.read_dataset("source", session=test_session).save("derived", query_script="")
    catalog.remove_dataset("source", version="1.0.0", keep_metadata=keep_metadata)

    info = describe(catalog, "derived")
    if keep_metadata:
        assert info["dependencies"][0]["name"] == f"{prefix}.source@1.0.0"
        assert [(e["num_objects"], e["removed"]) for e in info["dependencies"]] == [
            (2, True)
        ]
    else:
        assert info["dependencies"] == [None]

    describe_dataset_cmd(catalog, "derived")
    output = capsys.readouterr().out
    expected = (
        f"  {display_prefix}source@1.0.0 (removed)\n"
        if keep_metadata
        else "  unavailable dependency\n"
    )
    assert expected in output
    assert "Script:\nNo script recorded.\n" in output


def test_describe_text_includes_description(test_session, capsys):
    dc.read_values(value=[1], session=test_session).save(
        "source", description="Recorded labels"
    )
    describe_dataset_cmd(test_session.catalog, "source")
    assert "Description: Recorded labels\n" in capsys.readouterr().out


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
    assert len({e["name"] for e in info["dependencies"]}) == 2
    listing_prefix = (
        f"{metastore.system_namespace_name}.{metastore.listing_project_name}"
    )
    expected = {
        f"{listing_prefix}.lst__{uri}@1.0.0": 10,
        f"{listing_prefix}.lst__current__{uri}@1.0.0": 3,
    }
    assert {e["name"]: e["num_objects"] for e in info["dependencies"]} == expected
    assert all("listing_kind" not in e and "ref" not in e for e in info["dependencies"])
    describe_dataset_cmd(test_session.catalog, "top")
    output = capsys.readouterr().out
    assert output.count(f"{uri}@1.0.0") == 2
    assert f"{uri}@1.0.0 (all versions)" in output
    assert f"{uri}@1.0.0 (current only)" in output


@pytest.mark.parametrize("explicit_scope", [False, True])
def test_describe_python_api(lineage, test_session, prefix, capsys, explicit_scope):
    from datachain import describe as describe_saved

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
        info = describe_saved("source@1.0.0", **scope)
    assert isinstance(info, dc.DatasetDescription)
    assert isinstance(info.selected_version, DatasetInfo)
    assert info.name == f"{prefix}.source@1.0.0"
    assert info.selected_version.version == "1.0.0"
    assert info.selected_version.num_objects == 2
    assert info.selected_version.query_script == "MODEL = 'v1'\n"
    assert info.available_versions == ["1.0.0", "2.0.0"]
    assert [v.num_objects for v in info.versions] == [2, 1]
    assert info.schema.serialize()["value"] == "int"
    assert info.dependents[0].name == f"{prefix}.derived@1.0.0"
    assert capsys.readouterr().out == ""
    payload = info.to_dict()
    assert "ref" not in payload and "version" not in payload
    for item in [
        *payload["versions"],
        *payload["dependencies"],
        *payload["dependents"],
    ]:
        assert "name" in item
        assert "ref" not in item and "version" not in item
    assert json.loads(json.dumps(payload)) == payload


def test_describe_all_available_versions(test_session, capsys):
    metastore = test_session.catalog.metastore
    record = metastore.create_dataset(name="history")
    for major in range(1, 56):
        record, _ = metastore.create_dataset_version(
            record, f"{major}.0.0", status=DatasetStatus.COMPLETE, num_objects=major
        )
    info = dc.describe("history", session=test_session)
    assert info.selected_version.version == "55.0.0"
    assert info.available_versions == [f"{major}.0.0" for major in range(1, 56)]
    older = dc.describe("history@2.0.0", session=test_session)
    assert older.selected_version.num_objects == 2
    assert older.available_versions == info.available_versions
    describe_dataset_cmd(test_session.catalog, "history")
    assert (
        "Versions: " + ", ".join(info.available_versions) + "\n"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize("job_state", ["recorded", "missing", "none"])
def test_describe_recorded_jobs(test_session, capsys, monkeypatch, job_state):
    from datachain.job import Job

    monkeypatch.delenv("DATACHAIN_JOB_ID", raising=False)
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
    info = dc.describe("job_output", session=test_session)
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
        assert f"Job: {job_id}\nJob_status: COMPLETE\n" in output
        assert (
            "Job_finished: "
            + finished_at.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z").rstrip()
            + "\nWorkers: 4\nPython: 3.12\n"
            in output
        )
    elif job_state == "missing":
        assert f"Job: {saved_job_id} (unavailable)\n" in output
    else:
        assert "Job:" not in output


@pytest.mark.parametrize("json_output", [False, True])
def test_describe_cli_dispatch(lineage, capsys, prefix, display_prefix, json_output):
    from datachain.cli import get_parser, handle_dataset_command

    command = ["dataset", "describe", "source@1.0.0"]
    if json_output:
        command.append("--json")
    args = get_parser().parse_args(command)
    handle_dataset_command(args, lineage)
    output = capsys.readouterr().out
    if json_output:
        assert json.loads(output)["name"] == f"{prefix}.source@1.0.0"
    else:
        assert output.startswith(f"Name: {display_prefix}source@1.0.0\n")
        assert "Versions: 1.0.0, 2.0.0\n" in output


def test_describe_nondefault_namespace_and_project(test_session, monkeypatch, capsys):
    metastore = test_session.catalog.metastore
    monkeypatch.setenv("DATACHAIN_IS_STUDIO", "true")
    project = metastore.create_project("research", "analysis")
    record = metastore.create_dataset(name="scoped", project_id=project.id)
    metastore.create_dataset_version(
        record, "2.0.0", status=DatasetStatus.COMPLETE, num_objects=2
    )
    for info in [
        dc.describe("research.analysis.scoped", session=test_session),
        dc.describe(
            "scoped", namespace="research", project="analysis", session=test_session
        ),
        dc.describe("scoped", "research", "analysis", "2.0.0", test_session),
    ]:
        assert info.name == "research.analysis.scoped@2.0.0"
        assert info.selected_version.num_objects == 2
    describe_dataset_cmd(test_session.catalog, "research.analysis.scoped")
    assert capsys.readouterr().out.startswith("Name: research.analysis.scoped@2.0.0\n")
    with pytest.raises(DatasetNotFoundError):
        dc.describe("scoped", session=test_session)


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
    info = dc.describe("legacy", session=test_session)
    assert info.feature_schema == schema
    assert info.to_dict()["schema"] == schema
    assert describe(test_session.catalog, "legacy")["schema"] == schema


def test_describe_jobs_include_direct_dependencies_and_dependents(
    test_session, monkeypatch
):
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

    raw_job = metastore.create_job("raw", "")
    raw = record("raw", raw_job)
    source_job = metastore.create_job("source", "")
    derived_job = metastore.create_job("derived", "")
    report_job = metastore.create_job("report", "")
    source = record("source", source_job)
    derived = record("derived", derived_job)
    report = record("report", report_job)
    metastore.add_dataset_dependency(source, "1.0.0", raw, "1.0.0")
    metastore.add_dataset_dependency(derived, "1.0.0", source, "1.0.0")
    metastore.add_dataset_dependency(report, "1.0.0", derived, "1.0.0")
    original = test_session.catalog.get_dataset
    reads = []

    def direct_only(name, *args, **kwargs):
        assert name != "raw"
        reads.append(name)
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("describe must not traverse the dependency graph")

    monkeypatch.setattr(test_session.catalog, "get_dataset", direct_only)
    monkeypatch.setattr(
        test_session.catalog, "get_dataset_dependencies_by_ids", forbidden
    )
    info = dc.describe("derived", session=test_session)
    assert reads == ["source", "report"]
    assert info.job.id == derived_job
    assert info.dependencies[0].dataset.job_id == source_job
    assert info.dependents[0].dataset.job_id == report_job
    assert set(info.jobs) == {source_job, derived_job, report_job}
    payload = info.to_dict()
    assert [job["id"] for job in payload["jobs"]] == sorted(info.jobs)


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset is unavailable")
@pytest.mark.parametrize(
    "created_at,local_time",
    [
        (datetime(2026, 11, 1, 10, 2, tzinfo=UTC), "2026-11-01 02:02:00 PST"),
        (datetime(2026, 7, 1, 10, 2, tzinfo=UTC), "2026-07-01 03:02:00 PDT"),
        (datetime(2026, 11, 1, 10, 2), "2026-11-01 02:02:00 PST"),
    ],
)
def test_describe_prints_local_time_and_keeps_json_utc(
    test_session, monkeypatch, capsys, created_at, local_time
):
    from datachain.cli.commands.datasets import print_dataset_description

    dc.read_values(value=[1], session=test_session).save("clock")
    info = dc.describe("clock", session=test_session)
    info.selected_version.created_at = created_at
    info.selected_version.finished_at = created_at
    info.job.created_at = created_at
    info.job.finished_at = created_at
    expected_utc = created_at.replace(tzinfo=UTC).isoformat()
    try:
        with monkeypatch.context() as patch:
            patch.setenv("TZ", "America/Los_Angeles")
            time.tzset()
            print_dataset_description(info)
            output = capsys.readouterr().out
            assert f"Created: {local_time}\n" in output
            assert f"Job_created: {local_time}\n" in output
            assert f"Job_finished: {local_time}\n" in output
            payload = info.to_dict()
            assert payload["created_at"] == expected_utc
            assert payload["finished_at"] == expected_utc
            assert payload["jobs"][0]["created_at"] == expected_utc
            assert payload["jobs"][0]["finished_at"] == expected_utc
    finally:
        time.tzset()


def test_describe_json_normalizes_offsets_to_utc(test_session):
    dc.read_values(value=[1], session=test_session).save("clock")
    info = dc.describe("clock", session=test_session)
    timestamp = datetime(
        2026, 11, 1, 10, 2, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    info.selected_version.created_at = timestamp
    info.job.created_at = timestamp
    payload = info.to_dict()
    assert payload["created_at"] == "2026-11-01T04:32:00+00:00"
    assert payload["jobs"][0]["created_at"] == "2026-11-01T04:32:00+00:00"


def test_describe_uuids_survive_renames_and_distinguish_versions(
    test_session, describe
):
    first = dc.read_values(value=[1], session=test_session).save("stable")
    dc.read_dataset("stable", session=test_session).save("derived_stable")
    older = dc.describe("stable@1.0.0", session=test_session)
    dc.read_values(value=[2], session=test_session).save("stable", version="2.0.0")
    latest = dc.describe("stable", session=test_session)
    record = test_session.catalog.get_dataset("stable", versions=None)

    assert older.dataset_uuid == latest.dataset_uuid == record.uuid
    assert older.version_uuid == record.get_version("1.0.0").uuid
    assert latest.version_uuid == record.get_version("2.0.0").uuid
    assert older.version_uuid != latest.version_uuid
    payload = describe(test_session.catalog, "stable")
    assert payload["dataset_uuid"] == record.uuid
    assert payload["version_uuid"] == latest.version_uuid
    assert {v["dataset_uuid"] for v in payload["versions"]} == {record.uuid}
    assert [v["version_uuid"] for v in payload["versions"]] == [
        older.version_uuid,
        latest.version_uuid,
    ]
    derived = dc.describe("derived_stable", session=test_session)
    dependent = describe(test_session.catalog, "stable@1.0.0")["dependents"][0]
    assert dependent["dataset_uuid"] == derived.dataset_uuid
    assert dependent["version_uuid"] == derived.version_uuid
    dependency = describe(test_session.catalog, "derived_stable")["dependencies"][0]
    assert dependency["dataset_uuid"] == record.uuid
    assert dependency["version_uuid"] == older.version_uuid

    test_session.catalog.edit_dataset(
        "stable", first.dataset.project, new_name="renamed"
    )
    renamed = dc.describe("renamed@1.0.0", session=test_session)

    assert renamed.name != older.name
    assert renamed.dataset_uuid == older.dataset_uuid
    assert renamed.version_uuid == older.version_uuid
    assert dc.read_dataset("renamed", session=test_session).to_values("value") == [2]
