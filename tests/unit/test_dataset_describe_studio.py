import json
from dataclasses import asdict
from pathlib import Path

import pytest

import datachain as dc
from datachain.cli.commands.datasets import describe_dataset_cmd
from datachain.data_storage import JobQueryType, JobStatus
from datachain.dataset import DatasetStatus
from datachain.error import (
    DataChainError,
    DatasetNotFoundError,
    DatasetVersionNotFoundError,
)


@pytest.fixture
def studio_metadata(test_session, monkeypatch, requests_mock):
    monkeypatch.setenv("DATACHAIN_STUDIO_URL", "https://studio.test")
    monkeypatch.setenv("DATACHAIN_STUDIO_TOKEN", "test-token")
    monkeypatch.setenv("DATACHAIN_STUDIO_TEAM", "team")
    monkeypatch.setenv("DATACHAIN_IS_STUDIO", "false")
    metastore = test_session.catalog.metastore
    job_id = metastore.create_job(
        "produce",
        "process()",
        status=JobStatus.COMPLETE,
        python_version="3.12",
        params={"model": "v1"},
    )
    metastore.update_job(job_id, metrics={"score": 0.9})
    record = metastore.create_dataset(name="snapshot")
    for version, rows in [("1.0.0", 5), ("2.0.0", 8)]:
        record, _ = metastore.create_dataset_version(
            record,
            version,
            status=DatasetStatus.COMPLETE,
            num_objects=rows,
            feature_schema={"value": "int"},
            job_id=job_id,
            query_script="process()",
            sources="s3://bucket/photos/",
        )
    payload = metastore.get_dataset("snapshot", versions=None).to_dict()
    payload["name"] = "inventory"
    payload["project"]["name"] = "analytics"
    payload["project"]["namespace"]["name"] = "remote"
    payload = json.loads(json.dumps(payload, default=lambda v: v.isoformat()))
    requests_mock.get("https://studio.test/api/datachain/datasets/info", json=payload)
    job = asdict(metastore.get_job(job_id))
    job["status"] = JobStatus(job["status"]).name
    job["query_type"] = JobQueryType(job["query_type"]).name
    job["created_by"] = "somebody"
    job["environment"] = {"API_TOKEN": "not-a-real-token"}
    job["params"] = job["environment"]
    job["compute_cluster_id"] = "cluster-id"
    job["steps"] = []
    job["exit_code"] = 0
    job = json.loads(json.dumps(job, default=lambda v: v.isoformat()))
    requests_mock.get("https://studio.test/api/datachain/jobs/", json=[job])
    return payload, job


@pytest.mark.parametrize("version", [None, "1.0.0"])
def test_describe_studio_reads_metadata_only(
    test_session, studio_metadata, requests_mock, monkeypatch, capsys, version
):
    def forbidden(*args, **kwargs):
        raise AssertionError("describe must not read rows, storage, or pull data")

    monkeypatch.setattr(test_session.catalog.warehouse, "get_table", forbidden)
    monkeypatch.setattr(test_session.catalog, "pull_dataset", forbidden)
    name = "remote.analytics.inventory" + (f"@{version}" if version else "")
    info = dc.describe(name, session=test_session)
    assert info.name == "remote.analytics.inventory@" + (version or "2.0.0")
    assert info.available_versions == ["1.0.0", "2.0.0"]
    payload = studio_metadata[0]
    selected = next(
        v for v in payload["versions"] if v["version"] == (version or "2.0.0")
    )
    assert info.dataset_uuid == payload["uuid"]
    assert info.version_uuid == selected["uuid"]
    assert info.to_dict()["dataset_uuid"] == payload["uuid"]
    assert info.to_dict()["version_uuid"] == selected["uuid"]
    assert [v["version_uuid"] for v in info.to_dict()["versions"]] == [
        v["uuid"] for v in payload["versions"]
    ]
    assert info.selected_version.num_objects == (5 if version else 8)
    assert info.schema.serialize()["value"] == "int"
    assert info.selected_version.query_script == "process()"
    assert info.job.params == {}
    assert info.selected_version.params == {}
    assert "API_TOKEN" not in json.dumps(info.to_dict())
    assert info.job.metrics == {"score": 0.9}
    assert info.job.is_remote_execution is True
    assert not hasattr(info.job, "created_by")
    assert len(info.jobs) == 1
    assert info.dependencies is None and info.dependents is None
    assert info.to_dict()["dependencies"] is None
    assert len(requests_mock.request_history) == 2
    assert all(r.method == "GET" for r in requests_mock.request_history)
    describe_dataset_cmd(test_session.catalog, name)
    output = capsys.readouterr().out
    assert output.startswith(f"Name: {info.name}\n")
    assert "Dependencies: unavailable\nDependents: unavailable\n" in output


@pytest.mark.parametrize("status", [403, 429])
def test_describe_studio_jobs_unavailable(
    test_session, studio_metadata, requests_mock, status
):
    requests_mock.get(
        "https://studio.test/api/datachain/jobs/",
        status_code=status,
        json={"message": "Jobs access denied"},
    )
    info = dc.describe("remote.analytics.inventory", session=test_session)
    assert info.selected_version.num_objects == 8
    assert info.selected_version.job_id == studio_metadata[1]["id"]
    assert info.jobs is None and info.job is None
    assert info.to_dict()["jobs"] is None


@pytest.mark.parametrize("status", [403, 404])
def test_describe_studio_dataset_error(
    test_session, studio_metadata, requests_mock, status
):
    requests_mock.get(
        "https://studio.test/api/datachain/datasets/info",
        status_code=status,
        json={"message": "Dataset unavailable"},
    )
    error = DatasetNotFoundError if status == 404 else DataChainError
    with pytest.raises(error):
        dc.describe("remote.analytics.inventory", session=test_session)
    assert len(requests_mock.request_history) == 1


def test_describe_studio_missing_version(test_session, studio_metadata, requests_mock):
    with pytest.raises(DatasetVersionNotFoundError):
        dc.describe("remote.analytics.inventory@99.0.0", session=test_session)
    assert len(requests_mock.request_history) == 1


def test_describe_missing_local_dataset_does_not_contact_studio(
    test_session, requests_mock
):
    with pytest.raises(DatasetNotFoundError):
        dc.describe("missing", session=test_session)
    assert not requests_mock.called


@pytest.mark.parametrize(
    "name", ["remote.analytics.inventory", "remote/analytics/inventory"]
)
def test_skill_studio_snapshot_reads_metadata_once(
    test_session, studio_metadata, requests_mock, monkeypatch, name
):
    monkeypatch.syspath_prepend(
        str(
            Path(__file__).resolve().parents[2]
            / "src/datachain/skill/knowledge/scripts"
        )
    )
    import dataset

    reads = []

    def unavailable(name):
        reads.append(name)
        raise FileNotFoundError("rows unavailable")

    monkeypatch.setattr(dataset, "get_catalog", lambda: test_session.catalog)
    monkeypatch.setattr(dc, "read_dataset", unavailable)
    snapshot = dataset.fetch_dataset_snapshot(name, summarize=True)

    assert snapshot["name"] == "remote.analytics.inventory"
    assert snapshot["source"] == "studio"
    assert [v["version"] for v in snapshot["versions"]] == ["1.0.0", "2.0.0"]
    assert snapshot["versions"][-1]["schema"] == {
        "value": {"type": "int", "fields": None}
    }
    assert any("Dependencies are unavailable" in w for w in snapshot["warnings"])
    assert any("rows unavailable" in w for w in snapshot["warnings"])
    assert reads == ["remote.analytics.inventory@2.0.0"]
    assert len(requests_mock.request_history) == 1
    assert requests_mock.last_request.path == "/api/datachain/datasets/info"


def test_describe_metastore_without_dependents(test_session, monkeypatch):
    def unavailable(*args):
        raise NotImplementedError

    dc.read_values(value=[1], session=test_session).save("recorded")
    monkeypatch.setattr(
        test_session.catalog.metastore, "get_direct_dataset_dependents", unavailable
    )
    info = dc.describe("recorded", session=test_session)
    assert info.dependencies == []
    assert info.dependents is None
    assert info.to_dict()["dependents"] is None


@pytest.mark.parametrize("missing_version", [None, "9.0.0"])
def test_describe_missing_dependency_version_is_unavailable(
    test_session, monkeypatch, capsys, missing_version
):
    from datachain.dataset import DatasetDependency

    metastore = test_session.catalog.metastore
    record = metastore.create_dataset(name="output")
    metastore.create_dataset_version(record, "1.0.0", status=DatasetStatus.COMPLETE)
    dependency = DatasetDependency.parse(
        metastore.default_namespace_name,
        metastore.default_project_name,
        1,
        1,
        None,
        "input",
        missing_version,
        None,
    )
    monkeypatch.setattr(
        metastore, "get_direct_dataset_dependencies", lambda *args: [dependency]
    )
    info = dc.describe("output", session=test_session)
    assert info.to_dict()["dependencies"] == [None]
    describe_dataset_cmd(test_session.catalog, "output")
    assert "unavailable dependency" in capsys.readouterr().out


def test_describe_explicit_team_overrides_pulled_copy(
    test_session, studio_metadata, requests_mock, capsys
):
    metastore = test_session.catalog.metastore
    project = metastore.create_project("remote", "analytics")
    record = metastore.create_dataset(name="inventory", project_id=project.id)
    metastore.create_dataset_version(record, "1.0.0", status=DatasetStatus.COMPLETE)
    describe_dataset_cmd(
        test_session.catalog, "remote.analytics.inventory", team="chosen-team"
    )
    assert "Name: remote.analytics.inventory@2.0.0" in capsys.readouterr().out
    assert len(requests_mock.request_history) == 2
    assert all(
        request.qs["team_name"] == ["chosen-team"]
        for request in requests_mock.request_history
    )


def test_metadata_batches_and_reuses_direct_relation_records(test_session, monkeypatch):
    from datachain.lib.dataset_metadata import read_dataset_metadata

    metastore = test_session.catalog.metastore

    def record(name):
        dataset = metastore.create_dataset(name=name)
        return metastore.create_dataset_version(
            dataset, "1.0.0", status=DatasetStatus.COMPLETE
        )[0]

    selected = record("selected")
    for i in range(3):
        metastore.add_dataset_dependency(
            selected, "1.0.0", record(f"input{i}"), "1.0.0"
        )
        metastore.add_dataset_dependency(
            record(f"output{i}"), "1.0.0", selected, "1.0.0"
        )
    reads = []
    get_dataset = metastore.get_dataset

    def root_only(name, *args, **kwargs):
        reads.append(name)
        assert name == "selected"
        return get_dataset(name, *args, **kwargs)

    batches = []
    related = metastore.get_dataset_related_versions

    def batch(*args):
        batches.append(args)
        return related(*args)

    monkeypatch.setattr(metastore, "get_dataset", root_only)
    monkeypatch.setattr(metastore, "get_dataset_related_versions", batch)
    metadata = read_dataset_metadata(test_session.catalog, "selected")
    for _ in range(2):
        info = metadata.describe()
        assert len(info.dependencies) == len(info.dependents) == 3
    assert reads == ["selected"]
    assert len(batches) == 1


def test_metadata_groups_api_relations_by_dataset(test_session, monkeypatch):
    from datachain.dataset import DatasetDependency

    metastore = test_session.catalog.metastore
    source = metastore.create_dataset(name="input")
    for version in ["1.0.0", "2.0.0"]:
        metastore.create_dataset_version(source, version, status=DatasetStatus.COMPLETE)
    output = metastore.create_dataset(name="output")
    metastore.create_dataset_version(output, "1.0.0", status=DatasetStatus.COMPLETE)
    dependencies = [
        DatasetDependency.parse(
            metastore.default_namespace_name,
            metastore.default_project_name,
            i,
            source.id,
            i,
            source.name,
            version,
            None,
        )
        for i, version in enumerate(["1.0.0", "2.0.0"], start=1)
    ]

    def unavailable(*args):
        raise NotImplementedError

    monkeypatch.setattr(metastore, "get_dataset_related_versions", unavailable)
    monkeypatch.setattr(
        metastore, "get_direct_dataset_dependencies", lambda *a: dependencies
    )
    original = metastore.get_dataset
    calls = []

    def read(name, *args, **kwargs):
        calls.append((name, kwargs.get("versions")))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(metastore, "get_dataset", read)
    info = dc.describe("output", session=test_session)
    assert [entry.dataset.version for entry in info.dependencies] == ["1.0.0", "2.0.0"]
    assert [name for name, _ in calls] == ["output", "input"]
    assert set(calls[-1][1]) == {"1.0.0", "2.0.0"}


def test_describe_omits_temporary_dependents(test_session):
    metastore = test_session.catalog.metastore
    source = metastore.create_dataset(name="input")
    source, _ = metastore.create_dataset_version(
        source, "1.0.0", status=DatasetStatus.COMPLETE
    )
    for name in ["visible", "session_hidden"]:
        output = metastore.create_dataset(name=name)
        output, _ = metastore.create_dataset_version(
            output, "1.0.0", status=DatasetStatus.COMPLETE
        )
        metastore.add_dataset_dependency(output, "1.0.0", source, "1.0.0")
    info = dc.describe("input", session=test_session)
    assert [entry.dataset.name for entry in info.dependents] == ["visible"]


def test_describe_rejects_incomplete_only_dataset(test_session):
    metastore = test_session.catalog.metastore
    record = metastore.create_dataset(name="incomplete")
    metastore.create_dataset_version(record, "1.0.0", status=DatasetStatus.CREATED)

    with pytest.raises(DatasetVersionNotFoundError, match="no completed version"):
        dc.describe("incomplete", session=test_session)


@pytest.mark.parametrize("status", [DatasetStatus.CREATED, DatasetStatus.FAILED])
def test_describe_falls_back_from_incomplete_remote_copy(
    test_session, studio_metadata, requests_mock, status
):
    metastore = test_session.catalog.metastore
    project = metastore.create_project("remote", "analytics")
    cached = metastore.create_dataset(name="inventory", project_id=project.id)
    metastore.create_dataset_version(cached, "1.0.0", status=status)

    info = dc.describe("remote.analytics.inventory", session=test_session)

    assert info.selected_version.version == "2.0.0"
    assert info.selected_version.num_objects == 8
    assert info.available_versions == ["1.0.0", "2.0.0"]
    assert len(requests_mock.request_history) == 2


@pytest.mark.parametrize("version", [None, "2.0.0"])
def test_describe_prefers_pulled_copy_and_fetches_missing_version(
    test_session, studio_metadata, requests_mock, version
):
    metastore = test_session.catalog.metastore
    project = metastore.create_project("remote", "analytics")
    cached = metastore.create_dataset(name="inventory", project_id=project.id)
    metastore.create_dataset_version(
        cached, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=1
    )
    name = "remote.analytics.inventory" + (f"@{version}" if version else "")
    info = dc.describe(name, session=test_session)
    assert info.selected_version.version == ("2.0.0" if version else "1.0.0")
    assert info.selected_version.num_objects == (8 if version else 1)
    assert info.available_versions == (["1.0.0", "2.0.0"] if version else ["1.0.0"])
    assert info.dependencies == (None if version else [])
    assert len(requests_mock.request_history) == (2 if version else 0)


def test_describe_studio_missing_job(
    test_session, studio_metadata, requests_mock, capsys
):
    requests_mock.get("https://studio.test/api/datachain/jobs/", json=[])
    info = dc.describe("remote.analytics.inventory", session=test_session)
    assert info.jobs == {}
    assert info.selected_version.job_id == studio_metadata[1]["id"]
    assert info.job is None
    describe_dataset_cmd(test_session.catalog, "remote.analytics.inventory")
    assert (
        f"Job: {info.selected_version.job_id} (unavailable)" in capsys.readouterr().out
    )


def test_describe_studio_jobs_error(test_session, studio_metadata, requests_mock):
    requests_mock.get(
        "https://studio.test/api/datachain/jobs/",
        status_code=400,
        json={"message": "Job request failed"},
    )
    with pytest.raises(DataChainError, match="Job request failed"):
        dc.describe("remote.analytics.inventory", session=test_session)


def test_describe_studio_distinct_producing_jobs(
    test_session, studio_metadata, requests_mock
):
    payload, first = studio_metadata
    second = {
        **first,
        "id": "00000000-0000-0000-0000-000000000002",
        "name": "second",
        "metrics": {"score": 0.8},
    }
    payload["versions"][1]["job_id"] = second["id"]
    requests_mock.get("https://studio.test/api/datachain/datasets/info", json=payload)
    jobs = {first["id"]: first, second["id"]: second}

    def job_response(request, context):
        return [jobs[request.qs["job_id"][0]]]

    requests_mock.get("https://studio.test/api/datachain/jobs/", json=job_response)
    info = dc.describe("remote.analytics.inventory", session=test_session)
    assert set(info.jobs) == {second["id"]}
    assert info.job.id == second["id"]
    assert info.selected_version.metrics == {"score": 0.8}
    assert [v.metrics for v in info.versions] == [{}, {"score": 0.8}]
    assert len(requests_mock.request_history) == 2


def test_describe_uses_metastore_default_namespace(
    test_session, monkeypatch, requests_mock
):
    metastore = test_session.catalog.metastore
    monkeypatch.setattr(
        type(metastore), "default_namespace_name", property(lambda self: "teamuser")
    )
    project = metastore.create_project("teamuser", "sandbox")
    record = metastore.create_dataset(name="recorded", project_id=project.id)
    metastore.create_dataset_version(
        record, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=3
    )
    info = dc.describe("teamuser.sandbox.recorded", session=test_session)
    assert info.selected_version.num_objects == 3
    assert info.dependencies == []
    assert not requests_mock.called


def test_describe_prints_metastore_default_scope(test_session, monkeypatch, capsys):
    metastore = test_session.catalog.metastore
    project = metastore.create_project("teamuser", "sandbox")
    record = metastore.create_dataset(name="recorded", project_id=project.id)
    metastore.create_dataset_version(record, "1.0.0", status=DatasetStatus.COMPLETE)
    monkeypatch.setattr(
        type(metastore), "default_namespace_name", property(lambda s: "teamuser")
    )
    monkeypatch.setattr(
        type(metastore), "default_project_name", property(lambda s: "sandbox")
    )
    describe_dataset_cmd(test_session.catalog, "recorded")
    assert capsys.readouterr().out.startswith("Name: recorded@1.0.0\n")


def test_skill_pulled_snapshot_keeps_qualified_name(
    test_session, monkeypatch, requests_mock
):
    from datachain.lib.dataset_metadata import read_dataset_metadata
    from datachain.skill.knowledge.collect import snapshot_from_metadata

    metastore = test_session.catalog.metastore
    project = metastore.create_project("remote", "analytics")
    record = metastore.create_dataset(name="inventory", project_id=project.id)
    metastore.create_dataset_version(record, "1.0.0", status=DatasetStatus.COMPLETE)
    snapshot = snapshot_from_metadata(
        read_dataset_metadata(test_session.catalog, "remote.analytics.inventory")
    )
    assert snapshot["name"] == "remote.analytics.inventory"
    assert snapshot["source"] == "local"
    assert not requests_mock.called


@pytest.mark.parametrize("qualified", [False, True])
def test_describe_storage_name_roundtrip(test_session, requests_mock, qualified):
    metastore = test_session.catalog.metastore
    listing = metastore.create_dataset(
        name="lst__s3://bucket.example/photos/", project_id=metastore.listing_project.id
    )
    metastore.create_dataset_version(
        listing, "1.0.0", status=DatasetStatus.COMPLETE, num_objects=4
    )
    name = "lst__s3://bucket.example/photos/@1.0.0"
    canonical = (
        f"{metastore.system_namespace_name}.{metastore.listing_project_name}.{name}"
    )
    info = dc.describe(canonical if qualified else name, session=test_session)
    assert info.name == canonical
    assert info.to_dict()["name"] == canonical
    assert info.selected_version.num_objects == 4
    assert not requests_mock.called
