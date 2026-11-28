from datachain.query import metrics
from datachain.query.session import Session


def test_metrics(test_session, catalog, mocker, monkeypatch):
    mocker.patch("datachain.query.session.Session.get", return_value=test_session)
    mocker.patch.dict("datachain.query.metrics.metrics", {}, clear=True)

    job_id = catalog.metastore.create_job("test_metrics", "")

    job = catalog.metastore.get_job(job_id)
    assert job
    assert job.id == job_id
    assert job.metrics == {}

    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)

    metrics.set("foo", 42)
    assert metrics.get("foo") == 42

    job = catalog.metastore.get_job(job_id)
    assert job
    assert job.id == job_id
    assert job.metrics == metrics.metrics == {"foo": 42}


def test_metrics_set_inside_in_memory_session_updates_studio_job(
    catalog_tmpfile, mocker, monkeypatch
):
    mocker.patch.dict("datachain.query.metrics.metrics", {}, clear=True)
    job_id = catalog_tmpfile.metastore.create_job("studio-job", "")
    monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
    monkeypatch.setenv("DATACHAIN__METASTORE", catalog_tmpfile.metastore.serialize())

    close = mocker.spy(type(catalog_tmpfile.metastore), "close_on_exit")

    with Session("mem", in_memory=True):
        metrics.set("foo", 42)
        assert close.call_count == 1

    job = catalog_tmpfile.metastore.get_job(job_id)
    assert job
    assert job.metrics == {"foo": 42}
