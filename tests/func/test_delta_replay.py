from uuid import uuid4

import pytest

import datachain as dc
from datachain import C, DataModel, File
from datachain.error import SchemaDriftError
from datachain.toolkit import train_test_split
from tests.utils import reset_session_job_state


@pytest.mark.parametrize("static_first", [False, True])
@pytest.mark.parametrize("operation", ["map", "gen"])
@pytest.mark.parametrize("filter_at", [None, "before", "after"])
def test_delta_union_replaces_replayed_static_rows(
    test_session, static_first, operation, filter_at
):
    dc.read_values(id=[100], value=[1], session=test_session).save("union_static")

    def generate(id, value):
        for index in range(value):
            yield id, value, index

    def build(delta):
        changing = dc.read_dataset(
            "union_source",
            delta=delta,
            delta_on="id",
            delta_unsafe=True,
            session=test_session,
        )
        static = dc.read_dataset("union_static", session=test_session)
        chain = static.union(changing) if static_first else changing.union(static)
        if filter_at == "before":
            chain = chain.filter(C.value > 0)
        if operation == "gen":
            chain = chain.gen(generate, output={"id": int, "value": int, "index": int})
        else:
            chain = chain.map(index=lambda value: value, output=int)
        return chain.filter(C.value > 0) if filter_at == "after" else chain

    def rows(chain):
        return sorted(chain.to_iter("id", "value", "index"))

    dc.read_values(id=[1, 2], value=[1, 2], session=test_session).save("union_source")
    first = build(True).save("union_result")
    initial_rows = rows(first)
    for ids, values in [([1, 3], [0, 2]), ([1, 3, 4], [0, 1, 1])]:
        dc.read_values(id=ids, value=values, session=test_session).save("union_source")
        result = build(True).save("union_result")
        assert rows(result) == rows(build(False))
        assert result.filter(C.id == 100).count() == 1
    assert rows(first) == initial_rows
    assert build(True).save("union_result").version == result.version


@pytest.mark.parametrize("static_first", [False, True])
def test_storage_delta_union_does_not_duplicate_unchanged_files(
    test_session, tmp_path, static_first
):
    changing = tmp_path / "changing"
    static = tmp_path / "static"
    changing.mkdir()
    static.mkdir()
    (changing / "a.txt").write_text("2")
    (static / "b.txt").write_text("3")

    def read_number(file: File) -> int:
        return int(file.read_text())

    def build(delta):
        left = dc.read_storage(
            changing,
            update=True,
            delta=delta,
            delta_unsafe=True,
            session=test_session,
        )
        right = dc.read_storage(static, update=True, session=test_session)
        chain = right.union(left) if static_first else left.union(right)
        return chain.map(value=read_number).filter(C.value < 5)

    def rows(chain):
        return sorted(chain.to_iter("file.path", "value"))

    first = build(True).save("storage_union_result")
    (changing / "c.txt").write_text("4")
    added = build(True).save("storage_union_result")
    assert (
        rows(added)
        == rows(build(False))
        == [
            ("a.txt", 2),
            ("b.txt", 3),
            ("c.txt", 4),
        ]
    )
    (changing / "a.txt").write_text("99")
    modified = build(True).save("storage_union_result")
    assert rows(modified) == rows(build(False)) == [("b.txt", 3), ("c.txt", 4)]
    (changing / "c.txt").unlink()
    deleted = build(True).save("storage_union_result")
    assert rows(deleted) == rows(build(False)) == [("b.txt", 3)]
    assert rows(first) == [("a.txt", 2), ("b.txt", 3)]


def test_delta_split_is_disjoint_and_complete(test_session):
    def split():
        chain = dc.read_dataset(
            "split_source", delta=True, delta_on="id", session=test_session
        ).map(value=lambda id: id * 2, output=int)
        train, test = train_test_split(chain, [0.7, 0.3], seed=0)
        return (
            set(train.save("train").to_values("id")),
            set(test.save("test").to_values("id")),
        )

    dc.read_values(id=list(range(10)), session=test_session).save("split_source")
    split()
    dc.read_values(id=list(range(200)), session=test_session).save("split_source")
    train, test = split()
    assert not train & test
    assert train | test == set(range(200))


def test_delta_sibling_saves_reuse_udf(test_session):
    processed = []

    def record(id) -> int:
        processed.append(id)
        return id * 10

    def chain():
        return dc.read_dataset(
            "sibling_source", delta=True, delta_on="id", session=test_session
        ).map(value=record)

    def save():
        chain().filter(C.value < 50).save("small")
        chain().filter(C.value >= 50).save("big")

    dc.read_values(id=[1, 2, 3], session=test_session).save("sibling_source")
    save()
    dc.read_values(id=list(range(1, 9)), session=test_session).save("sibling_source")
    processed.clear()
    save()
    assert sorted(processed) == [4, 5, 6, 7, 8]
    assert sorted(dc.read_dataset("small").to_values("id")) == [1, 2, 3, 4]
    assert sorted(dc.read_dataset("big").to_values("id")) == [5, 6, 7, 8]


def test_delta_resumes_partial_udf(test_session, monkeypatch):
    monkeypatch.setattr("datachain.query.session.is_script_run", lambda: True)
    metastore = test_session.catalog.metastore
    state = {"fail": False, "calls": 0}

    def record(id) -> int:
        state["calls"] += 1
        if state["fail"] and state["calls"] > 3:
            raise RuntimeError("interrupted delta")
        return id * 2

    def save():
        return (
            dc.read_dataset(
                "resume_source", delta=True, delta_on="id", session=test_session
            )
            .settings(batch_size=1)
            .map(value=record)
            .save("resume_result")
        )

    def new_job(**kwargs):
        reset_session_job_state()
        job_id = str(uuid4())
        metastore.create_job("delta", "", job_id=job_id, **kwargs)
        monkeypatch.setenv("DATACHAIN_JOB_ID", job_id)
        return job_id

    try:
        dc.read_values(id=[1, 2, 3], session=test_session).save("resume_source")
        new_job()
        save()
        dc.read_values(id=list(range(1, 10)), session=test_session).save(
            "resume_source"
        )
        failed_job = new_job()
        state.update(fail=True, calls=0)
        with pytest.raises(RuntimeError, match="interrupted delta"):
            save()
        new_job(rerun_from_job_id=failed_job)
        state.update(fail=False, calls=0)
        result = save()
        assert state["calls"] == 3
        assert sorted(result.to_iter("id", "value")) == [
            (id, id * 2) for id in range(1, 10)
        ]
    finally:
        reset_session_job_state()


@pytest.mark.parametrize("operation", ["filter_before", "filter_after", "gen"])
def test_delta_empty_retry_reuses_result_version(test_session, operation):
    def generate(id):
        if id < 3:
            yield id, id * 2

    def build():
        chain = dc.read_dataset(
            "retry_source",
            delta=True,
            delta_on="id",
            delta_retry=True,
            session=test_session,
        )
        if operation == "gen":
            return chain.gen(generate, output={"id": int, "value": int})
        if operation == "filter_before":
            chain = chain.filter(C.id < 3)
        chain = chain.map(value=lambda id: id * 2, output=int)
        return chain.filter(C.value < 6) if operation == "filter_after" else chain

    dc.read_values(id=[1, 2, 3], session=test_session).save("retry_source")
    first = build().save("retry_result")
    for _ in range(2):
        result = build().save("retry_result")
        assert result.version == first.version
        assert sorted(result.to_iter("id", "value")) == [(1, 2), (2, 4)]


def test_delta_retry_can_remove_existing_result(test_session):
    state = {"offset": 0}

    def record(id) -> int:
        return id + state["offset"]

    def build():
        return (
            dc.read_dataset(
                "retry_removal_source",
                delta=True,
                delta_on="id",
                delta_retry="error",
                session=test_session,
            )
            .map(value=record)
            .map(error=lambda: "retry", output=str)
            .filter(C.value < 2)
        )

    dc.read_values(id=[1, 2], session=test_session).save("retry_removal_source")
    first = build().save("retry_removal_result")
    assert first.count() == 1
    state["offset"] = 10
    result = build().save("retry_removal_result")
    assert result.version != first.version
    assert result.count() == 0


class DeltaKey(DataModel):
    a: int
    b: str


class ReorderedDeltaKey(DataModel):
    b: str
    a: int


@pytest.mark.parametrize("source_optional", [False, True])
@pytest.mark.parametrize("result_key", [DeltaKey, DeltaKey | None, ReorderedDeltaKey])
@pytest.mark.parametrize("rename", [False, True])
def test_delta_model_key_matches_fields(
    test_session, source_optional, result_key, rename
):
    source_key = DeltaKey | None if source_optional else DeltaKey
    result_field = "result_key" if rename else "key"

    def source(keys, values):
        dc.read_values(
            key=keys,
            value=values,
            output={"key": source_key, "value": int},
            session=test_session,
        ).save("model_source")

    def generate(key, value):
        model = ReorderedDeltaKey if result_key is ReorderedDeltaKey else DeltaKey
        yield model(a=key.a, b=key.b), value

    def build(delta):
        return dc.read_dataset(
            "model_source",
            delta=delta,
            delta_on="key",
            delta_result_on=result_field,
            session=test_session,
        ).gen(generate, output={result_field: result_key, "value": int})

    keys = [DeltaKey(a=1, b="x"), DeltaKey(a=2, b="y"), DeltaKey(a=3, b="z")]
    source(keys, [10, 20, 30])
    build(True).save("model_result")
    source(keys[:2], [10, 50])
    result = build(True).save("model_result")

    def rows(chain):
        return sorted(
            (key.a, key.b, value) for key, value in chain.to_iter(result_field, "value")
        )

    assert rows(result) == rows(build(False)) == [(1, "x", 10), (2, "y", 50)]


class IncompatibleDeltaKey(DataModel):
    a: int
    c: str


def test_delta_model_key_rejects_incompatible_fields(test_session):
    def generate(key, value):
        yield IncompatibleDeltaKey(a=key.a, c=key.b), value

    def save():
        return (
            dc.read_dataset(
                "incompatible_source", delta=True, delta_on="key", session=test_session
            )
            .gen(generate, output={"key": IncompatibleDeltaKey, "value": int})
            .save("incompatible_result")
        )

    for value in [10, 20]:
        dc.read_values(
            key=[DeltaKey(a=1, b="x")], value=[value], session=test_session
        ).save("incompatible_source")
        if value == 10:
            save()
        else:
            with pytest.raises(SchemaDriftError, match="Delta key fields do not match"):
                save()
