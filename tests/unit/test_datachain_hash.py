from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

import datachain as dc
from datachain import func
from datachain.lib.dc import C

DF_DATA = {
    "first_name": ["Alice", "Bob", "Charlie", "David", "Eva"],
    "age": [25, 30, 35, 40, 45],
}


class Person(BaseModel):
    name: str
    age: float


class PersonAgg(BaseModel):
    name: str
    ages: float


class Player(BaseModel):
    name: str
    sport: str


class Worker(BaseModel):
    name: str
    age: float
    title: str


persons = [
    Person(name="p1", age=10),
    Person(name="p2", age=20),
    Person(name="p3", age=30),
    Person(name="p4", age=40),
    Person(name="p5", age=40),
    Person(name="p6", age=60),
]


players = [
    Player(name="p1", sport="baksetball"),
    Player(name="p2", sport="soccer"),
    Player(name="p3", sport="baseball"),
    Player(name="p4", sport="tennis"),
]


@pytest.fixture
def mock_get_listing():
    with patch("datachain.lib.dc.storage.get_listing") as mock:
        mock.return_value = ("lst__s3://my-bucket", "", "", True)
        yield mock


def _set_stable_uuid(test_session, name, uuid):
    """Set a stable UUID on a dataset version for deterministic hash tests."""
    parts = name.split(".")
    ds_name = parts[-1]
    ns = parts[0] if len(parts) > 1 else "default"
    proj = parts[1] if len(parts) > 2 else "default"
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset(
            ds_name,
            namespace_name=ns,
            project_name=proj,
            versions=["1.0.0"],
        ),
        "1.0.0",
        uuid=uuid,
    )


def test_read_values():
    """
    Hash of the chain started with read_values is currently inconsistent.
    Goal of this test is just to check it doesn't break.
    """
    assert dc.read_values(num=[1, 2, 3]).hash() is not None


def test_read_csv(test_session, tmp_dir):
    """
    Hash of the chain started with read_csv is currently inconsistent.
    Goal of this test is just to check it doesn't break.
    """
    path = tmp_dir / "test.csv"
    pd.DataFrame(DF_DATA).to_csv(path, index=False)
    assert dc.read_csv(path.as_uri(), session=test_session).hash() is not None


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_read_json(test_session, tmp_dir):
    """
    Hash of the chain started with read_json is currently inconsistent.
    Goal of this test is just to check it doesn't break.
    """
    path = tmp_dir / "test.jsonl"
    dc.read_pandas(pd.DataFrame(DF_DATA), session=test_session).to_jsonl(path)
    assert (
        dc.read_json(path.as_uri(), format="jsonl", session=test_session).hash()
        is not None
    )


def test_read_pandas(test_session, tmp_dir):
    """
    Hash of the chain started with read_pandas is currently inconsistent.
    Goal of this test is just to check it doesn't break.
    """
    df = pd.DataFrame(DF_DATA)
    assert dc.read_pandas(df, session=test_session).hash() is not None


def test_read_parquet(test_session, tmp_dir):
    """
    Hash of the chain started with read_parquet is currently inconsistent.
    Goal of this test is just to check it doesn't break.
    """
    df = pd.DataFrame(DF_DATA)
    path = tmp_dir / "test.parquet"
    dc.read_pandas(df, session=test_session).to_parquet(path)
    assert dc.read_parquet(path.as_uri(), session=test_session).hash() is not None


def test_read_storage(mock_get_listing, test_session):
    assert dc.read_storage("s3://bucket", session=test_session).hash() == (
        "811e7089ead93a572d75d242220f6b94fd30f21def1bbcf37f095f083883bc41"
    )


def test_read_dataset(test_session):
    dc.read_values(num=[1, 2, 3], session=test_session).save("dev.animals.cats")
    _set_stable_uuid(
        test_session, "dev.animals.cats", "b1c2d3e4-f5a6-4b1c-8d3e-4f5a6b1c2d3e"
    )
    assert dc.read_dataset(
        name="dev.animals.cats", version="1.0.0", session=test_session
    ).hash() == ("58c939b8626443e5d68e9e419a9a5fd1bc7282ca0c5e06cfeb578635e9703a06")


def test_read_dataset_delta_hash_changes_with_delta_spec(test_session):
    dataset_name = "dev.animals.delta_hash"
    dc.read_values(id=[1, 2, 3], value=[10, 20, 30], session=test_session).save(
        dataset_name
    )

    base_hash = dc.read_dataset(
        name=dataset_name,
        version="1.0.0",
        session=test_session,
    ).hash()
    delta_hash = dc.read_dataset(
        name=dataset_name,
        version="1.0.0",
        session=test_session,
        delta=True,
        delta_on="id",
    ).hash()
    delta_compare_hash = dc.read_dataset(
        name=dataset_name,
        version="1.0.0",
        session=test_session,
        delta=True,
        delta_on="id",
        delta_compare="value",
    ).hash()
    delta_unsafe_hash = dc.read_dataset(
        name=dataset_name,
        version="1.0.0",
        session=test_session,
        delta=True,
        delta_on="id",
        delta_unsafe=True,
    ).hash()

    assert base_hash != delta_hash
    assert delta_hash != delta_compare_hash
    assert delta_hash != delta_unsafe_hash


def test_order_of_steps(mock_get_listing):
    assert (
        dc.read_storage("s3://bucket").mutate(new=10).filter(C("age") > 20).hash()
    ) == "b07f11244f1f84e4ecde87976fc380b4b8b656b0202294179e30be2112df7d3a"

    assert (
        dc.read_storage("s3://bucket").filter(C("age") > 20).mutate(new=10).hash()
    ) == "82780df484ce63e499ceed6ef3418920fdf68461a6b5f24234d3c0628c311c02"


def test_all_possible_steps(test_session):
    persons_ds_name = "dev.my_pr.persons"
    players_ds_name = "dev.my_pr.players"

    def map_worker(person: Person) -> Worker:
        return Worker(
            name=person.name,
            age=person.age,
            title="worker",
        )

    def gen_persons(person):
        yield Person(
            age=person.age * 2,
            name=person.name + "_suf",
        )

    def agg_persons(persons):
        return PersonAgg(ages=sum(p.age for p in persons), name=persons[0].age)

    dc.read_values(person=persons, session=test_session).save(persons_ds_name)
    dc.read_values(player=players, session=test_session).save(players_ds_name)
    _set_stable_uuid(
        test_session, persons_ds_name, "a1a1a1a1-b2b2-4c3c-8d4d-e5e5e5e5e5e5"
    )
    _set_stable_uuid(
        test_session, players_ds_name, "f6f6f6f6-a7a7-4b8b-8c9c-d0d0d0d0d0d0"
    )

    players_chain = dc.read_dataset(
        players_ds_name, version="1.0.0", session=test_session
    )

    assert (
        dc.read_dataset(persons_ds_name, version="1.0.0", session=test_session)
        .mutate(age_double=C("person.age") * 2)
        .filter(C("person.age") > 20)
        .order_by("person.name", "person.age")
        .gen(
            person=gen_persons,
            output=Person,
        )
        .map(
            worker=map_worker,
            params="person",
            output={"worker": Worker},
        )
        .agg(
            persons=agg_persons,
            partition_by=C.person.name,
            params="person",
            output={"persons": PersonAgg},
        )
        .merge(players_chain, "persons.name", "player.name")
        .distinct("persons.name")
        .sample(10)
        .offset(2)
        .limit(5)
        .group_by(age_avg=func.avg("persons.ages"), partition_by="persons.name")
        .select("persons.name", "age_avg")
        .subtract(
            players_chain,
            on=["persons.name"],
            right_on=["player.name"],
        )
        .hash()
    ) == "8e5cf0a718406a94c99ab7ffafc67aa5430f79a276deb1f1d87e5a5991bc56bf"


def test_diff(test_session):
    persons_ds_name = "dev.my_pr.persons"
    players_ds_name = "dev.my_pr.players"

    dc.read_values(person=persons, session=test_session).save(persons_ds_name)
    dc.read_values(player=players, session=test_session).save(players_ds_name)
    _set_stable_uuid(
        test_session, persons_ds_name, "a1a1a1a1-b2b2-4c3c-8d4d-e5e5e5e5e5e5"
    )
    _set_stable_uuid(
        test_session, players_ds_name, "f6f6f6f6-a7a7-4b8b-8c9c-d0d0d0d0d0d0"
    )

    players_chain = dc.read_dataset(
        players_ds_name, version="1.0.0", session=test_session
    )

    assert (
        dc.read_dataset(persons_ds_name, version="1.0.0", session=test_session)
        .diff(
            players_chain,
            on=["person.name"],
            right_on=["player.name"],
            status_col="diff",
        )
        .hash()
    ) == "5d74e62f9722b62ba7a5ab0a9661570920cc08c68aa8102a0876390e376ba99b"
