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
    assert dc.read_dataset(
        name="dev.animals.cats", version="1.0.0", session=test_session
    ).hash() == ("51f2e5b81e40a22062a75c1590d0ccab880d182df9b39f610c6ccc503a5eb33c")


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
    ) == "bd685bd97746a8e0e012c7029c7f2c8b17fc7eb5b7a5cd8fa5dacada57d75a07"


def test_diff(test_session):
    persons_ds_name = "dev.my_pr.persons"
    players_ds_name = "dev.my_pr.players"

    dc.read_values(person=persons, session=test_session).save(persons_ds_name)
    dc.read_values(player=players, session=test_session).save(players_ds_name)

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
    ) == "02a8596af8465a78b6369bb3d09c823c5a5fdcbbfd3695abd57ed1a01c9bb5f3"
