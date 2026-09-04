import hashlib
import math
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from pydantic import BaseModel

import datachain as dc
import datachain.lib.udf
from datachain import C, func
from datachain.dataset import DatasetRecord, DatasetVersion
from datachain.func.func import Func
from datachain.lib.signal_schema import SignalSchema
from datachain.lib.udf import (
    Aggregator,
    Generator,
    Mapper,
    _physical_schema_hash,
)
from datachain.lib.udf_signature import UdfSignature
from datachain.query.dataset import (
    QueryStep,
    RowGenerator,
    SQLCount,
    SQLDistinct,
    SQLFilter,
    SQLGroupBy,
    SQLJoin,
    SQLLimit,
    SQLMutate,
    SQLOffset,
    SQLOrderBy,
    SQLSelect,
    SQLSelectExcept,
    SQLUnion,
    Subtract,
    UDFSignal,
)


class CustomFeature(BaseModel):
    sqrt: float
    my_name: str


def double(x):
    return x * 2


def double2(y):
    return 7 * 2


def double_gen(x):
    yield x * 2


def double_gen_multi_arg(x, y):
    yield x * 2
    yield y * 2


def double_default(x, y=2):
    return x * y


def double_kwonly(x, *, factor=3):
    return x * factor


def map_custom_feature(m_fr):
    return CustomFeature(
        sqrt=math.sqrt(m_fr.count),
        my_name=m_fr.nnn + "_suf",
    )


def custom_feature_gen(m_fr):
    yield CustomFeature(
        sqrt=math.sqrt(m_fr.count),
        my_name=m_fr.nnn + "_suf",
    )


# Class-based UDFs for testing hash calculation
class DoubleMapper(Mapper):
    """Class-based Mapper that overrides process()."""

    def process(self, x):
        return x * 2


class TripleGenerator(Generator):
    """Class-based Generator that overrides process()."""

    def process(self, x):
        yield x * 3
        yield x * 3 + 1


@pytest.fixture
def numbers_dataset(test_session):
    """
    Fixture to create dataset with stable / constant UUID to have consistent
    hash values in tests as it goes into chain hash calculation
    """
    test_session.catalog.metastore.create_project("dev", "num")
    dc.read_values(num=list(range(100)), session=test_session).save("dev.num.numbers")
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset(
            "numbers",
            namespace_name="dev",
            project_name="num",
            versions=["1.0.0"],
        ),
        "1.0.0",
        uuid="9045d46d-7c57-4442-aae3-3ca9e9f286c4",
    )

    return test_session.catalog.get_dataset(
        "numbers",
        namespace_name="dev",
        project_name="num",
        versions=["1.0.0"],
    )


@pytest.mark.parametrize(
    "inputs",
    [
        (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
        (),
        (C("name"),),
        (func.rand().label("random"),),
        ("name",),
    ],
)
def test_select_hash(inputs):
    assert SQLSelect(inputs).hash() == SQLSelect(inputs).hash()


def test_select_hash_different_inputs():
    assert SQLSelect((C("name"),)).hash() != SQLSelect((C("age"),)).hash()


@pytest.mark.parametrize(
    "inputs",
    [
        (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
        (),
        (C("name"),),
        ("name",),
    ],
)
def test_select_except_hash(inputs):
    assert SQLSelectExcept(inputs).hash() == SQLSelectExcept(inputs).hash()


def test_select_except_hash_different_inputs():
    assert SQLSelectExcept((C("name"),)).hash() != SQLSelectExcept((C("age"),)).hash()


@pytest.mark.parametrize(
    "inputs",
    [
        (sa.and_(C("name") != "John", C("age") * 10 > 100)),
        (),
        (C("files.path").glob("*.jpg"),),
        sa.or_(C("age") > 50, C("country") == "US"),
    ],
)
def test_filter_hash(inputs):
    assert SQLFilter(inputs).hash() == SQLFilter(inputs).hash()


def test_filter_hash_different_inputs():
    assert SQLFilter((C("age") > 20,)).hash() != SQLFilter((C("age") > 30,)).hash()


def test_mutate_hash():
    schema = SignalSchema({"id": int})

    def _mutate(inputs):
        cols = (
            v.label(k).get_column(schema) if isinstance(v, Func) else v.label(k)
            for k, v in inputs.items()
        )
        return SQLMutate(cols, new_schema=None).hash()

    h1 = _mutate({"new_id": func.sum("id")})
    h2 = _mutate({"new_id": C("id") * 10, "old_id": C("id")})
    h3 = _mutate({})

    assert h1 == _mutate({"new_id": func.sum("id")})
    assert len({h1, h2, h3}) == 3


@pytest.mark.parametrize(
    "inputs", [(C("name"), C("age")), ("name",), (sa.desc(C("name")),), ()]
)
def test_order_by_hash(inputs):
    assert SQLOrderBy(inputs).hash() == SQLOrderBy(inputs).hash()


def test_order_by_hash_different_inputs():
    assert SQLOrderBy((C("name"),)).hash() != SQLOrderBy((C("age"),)).hash()


def test_limit_hash():
    assert SQLLimit(5).hash() == SQLLimit(5).hash()
    assert SQLLimit(5).hash() != SQLLimit(0).hash()


def test_offset_hash():
    assert SQLOffset(5).hash() == SQLOffset(5).hash()
    assert SQLOffset(5).hash() != SQLOffset(0).hash()


def test_count_hash():
    assert SQLCount().hash() == SQLCount().hash()


def test_distinct_hash():
    assert (
        SQLDistinct(("name",), dialect=None).hash()
        == SQLDistinct(("name",), dialect=None).hash()
    )
    assert (
        SQLDistinct(("name",), dialect=None).hash()
        != SQLDistinct(("age",), dialect=None).hash()
    )


def test_union_hash(test_session, numbers_dataset):
    chain1 = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("dev.num.numbers").filter(C("num") < 50).limit(20)

    h = SQLUnion(chain1._query, chain2._query).hash()
    assert h == SQLUnion(chain1._query, chain2._query).hash()


def test_join_hash(test_session, numbers_dataset):
    chain1 = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("dev.num.numbers").filter(C("num") < 50).limit(20)

    def _join(predicates, inner, full, rname):
        return SQLJoin(
            test_session.catalog,
            chain1._query,
            chain2._query,
            predicates,
            inner,
            full,
            rname,
        ).hash()

    h1 = _join("id", True, False, "{name}_right")
    h2 = _join(("id", "name"), False, True, "{name}_r")
    h3 = _join(sa.column("id"), True, False, "{name}_right")

    assert h1 == _join("id", True, False, "{name}_right")
    assert len({h1, h2, h3}) == 3


def test_group_by_hash():
    schema = SignalSchema({"id": int})

    def _group_by(columns, partition_by):
        cols = [v.get_column(schema, label=k) for k, v in columns.items()]
        return SQLGroupBy(cols, partition_by).hash()

    h1 = _group_by({"cnt": func.count(), "sum": func.sum("id")}, [C("id")])
    h2 = _group_by({"cnt": func.count(), "sum": func.sum("id")}, [C("id"), C("name")])
    h3 = _group_by({"cnt": func.count()}, [])

    assert h1 == _group_by({"cnt": func.count(), "sum": func.sum("id")}, [C("id")])
    assert len({h1, h2, h3}) == 3


@pytest.mark.parametrize(
    "on",
    [
        [("id", "id")],
        [("id", "id"), ("name", "name")],
        [],
    ],
)
def test_subtract_hash(test_session, numbers_dataset, on):
    chain = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(20)
    h = Subtract(chain._query, test_session.catalog, on).hash()
    assert h == Subtract(chain._query, test_session.catalog, on).hash()


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            double,
            ["x"],
            {"double": int},
            "ad0e316d05fd6532cd3e93f9861e0a7eb0edd673ac9b51c211a6a911981df6c9",
        ),
        (
            double2,
            ["y"],
            {"double": int},
            "e07754540c5873f16682684051e8dd8f0e90f8a41294f37b05e26f9f8554dca5",
        ),
        (
            double_default,
            ["x"],
            {"double": int},
            "e9fb2d020370fd91046405ae635fbf40bc886c5f5109c3381055563a41e7a6e9",
        ),
        (
            double_kwonly,
            ["x"],
            {"double": int},
            "214c2dff7b31eaf3cf0d18d5b247c9f84412524bd7642fabc7535e236d2a014f",
        ),
        (
            map_custom_feature,
            ["t1"],
            {"x": CustomFeature},
            "eaa42c165a774777cdb16ec094faa27d4c861d68ae7723129a513a29b76a14b5",
        ),
        (
            DoubleMapper(),
            ["x"],
            {"double": int},
            "2510ed7fd1958281cfed724b1e786851552a4c22522b7eacdfee27beec601be4",
        ),
    ],
)
def test_udf_mapper_hash(
    func,
    params,
    output,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Mapper._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert UDFSignal(udf_adapter, None).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            double_gen,
            ["x"],
            {"double": int},
            "60b0c79ad0fdaa4fc40e16c18cc9a849854eba979ed62d1c7c08caf51b44c338",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "2b6edd763d8304362b5831921f93189173266098423a7880b4debd9027e0d50a",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "4ef989cdce2eade740b886331a4a5919fde1f0963c0a2410881904666e7e6966",
        ),
        (
            TripleGenerator(),
            ["x"],
            {"triple": int},
            "8a977a38c3d9ff5f9f23c2eb2e85b425fd4b4e74a068b28f73d861d6b7d3561b",
        ),
    ],
)
def test_udf_generator_hash(
    func,
    params,
    output,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Generator._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert RowGenerator(udf_adapter, None).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,partition_by,_hash",
    [
        (
            double_gen,
            ["x"],
            {"double": int},
            [C("x")],
            "c55b45ab500518a2286aa6b4be7a08c401ce3cdf59582388cada0e622c68bd8c",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "67e85bdb21bbc4e3c6efc8afd159677dedd614d813c253a14df2546b3b36ce65",
        ),
    ],
)
def test_udf_aggregator_hash(
    func,
    params,
    output,
    partition_by,
    _hash,
):
    sign = UdfSignature.parse("", {}, func, params, output, False)
    udf_adapter = Aggregator._create(sign, SignalSchema(sign.params)).to_udf_wrapper()
    assert RowGenerator(udf_adapter, None, partition_by=partition_by).hash() == _hash


def test_query_step_hash_uses_version_uuid():
    """QueryStep hash is based on dataset version UUID, not name/version string."""
    uuid1 = "a1b2c3d4-e5f6-4a1b-8c3d-4e5f6a1b2c3d"
    uuid2 = "f6e5d4c3-b2a1-4f6e-8d4c-3b2a1f6e5d4c"

    ds = DatasetRecord(
        id=1,
        uuid="d4e5f6a1-b2c3-4d4e-8f6a-1b2c3d4e5f6a",
        name="test_ds",
        description="",
        attrs=[],
        _versions=[
            DatasetVersion(
                id=1,
                uuid=uuid1,
                dataset_id=1,
                version="1.0.0",
                status=1,
                created_at=None,
                finished_at=None,
                error_message="",
                error_stack="",
                num_objects=0,
                size=0,
                feature_schema=None,
                script_output="",
                schema=None,
                _preview_data=[],
                _preview_loaded=True,
            ),
        ],
        _versions_loaded=True,
        status=1,
        schema={},
        feature_schema={},
        project=None,
    )

    hash1 = QueryStep(None, ds, "1.0.0").hash()
    assert hash1 == hashlib.sha256(uuid1.encode()).hexdigest()

    # Same name/version but different UUID produces different hash
    ds.versions[0].uuid = uuid2
    hash2 = QueryStep(None, ds, "1.0.0").hash()
    assert hash2 == hashlib.sha256(uuid2.encode()).hexdigest()
    assert hash1 != hash2

    # Same UUID with different dataset name produces same hash
    ds.versions[0].uuid = uuid1
    ds.name = "completely_different_name"
    assert QueryStep(None, ds, "1.0.0").hash() == hash1


def test_a_udf_hash_follows_the_physical_column_types():
    # The logical schema does not move when an annotation starts mapping to a
    # different SQL type, so without this a checkpoint written before such a
    # change would be reused under the new one and read back as the wrong type.
    as_json = SignalSchema({"v": int})
    assert _physical_schema_hash(as_json) != _physical_schema_hash(
        SignalSchema({"v": str})
    )
    assert _physical_schema_hash(as_json) == _physical_schema_hash(
        SignalSchema({"v": int})
    )


@pytest.mark.parametrize("multi_output", [False, True], ids=["single", "multi"])
def test_the_physical_fingerprint_reaches_every_udf_hash(multi_output):
    # A UDF that overrides hash() -- _MultiSignalMapper does, and a user
    # subclass may -- must not be able to drop the framework's own key, or a
    # checkpoint written under different physical types would be reused.
    def one(i: int) -> int:
        return i

    def two(i: int) -> int:
        return i

    chain = dc.read_values(i=[1])
    chain = chain.map(a=one, b=two) if multi_output else chain.map(a=one)
    udf = [s for s in chain._query.steps if hasattr(s, "udf")][-1].udf

    before = udf.hash()
    with patch.object(datachain.lib.udf, "_physical_schema_hash", lambda _: "ff" * 32):
        assert udf.hash() != before
