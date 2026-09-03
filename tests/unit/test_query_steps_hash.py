import hashlib
import math

import pytest
import sqlalchemy as sa
from pydantic import BaseModel

import datachain as dc
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
            "e9e4cdfede6099e64e14ff7d430e06d5a9a4d5dc8f8fba78a4f4c40028ef0e8f",
        ),
        (
            double2,
            ["y"],
            {"double": int},
            "96f977f42fe3c62fad9be25a531640dced494853688d470d3bf52db6d2b7b060",
        ),
        (
            double_default,
            ["x"],
            {"double": int},
            "e378641fcae870650e1df5ce85d09958d9e37471885758aec09db02bf69b694d",
        ),
        (
            double_kwonly,
            ["x"],
            {"double": int},
            "e56df26ab8c36a91849fa83db952050eb6a8ab34ebd7f0f17b85cfe1672d2cd6",
        ),
        (
            map_custom_feature,
            ["t1"],
            {"x": CustomFeature},
            "8184307469b77db524fd7c894f198f69c8df59656f8ff1c9dbbfb72ce8463d82",
        ),
        (
            DoubleMapper(),
            ["x"],
            {"double": int},
            "b637c07c491ef521674e859c6c97ec45535866775ab69063e7ac86f2f86f0f15",
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
            "0eb1120e9508b7fb89349546ede72041945b51f3c6fcb6687be37aa9ce7aeeb2",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "137ebc9250861f45192914d7e111546fa010d726ccea2b75b12384a16c9eb73f",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "15b296e6096cac42ac39d234f943e34830cda0d4005440a6f269ef3fc2d8bad4",
        ),
        (
            TripleGenerator(),
            ["x"],
            {"triple": int},
            "87395518564cfd14c04cb10f145f8ed919126becccdf00646ffdbbd2874ef224",
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
            "ada89641522f11fb504662ec1cae6247073ef123604e64a8fd56361bf234b194",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "7fd56d4a2af9dbd9506023bf30aab77673dade2610b7f12d66cb2377de4b59fe",
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
