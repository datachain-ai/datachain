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
from datachain.lib.udf import Aggregator, Generator, Mapper
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
            "8ffe5a423a22de6bf8153624392763d01bc64ce2438b102c4301111a5ae2a0c1",
        ),
        (
            double2,
            ["y"],
            {"double": int},
            "adcda7fd17b46b3ad761b611258c11a743dc7ef71ed007b416a448b1391ba7bb",
        ),
        (
            double_default,
            ["x"],
            {"double": int},
            "073c4425ef9a2a744174537de6e35a1e5f743da38bbb1a6b23347a760540eff6",
        ),
        (
            double_kwonly,
            ["x"],
            {"double": int},
            "e9a57bc156375e45ed9ff9446d31b7aa130d67f3dcaf89e8c534c237772d616a",
        ),
        (
            map_custom_feature,
            ["t1"],
            {"x": CustomFeature},
            "95cc34a11e91a01c33f42c5a883779ba8e4fd1758a9aada2e318e4061cc00a6e",
        ),
        (
            DoubleMapper(),
            ["x"],
            {"double": int},
            "9a8edb2936d17664f2310d422a7454fc72eb735ca13da17e8f940ea408875694",
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
            "bda9ea24a9180a99cbab557af9ac91138405be78bd67a2f750adc2faffd2230d",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "d1087033394c8d6aed4ba6a6a95de8c61b0d2c804a9fee95e94d6839e96699dd",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "31d256650c1e124981ed0781040155af94db009c1b60138a38d0fd45731856a1",
        ),
        (
            TripleGenerator(),
            ["x"],
            {"triple": int},
            "8a343fc97cfeac9a235edf9cbd205fce9dc446e7fc726aac9333bd997550a577",
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
            "0232b9e988ed6b0baecc7e849058c173d3d9b1cdcfca2f034b1dd0e7dc2f76b9",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "86537b34c5e6785b67ede19185c2da756fcdf7fe5a2a38e83fdcf1c351d52552",
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
