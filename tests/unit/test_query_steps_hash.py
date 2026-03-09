import math
from dataclasses import replace

import pytest
import sqlalchemy as sa
from pydantic import BaseModel

import datachain as dc
from datachain import C, func
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
    dc.read_values(num=list(range(100)), session=test_session).save("dev.num.numbers")
    test_session.catalog.metastore.update_dataset_version(
        test_session.catalog.get_dataset(
            "numbers", namespace_name="dev", project_name="num"
        ),
        "1.0.0",
        uuid="9045d46d-7c57-4442-aae3-3ca9e9f286c4",
    )

    return test_session.catalog.get_dataset(
        "numbers", namespace_name="dev", project_name="num"
    )


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "d53cd8431f00e29ae1b31df6ef39ca206d2918555fe26877a9c6bb058fb77097",
        ),
        ((), "3245ba76bc1e4b1b1d4d775b88448ff02df9473bd919929166c70e9e2b245345"),
        (
            (C("name"),),
            "5da26d0f27cba01ae3464da25d5ca0d66ff57deb71eaecc549ffdcf0dfe471a4",
        ),
        (
            (func.rand().label("random"),),
            "f6706531fb15662eec9a28845e8a460f1c5a2d9898cac0adb68568f7a16764ba",
        ),
        (("name",), "46eeec88c5f842bd478d3ec87032c49b22adcdd46572463b0acde4b2bac0900a"),
    ],
)
def test_select_hash(inputs, _hash):
    assert SQLSelect(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (
            (C("name"), C("age") * 10, func.avg("id"), C("country").label("country")),
            "1c8d29ed3c4c0e0f3344257a655a6d82b8bb53f3c0fa322f89cca0b5fc13d498",
        ),
        ((), "0d27e4cfa3801628afc535190c64a426d9db66e5145c57129b9f5ca0935ef29e"),
        (
            (C("name"),),
            "84a4280453505d3d7704b75a78a7a861b130c4d06dd6b351a821bf17b3647e33",
        ),
        (("name",), "e26923a0433e549e680a4bcbc5cb95bb9a523c4b47ae23b07b2a928a609fc498"),
    ],
)
def test_select_except_hash(inputs, _hash):
    assert SQLSelectExcept(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (
            (sa.and_(C("name") != "John", C("age") * 10 > 100)),
            "c23048c8b931078d0d2dfc81fbde32f663a81f96a9592e7e39d9e88866f6cf73",
        ),
        ((), "19e718af35ddc311aa892756fa4f95413ce17db7c8b27f68200d9c3ce0fc8dbf"),
        (
            (C("files.path").glob("*.jpg"),),
            "b85dfb62d62f7b1e142c13544919e2cf8d4d47fa1d9da90cc41197b1d03e3ac4",
        ),
        (
            sa.or_(C("age") > 50, C("country") == "US"),
            "69102e1955786abc8e985b3ca88047ea04b340d9e0b5cde17f84a0c91db91775",
        ),
    ],
)
def test_filter_hash(inputs, _hash):
    assert SQLFilter(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,schema,_hash",
    [
        (
            {"new_id": func.sum("id")},
            SignalSchema({"id": int}),
            "60f3a9e31aa77dabb8045060659e75f050133278fd613f0a0075b309747eb80e",
        ),
        (
            {"new_id": C("id") * 10, "old_id": C("id")},
            SignalSchema({"id": int}),
            "d124ce7453b399e15a65bec1887d734115f0c1af3987f26d1df782ec1a29e879",
        ),
        (
            {},
            SignalSchema({"id": int}),
            "b9717325e70a10ccd55c7faa22d5099ac8d5726d1a3c0eb3cfb001c7f628ce7f",
        ),
    ],
)
def test_mutate_hash(inputs, schema, _hash):
    # transforming input into format SQLMutate expects
    inputs = (
        v.label(k).get_column(schema) if isinstance(v, Func) else v.label(k)
        for k, v in inputs.items()
    )
    assert SQLMutate(inputs, new_schema=None).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (
            (C("name"), C("age")),
            "47646b4046685f7f988b93e40cf72c8ea43678bbb2b68cfbc017fdb574bc428f",
        ),
        (("name",), "b3562b4508052e5a57bc84ae862255939df294eb079e124c5af61fc21044343e"),
        (
            (sa.desc(C("name")),),
            "8e64f7694349f0e7487f662d4e24edff2fc42007d9d19b0e08aa504160c1f689",
        ),
        ((), "c525013178ef24a807af6d4dd44d108c20a5224eb3ab88b84c55c635ec32ba04"),
    ],
)
def test_order_by_hash(inputs, _hash):
    assert SQLOrderBy(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (5, "9fc462c7b5fe66106c8056b9f361817523de5c9f8d4e4b847e79cb02feba1351"),
        (0, "1da7ad424bfdb853e852352fbb853722eb5fdc119592a778679aa00ba29f971a"),
    ],
)
def test_limit_hash(inputs, _hash):
    assert SQLLimit(inputs).hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (5, "ff65be6bef149f6f2568f33c2bd0ac3362018a504caadf52c221a2e64acc5bb3"),
        (0, "e88121711a1fa5da46ea2305e0d6fbeebe63f5b575450c628e7bf6f81e73aa46"),
    ],
)
def test_offset_hash(inputs, _hash):
    assert SQLOffset(inputs).hash() == _hash


@pytest.mark.parametrize(
    "_hash",
    [
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
        "8867973da58bd4d14c023fa9bad98dc50c18ba69240347216f7a8a1c7e70d377",
    ],
)
def test_count_hash(_hash):
    assert SQLCount().hash() == _hash


@pytest.mark.parametrize(
    "inputs,_hash",
    [
        (("name",), "bb0a1acba3bce39d31cc05dc01e57fc7265e451154187a6f93fbcf2001525c51"),
        (
            ("name", "age"),
            "29203756f44599f2728c70d75d92ff7af6110c8602e25839127c736d25a30c4b",
        ),
        ((), "7d4efeefbe9d1694bb89e7bf8b2d3f1d96ed0603e312b48d247d0ed3c881bf48"),
    ],
)
def test_distinct_hash(inputs, _hash):
    assert SQLDistinct(inputs, dialect=None).hash() == _hash


def test_union_hash(test_session, numbers_dataset):
    chain1 = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("dev.num.numbers").filter(C("num") < 50).limit(20)

    assert SQLUnion(chain1._query, chain2._query).hash() == (
        "a8bf8e31e33af266201985ac5e75b3d5e05f1b3b058ed5f87c173f888e0f0154"
    )


@pytest.mark.parametrize(
    "predicates,inner,full,rname,_hash",
    [
        (
            "id",
            True,
            False,
            "{name}_right",
            "8dd0ed4b89968a76e0f674f9ce00ae5a31192da828c7d1ecfdf1af55e2f215b0",
        ),
        (
            ("id", "name"),
            False,
            True,
            "{name}_r",
            "e0d5d0cd0ed8b45053edaf159390645baf44671b41bcb9662b19c9caed85b64e",
        ),
        (
            sa.column("id"),
            True,
            False,
            "{name}_right",
            "6c202f10e09a90ffd1edb2ae3a806cd7cd9aec391e00c7d3a0b970f7f7bba795",
        ),
    ],
)
def test_join_hash(
    test_session, numbers_dataset, predicates, inner, full, rname, _hash
):
    chain1 = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(10)
    chain2 = dc.read_dataset("dev.num.numbers").filter(C("num") < 50).limit(20)

    assert (
        SQLJoin(
            test_session.catalog,
            chain1._query,
            chain2._query,
            predicates,
            inner,
            full,
            rname,
        ).hash()
        == _hash
    )


@pytest.mark.parametrize(
    "columns,partition_by,_hash",
    [
        (
            {"cnt": func.count(), "sum": func.sum("id")},
            [
                C("id"),
            ],
            "dad35b97c6a47beaa605df6ccc46225c7279180d93f01b43cf2918fccca0a7ed",
        ),
        (
            {"cnt": func.count(), "sum": func.sum("id")},
            [C("id"), C("name")],
            "5ebb814165256f4cd4717d8ec255d6d67bc4abb165bffaa95b09b49b0f90c6e5",
        ),
        (
            {"cnt": func.count()},
            [],
            "96512eb2367f9940e53e37d450d79f9a08d3de19b6c36d79a6939b55487d657c",
        ),
    ],
)
def test_group_by_hash(columns, partition_by, _hash):
    schema = SignalSchema({"id": int})
    # transforming inputs into format SQLGroupBy expects
    columns = [v.get_column(schema, label=k) for k, v in columns.items()]
    assert SQLGroupBy(columns, partition_by).hash() == _hash


@pytest.mark.parametrize(
    "on,_hash",
    [
        (
            [("id", "id")],
            "5f90fc7c0a5287c8665c0fd912b0d91fb2a8baca416e68dd7bb38f75ad8926a1",
        ),
        (
            [("id", "id"), ("name", "name")],
            "f595869a8990a259023ec5cefa9d27868750931ebaaf91c13e2296a0d5ded990",
        ),
        (
            [],
            "0f1100306f3029d8897dc826ef1ebb2c950673682b2479582bf19e38c12a3f5d",
        ),
    ],
)
def test_subtract_hash(test_session, numbers_dataset, on, _hash):
    chain = dc.read_dataset("dev.num.numbers").filter(C("num") > 50).limit(20)
    assert Subtract(chain._query, test_session.catalog, on).hash() == _hash


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            double,
            ["x"],
            {"double": int},
            "c62dcb3c110b1cadb47dd3b6499d7f4da351417fbe806a3e835237928a468708",
        ),
        (
            double2,
            ["y"],
            {"double": int},
            "674838e9557ad24b9fc68c6146b781e02fd7e0ad64361cc20c055f47404f0a95",
        ),
        (
            double_default,
            ["x"],
            {"double": int},
            "f25afd25ebb5f054bab721bea9126c5173c299abb0cbb3fd37d5687a7693a655",
        ),
        (
            double_kwonly,
            ["x"],
            {"double": int},
            "12f3620f703c541e0913c27cd828a8fe6e446f62f3d0b2a4ccfa5a1d9e2472e7",
        ),
        (
            map_custom_feature,
            ["t1"],
            {"x": CustomFeature},
            "b4edceaa18ed731085e1c433a6d21deabec8d92dfc338fb1d709ed7951977fc5",
        ),
        (
            DoubleMapper(),
            ["x"],
            {"double": int},
            "7994436106fef0486b04078b02ee437be3aa73ade2d139fb8c020e2199515e26",
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
            "c7ae1a50df841da2012c8422be87bfb29b101113030c43309ab6619011cdcc1c",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "850352183532e057ec9c914bda906f15eb2223298e2cbd0c3585bf95a54e15e9",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "7ff702d242612cbb83cbd1777aa79d2792fb2a341db5ea406cd9fd3f42543b9c",
        ),
        (
            TripleGenerator(),
            ["x"],
            {"triple": int},
            "02b4c6bf98ffa011b7c62f3374f219f21796ece5b001d99e4c2f69edf0a94f4a",
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
            "9f0ffa47038bfea164b8b6aa87a9f7ee245a8a39506596cd132d9c0ec65a50ec",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "2f782a9ec575cb3a9042c7683184ec5d9d2c9db56488f1a66922341048e9d688",
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


@pytest.mark.parametrize(
    "namespace_name,project_name,name,version,_hash",
    [
        (
            "default",
            "default",
            "numbers",
            "1.0.4",
            "8173fb1d88df5cca3e904cbd17a9b80a0c8a682425c32cd95e32e1e196b7eff8",
        ),
        (
            "dev",
            "animals",
            "cats",
            "1.0.1",
            "e0aec7fe323ae3482ee2e74030a87ebb73dbb823ce970e15fdfcbd43e7abe2da",
        ),
        (
            "system",
            "listing",
            "lst__s3://bucket",
            "1.0.1",
            "19dff9f21030312c7469de7284cac2841063c22c62a7948a68f25ca018777c6d",
        ),
    ],
)
def test_query_step_hash(
    dataset_record, namespace_name, project_name, name, version, _hash
):
    namespace = replace(dataset_record.project.namespace, name=namespace_name)
    project = dataset_record.project
    project = replace(project, namespace=namespace)
    project = replace(project, name=project_name)
    dataset_record.project = project
    dataset_record.name = name
    dataset_record.versions[0].version = version

    assert QueryStep(None, dataset_record, version).hash() == _hash
