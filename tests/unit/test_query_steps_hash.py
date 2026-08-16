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
from datachain.lib.udf import Aggregator, Generator, Mapper, _stable_state_value
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
            "4004b6ee6ef90934d0f48fcb337d73c6552fcf6e5d8250d652d310b849dbdca7",
        ),
        (
            double2,
            ["y"],
            {"double": int},
            "98eed82d9e4ca5217d325f1182c96789390b2743d3b09739b84e50420f10cb4f",
        ),
        (
            double_default,
            ["x"],
            {"double": int},
            "40e456055a765697bfebb8371e4b3c7c3125aea100e25505d8af09155c9b6a8e",
        ),
        (
            double_kwonly,
            ["x"],
            {"double": int},
            "221364c9949afdb25aff731ee6b2db815ecd62ed5a713f58190d341ffe608ac4",
        ),
        (
            map_custom_feature,
            ["t1"],
            {"x": CustomFeature},
            "728657607131969e66f374374c353987801e6abc724b8a8c12edd44bccc33380",
        ),
        (
            DoubleMapper(),
            ["x"],
            {"double": int},
            "b58c9679ed454d3f54b4a754585727697a9aea9e4725bd12a842a774b5087963",
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


class ScaleMapper(Mapper):
    def __init__(self, factor: int):
        self.factor = factor

    def process(self, x):
        return x * self.factor


class CountAbove(Aggregator):
    def __init__(self, limit: int):
        self.limit = limit

    def process(self, key, value):
        yield key[0], sum(1 for v in value if v > self.limit)


def _udf_hash(cls, udf, params, output):
    sign = UdfSignature.parse("", {}, udf, params, output, False)
    return cls._create(sign, SignalSchema(sign.params)).hash()


def test_class_mapper_hash_includes_constructor_state():
    h2 = _udf_hash(Mapper, ScaleMapper(2), ["x"], {"y": int})
    h3 = _udf_hash(Mapper, ScaleMapper(3), ["x"], {"y": int})
    h2_again = _udf_hash(Mapper, ScaleMapper(2), ["x"], {"y": int})
    assert h2 == h2_again
    assert h2 != h3


def test_class_aggregator_hash_includes_constructor_state():
    h0 = _udf_hash(Aggregator, CountAbove(0), ["key", "value"], {"n": int})
    h3 = _udf_hash(Aggregator, CountAbove(3), ["key", "value"], {"n": int})
    h0_again = _udf_hash(Aggregator, CountAbove(0), ["key", "value"], {"n": int})
    assert h0 == h0_again
    assert h0 != h3


def test_class_udf_hash_ignores_setup_runtime_state():
    class Scale(Mapper):
        def __init__(self, factor: int):
            self.factor = factor

        def process(self, x):
            return x * self.factor

        def setup(self):
            self.ready = True

    sign = UdfSignature.parse("", {}, Scale(2), ["x"], {"y": int}, False)
    udf = Mapper._create(sign, SignalSchema(sign.params))
    before = udf.hash()
    udf.setup()
    assert udf.hash() == before
    assert before != _udf_hash(Mapper, Scale(3), ["x"], {"y": int})


def test_class_udf_hash_includes_type_constructor_state():
    class Cast(Mapper):
        def __init__(self, typ):
            self.typ = typ

        def process(self, x):
            return self.typ(x)

    hint = _udf_hash(Mapper, Cast(int), ["x"], {"y": int})
    hbool = _udf_hash(Mapper, Cast(bool), ["x"], {"y": int})
    hint_again = _udf_hash(Mapper, Cast(int), ["x"], {"y": int})
    assert hint == hint_again
    assert hint != hbool


def test_class_udf_hash_canonicalizes_unordered_constructor_state():
    class HasLabels(Mapper):
        def __init__(self, labels):
            self.labels = labels

        def process(self, x):
            return x in self.labels

    left = _udf_hash(
        Mapper, HasLabels(frozenset({"alpha", "beta", "gamma"})), ["x"], {"y": bool}
    )
    right = _udf_hash(
        Mapper, HasLabels(frozenset({"gamma", "beta", "alpha"})), ["x"], {"y": bool}
    )
    assert _stable_state_value(frozenset({"b", "a", "c"})) == (
        "frozenset",
        ["a", "b", "c"],
    )
    assert left == right


def test_class_udf_hash_distinguishes_dict_from_pair_list():
    class Hold(Mapper):
        def __init__(self, payload):
            self.payload = payload

        def process(self, x):
            return x

    hdict = _udf_hash(Mapper, Hold({"a": 1}), ["x"], {"y": int})
    hlist = _udf_hash(Mapper, Hold([("a", 1)]), ["x"], {"y": int})
    assert hdict != hlist


def test_class_udf_hash_updates_when_rebound_after_mutation():
    class Scale(Mapper):
        def __init__(self, factor: int):
            self.factor = factor

        def process(self, x):
            return x * self.factor

    sign = UdfSignature.parse("", {}, Scale(2), ["x"], {"y": int}, False)
    udf = Mapper._create(sign, SignalSchema(sign.params))
    before = udf.hash()
    udf.factor = 3
    rebound = Mapper._create(sign, SignalSchema(sign.params))
    assert rebound.hash() != before


def test_class_udf_hash_includes_new_attr_on_rebind():
    class Scale(Mapper):
        def __init__(self, factor: int):
            self.factor = factor

        def process(self, x):
            return x * self.factor + getattr(self, "bonus", 0)

    sign = UdfSignature.parse("", {}, Scale(2), ["x"], {"y": int}, False)
    udf = Mapper._create(sign, SignalSchema(sign.params))
    before = udf.hash()
    udf.bonus = 3
    assert udf.hash() == before
    rebound = Mapper._create(sign, SignalSchema(sign.params))
    assert rebound.hash() != before


def test_class_udf_hash_includes_slotted_constructor_state():
    class Scale(Mapper):
        __slots__ = ("factor",)

        def __init__(self, factor: int):
            self.factor = factor

        def process(self, x):
            return x * self.factor

    assert _udf_hash(Mapper, Scale(2), ["x"], {"y": int}) != _udf_hash(
        Mapper, Scale(3), ["x"], {"y": int}
    )


def test_class_udf_hash_preserves_dict_key_types():
    class Lookup(Mapper):
        def __init__(self, table):
            self.table = table

        def process(self, x):
            return self.table.get(x)

    hint = _udf_hash(Mapper, Lookup({1: "a"}), ["x"], {"y": str})
    hstr = _udf_hash(Mapper, Lookup({"1": "a"}), ["x"], {"y": str})
    assert hint != hstr


def test_class_udf_hash_distinguishes_local_type_state():
    def make(factor: int):
        class Kind:
            def __call__(self, x):
                return x * factor

        class Scale(Mapper):
            def __init__(self):
                self.kind = Kind

            def process(self, x):
                return self.kind()(x)

        return _udf_hash(Mapper, Scale(), ["x"], {"y": int})

    assert make(2) != make(3)


def test_class_udf_hash_tolerates_unpickleable_setup_state():
    import threading

    class Scale(Mapper):
        def __init__(self, factor: int):
            self.factor = factor

        def process(self, x):
            return x * self.factor

        def setup(self):
            self.lock = threading.Lock()

    sign = UdfSignature.parse("", {}, Scale(2), ["x"], {"y": int}, False)
    udf = Mapper._create(sign, SignalSchema(sign.params))
    before = udf.hash()
    udf.setup()
    assert udf.hash() == before
    rebound = Mapper._create(sign, SignalSchema(sign.params))
    assert rebound.hash()  # rebind after Lock must not raise


def test_class_udf_hash_includes_unpickleable_constructor_state():
    import threading

    class LockMode(Mapper):
        def __init__(self, lock):
            self.lock = lock

        def process(self, x):
            return x if self.lock.locked() else -x

    open_lock = threading.Lock()
    locked_lock = threading.Lock()
    locked_lock.acquire()
    opened = _udf_hash(Mapper, LockMode(open_lock), ["x"], {"y": int})
    locked = _udf_hash(Mapper, LockMode(locked_lock), ["x"], {"y": int})
    opened_again = _udf_hash(Mapper, LockMode(threading.Lock()), ["x"], {"y": int})
    assert opened != locked
    assert opened == opened_again


def test_class_udf_hash_walks_unpickleable_instance_state():
    import threading

    class Holder:
        def __init__(self, n: int):
            self.n = n
            self.lock = threading.Lock()

        def __repr__(self):
            return "Holder()"

    class Scale(Mapper):
        def __init__(self, holder):
            self.holder = holder

        def process(self, x):
            return x * self.holder.n

    left = _udf_hash(Mapper, Scale(Holder(2)), ["x"], {"y": int})
    right = _udf_hash(Mapper, Scale(Holder(3)), ["x"], {"y": int})
    left_again = _udf_hash(Mapper, Scale(Holder(2)), ["x"], {"y": int})
    assert left != right
    assert left == left_again


def test_class_udf_hash_handles_cyclic_constructor_state():
    class Hold(Mapper):
        def __init__(self, payload):
            self.payload = payload

        def process(self, x):
            return x

    cyclic = []
    cyclic.append(cyclic)
    other = []
    other.append(other)
    left = _udf_hash(Mapper, Hold(cyclic), ["x"], {"y": int})
    right = _udf_hash(Mapper, Hold(other), ["x"], {"y": int})
    assert left == right


def test_function_udf_hash_includes_callable_instance_state():
    class Scale:
        def __init__(self, factor: int):
            self.factor = factor

        def __call__(self, x):
            return x * self.factor

    h2 = _udf_hash(Mapper, Scale(2), ["x"], {"y": int})
    h3 = _udf_hash(Mapper, Scale(3), ["x"], {"y": int})
    h2_again = _udf_hash(Mapper, Scale(2), ["x"], {"y": int})
    assert h2 == h2_again
    assert h2 != h3


def test_function_udf_hash_includes_bound_method_owner_state():
    class Scale:
        def __init__(self, factor: int):
            self.factor = factor

        def apply(self, x):
            return x * self.factor

    h2 = _udf_hash(Mapper, Scale(2).apply, ["x"], {"y": int})
    h3 = _udf_hash(Mapper, Scale(3).apply, ["x"], {"y": int})
    h2_again = _udf_hash(Mapper, Scale(2).apply, ["x"], {"y": int})
    assert h2 == h2_again
    assert h2 != h3


def test_class_udf_hash_distinguishes_aliased_constructor_state():
    class Alias(Mapper):
        def __init__(self, left, right):
            self.left = left
            self.right = right

        def process(self, x):
            return x if self.left is self.right else -x

    shared = []
    other = []
    h_shared = _udf_hash(Mapper, Alias(shared, shared), ["x"], {"y": int})
    h_shared_again = _udf_hash(Mapper, Alias(other, other), ["x"], {"y": int})
    h_separate = _udf_hash(Mapper, Alias([], []), ["x"], {"y": int})
    assert h_shared == h_shared_again
    assert h_shared != h_separate


@pytest.mark.parametrize(
    "func,params,output,_hash",
    [
        (
            double_gen,
            ["x"],
            {"double": int},
            "36c34c7c957ab0ba6210a49542ec3c9d6cc9d3c632a4752b1eee04e5c2ffdc2f",
        ),
        (
            double_gen_multi_arg,
            ["x", "y"],
            {"double": int},
            "1a39881c75054e4c548233d1ea0dff8f57488b060015fb820ff6ccf054fc60d3",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            "990f4218dfcdcf9e5cecb07d6996e571c1144d9a52b207ca026de8b89918f091",
        ),
        (
            TripleGenerator(),
            ["x"],
            {"triple": int},
            "01201327b1926788e6242d2be5383c63b97ec018232ab0844f047cf64ec2dfca",
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
            "037a9753bfc2921557b48e6fbddc3ddadb6b1ac4e5a134e565d9d9181e5d930d",
        ),
        (
            custom_feature_gen,
            ["t1"],
            {"x": CustomFeature},
            [C.t1.my_name],
            "fbdacb85da356170053b297f887bcb3a70c9469a2fd65bcf935809e66a014860",
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
