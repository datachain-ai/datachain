import pytest
from sqlalchemy import (
    Float,
    Integer,
    String,
    and_,
    case,
    cast,
    distinct,
    literal,
    not_,
    null,
    or_,
    tuple_,
)
from sqlalchemy import func as sa_func

from datachain import C, func
from datachain.hash_utils import hash_callable, hash_column_elements


def double(x):
    return x * 2


def double_arg_annot(x: int):
    return x * 2


def double_arg_and_return_annot(x: int) -> int:
    return x * 2


lambda1 = lambda x: x * 2  # noqa: E731
lambda2 = lambda y: y + 1  # noqa: E731
lambda3 = lambda z: z - 1  # noqa: E731


COLUMN_EXPRESSIONS = [
    [C("name")],
    [C("name"), C("age")],
    [func.avg("age")],
    [func.count()],
    [func.sum(C("age"))],
    [func.min(C("age")), func.max(C("age"))],
    [sa_func.coalesce(C("name"), "unknown")],
    [sa_func.nullif(C("age"), 0)],
    [
        func.row_number().over(
            func.window(partition_by="file.name", order_by="file.name")
        )
    ],
    [C("age").label("user_age")],
    [C("age") + 10],
    [C("age") * 2],
    [C("age") % 10],
    [-C("age")],
    [C("age") > 20],
    [C("age") >= 21],
    [C("age") == 25],
    [C("name") != ""],
    [and_(C("age") > 20, C("name") != "")],
    [or_(C("age") < 18, C("age") > 65)],
    [not_(C("active"))],
    [and_(C("age") > 18, or_(C("city") == "NYC", C("city") == "LA"))],
    [C("name").like("John%")],
    [C("name").startswith("A")],
    [C("name").contains("John")],
    [sa_func.concat(C("first_name"), " ", C("last_name"))],
    [sa_func.lower(C("name"))],
    [C("name").is_(None)],
    [C("name").isnot(None)],
    [C("age").in_([18, 21, 65])],
    [C("name").in_(["Alice", "Bob", "Charlie"])],
    [C("age").notin_([0, 1])],
    [C("age").between(18, 65)],
    [cast(C("age"), Integer)],
    [cast(C("price"), Float)],
    [cast(C("id"), String)],
    [case((C("age") > 20, "adult"), else_="child")],
    [
        case(
            (C("age") < 13, "child"),
            (C("age") < 20, "teen"),
            (C("age") < 65, "adult"),
            else_="senior",
        )
    ],
    [case({1: "one", 2: "two", 3: "three"}, value=C("num"), else_="other")],
    [tuple_(C("name"), C("age"))],
    [(C("age") + 10).self_group()],
    [distinct(C("name"))],
    [literal(42)],
    [literal("hello")],
    [literal(True)],
    [C("data") == b"hello"],
    [C("data") == b"\x00\x01\x02\xff"],
    [C("data") == b""],
    [null()],
    [C("age").desc()],
    [C("name").asc()],
    [
        case(
            (C("age") > 65, "senior"),
            else_=case((C("age") >= 18, "adult"), else_="minor"),
        )
    ],
    [((C("price") * C("quantity")) + C("tax")) - C("discount")],
    [
        and_(
            C("active") == True,  # noqa: E712
            or_(C("age") >= 18, C("parent_consent") == True),  # noqa: E712
            C("verified").isnot(None),
        )
    ],
    [],
    [C("empty_string") == ""],
    [C("zero") == 0],
    [C("negative") < -1000],
    [sa_func.abs(C("value"))],
    [sa_func.round(C("price"), 2)],
]


@pytest.mark.parametrize("expr", COLUMN_EXPRESSIONS)
def test_hash_column_elements_deterministic(expr):
    """Same expression always produces the same hash."""
    assert hash_column_elements(expr) == hash_column_elements(expr)


def test_hash_column_elements_unique():
    """All different expressions produce different hashes."""
    hashes = [hash_column_elements(expr) for expr in COLUMN_EXPRESSIONS]
    assert len(set(hashes)) == len(hashes)


def test_hash_column_elements_near_miss():
    """Minor modifications produce different hashes."""
    assert hash_column_elements([C("name")]) != hash_column_elements([C("name2")])
    assert hash_column_elements([C("age") > 20]) != hash_column_elements(
        [C("age") > 21]
    )
    assert hash_column_elements([C("age") + 10]) != hash_column_elements(
        [C("age") + 11]
    )
    assert hash_column_elements([C("age") * 2]) != hash_column_elements([C("age") * 3])
    assert hash_column_elements([literal(42)]) != hash_column_elements([literal(43)])
    assert hash_column_elements([literal("hello")]) != hash_column_elements(
        [literal("hell")]
    )
    assert hash_column_elements([C("name").like("A%")]) != hash_column_elements(
        [C("name").like("B%")]
    )
    assert hash_column_elements([C("age").in_([1, 2])]) != hash_column_elements(
        [C("age").in_([1, 3])]
    )


def test_hash_named_functions():
    h1 = hash_callable(double)
    h2 = hash_callable(double_arg_annot)
    h3 = hash_callable(double_arg_and_return_annot)

    assert h1 == hash_callable(double)
    assert h2 == hash_callable(double_arg_annot)
    assert h3 == hash_callable(double_arg_and_return_annot)
    assert len({h1, h2, h3}) == 3


@pytest.mark.parametrize(
    "func",
    [
        lambda1,
        lambda2,
        lambda3,
    ],
)
def test_lambda_same_hash(func):
    h1 = hash_callable(func)
    h2 = hash_callable(func)
    assert h1 == h2  # same object produces same hash


def test_lambda_different_hashes():
    h1 = hash_callable(lambda1)
    h2 = hash_callable(lambda2)
    h3 = hash_callable(lambda3)

    # Ensure hashes are all different
    assert len({h1, h2, h3}) == 3


def test_hash_callable_objects():
    """Test hashing of callable objects (instances with __call__)."""

    class MyCallable:
        def __call__(self, x):
            return x * 2

    class AnotherCallable:
        def __call__(self, y):
            return y + 1

    obj1 = MyCallable()
    obj2 = AnotherCallable()

    h1 = hash_callable(obj1)
    h2 = hash_callable(obj2)
    assert h1 == hash_callable(obj1)
    assert h2 == hash_callable(obj2)
    assert h1 != h2


@pytest.mark.parametrize("value", ["not a callable", 42, None, [1, 2, 3]])
def test_hash_callable_not_callable(value):
    with pytest.raises(TypeError, match="Expected a callable"):
        hash_callable(value)


def test_hash_callable_builtin_functions():
    h1 = hash_callable(len)
    h2 = hash_callable(len)
    # Built-ins return random hash each time
    assert h1 != h2
    assert len(h1) == 64


def test_hash_callable_no_name_attribute():
    from unittest.mock import MagicMock

    mock_callable = MagicMock()
    del mock_callable.__name__
    h = hash_callable(mock_callable)
    assert len(h) == 64


def test_hash_column_elements_single_element():
    single_hash = hash_column_elements(C("name"))
    list_hash = hash_column_elements([C("name")])
    assert single_hash == list_hash


def test_hash_callable_include_body_false_ignores_body():
    """Same qualname + different body → same hash when include_body=False."""

    def make(variant):
        if variant == 1:

            def inner(x):
                return x + 1

        else:

            def inner(x):
                return x * 100

        return inner

    f1 = make(1)
    f2 = make(2)
    assert f1.__qualname__ == f2.__qualname__
    # Different source bodies → different body-based hashes.
    assert hash_callable(f1) != hash_callable(f2)
    # Identity-only → same hash.
    assert hash_callable(f1, include_body=False) == hash_callable(
        f2, include_body=False
    )


def test_hash_callable_include_body_false_different_qualname():
    """Different qualname → different hash even with include_body=False."""
    h1 = hash_callable(double, include_body=False)
    h2 = hash_callable(double_arg_annot, include_body=False)
    assert h1 != h2


def test_hash_callable_lambda_include_body_false_still_uses_body():
    """Lambdas have no meaningful name — include_body flag doesn't change them."""
    assert hash_callable(lambda1, include_body=False) == hash_callable(lambda1)
    assert hash_callable(lambda1, include_body=False) != hash_callable(
        lambda2, include_body=False
    )


def test_hash_callable_include_body_false_defaults_matter():
    """Default values are part of identity even when body is excluded."""

    def make(d):
        def inner(x, y=d):
            return x + y

        return inner

    f1 = make(1)
    f2 = make(1)
    f3 = make(2)

    # Same qualname + same defaults → same hash.
    assert hash_callable(f1, include_body=False) == hash_callable(
        f2, include_body=False
    )
    # Same qualname but different defaults → different hash.
    assert hash_callable(f1, include_body=False) != hash_callable(
        f3, include_body=False
    )


def test_hash_callable_include_body_false_module_distinguishes_qualname():
    """Two functions with the same qualname but different modules hash differently."""

    def func_a():
        return 1

    def func_b():
        return 1

    # Force identical qualname but different module — simulates two functions
    # with the same name defined in different modules.
    func_b.__qualname__ = func_a.__qualname__
    func_a.__module__ = "pkg.module_a"
    func_b.__module__ = "pkg.module_b"

    assert func_a.__qualname__ == func_b.__qualname__
    assert hash_callable(func_a, include_body=False) != hash_callable(
        func_b, include_body=False
    )


def test_hash_callable_include_body_false_annotations_ignored():
    """Annotations do not affect the identity-only hash."""

    def make_plain():
        def inner(x):
            return x

        return inner

    def make_annotated():
        def inner(x: int) -> int:
            return x

        inner.__qualname__ = make_plain().__qualname__
        return inner

    f_plain = make_plain()
    f_annot = make_annotated()
    assert f_plain.__qualname__ == f_annot.__qualname__
    assert hash_callable(f_plain, include_body=False) == hash_callable(
        f_annot, include_body=False
    )
