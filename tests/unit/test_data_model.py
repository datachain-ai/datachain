import copy
import gc
import sys
from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import Enum, IntEnum
from typing import (  # noqa: UP035
    Annotated,
    Any,
    Dict,
    Generic,
    List,
    Literal,
    NewType,
    Optional,
    TypeVar,
    Union,
)

import pytest

# typing.TypedDict cannot be combined with Generic on Python 3.10
from typing_extensions import TypedDict

from datachain.lib.data_model import (
    DataModel,
    compute_model_fingerprint,
    is_chain_type,
    key_is_stored_verbatim,
    key_needs_json_decode,
    resolve_literal_member,
    unwrap_alias,
)
from datachain.lib.model_store import ModelStore


@pytest.fixture(autouse=True)
def restore_model_store():
    snapshot = copy.deepcopy(ModelStore.store)
    ModelStore.store = {}
    try:
        yield
    finally:
        ModelStore.store = snapshot


def test_compute_model_fingerprint_missing_field():
    class Sample(DataModel):
        a: int

    with pytest.raises(ValueError, match="Field missing not found in Sample"):
        compute_model_fingerprint(Sample, {"missing": None})


def test_compute_model_fingerprint_non_model_child():
    class Sample(DataModel):
        a: int

    with pytest.raises(ValueError, match="Field a in Sample is not a model"):
        compute_model_fingerprint(Sample, {"a": {"child": None}})


def test_compute_model_fingerprint_stable_for_same_selection():
    class Sample(DataModel):
        a: int
        b: int

    sel = {"a": None}
    fp1 = compute_model_fingerprint(Sample, sel)
    fp2 = compute_model_fingerprint(Sample, sel)
    assert fp1 == fp2


def test_compute_model_fingerprint_changes_with_selection():
    class Sample(DataModel):
        a: int
        b: int

    fp_a = compute_model_fingerprint(Sample, {"a": None})
    fp_b = compute_model_fingerprint(Sample, {"b": None})
    assert fp_a != fp_b


def test_compute_model_fingerprint_nested_model():
    class Child(DataModel):
        x: int
        y: int

    class Parent(DataModel):
        child: Child
        z: int

    fp_child_x = compute_model_fingerprint(Parent, {"child": {"x": None}})
    fp_child_y = compute_model_fingerprint(Parent, {"child": {"y": None}})
    fp_child_all = compute_model_fingerprint(Parent, {"child": {"x": None, "y": None}})

    assert fp_child_x != fp_child_y
    assert fp_child_all != fp_child_x
    assert fp_child_all != fp_child_y


def test_compute_model_fingerprint_required_vs_optional_differs():
    class Required(DataModel):
        value: int

    class OptionalField(DataModel):
        value: int | None = None

    fp_required = compute_model_fingerprint(Required, {"value": None})
    fp_optional = compute_model_fingerprint(OptionalField, {"value": None})
    assert fp_required != fp_optional


def test_is_chain_type_handles_generics_that_reject_issubclass():
    T = TypeVar("T")

    class GenericRow(TypedDict, Generic[T]):
        x: T

    assert is_chain_type(GenericRow[int]) is False


@pytest.mark.parametrize(
    "annotation",
    [
        # Abstract origins serialize as their bare origin name, losing their args,
        # so a saved signal would not survive a reload.
        Sequence[int],
        Mapping[str, int],
        Collection[int],
        Iterable[int],
    ],
    ids=["Sequence", "Mapping", "Collection", "Iterable"],
)
def test_is_chain_type_rejects_abstract_collection_roots(annotation):
    assert is_chain_type(annotation) is False


@pytest.mark.parametrize(
    "annotation",
    [list[int], dict[str, int], list[list[int]]],
    ids=["list", "dict", "nested-list"],
)
def test_is_chain_type_accepts_serializable_collections(annotation):
    assert is_chain_type(annotation) is True


@pytest.mark.parametrize(
    "annotation",
    [
        # `type_to_str` normalises a wrong arity rather than refusing it, so
        # `dict[str]` would be written back out as `dict[str, Any]`.
        list[int, str],  # type: ignore[misc]
        dict[str],  # type: ignore[misc]
        dict[str, int, float],  # type: ignore[misc]
        # `python_to_sql` mis-types tuples, so they are not valid root signals.
        tuple[int, ...],
        tuple[int, str],
        # `set` has no column type, and rejection must hold through nesting
        set[int],
        list[set[int]],
        # both arguments are validated, not just one
        dict[str, set[int]],
        dict[set[int], int],
        # Ellipsis counts toward arity during validation; it must not be
        # normalised away first.
        list[int, ...],  # type: ignore[misc]
        dict[str, int, ...],  # type: ignore[misc]
        list[..., int],  # type: ignore[misc]
    ],
    ids=[
        "list-two-args",
        "dict-one-arg",
        "dict-three-args",
        "variadic-tuple",
        "fixed-tuple",
        "set",
        "nested-set",
        "set-as-dict-value",
        "set-as-dict-key",
        "list-ellipsis",
        "dict-ellipsis",
        "list-leading-ellipsis",
    ],
)
def test_is_chain_type_rejects_unsupported_or_malformed_collection_annotations(
    annotation,
):
    assert is_chain_type(annotation) is False


def test_is_chain_type_accepts_bare_dict():
    assert is_chain_type(dict) is True


@pytest.mark.parametrize(
    "annotation",
    [
        List,  # noqa: UP006
        Dict,  # noqa: UP006
        tuple,
    ],
    ids=["typing-List", "typing-Dict", "tuple"],
)
def test_is_chain_type_rejects_unsupported_bare_collection_roots(annotation):
    assert is_chain_type(annotation) is False


class StringKey(str, Enum):
    NULL = "null"
    OK = "ok"


class NumberKey(IntEnum):
    ONE = 1


class PlainValueKey(Enum):
    A = "a"


class StringKey2(str, Enum):
    A = "a"


StringAlias = NewType("StringAlias", str)
NumberAlias = NewType("NumberAlias", int)


@pytest.mark.parametrize(
    "annotation,needs_decode",
    [
        (str, False),
        (Any, False),
        (str | None, False),
        (Literal["null", "ok"], False),
        (Literal["a", 1], True),
        (Annotated[str, "meta"], False),
        (StringKey, False),
        (int, True),
        (Literal[1, 2], True),
        (Annotated[int, "meta"], True),
        (NumberKey, True),
        (tuple[int, int], True),
        (StringAlias, False),
        (NumberAlias, True),
    ],
    ids=[
        "str",
        "any",
        "optional-str",
        "literal-str",
        "literal-mixed",
        "annotated-str",
        "str-enum",
        "int",
        "literal-int",
        "annotated-int",
        "int-enum",
        "tuple",
        "newtype-str",
        "newtype-int",
    ],
)
def test_key_needs_json_decode(annotation, needs_decode):
    assert key_needs_json_decode(annotation) is needs_decode


@pytest.mark.parametrize(
    "annotation,key,verbatim",
    [
        (Literal["a", 1], "a", True),
        (Literal["a", 1], "1", False),
        (Literal["1", 1], "1", True),
        (Literal[1, 2], "1", False),
        (str, "1", True),
        (int, "1", False),
        (StringKey, "null", True),
        # wrappers and unions must not hide the string members
        (Optional[Literal["null", 1]], "null", True),  # noqa: UP045
        (Literal["null", 1] | None, "1", False),
        (Literal["[]", 1] | None, "[]", True),
        (Union[Literal["a"], int], "a", True),  # noqa: UP007
        (Literal["a"] | int, "1", False),
        (Annotated[Literal["a", 1], "meta"], "a", True),
        (PlainValueKey, "a", True),
    ],
    ids=[
        "mixed-string-arm",
        "mixed-int-arm",
        "ambiguous-prefers-string",
        "all-int",
        "str",
        "int",
        "str-enum",
        "optional-literal-string-arm",
        "optional-literal-int-arm",
        "optional-literal-bracket",
        "union-literal-string-arm",
        "union-literal-int-arm",
        "annotated-literal",
        "plain-enum-with-string-values",
    ],
)
def test_key_is_stored_verbatim(annotation, key, verbatim):
    assert key_is_stored_verbatim(annotation, key) is verbatim


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type aliases need Python 3.12"
)
@pytest.mark.parametrize(
    "alias_source,needs_decode",
    [("type Alias = str", False), ("type Alias = int", True)],
    ids=["str", "int"],
)
def test_key_needs_json_decode_pep695_alias(alias_source, needs_decode):
    namespace: dict = {}
    exec(alias_source, namespace)  # noqa: S102

    assert key_needs_json_decode(namespace["Alias"]) is needs_decode


@pytest.mark.parametrize(
    "annotation,value,expected",
    [
        (Literal[StringKey.NULL, "null"], "null", "null"),
        (Literal["null", StringKey.NULL], "null", "null"),
        (Literal[StringKey.NULL], "null", StringKey.NULL),
        (Literal[NumberKey.ONE], 1, NumberKey.ONE),
        (Literal["a", 1], 1, 1),
        (Literal["a"] | None, "a", "a"),
        (int, "1", None),
        # True == 1 == 1.0 in Python, so an exact-type arm must win
        (Literal[1] | bool, True, None),
        (Literal[True] | int, 1, None),
        (Literal[1] | float, 1.0, None),
        (Literal[NumberKey.ONE] | int, 1, None),
        (Literal[1], True, None),  # a bool must not match an int member
    ],
    ids=[
        "enum-then-string",
        "string-then-enum",
        "enum-only",
        "int-enum-only",
        "mixed-int",
        "through-optional",
        "not-a-literal",
        "int-literal-with-bool-arm",
        "bool-literal-with-int-arm",
        "int-literal-with-float-arm",
        "int-enum-literal-with-int-arm",
        "bool-value-against-int-literal",
    ],
)
def test_resolve_literal_member(annotation, value, expected):
    resolved = resolve_literal_member(annotation, value)

    assert resolved == expected
    if expected is not None:
        assert type(resolved) is type(expected)


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type aliases need Python 3.12"
)
@pytest.mark.parametrize(
    "source,expression,expected",
    [
        ("type Key[T] = T", "Key[str]", str),
        ("type Key[T] = T", "Key[int]", int),
        ("type Pair[T] = dict[str, T]", "Pair[int]", dict[str, int]),
        ("type Plain = str", "Plain", str),
        ("type Wrapped[T] = Annotated[T, 'meta']", "Wrapped[int]", int),
        ("type MaybeItems[T] = list[T] | None", "MaybeItems[int]", list[int] | None),
        ("type Fixed[T] = int", "Fixed", int),
    ],
    ids=[
        "param-str",
        "param-int",
        "param-nested",
        "unparameterised",
        "through-annotated",
        "through-union",
        "body-without-the-parameter",
    ],
)
def test_unwrap_alias_substitutes_parameters(source, expression, expected):
    namespace: dict = dict(globals())
    exec(source, namespace)  # noqa: S102

    assert unwrap_alias(eval(expression, namespace)) == expected  # noqa: S307


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type aliases need Python 3.12"
)
@pytest.mark.parametrize(
    "source,name,needs_decode",
    [
        ("type Fixed[T] = int", "Fixed", True),
        ("type TupleKey[T] = tuple[T, ...]", "TupleKey", True),
        ("type Key[T] = T", "Key", False),
    ],
    ids=["body-ignores-the-parameter", "structure-known", "body-is-the-parameter"],
)
def test_unsubscripted_alias_is_judged_on_its_body(source, name, needs_decode):
    namespace: dict = dict(globals())
    exec(source, namespace)  # noqa: S102

    assert key_needs_json_decode(namespace[name]) is needs_decode


@pytest.mark.parametrize(
    "type_var,needs_decode",
    [
        (TypeVar("Bound", bound=int), True),
        (TypeVar("Constrained", int, float), True),
        (TypeVar("Free"), False),
    ],
    ids=["bound", "constrained", "unconstrained"],
)
def test_type_var_is_judged_on_its_bound(type_var, needs_decode):
    assert key_needs_json_decode(type_var) is needs_decode


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type aliases need Python 3.12"
)
@pytest.mark.parametrize(
    "source,name",
    [("type Key[T] = T", "Key"), ("type Loop = Loop", "Loop")],
    ids=["unsubscripted-parameterised", "self-referential"],
)
def test_unresolvable_aliases_are_left_verbatim(source, name):
    namespace: dict = dict(globals())
    exec(source, namespace)  # noqa: S102
    alias = namespace[name]

    # guessing wrong here corrupts data: decoding would turn "null" into None
    assert key_is_stored_verbatim(alias, "null") is True
    assert key_needs_json_decode(alias) is False


def test_unresolved_type_var_is_left_verbatim():
    Unknown = TypeVar("Unknown")

    assert key_is_stored_verbatim(Unknown, "null") is True


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="PEP 695 type aliases need Python 3.12"
)
def test_alias_chain_resolves_the_same_way_every_time():
    namespace: dict = dict(globals())
    exec(  # noqa: S102
        "type A0[T] = A1[T]\ntype A1[T] = A2[T]\ntype A2[T] = A3[T]\ntype A3[T] = T",
        namespace,
    )

    # intermediate aliases are freed between calls; an identity-based cycle
    # guard would reuse their ids and stop early on later passes
    results = []
    for _ in range(5):
        gc.collect()
        results.append(unwrap_alias(namespace["A0"][int]))

    assert results == [int] * 5


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="PEP 696 defaults need Python 3.13"
)
@pytest.mark.parametrize(
    "source,expression,expected",
    [
        ("type Key[T = int] = T", "Key", int),
        ("type Key[T = int] = T", "Key[str]", str),
        ("type Pair[A, B = int] = dict[A, B]", "Pair[str]", dict[str, int]),
    ],
    ids=["default-only", "default-overridden", "partial-specialisation"],
)
def test_unwrap_alias_fills_pep696_defaults(source, expression, expected):
    namespace: dict = dict(globals())
    exec(source, namespace)  # noqa: S102

    assert unwrap_alias(eval(expression, namespace)) == expected  # noqa: S307


@pytest.mark.skipif(
    sys.version_info < (3, 13), reason="PEP 696 defaults need Python 3.13"
)
def test_type_var_default_narrows_the_decode_decision():
    assert key_needs_json_decode(TypeVar("T", default=int)) is True


def test_alias_detection_ignores_lookalike_classes():
    class Lookalike:
        __value__ = int
        __supertype__ = int

    assert unwrap_alias(Lookalike) is Lookalike


@pytest.mark.parametrize(
    "key,verbatim",
    [("a", True), ("1", False), ("zz", False)],
    ids=["member-value", "non-member", "unknown"],
)
def test_enum_key_is_verbatim_only_for_an_actual_member(key, verbatim):
    assert key_is_stored_verbatim(StringKey2, key) is verbatim
