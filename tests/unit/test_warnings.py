import warnings

import pytest

from datachain.warnings import _reset_warned, warn_deprecated


@pytest.fixture(autouse=True)
def clean_warned():
    _reset_warned()
    yield
    _reset_warned()


def test_warn_deprecated_default_category():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("something")

    assert len(record) == 1
    assert str(record[0].message) == "something is deprecated."
    assert issubclass(record[0].category, FutureWarning)


def test_warn_deprecated_with_instead():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("DataChain.print_schema()", instead="print(chain.schema)")

    assert len(record) == 1
    assert (
        str(record[0].message)
        == "DataChain.print_schema() is deprecated; use print(chain.schema) instead."
    )


def test_warn_deprecated_with_removal():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("feature_x", removal="1.0")

    assert len(record) == 1
    assert (
        str(record[0].message) == "feature_x is deprecated and will be removed in 1.0."
    )


def test_warn_deprecated_with_instead_and_removal():
    with pytest.warns(FutureWarning) as record:
        warn_deprecated("func_a()", instead="func_b()", removal="v2.0")

    assert len(record) == 1
    assert (
        str(record[0].message)
        == "func_a() is deprecated and will be removed in v2.0; use func_b() instead."
    )


def test_warn_deprecated_custom_category():
    with pytest.warns(DeprecationWarning) as record:
        warn_deprecated("internal_hook", category=DeprecationWarning)

    assert len(record) == 1
    assert issubclass(record[0].category, DeprecationWarning)
    assert not issubclass(record[0].category, FutureWarning)
    assert str(record[0].message) == "internal_hook is deprecated."


def test_warn_deprecated_once():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(5):
            warn_deprecated("hot_path_arm", instead="new_arm", once=True)

    assert len(record) == 1
    assert str(record[0].message) == "hot_path_arm is deprecated; use new_arm instead."


def test_warn_deprecated_without_once_warns_each_time():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            warn_deprecated("repeated_warning", instead="alt", once=False)

    assert len(record) == 3


def test_warn_deprecated_single_line_format():
    cases = [
        ("X", None, None),
        ("X", "Y", None),
        ("X", None, "Z"),
        ("X", "Y", "Z"),
    ]
    for what, instead, removal in cases:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            warn_deprecated(what, instead=instead, removal=removal)
            msg = str(record[0].message)
            assert "\n" not in msg
            assert "\r" not in msg
            assert msg.endswith(".")


def test_warn_deprecated_stacklevel():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        # Line right below must be the reported lineno
        warn_deprecated("stack_check")
        expected_line = test_warn_deprecated_stacklevel.__code__.co_firstlineno + 4

    assert len(record) == 1
    assert record[0].lineno == expected_line
    assert record[0].filename == __file__
