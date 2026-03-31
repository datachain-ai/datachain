from typing import ClassVar

import datachain as dc
from datachain.lib.signal_schema import SignalSchema


class InnerClass(dc.DataModel):
    inner_value: float
    hide_inner: float

    _hidden_fields: ClassVar[list[str]] = ["hide_inner"]


class OuterClass(dc.DataModel):
    outer_value: float
    hide_outer: float

    inner_object: InnerClass
    _hidden_fields: ClassVar[list[str]] = ["hide_outer"]


def test_datachain_show(capsys, test_session):
    inner = InnerClass(inner_value=1.1, hide_inner=1.2)
    outer = OuterClass(outer_value=1.3, hide_outer=1.4, inner_object=inner)

    expected = """
       outer        outer nums
  outer_value inner_object
               inner_value
0         1.3          1.1    1
"""
    dc.read_values(outer=[outer], nums=[1]).show()

    captured = capsys.readouterr()
    output_lines = [line.strip() for line in captured.out.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]

    assert output_lines == expected_lines


def test_datachain_show_include_hidden(capsys, test_session):
    inner = InnerClass(inner_value=1.1, hide_inner=1.2)
    outer = OuterClass(outer_value=1.3, hide_outer=1.4, inner_object=inner)

    expected = """
        outer      outer        outer        outer nums
  outer_value hide_outer inner_object inner_object
                          inner_value   hide_inner
0         1.3        1.4          1.1          1.2    1
"""
    dc.read_values(outer=[outer], nums=[1]).show(include_hidden=True)

    captured = capsys.readouterr()
    output_lines = [line.strip() for line in captured.out.strip().split("\n")]
    expected_lines = [line.strip() for line in expected.strip().split("\n")]

    assert output_lines == expected_lines


def test_datachain_save(test_session):
    inner = InnerClass(inner_value=1.1, hide_inner=1.2)
    outer = OuterClass(outer_value=1.3, hide_outer=1.4, inner_object=inner)

    ds = dc.read_values(outer=[outer], nums=[1], session=test_session).persist()

    version = test_session.catalog.get_dataset(ds.name, versions=["1.0.0"]).get_version(
        "1.0.0"
    )
    feature_schema = version.feature_schema

    hidden_fields = SignalSchema.get_flatten_hidden_fields(feature_schema)
    assert hidden_fields == ["outer__hide_outer", "outer__inner_object__hide_inner"]
