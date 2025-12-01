import itertools
from collections.abc import Sequence
from typing import Any

from datachain.lib.data_model import DataType, DataTypeNames, DataValue, is_chain_type
from datachain.lib.utils import DataChainParamsError


class ValuesToTupleError(DataChainParamsError):
    def __init__(self, ds_name: str, msg: str):
        self.ds_name = ds_name
        self.msg = msg

        if ds_name:
            ds_name = f"' {ds_name}'"

        super().__init__(f"Cannot convert signals for dataset{ds_name}: {msg}")

    def __reduce__(self):
        return ValuesToTupleError, (self.ds_name, self.msg)


def _find_first_non_none(sequence: Sequence[Any]) -> Any | None:
    """Find the first non-None element in a sequence."""
    try:
        return next(itertools.dropwhile(lambda i: i is None, sequence))
    except StopIteration:
        return None


def _infer_list_item_type(lst: list) -> type:
    """Infer the item type of a list, handling None values and nested lists."""
    if len(lst) == 0:
        # Default to str when list is empty to avoid generic list
        return str

    first_item = _find_first_non_none(lst)
    if first_item is None:
        # Default to str when all items are None
        return str

    item_type = type(first_item)

    # Handle nested lists one level deep
    if isinstance(first_item, list) and len(first_item) > 0:
        nested_item = _find_first_non_none(first_item)
        if nested_item is not None:
            return list[type(nested_item)]  # type: ignore[misc, return-value]
        # Default to str for nested lists with all None
        return list[str]  # type: ignore[return-value]

    return item_type


def _infer_dict_value_type(dct: dict) -> type:
    """Infer the value type of a dict, handling None values and list values."""
    if len(dct) == 0:
        # Default to str when dict is empty to avoid generic dict values
        return str

    # Find first non-None value
    first_value = None
    for val in dct.values():
        if val is not None:
            first_value = val
            break

    if first_value is None:
        # Default to str when all values are None
        return str

    # Handle list values
    if isinstance(first_value, list) and len(first_value) > 0:
        list_item = _find_first_non_none(first_value)
        if list_item is not None:
            return list[type(list_item)]  # type: ignore[misc, return-value]
        # Default to str for lists with all None
        return list[str]  # type: ignore[return-value]

    return type(first_value)


def _infer_type_from_sequence(
    sequence: Sequence[DataValue], signal_name: str, ds_name: str
) -> type:
    """
    Infer the type from a sequence of values.

    Returns str if all values are None, otherwise infers from the first non-None value.
    Handles lists and dicts with proper type inference for nested structures.
    """
    first_element = _find_first_non_none(sequence)

    if first_element is None:
        # Default to str if column is empty or all values are None
        return str

    typ = type(first_element)

    if not is_chain_type(typ):
        raise ValuesToTupleError(
            ds_name,
            f"signal '{signal_name}' has unsupported type '{typ.__name__}'."
            f" Please use DataModel types: {DataTypeNames}",
        )

    if isinstance(first_element, list):
        item_type = _infer_list_item_type(first_element)
        return list[item_type]  # type: ignore[valid-type, return-value]

    if isinstance(first_element, dict):
        # If the first dict is empty, use str as default key/value types
        if len(first_element) == 0:
            return dict[str, str]  # type: ignore[return-value]
        first_key = next(iter(first_element.keys()))
        value_type = _infer_dict_value_type(first_element)
        return dict[type(first_key), value_type]  # type: ignore[misc, return-value]

    return typ


def _validate_and_normalize_output(
    output: DataType | Sequence[str] | dict[str, DataType] | None,
    fr_map: dict[str, Sequence[DataValue]],
    ds_name: str,
) -> dict[str, DataType] | None:
    """Validate and normalize the output parameter to a dict format."""
    if not output:
        return None

    if not isinstance(output, (Sequence, str, dict)):
        if len(fr_map) != 1:
            raise ValuesToTupleError(
                ds_name,
                f"only one output type was specified, {len(fr_map)} expected",
            )
        if not isinstance(output, type):
            raise ValuesToTupleError(
                ds_name,
                f"output must specify a type while '{output}' was given",
            )

        key: str = next(iter(fr_map.keys()))
        return {key: output}  # type: ignore[dict-item]

    if not isinstance(output, dict):
        raise ValuesToTupleError(
            ds_name,
            "output type must be dict[str, DataType] while "
            f"'{type(output).__name__}' is given",
        )

    if len(output) != len(fr_map):
        raise ValuesToTupleError(
            ds_name,
            f"number of outputs '{len(output)}' should match"
            f" number of signals '{len(fr_map)}'",
        )

    return output  # type: ignore[return-value]


def values_to_tuples(
    ds_name: str = "",
    output: DataType | Sequence[str] | dict[str, DataType] | None = None,
    **fr_map: Sequence[DataValue],
) -> tuple[Any, Any, Any]:
    output = _validate_and_normalize_output(output, fr_map, ds_name)

    types_map: dict[str, type] = {}
    length = -1
    for k, v in fr_map.items():
        if not isinstance(v, Sequence) or isinstance(v, str):  # type: ignore[unreachable]
            raise ValuesToTupleError(ds_name, f"signal '{k}' is not a sequence")
        len_ = len(v)

        if output:
            if k not in output:  # type: ignore[operator]
                raise ValuesToTupleError(
                    ds_name,
                    f"signal '{k}' is not present in the output",
                )
        else:
            # FIXME: Stops as soon as it finds the first non-None value.
            # If a non-None value appears early, it won't check the remaining items for
            # `None` values.
            typ = _infer_type_from_sequence(v, k, ds_name)
            types_map[k] = typ

        if length < 0:
            length = len_
        elif length != len_:
            raise ValuesToTupleError(
                ds_name,
                f"signal '{k}' should have length {length} while {len_} is given",
            )

    if not output:
        output = types_map  # type: ignore[assignment]

    if not output:
        raise ValuesToTupleError(
            ds_name,
            "output type must be dict[str, DataType] while empty is given"
            " and no signals are provided",
        )

    output_types: list[type] = list(output.values())  # type: ignore[union-attr,call-arg,arg-type]
    if len(output) > 1:  # type: ignore[arg-type]
        tuple_type = tuple(output_types)
        res_type = tuple[tuple_type]  # type: ignore[valid-type]
        res_values: Sequence[Any] = list(zip(*fr_map.values(), strict=False))
    else:
        res_type = output_types[0]  # type: ignore[misc]
        res_values = next(iter(fr_map.values()))

    return res_type, output, res_values
