from collections.abc import Sequence
from enum import Enum
from typing import TYPE_CHECKING

from datachain.func import case, ifelse, isnone, or_
from datachain.lib.signal_schema import SignalSchema
from datachain.query.schema import Column

if TYPE_CHECKING:
    from datachain.lib.dc import DataChain

C = Column


STATUS_COL_NAME = "diff_7aeed3aa17ba4d50b8d1c368c76e16a6"
LEFT_DIFF_COL_NAME = "diff_95f95344064a4b819c8625cd1a5cfc2b"
RIGHT_DIFF_COL_NAME = "diff_5808838a49b54849aa461d7387376d34"


class CompareStatus(str, Enum):
    ADDED = "A"
    DELETED = "D"
    MODIFIED = "M"
    SAME = "S"


def _compare(  # noqa: C901
    left: "DataChain",
    right: "DataChain",
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None = None,
    compare: str | Sequence[str] | None = None,
    right_compare: str | Sequence[str] | None = None,
    added: bool = True,
    deleted: bool = True,
    modified: bool = True,
    same: bool = True,
    status_col: str | None = None,
) -> "DataChain":
    """Comparing two chains by identifying rows that are added, deleted, modified
    or same"""
    rname = "right_"
    schema = left.signals_schema  # final chain must have schema from left chain

    def _to_list(obj: str | Sequence[str] | None) -> list[str] | None:
        if obj is None:
            return None
        return [obj] if isinstance(obj, str) else list(obj)

    on = _to_list(on)  # type: ignore[assignment]
    right_on = _to_list(right_on)
    compare = _to_list(compare)
    right_compare = _to_list(right_compare)

    if not any([added, deleted, modified, same]):
        raise ValueError(
            "At least one of added, deleted, modified, same flags must be set"
        )
    if on is None:
        raise ValueError("'on' must be specified")
    if right_on and len(on) != len(right_on):
        raise ValueError("'on' and 'right_on' must be have the same length")
    if right_compare and not compare:
        raise ValueError("'compare' must be defined if 'right_compare' is defined")
    if compare and right_compare and len(compare) != len(right_compare):
        raise ValueError("'compare' and 'right_compare' must have the same length")

    # all left and right columns
    cols = left.signals_schema.clone_without_sys_signals().db_signals()
    right_cols = right.signals_schema.clone_without_sys_signals().db_signals()
    cols_select = list(left.signals_schema.clone_without_sys_signals().values.keys())

    # getting correct on and right_on column names
    on_ = on
    on = left.signals_schema.resolve(*on).db_signals()  # type: ignore[assignment]
    right_on = right.signals_schema.resolve(*(right_on or on_)).db_signals()  # type: ignore[assignment]

    # getting correct compare and right_compare column names if they are defined
    if compare:
        compare_ = compare
        compare = left.signals_schema.resolve(*compare).db_signals()  # type: ignore[assignment]
        right_compare = right.signals_schema.resolve(
            *(right_compare or compare_)
        ).db_signals()  # type: ignore[assignment]
    elif not compare and len(cols) != len(right_cols):
        # here we will mark all rows that are not added or deleted as modified since
        # there was no explicit list of compare columns provided (meaning we need
        # to check all columns to determine if row is modified or same), but
        # the number of columns on left and right is not the same (one of the chains
        # have additional column)
        compare = None
        right_compare = None
    else:
        # we are checking all columns as explicit compare is not defined
        compare = right_compare = [c for c in cols if c in right_cols and c not in on]  # type: ignore[misc]

    # get diff column names
    diff_col = status_col or STATUS_COL_NAME
    ldiff_col = LEFT_DIFF_COL_NAME
    rdiff_col = RIGHT_DIFF_COL_NAME

    # adding helper diff columns, which will be removed after
    left = left.mutate(**{ldiff_col: 1})
    right = right.mutate(**{rdiff_col: 1})

    if compare is None:
        modified_cond = True
    elif len(compare) == 0:
        modified_cond = False
    else:
        modified_cond = or_(  # type: ignore[assignment]
            *[
                C(c) != (C(f"{rname}{rc}") if c == rc else C(rc))
                for c, rc in zip(compare, right_compare, strict=False)  # type: ignore[arg-type]
            ]
        )

    dc_diff = (
        left.merge(right, on=on, right_on=right_on, rname=rname, full=True)
        .mutate(
            **{
                diff_col: case(
                    (isnone(ldiff_col), CompareStatus.DELETED),
                    (isnone(rdiff_col), CompareStatus.ADDED),
                    (modified_cond, CompareStatus.MODIFIED),
                    else_=CompareStatus.SAME,
                )
            }
        )
        # when the row is deleted, we need to take column values from the right chain
        .mutate(
            **{
                f"{l_on}": ifelse(
                    C(diff_col) == CompareStatus.DELETED,
                    C(f"{rname + l_on if on == right_on else r_on}"),
                    C(l_on),
                )
                for l_on, r_on in zip(on, right_on, strict=False)  # type: ignore[arg-type]
            }
        )
        .select_except(ldiff_col, rdiff_col)
    )

    if not added:
        dc_diff = dc_diff.filter(C(diff_col) != CompareStatus.ADDED)
    if not modified:
        dc_diff = dc_diff.filter(C(diff_col) != CompareStatus.MODIFIED)
    if not same:
        dc_diff = dc_diff.filter(C(diff_col) != CompareStatus.SAME)
    if not deleted:
        dc_diff = dc_diff.filter(C(diff_col) != CompareStatus.DELETED)

    if status_col:
        cols_select.append(diff_col)

    dc_diff = dc_diff.select(*cols_select)

    # final schema is schema from the left chain with status column added if needed
    dc_diff.signals_schema = (
        schema if not status_col else SignalSchema({status_col: str}) | schema
    )

    return dc_diff


def compare_and_split(
    left: "DataChain",
    right: "DataChain",
    on: str | Sequence[str],
    right_on: str | Sequence[str] | None = None,
    compare: str | Sequence[str] | None = None,
    right_compare: str | Sequence[str] | None = None,
    added: bool = True,
    deleted: bool = True,
    modified: bool = True,
    same: bool = False,
) -> dict[str, "DataChain"]:
    """Comparing two chains and returning multiple chains, one for each of `added`,
    `deleted`, `modified` and `same` status. Result is returned in form of
    dictionary where each item represents one of the statuses and key values
    are `A`, `D`, `M`, `S` corresponding. Note that status column is not in the
    resulting chains.

    Parameters:
        left: Chain to calculate diff on.
        right: Chain to calculate diff from.
        on: Column or list of columns to match on. If both chains have the
            same columns then this column is enough for the match. Otherwise,
            `right_on` parameter has to specify the columns for the other chain.
            This value is used to find corresponding row in other dataset. If not
            found there, row is considered as added (or removed if vice versa), and
            if found then row can be either modified or same.
        right_on: Optional column or list of columns
            for the `other` to match.
        compare: Column or list of columns to compare on. If both chains have
            the same columns then this column is enough for the compare. Otherwise,
            `right_compare` parameter has to specify the columns for the other
            chain. This value is used to see if row is modified or same. If
            not set, all columns will be used for comparison
        right_compare: Optional column or list of columns
                for the `other` to compare to.
        added (bool): Whether to return chain containing only added rows.
        deleted (bool): Whether to return chain containing only deleted rows.
        modified (bool): Whether to return chain containing only modified rows.
        same (bool): Whether to return chain containing only same rows.

    Example:
        ```py
        chains = compare(
            persons,
            new_persons,
            on=["id"],
            right_on=["other_id"],
            compare=["name"],
            added=True,
            deleted=True,
            modified=True,
            same=True,
        )
        ```
    """
    status_col = STATUS_COL_NAME

    res = _compare(
        left,
        right,
        on,
        right_on=right_on,
        compare=compare,
        right_compare=right_compare,
        added=added,
        deleted=deleted,
        modified=modified,
        same=same,
        status_col=status_col,
    )

    chains = {}

    def filter_by_status(compare_status) -> "DataChain":
        return res.filter(C(status_col) == compare_status).select_except(status_col)

    if added:
        chains[CompareStatus.ADDED.value] = filter_by_status(CompareStatus.ADDED)
    if deleted:
        chains[CompareStatus.DELETED.value] = filter_by_status(CompareStatus.DELETED)
    if modified:
        chains[CompareStatus.MODIFIED.value] = filter_by_status(CompareStatus.MODIFIED)
    if same:
        chains[CompareStatus.SAME.value] = filter_by_status(CompareStatus.SAME)

    return chains
