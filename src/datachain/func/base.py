from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy import TableClause

    from datachain.lib.signal_schema import SignalSchema
    from datachain.query.schema import Column


class Function:
    __metaclass__ = ABCMeta

    name: str
    cols: Sequence
    args: Sequence

    @abstractmethod
    def get_column(
        self,
        signals_schema: "SignalSchema | None" = None,
        label: str | None = None,
        table: "TableClause | None" = None,
    ) -> "Column":
        pass
