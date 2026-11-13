from datetime import UTC, datetime, timedelta

from datachain.lib.dataset_info import DatasetInfo
from datachain.lib.listing import (
    CURRENT_LISTING_PREFIX,
    LISTING_TTL,
    listing_uri_from_name,
)


class ListingInfo(DatasetInfo):
    @property
    def uri(self) -> str:
        return listing_uri_from_name(self.name)

    @property
    def versions(self) -> bool:
        """False for a listing of current objects only."""
        return not self.name.startswith(CURRENT_LISTING_PREFIX)

    @property
    def expires(self) -> datetime | None:
        if not self.finished_at:
            return None
        return self.finished_at + timedelta(seconds=LISTING_TTL)

    @property
    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.expires if self.expires else False

    @property
    def last_inserted_at(self):
        # TODO we need to add updated_at to dataset version or explicit last_inserted_at
        raise NotImplementedError

    def contains(self, other_name: str) -> bool:
        """Checks if this listing contains another one"""
        return other_name.startswith(self.name)
