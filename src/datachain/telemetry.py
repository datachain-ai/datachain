import logging
import os
from importlib.metadata import PackageNotFoundError, version

from iterative_telemetry import IterativeTelemetryLogger

from datachain.utils import env2bool

logger = logging.getLogger(__name__)


def is_enabled():
    """
    Determine if telemetry is enabled based on environment variables and configuration.
    """
    # Disable telemetry if running in test mode
    if env2bool("DATACHAIN_TEST"):
        return False

    # Check if telemetry is disabled by environment variable
    disabled = bool(os.getenv("DATACHAIN_NO_ANALYTICS"))
    if disabled:
        logger.debug("Telemetry is disabled by environment variable.")
        return False

    logger.debug("Telemetry is enabled.")
    return True


class DataChainTelemetryLogger(IterativeTelemetryLogger):
    def send(self, payload: dict, use_thread: bool = False, use_daemon: bool = True):
        try:
            return super().send(payload, use_thread=use_thread, use_daemon=use_daemon)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to send telemetry", exc_info=True)
            return None


# Try to get the version of the datachain package
try:
    __version__ = version("datachain")
except PackageNotFoundError:
    __version__ = "unknown"

# Initialize telemetry logger
telemetry = DataChainTelemetryLogger("datachain", __version__, is_enabled)
