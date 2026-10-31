import pytest

import datachain as dc
from datachain.cli import main
from datachain.error import DataChainError
from datachain.telemetry import telemetry


def test_is_enabled():
    assert not telemetry.is_enabled()


def test_telemetry_api_call(mocker, tmp_dir):
    patch_send = mocker.patch("iterative_telemetry.IterativeTelemetryLogger.send")
    mocker.patch.object(telemetry, "_event_sent", False)

    dc.read_storage(tmp_dir.as_uri())
    assert patch_send.call_count == 1
    args = patch_send.call_args_list[0].args[0]
    extra = args.pop("extra")

    assert args == {"interface": "class", "action": "datachain_init", "error": None}

    assert "name" in extra


@pytest.fixture
def failing_telemetry(mocker):
    mocker.patch.object(telemetry, "_event_sent", False)
    return mocker.patch(
        "iterative_telemetry.IterativeTelemetryLogger.send",
        side_effect=PermissionError("telemetry config is read-only"),
    )


def test_telemetry_error_does_not_break_read_dataset(failing_telemetry, test_session):
    dc.read_values(value=[1, 2], session=test_session).save("saved")
    assert dc.read_dataset("saved", session=test_session).to_values("value") == [1, 2]
    failing_telemetry.assert_called_once()


@pytest.mark.parametrize(
    "command_error,exit_code", [(None, 0), (DataChainError("command failed"), 1)]
)
def test_telemetry_error_keeps_cli_exit_code(
    failing_telemetry, mocker, cli_logger, command_error, exit_code
):
    mocker.patch("datachain.catalog.get_catalog")
    mocker.patch(
        "datachain.cli.handle_command", return_value=0, side_effect=command_error
    )
    assert main(["dataset", "ls"]) == exit_code
    failing_telemetry.assert_called_once()
