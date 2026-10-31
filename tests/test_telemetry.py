import datachain as dc
from datachain.telemetry import telemetry


def test_is_enabled():
    assert not telemetry.is_enabled()


def test_telemetry_api_call(mocker, tmp_dir):
    patch_send = mocker.patch("iterative_telemetry.IterativeTelemetryLogger.send")
    telemetry._event_sent = False

    dc.read_storage(tmp_dir.as_uri())
    assert patch_send.call_count == 1
    args = patch_send.call_args_list[0].args[0]
    extra = args.pop("extra")

    assert args == {"interface": "class", "action": "datachain_init", "error": None}

    assert "name" in extra


def test_telemetry_cache_error_does_not_block_reading(mocker, test_session):
    dc.read_values(value=[1, 2], session=test_session).save("saved")
    mocker.patch.object(telemetry, "is_enabled", return_value=True)
    runtime_info = mocker.patch(
        "iterative_telemetry.IterativeTelemetryLogger._runtime_info",
        side_effect=PermissionError("telemetry cache is read-only"),
    )
    telemetry._event_sent = False
    assert dc.read_dataset("saved", session=test_session).to_list("value") == [
        (1,),
        (2,),
    ]
    runtime_info.assert_called_once()
