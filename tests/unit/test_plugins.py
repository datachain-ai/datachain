import logging
from types import SimpleNamespace

import pytest

from datachain import plugins


@pytest.mark.parametrize("failure_stage", ["load", "initialize"])
def test_plugin_failure_warns_and_continues(mocker, caplog, failure_stage):
    error = RuntimeError("broken plugin")
    broken_func = mocker.Mock(
        side_effect=error if failure_stage == "initialize" else None
    )
    broken_load = mocker.Mock(
        side_effect=error if failure_stage == "load" else None,
        return_value=broken_func,
    )
    working_func = mocker.Mock()
    entry_points = [
        SimpleNamespace(
            name="broken", dist=SimpleNamespace(name="broken-dist"), load=broken_load
        ),
        SimpleNamespace(
            name="working",
            dist=SimpleNamespace(name="working-dist"),
            load=mocker.Mock(return_value=working_func),
        ),
    ]
    mocker.patch.object(
        plugins.importlib_metadata,
        "entry_points",
        return_value=SimpleNamespace(select=lambda **kwargs: entry_points),
    )
    mocker.patch.object(plugins, "_plugins_loaded", False)

    with caplog.at_level(logging.WARNING, logger="datachain"):
        plugins.ensure_plugins_loaded()

    working_func.assert_called_once_with()
    assert "broken" in caplog.text
    assert "broken-dist" in caplog.text
    assert "broken plugin" in caplog.text
