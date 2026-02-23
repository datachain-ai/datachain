from unittest.mock import MagicMock, patch

import pytest

import datachain as dc
from datachain.catalog import (
    QUERY_SCRIPT_ABORTED_EXIT_CODE,
    QUERY_SCRIPT_CANCELED_EXIT_CODE,
)


def _make_mock_popen(poll_return_value=0, communicate_side_effect=None):
    mock_process = MagicMock()
    mock_process.__enter__ = MagicMock(return_value=mock_process)
    mock_process.__exit__ = MagicMock(return_value=False)
    if communicate_side_effect:
        mock_process.communicate.side_effect = communicate_side_effect
    else:
        mock_process.communicate.return_value = (b"", b"")
    mock_process.poll.return_value = poll_return_value
    return mock_process


def _build_parallel_chain(session):
    def identity(x: int) -> int:
        return x

    return (
        dc.read_values(x=list(range(5)), session=session)
        .settings(parallel=2)
        .map(identity, output={"result": int})
    )


def test_aborted_exit_code_causes_sys_exit(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(poll_return_value=QUERY_SCRIPT_ABORTED_EXIT_CODE)

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(SystemExit) as exc_info:
            chain.to_list()

    assert exc_info.value.code == QUERY_SCRIPT_ABORTED_EXIT_CODE


def test_canceled_exit_code_causes_sys_exit(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(poll_return_value=QUERY_SCRIPT_CANCELED_EXIT_CODE)

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(SystemExit) as exc_info:
            chain.to_list()

    assert exc_info.value.code == QUERY_SCRIPT_CANCELED_EXIT_CODE


def test_unknown_exit_code_raises_runtime_error(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(poll_return_value=42)

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(RuntimeError, match="UDF Execution Failed! Exit code: 42"):
            chain.to_list()


def test_keyboard_interrupt_causes_cancel_sys_exit(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(
        communicate_side_effect=KeyboardInterrupt("simulated ctrl-c")
    )

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(SystemExit) as exc_info:
            chain.to_list()

    assert exc_info.value.code == QUERY_SCRIPT_CANCELED_EXIT_CODE


def test_aborted_exit_code_closes_warehouse(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(poll_return_value=QUERY_SCRIPT_ABORTED_EXIT_CODE)
    warehouse = test_session_tmpfile.catalog.warehouse
    original_close = warehouse.close
    close_called = False

    def tracking_close():
        nonlocal close_called
        close_called = True
        original_close()

    monkeypatch.setattr(warehouse, "close", tracking_close)

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(SystemExit):
            chain.to_list()

    assert close_called


def test_canceled_exit_code_closes_warehouse(test_session_tmpfile, monkeypatch):
    monkeypatch.delenv("DATACHAIN_DISTRIBUTED", raising=False)
    chain = _build_parallel_chain(test_session_tmpfile)

    mock_process = _make_mock_popen(poll_return_value=QUERY_SCRIPT_CANCELED_EXIT_CODE)
    warehouse = test_session_tmpfile.catalog.warehouse
    original_close = warehouse.close
    close_called = False

    def tracking_close():
        nonlocal close_called
        close_called = True
        original_close()

    monkeypatch.setattr(warehouse, "close", tracking_close)

    with patch("datachain.query.dataset.subprocess.Popen", return_value=mock_process):
        with pytest.raises(SystemExit):
            chain.to_list()

    assert close_called
