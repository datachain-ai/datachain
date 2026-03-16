import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import datachain as dc
from datachain.catalog.catalog import Catalog


def expected_parties_for_attempt(
    attempt: int,
    parties: int,
    sync_all_attempts: bool,
) -> int:
    if not sync_all_attempts:
        return parties
    return parties - (attempt - 1)


def wait_for_other_writers(
    barrier_dir: Path,
    worker_id: str,
    attempt: int,
    parties: int,
) -> None:
    attempt_dir = barrier_dir / f"attempt-{attempt}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / f"{worker_id}.ready").write_text("ready\n", encoding="utf-8")

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if len(list(attempt_dir.glob("*.ready"))) >= parties:
            return
        time.sleep(0.01)

    raise TimeoutError("Timed out waiting for concurrent save writers")


def build_result(worker_id: str, attempts: int, **kwargs) -> dict[str, object]:
    return {
        "worker": worker_id,
        "attempts": attempts,
        **kwargs,
    }


def main() -> None:
    dataset_name = os.environ["DATACHAIN_CONCURRENT_SAVE_DATASET"]
    worker_id = os.environ["DATACHAIN_CONCURRENT_SAVE_WORKER"]
    barrier_dir = Path(os.environ["DATACHAIN_CONCURRENT_SAVE_BARRIER_DIR"])
    parties = int(os.environ.get("DATACHAIN_CONCURRENT_SAVE_PARTIES", "2"))
    sync_all_attempts = os.environ.get("DATACHAIN_CONCURRENT_SAVE_SYNC_ALL", "0") == "1"

    original_create_dataset_version = Catalog.create_dataset_version
    attempts = 0

    def wrapped_create_dataset_version(self, *args, **kwargs):
        nonlocal attempts
        dataset = args[0]
        if dataset.name == dataset_name:
            attempts += 1
            if attempts == 1 or sync_all_attempts:
                wait_for_other_writers(
                    barrier_dir,
                    worker_id,
                    attempts,
                    expected_parties_for_attempt(
                        attempts,
                        parties,
                        sync_all_attempts,
                    ),
                )
        return original_create_dataset_version(self, *args, **kwargs)

    try:
        with patch.object(
            Catalog,
            "create_dataset_version",
            new=wrapped_create_dataset_version,
        ):
            saved = dc.read_values(num=[int(worker_id)], session=None).save(
                dataset_name
            )
        result = build_result(
            worker_id,
            attempts,
            version=saved.version,
            status="success",
        )
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        result = build_result(
            worker_id,
            attempts,
            status="error",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        exit_code = 1

    print(json.dumps(result))
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
