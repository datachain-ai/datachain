import argparse
import cProfile
import json
import os
import re
import sys
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import datachain as dc
from datachain.lib.udf import Mapper
from datachain.query.dataset import UDFStep
from datachain.query.session import Session

PATTERN = re.compile(r"([^/]+)/(\d{4}-\d{2})/(\d{2})/")


def keys(path):
    match = PATTERN.search(path)
    return match.groups() if match else ("", "", "")


def heavy(path):
    time.sleep(0.02)
    return keys(path)


class ProfileMapper(Mapper):
    def setup(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def process(self, path):
        return keys(path)

    def teardown(self):
        self.profiler.disable()
        directory = Path(os.environ["UDF_PROFILE_DIR"])
        self.profiler.dump_stats(str(directory / f"worker-{os.getpid()}.prof"))


def timed_step(function, name, timings):
    def run(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0) + time.perf_counter() - start

    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--heavy", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        os.environ["UDF_PROFILE_DIR"] = str(args.profile_dir.resolve())
    os.environ["DATACHAIN_NO_ANALYTICS"] = "1"
    os.environ["DATACHAIN_IGNORE_CHECKPOINTS"] = "True"

    with tempfile.TemporaryDirectory(prefix="dc-udf-bench-") as root:
        os.environ["DATACHAIN_DIR"] = root
        try:
            with Session() as session:
                paths = [
                    f"B{i % 12}/2026-10/{i % 28 + 1:02d}/{i}.tif"
                    for i in range(args.rows)
                ]
                source = dc.read_values(path=paths, session=session).save("source")
                function = ProfileMapper() if args.profile_dir else keys
                if args.heavy:
                    function = heavy
                timings = {}
                dispatcher_profile = (
                    patch(
                        "datachain.query.dataset.get_datachain_executable",
                        return_value=[
                            sys.executable,
                            "-m",
                            "cProfile",
                            "-o",
                            str(args.profile_dir.resolve() / "dispatcher.prof"),
                            "-m",
                            "datachain",
                        ],
                    )
                    if args.profile_dir and args.parallel
                    else nullcontext()
                )
                with (
                    dispatcher_profile,
                    patch.object(
                        UDFStep,
                        "get_or_create_input_table",
                        timed_step(
                            UDFStep.get_or_create_input_table,
                            "prepare_seconds",
                            timings,
                        ),
                    ),
                    patch.object(
                        UDFStep,
                        "populate_udf_output_table",
                        timed_step(
                            UDFStep.populate_udf_output_table, "udf_seconds", timings
                        ),
                    ),
                ):
                    cpu_start = os.times()
                    start = time.perf_counter()
                    result = (
                        source.settings(parallel=args.parallel)
                        .map(
                            function,
                            params=["path"],
                            output={"band": str, "month": str, "day": str},
                        )
                        .save("result")
                    )
                    elapsed = time.perf_counter() - start
                    cpu_end = os.times()
                assert result.count() == args.rows
                records = result.order_by("sys.id").select(
                    "path", "band", "month", "day"
                )
                for record in records.to_records():
                    assert (record["band"], record["month"], record["day"]) == keys(
                        record["path"]
                    )
                print(
                    json.dumps(
                        {
                            **timings,
                            "rows": args.rows,
                            "parallel": args.parallel,
                            "heavy": args.heavy,
                            "profile": bool(args.profile_dir),
                            "seconds": elapsed,
                            "cpu_seconds": sum(cpu_end[:4]) - sum(cpu_start[:4]),
                            "rows_per_second": args.rows / elapsed,
                        }
                    ),
                    flush=True,
                )
        finally:
            Session.cleanup_for_tests()


if __name__ == "__main__":
    main()
