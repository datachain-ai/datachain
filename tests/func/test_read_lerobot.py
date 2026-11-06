import json
import logging

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import datachain as dc

av = pytest.importorskip("av")

FPS = 10
TOP = "observation.images.top"
WRIST = "observation.images.wrist"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"


def _write_video(path, n_frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=FPS)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            image = np.full((16, 16, 3), i * 20, dtype=np.uint8)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(image, "rgb24")))
        container.mux(stream.encode())


def _make_dataset(root, version="v3.0"):
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": version,
        "fps": FPS,
        "features": {
            TOP: {"dtype": "video", "shape": [16, 16, 3]},
            WRIST: {"dtype": "video", "shape": [16, 16, 3]},
            "action": {"dtype": "float32", "shape": [2]},
        },
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": VIDEO_PATH,
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    # Episodes of 5, 3 and 4 frames: TOP holds all in one file, WRIST splits 0-1 | 2.
    episodes = {
        "episode_index": [0, 1, 2],
        "tasks": [["pick"], ["place"], ["pick", "place"]],
        "length": [5, 3, 4],
        "data/chunk_index": [0, 0, 0],
        "data/file_index": [0, 0, 0],
        "dataset_from_index": [0, 5, 8],
        "dataset_to_index": [5, 8, 12],
        f"videos/{TOP}/chunk_index": [0, 0, 0],
        f"videos/{TOP}/file_index": [0, 0, 0],
        f"videos/{TOP}/from_timestamp": [0.0, 0.5, 0.8],
        f"videos/{TOP}/to_timestamp": [0.5, 0.8, 1.2],
        f"videos/{WRIST}/chunk_index": [0, 0, 0],
        f"videos/{WRIST}/file_index": [0, 0, 1],
        f"videos/{WRIST}/from_timestamp": [0.0, 0.5, 0.0],
        f"videos/{WRIST}/to_timestamp": [0.5, 0.8, 0.4],
        "stats/action/mean": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    }
    pq.write_table(
        pa.table(episodes), root / "meta/episodes/chunk-000/file-000.parquet"
    )
    _write_video(
        root / VIDEO_PATH.format(video_key=TOP, chunk_index=0, file_index=0), 12
    )
    _write_video(
        root / VIDEO_PATH.format(video_key=WRIST, chunk_index=0, file_index=0), 8
    )
    _write_video(
        root / VIDEO_PATH.format(video_key=WRIST, chunk_index=0, file_index=1), 4
    )


def _frame_values(fragment):
    start, end = round(fragment.start * FPS), round(fragment.end * FPS)
    return [
        round(frame.get_np().mean() / 20) * 20
        for frame in fragment.video.get_frames(start, end)
    ]


@pytest.fixture
def lerobot_dataset(tmp_dir):
    root = tmp_dir / "ds"
    _make_dataset(root)
    return root


def test_read_lerobot_episode_metadata(lerobot_dataset, test_session):
    chain = dc.read_lerobot(lerobot_dataset.as_uri(), session=test_session)

    assert set(chain.schema) == {
        "episode_index",
        "tasks",
        "length",
        "data_chunk_index",
        "data_file_index",
        "dataset_from_index",
        "dataset_to_index",
        "videos",
    }
    assert chain.order_by("episode_index").to_list(
        "episode_index", "length", "tasks", "data_file_index", "dataset_to_index"
    ) == [
        (0, 5, ["pick"], 0, 5),
        (1, 3, ["place"], 0, 8),
        (2, 4, ["pick", "place"], 0, 12),
    ]


def test_read_lerobot_video_fragments(lerobot_dataset, test_session):
    chain = dc.read_lerobot(lerobot_dataset.as_uri(), session=test_session)
    rows = chain.order_by("episode_index").to_list(
        "videos.observation_images_top", "videos.observation_images_wrist"
    )

    assert [(t.start, t.end, w.start, w.end) for t, w in rows] == [
        (0.0, 0.5, 0.0, 0.5),
        (0.5, 0.8, 0.5, 0.8),
        (0.8, 1.2, 0.0, 0.4),
    ]
    top, wrist = rows[2]
    assert top.video.path == "videos/observation.images.top/chunk-000/file-000.mp4"
    assert wrist.video.path == "videos/observation.images.wrist/chunk-000/file-001.mp4"
    assert isinstance(wrist.video, dc.VideoFile)
    assert wrist.video.size == (lerobot_dataset / wrist.video.path).stat().st_size
    assert wrist.video.etag


def test_read_lerobot_decodes_episode_frames(lerobot_dataset, test_session):
    chain = dc.read_lerobot(lerobot_dataset.as_uri(), session=test_session)
    top, wrist = chain.order_by("episode_index").to_list(
        "videos.observation_images_top", "videos.observation_images_wrist"
    )[1]

    assert _frame_values(top) == [100, 120, 140]
    assert _frame_values(wrist) == [100, 120, 140]


def test_read_lerobot_save_round_trip(lerobot_dataset, test_session):
    dc.read_lerobot(lerobot_dataset.as_uri(), session=test_session).save("eps")
    chain = dc.read_dataset("eps", session=test_session)

    (wrist,) = chain.filter(dc.C("episode_index") == 2).to_values(
        "videos.observation_images_wrist"
    )
    assert (wrist.start, wrist.end) == (0.0, 0.4)
    assert _frame_values(wrist) == [0, 20, 40, 60]


def test_read_lerobot_skips_episodes_with_missing_videos(
    lerobot_dataset, test_session, caplog
):
    (
        lerobot_dataset
        / VIDEO_PATH.format(video_key=WRIST, chunk_index=0, file_index=1)
    ).unlink()

    with caplog.at_level(logging.WARNING, logger="datachain"):
        chain = dc.read_lerobot(lerobot_dataset.as_uri(), session=test_session)

    assert sorted(chain.to_values("episode_index")) == [0, 1]
    assert "Skipped 1 of 3 episodes" in caplog.text


def test_read_lerobot_rejects_non_v3(tmp_dir, test_session):
    _make_dataset(tmp_dir / "old", version="v2.1")

    with pytest.raises(ValueError, match=r"LeRobot v2\.1, only v3 is supported"):
        dc.read_lerobot((tmp_dir / "old").as_uri(), session=test_session)


def test_read_lerobot_requires_info_json(tmp_dir, test_session):
    (tmp_dir / "empty" / "data").mkdir(parents=True)
    (tmp_dir / "empty" / "data" / "x.parquet").write_bytes(b"x")

    with pytest.raises(ValueError, match=r"Expected one meta/info\.json"):
        dc.read_lerobot((tmp_dir / "empty").as_uri(), session=test_session)
