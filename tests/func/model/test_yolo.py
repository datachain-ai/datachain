import os

import numpy as np
import pytest
import torch
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from PIL import Image
from ultralytics.engine.results import Results

from datachain.model.ultralytics import (
    YoloBBox,
    YoloBBoxes,
    YoloOBBox,
    YoloOBBoxes,
    YoloPose,
    YoloPoses,
    YoloSegment,
    YoloSegments,
)


@pytest.fixture
def running_img() -> np.ndarray:
    img_file = os.path.join(os.path.dirname(__file__), "data", "running.jpg")
    with Image.open(img_file) as img:
        return np.array(img)


@pytest.fixture
def ships_img() -> np.ndarray:
    img_file = os.path.join(os.path.dirname(__file__), "data", "ships.jpg")
    with Image.open(img_file) as img:
        return np.array(img)


@pytest.fixture
def running_img_masks() -> torch.Tensor:
    mask0_file = os.path.join(os.path.dirname(__file__), "data", "running-mask0.png")
    with Image.open(mask0_file) as mask0_img:
        mask0_np = np.array(mask0_img)

    mask1_file = os.path.join(os.path.dirname(__file__), "data", "running-mask1.png")
    with Image.open(mask1_file) as mask1_img:
        mask1_np = np.array(mask1_img)

    stacked = np.stack(
        [mask0_np.astype(np.float32), mask1_np.astype(np.float32)],
        axis=0,
    )
    return torch.from_numpy(stacked)


def _assert_segment_shape(segment, expected_title, bbox, tol=2):
    assert set(segment) == {"title", "x", "y"}
    assert segment["title"] == expected_title
    x, y = segment["x"], segment["y"]
    assert len(x) > 0 and len(x) == len(y)
    x1, y1, x2, y2 = bbox
    assert min(x) >= x1 - tol and max(x) <= x2 + tol
    assert min(y) >= y1 - tol and max(y) <= y2 + tol


def test_yolo_bbox_from_results_empty(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={},
        boxes=torch.empty((0, 6)),
    )

    model = YoloBBox.from_result(result)
    assert model.model_dump() == {
        "cls": -1,
        "name": "",
        "confidence": 0,
        "box": {
            "coords": [],
            "title": "",
        },
    }

    model = YoloBBoxes.from_results([result])
    assert model.model_dump() == {
        "cls": [],
        "name": [],
        "confidence": [],
        "box": [],
    }


def test_yolo_bbox_from_results(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={0: "person", 16: "dog"},
        boxes=torch.tensor(
            [
                [100.2483, 83.7399, 183.179, 238.1918, 0.9057, 0.0],
                [10.8968, 177.1552, 71.1275, 239.0617, 0.8919, 16.0],
            ],
        ),
    )

    model = YoloBBox.from_result(result)
    model_json = model.model_dump()
    assert set(model_json.keys()) == {"cls", "name", "confidence", "box"}
    assert model_json["cls"] == 0
    assert model_json["name"] == "person"
    assert_almost_equal(model_json["confidence"], 0.9057, decimal=3)
    assert model_json["box"] == {"coords": [100, 84, 183, 238], "title": "person"}

    model = YoloBBoxes.from_results([result])
    model_json = model.model_dump()
    assert set(model_json.keys()) == {"cls", "name", "confidence", "box"}
    assert model_json["cls"] == [0, 16]
    assert model_json["name"] == ["person", "dog"]
    assert_array_almost_equal(model_json["confidence"], [0.9057, 0.8919], decimal=3)
    assert model_json["box"] == [
        {"coords": [100, 84, 183, 238], "title": "person"},
        {"coords": [11, 177, 71, 239], "title": "dog"},
    ]


def test_yolo_obbox_from_results_empty(ships_img):
    result = Results(
        orig_img=ships_img,
        path="ships.jpeg",
        names={},
        obb=torch.empty((0, 7)),
    )

    model = YoloOBBox.from_result(result)
    assert model.model_dump() == {
        "cls": -1,
        "name": "",
        "confidence": 0,
        "box": {
            "coords": [],
            "title": "",
        },
    }

    model = YoloOBBoxes.from_results([result])
    assert model.model_dump() == {
        "cls": [],
        "name": [],
        "confidence": [],
        "box": [],
    }


def test_yolo_obbox_from_results(ships_img):
    result = Results(
        orig_img=ships_img,
        path="ships.jpeg",
        names={1: "ship"},
        obb=torch.tensor(
            [
                [272.7724, 83.4637, 80.2673, 41.5667, 0.5629, 0.7289, 1.0],
                [70.2, 95.1912, 143.2061, 48.404, 0.494, 0.5785, 1.0],
                [120.6607, 76.8303, 115.6623, 42.3174, 0.7043, 0.5677, 1.0],
            ]
        ),
    )

    model = YoloOBBox.from_result(result)
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {"cls", "name", "confidence", "box"}
    assert model_dict["cls"] == 1
    assert model_dict["name"] == "ship"
    assert_almost_equal(model_dict["confidence"], 0.7289, decimal=3)
    assert model_dict["box"] == {
        "coords": [296, 122, 318, 87, 250, 44, 228, 80],
        "title": "ship",
    }

    model = YoloOBBoxes.from_results([result])
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {"cls", "name", "confidence", "box"}
    assert model_dict["cls"] == [1, 1, 1]
    assert model_dict["name"] == ["ship", "ship", "ship"]
    assert_array_almost_equal(
        model_dict["confidence"], [0.7289, 0.5785, 0.5677], decimal=3
    )
    assert model_dict["box"] == [
        {"coords": [296, 122, 318, 87, 250, 44, 228, 80], "title": "ship"},
        {"coords": [122, 150, 145, 108, 19, 40, -4, 83], "title": "ship"},
        {"coords": [151, 130, 178, 98, 90, 23, 63, 56], "title": "ship"},
    ]


def test_yolo_segment_from_results_empty(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={},
        obb=torch.empty((0, 7)),
    )

    model = YoloSegment.from_result(result)
    assert model.model_dump() == {
        "cls": -1,
        "name": "",
        "confidence": 0,
        "box": {
            "coords": [],
            "title": "",
        },
        "segment": {
            "title": "",
            "x": [],
            "y": [],
        },
    }

    model = YoloSegments.from_results([result])
    assert model.model_dump() == {
        "cls": [],
        "name": [],
        "confidence": [],
        "box": [],
        "segment": [],
    }


def test_yolo_segment_from_results_empty_segments(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={0: "person"},
        boxes=torch.tensor([[102.0, 84.0, 183.0, 238.0, 0.9078, 0.0]]),
    )

    model = YoloSegment.from_result(result)
    assert model.model_dump() == {
        "cls": 0,
        "name": "person",
        "confidence": 0.9078,
        "box": {
            "coords": [102, 84, 183, 238],
            "title": "person",
        },
        "segment": {
            "title": "",
            "x": [],
            "y": [],
        },
    }

    model = YoloSegments.from_results([result])
    assert model.model_dump() == {
        "cls": [0],
        "name": ["person"],
        "confidence": [0.9078],
        "box": [
            {
                "coords": [102, 84, 183, 238],
                "title": "person",
            }
        ],
        "segment": [],
    }


def test_yolo_seg_from_results(running_img, running_img_masks):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={0: "person", 16: "dog"},
        boxes=torch.tensor(
            [
                [11.6684, 178.2727, 71.4605, 238.8026, 0.9157, 16.0],
                [100.8046, 84.3006, 183.2971, 238.1916, 0.8578, 0.0],
            ],
        ),
        masks=running_img_masks,
    )

    model = YoloSegment.from_result(result)
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {
        "cls",
        "name",
        "confidence",
        "box",
        "segment",
    }
    assert model_dict["cls"] == 16
    assert model_dict["name"] == "dog"
    assert_almost_equal(model_dict["confidence"], 0.9157, decimal=3)
    assert model_dict["box"] == {
        "coords": [12, 178, 71, 239],
        "title": "dog",
    }
    _assert_segment_shape(model_dict["segment"], "dog", model_dict["box"]["coords"])

    model = YoloSegments.from_results([result])
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {
        "cls",
        "name",
        "confidence",
        "box",
        "segment",
    }
    assert model_dict["cls"] == [16, 0]
    assert model_dict["name"] == ["dog", "person"]
    assert_array_almost_equal(model_dict["confidence"], [0.9157, 0.8578], decimal=3)
    assert model_dict["box"] == [
        {"coords": [12, 178, 71, 239], "title": "dog"},
        {"coords": [101, 84, 183, 238], "title": "person"},
    ]
    assert len(model_dict["segment"]) == 2
    _assert_segment_shape(
        model_dict["segment"][0], "dog", model_dict["box"][0]["coords"]
    )
    _assert_segment_shape(
        model_dict["segment"][1], "person", model_dict["box"][1]["coords"]
    )


def test_yolo_pose_from_results_empty(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={},
        obb=torch.empty((0, 7)),
    )

    model = YoloPose.from_result(result)
    assert model.model_dump() == {
        "cls": -1,
        "name": "",
        "confidence": 0,
        "box": {
            "coords": [],
            "title": "",
        },
        "pose": {
            "x": [],
            "y": [],
            "visible": [],
        },
    }

    model = YoloPoses.from_results([result])
    assert model.model_dump() == {
        "cls": [],
        "name": [],
        "confidence": [],
        "box": [],
        "pose": [],
    }


def test_yolo_pose_from_results_empty_poses(running_img):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={0: "person"},
        boxes=torch.tensor([[102.0, 84.0, 183.0, 238.0, 0.9078, 0.0]]),
    )

    model = YoloPose.from_result(result)
    assert model.model_dump() == {
        "cls": 0,
        "name": "person",
        "confidence": 0.9078,
        "box": {
            "coords": [102, 84, 183, 238],
            "title": "person",
        },
        "pose": {
            "x": [],
            "y": [],
            "visible": [],
        },
    }

    model = YoloPoses.from_results([result])
    assert model.model_dump() == {
        "cls": [0],
        "name": ["person"],
        "confidence": [0.9078],
        "box": [{"coords": [102, 84, 183, 238], "title": "person"}],
        "pose": [],
    }


def test_yolo_pose_from_results(running_img, running_img_masks):
    result = Results(
        orig_img=running_img,
        path="running.jpeg",
        names={0: "person"},
        boxes=torch.tensor([[102.0, 84.0, 183.0, 238.0, 0.9078, 0.0]]),
        keypoints=torch.tensor(
            [
                [
                    [138.4593, 103.4513, 0.991],
                    [141.6461, 100.4834, 0.9863],
                    [136.6068, 100.2756, 0.8826],
                    [150.1754, 99.0366, 0.9691],
                    [0.0, 0.0, 0.2733],
                    [163.061, 114.3064, 0.9993],
                    [134.8462, 113.5232, 0.9977],
                    [179.6755, 136.7466, 0.9974],
                    [127.26, 130.8018, 0.9806],
                    [172.8256, 158.7771, 0.9962],
                    [118.0251, 146.8072, 0.9786],
                    [154.7233, 160.504, 0.9999],
                    [139.7511, 158.7988, 0.9997],
                    [143.9287, 193.5594, 0.9995],
                    [145.329, 190.7517, 0.9991],
                    [137.9563, 223.4, 0.9941],
                    [156.7822, 218.5813, 0.9921],
                ]
            ],
        ),
    )

    model = YoloPose.from_result(result)
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {
        "cls",
        "name",
        "confidence",
        "box",
        "pose",
    }
    assert model_dict["cls"] == 0
    assert model_dict["name"] == "person"
    assert_almost_equal(model_dict["confidence"], 0.9078, decimal=3)
    assert model_dict["box"] == {
        "coords": [102, 84, 183, 238],
        "title": "person",
    }
    assert set(model_dict["pose"]) == {"x", "y", "visible"}
    assert model_dict["pose"]["x"] == [
        138,
        142,
        137,
        150,
        0,
        163,
        135,
        180,
        127,
        173,
        118,
        155,
        140,
        144,
        145,
        138,
        157,
    ]
    assert model_dict["pose"]["y"] == [
        103,
        100,
        100,
        99,
        0,
        114,
        114,
        137,
        131,
        159,
        147,
        161,
        159,
        194,
        191,
        223,
        219,
    ]
    assert_array_almost_equal(
        model_dict["pose"]["visible"],
        [
            0.991,
            0.9862,
            0.8826,
            0.9691,
            0.2733,
            0.9993,
            0.9977,
            0.9974,
            0.9806,
            0.9962,
            0.9786,
            0.9999,
            0.9997,
            0.9995,
            0.9991,
            0.9941,
            0.9921,
        ],
        decimal=3,
    )

    model = YoloPoses.from_results([result])
    model_dict = model.model_dump()
    assert set(model_dict.keys()) == {
        "cls",
        "name",
        "confidence",
        "box",
        "pose",
    }
    assert model_dict["cls"] == [0]
    assert model_dict["name"] == ["person"]
    assert_array_almost_equal(model_dict["confidence"], [0.9078], decimal=3)
    assert model_dict["box"] == [{"coords": [102, 84, 183, 238], "title": "person"}]
    assert len(model_dict["pose"]) == 1
    assert set(model_dict["pose"][0]) == {"x", "y", "visible"}
    assert model_dict["pose"][0]["x"] == [
        138,
        142,
        137,
        150,
        0,
        163,
        135,
        180,
        127,
        173,
        118,
        155,
        140,
        144,
        145,
        138,
        157,
    ]
    assert model_dict["pose"][0]["y"] == [
        103,
        100,
        100,
        99,
        0,
        114,
        114,
        137,
        131,
        159,
        147,
        161,
        159,
        194,
        191,
        223,
        219,
    ]
    assert_array_almost_equal(
        model_dict["pose"][0]["visible"],
        [
            0.991,
            0.9862,
            0.8826,
            0.9691,
            0.2733,
            0.9993,
            0.9977,
            0.9974,
            0.9806,
            0.9962,
            0.9786,
            0.9999,
            0.9997,
            0.9995,
            0.9991,
            0.9941,
            0.9921,
        ],
        decimal=3,
    )
