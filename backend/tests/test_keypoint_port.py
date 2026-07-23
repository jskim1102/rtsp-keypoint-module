"""rtsp-detection 신형 추론 인프라에 재이식한 pose 계약 회귀 테스트."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from app.inference.worker import (
    Detection,
    FrameRequest,
    InferenceResult,
    _merge_partial_results,
    _parse_results,
    _rescale_result,
)
from app.streaming.manager import detections_to_json


class _Tensor:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Boxes:
    def __init__(self) -> None:
        self.xyxy = _Tensor([[10.2, 20.4, 110.6, 220.8]])
        self.conf = _Tensor([0.91])
        self.cls = _Tensor([0])

    def __len__(self) -> int:
        return 1


def _keypoints() -> list[tuple[int, int, float]]:
    return [(i + 1, i + 2, 0.8) for i in range(17)]


def test_parse_results_returns_empty_when_pose_keypoints_are_missing():
    result = SimpleNamespace(boxes=_Boxes(), keypoints=None)

    assert _parse_results(result, {0: "person"}, "yolo26n-pose.pt") == []


def test_parse_results_keeps_person_box_and_extracts_17_keypoints():
    data = np.zeros((1, 17, 3), dtype=float)
    data[0, :, 0] = np.arange(17) + 1.2
    data[0, :, 1] = np.arange(17) + 2.7
    data[0, :, 2] = 0.8
    result = SimpleNamespace(
        boxes=_Boxes(),
        keypoints=SimpleNamespace(data=_Tensor(data)),
    )

    detections = _parse_results(result, {0: "person"}, "yolo26n-pose.pt")

    assert len(detections) == 1
    assert detections[0].xyxy == (10, 20, 110, 220)
    assert detections[0].keypoints[0] == (1, 2, 0.8)
    assert len(detections[0].keypoints) == 17


def test_oom_rescale_restores_box_and_keypoints_to_request_coordinates():
    smaller = InferenceResult(
        "cam",
        1.0,
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                xyxy=(10, 20, 100, 150),
                model="yolo26n-pose.pt",
                keypoints=_keypoints(),
            )
        ],
        frame_w=320,
        frame_h=180,
        effective_imgsz=320,
        oom_recovered=True,
    )

    recovered = _rescale_result(smaller, frame_w=640, frame_h=360)

    assert recovered.detections[0].xyxy == (20, 40, 200, 300)
    assert recovered.detections[0].keypoints[0] == (2, 4, 0.8)
    assert recovered.frame_w == 640
    assert recovered.frame_h == 360


def test_multimodel_merge_preserves_each_detection_keypoints():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    request = FrameRequest("cam", frame, 123.0, request_id=9, imgsz=640)
    partials = [
        InferenceResult(
            "cam",
            123.0,
            detections=[
                Detection(
                    0,
                    "person",
                    0.9,
                    (1, 2, 30, 40),
                    model=model,
                    keypoints=_keypoints(),
                )
            ],
            frame_w=640,
            frame_h=360,
            request_id=9,
            model=model,
        )
        for model in ("yolo26n-pose.pt", "yolo26x-pose.pt")
    ]

    merged = _merge_partial_results(request, partials)

    assert [item.model for item in merged.detections] == [
        "yolo26n-pose.pt",
        "yolo26x-pose.pt",
    ]
    assert all(len(item.keypoints) == 17 for item in merged.detections)


def test_ws_payload_contains_box_keypoints_and_frame_seam():
    result = InferenceResult(
        source_id="cam",
        timestamp=1.0,
        frame_w=640,
        frame_h=360,
        detections=[
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.9,
                xyxy=(1, 2, 30, 40),
                model="yolo26n-pose.pt",
                keypoints=_keypoints(),
            )
        ],
    )

    payload = json.loads(detections_to_json(result))

    assert payload["frame"] == {"w": 640, "h": 360}
    assert payload["items"][0]["xyxy"] == [1, 2, 30, 40]
    assert len(payload["items"][0]["keypoints"]) == 17
