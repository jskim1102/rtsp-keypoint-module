"""모델별 worker pool + micro-batch 계약."""

from __future__ import annotations

import queue
import threading
from collections import OrderedDict

import numpy as np

from app.inference.worker import (
    Detection,
    FrameRequest,
    InferenceResult,
    InferenceWorker,
    _ModelLane,
    _group_requests_by_imgsz,
    _merge_partial_results,
    _next_lower_imgsz,
    _rescale_result,
)


def _request(source_id: str, request_id: int, *, imgsz: int = 640) -> FrameRequest:
    return FrameRequest(
        source_id=source_id,
        frame=np.zeros((imgsz // 2, imgsz, 3), dtype=np.uint8),
        timestamp=float(request_id),
        model_names=["yolo26n-pose.pt"],
        request_id=request_id,
        imgsz=imgsz,
    )


def test_lane_pending_is_source_bounded_latest_wins_without_reordering():
    lane = object.__new__(_ModelLane)
    lane._pending_lock = threading.Lock()
    lane._pending = OrderedDict()

    assert lane.enqueue(_request("a", 1)) is None
    assert lane.enqueue(_request("b", 2)) is None
    assert lane.enqueue(_request("a", 3)) == 1

    assert list(lane._pending) == ["a", "b"]
    assert [request.request_id for request in lane._pending.values()] == [3, 2]


def test_microbatch_groups_only_equal_inference_sizes():
    groups = _group_requests_by_imgsz(
        [_request("a", 1, imgsz=640), _request("b", 2, imgsz=512), _request("c", 3, imgsz=640)]
    )

    assert [[request.source_id for request in group] for group in groups] == [
        ["a", "c"],
        ["b"],
    ]


def test_multimodel_merge_preserves_request_frame_coordinate_contract():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    request = FrameRequest("cam", frame, 123.0, request_id=9, imgsz=640)
    first = InferenceResult(
        "cam",
        123.0,
        detections=[Detection(0, "person", 0.9, (1, 2, 30, 40), "yolo26n-pose.pt")],
        frame_w=640,
        frame_h=360,
        infer_ms=4.0,
        idle_ms=2.0,
        request_id=9,
        model="yolo26n-pose.pt",
        batch_size=4,
        effective_imgsz=640,
        device="cuda:0",
    )
    second = InferenceResult(
        "cam",
        123.0,
        detections=[Detection(0, "person", 0.8, (10, 20, 300, 200), "yolo26x-pose.pt")],
        frame_w=640,
        frame_h=360,
        infer_ms=12.0,
        idle_ms=1.0,
        request_id=9,
        model="yolo26x-pose.pt",
        batch_size=2,
        effective_imgsz=512,
        oom_recovered=True,
        device="cuda:0",
    )

    merged = _merge_partial_results(request, [first, second])

    assert merged.frame_w == 640
    assert merged.frame_h == 360
    assert [d.model for d in merged.detections] == [
        "yolo26n-pose.pt",
        "yolo26x-pose.pt",
    ]
    assert merged.infer_ms == 16.0
    assert merged.batch_size == 4
    assert merged.effective_imgsz == 512
    assert merged.oom_recovered is True


def test_inflight_aggregates_are_source_bounded_and_do_not_retain_frames():
    class SinkQueue:
        def put_nowait(self, item) -> None:
            return None

    worker = InferenceWorker()
    lane = worker._make_lane("yolo26n-pose.pt")
    lane.in_q = SinkQueue()
    worker._lanes["yolo26n-pose.pt"] = lane

    for request_id in range(10):
        assert worker.submit(_request("cam", request_id)) is True
        worker.flush_pending()

    assert len(worker._aggregates) == worker._MAX_INFLIGHT_PER_SOURCE
    assert all(
        not hasattr(aggregate.request, "frame")
        for aggregate in worker._aggregates.values()
    )
    assert worker.get_queue_stats() == {"submitted": 10, "dropped": 6}


def test_pool_aggregates_two_model_lane_results_once():
    worker = InferenceWorker()
    for model in ("yolo26n-pose.pt", "yolo26x-pose.pt"):
        lane = worker._make_lane(model)
        lane.in_q = queue.Queue()
        lane.out_q = queue.Queue()
        worker._lanes[model] = lane

    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    worker.submit(
        FrameRequest(
            "cam",
            frame,
            42.0,
            model_names=["yolo26n-pose.pt", "yolo26x-pose.pt"],
            imgsz=640,
        )
    )
    worker.flush_pending()
    for model in ("yolo26n-pose.pt", "yolo26x-pose.pt"):
        worker._lanes[model].out_q.put_nowait(
            InferenceResult(
                "cam",
                42.0,
                detections=[Detection(0, "person", 0.9, (1, 2, 3, 4), model)],
                frame_w=640,
                frame_h=360,
                infer_ms=5.0,
                request_id=1,
                model=model,
                batch_size=2,
                effective_imgsz=640,
                device="cuda:0",
            )
        )

    results = worker.drain_results()

    assert len(results) == 1
    assert results[0].request_id == 1
    assert [item.model for item in results[0].detections] == [
        "yolo26n-pose.pt",
        "yolo26x-pose.pt",
    ]


def test_oom_lower_stage_result_is_rescaled_to_request_coordinates():
    smaller = InferenceResult(
        "cam",
        1.0,
        detections=[Detection(0, "person", 0.9, (10, 20, 100, 150))],
        frame_w=320,
        frame_h=180,
        effective_imgsz=320,
        oom_recovered=True,
    )

    recovered = _rescale_result(smaller, frame_w=640, frame_h=360)

    assert recovered.frame_w == 640
    assert recovered.frame_h == 360
    assert recovered.detections[0].xyxy == (20, 40, 200, 300)
    assert _next_lower_imgsz(640, 320) == 512
    assert _next_lower_imgsz(512, 320) == 416
    assert _next_lower_imgsz(320, 320) == 320
