"""Inference autotune 계측·cadence·bounded-state 회귀 테스트."""

from __future__ import annotations

import queue

import numpy as np
import pytest

import app.streaming.capture as capture_mod
from app.inference import FrameRequest, InferenceResult
from app.inference.worker import InferenceWorker
from app.streaming.capture import VideoCaptureThread
from app.streaming.manager import InferenceTelemetry


def test_worker_telemetry_ewma_converges_and_capacity_is_exact():
    telemetry = InferenceTelemetry(alpha=0.5, min_samples=3)

    # 0은 모델별 warm-up 폐기 표본 — 집계하지 않는다.
    assert telemetry.observe(InferenceResult("s", 0.0, infer_ms=0.0, idle_ms=0.0)) is False
    assert telemetry.samples == 0

    assert telemetry.observe(InferenceResult("s", 0.1, infer_ms=20.0, idle_ms=20.0)) is True
    assert telemetry.observe(InferenceResult("s", 0.2, infer_ms=10.0, idle_ms=30.0)) is True
    assert telemetry.observe(InferenceResult("s", 0.3, infer_ms=10.0, idle_ms=10.0)) is True

    assert telemetry.infer_ms_ewma == pytest.approx(12.5)
    assert telemetry.duty_ewma == pytest.approx(0.4375)
    assert telemetry.capacity_fps == pytest.approx(80.0)
    assert telemetry.ready is True


def test_worker_submit_counts_drop_oldest_without_unbounded_history():
    worker = InferenceWorker()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    assert worker.submit(FrameRequest("s", frame, 0.0)) is True
    assert worker.submit(FrameRequest("s", frame, 0.1)) is True
    assert worker.get_queue_stats() == {"submitted": 2, "dropped": 1}
    assert not any(isinstance(v, list) for v in worker.__dict__.values())


def test_cadence_speedup_pulls_next_submit_deadline_to_stable_source_phase(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(capture_mod.time, "time", lambda: clock[0])
    thread = VideoCaptureThread("s", "rtsp://fake", inference_interval=10.0)
    thread._next_submit_ts = 109.0

    thread.set_inference_interval(0.25)

    assert thread._inference_interval == pytest.approx(0.25)
    assert thread._next_submit_ts == pytest.approx(100.0 + thread._submit_phase * 0.25)


def test_simultaneous_capture_phases_prevent_latest_wins_drops(monkeypatch):
    """16캠 동시 시작도 26.7% service budget에서 pending 교체가 없어야 한다."""
    clock = [100.0]
    monkeypatch.setattr(capture_mod.time, "time", lambda: clock[0])
    interval = 0.5
    threads = [
        VideoCaptureThread(f"cam-{i}", "rtsp://fake", inference_interval=interval)
        for i in range(16)
    ]

    class ServiceQueue:
        """in_q size=1 + 별도 consumer의 deterministic service-time 모사."""

        def __init__(self, service_sec: float) -> None:
            self.service_sec = service_sec
            self.now = 0.0
            self.busy_until = 0.0
            self.queued = None

        def advance(self, now: float) -> None:
            self.now = now
            while self.queued is not None and self.busy_until <= now:
                self.queued = None
                self.busy_until += self.service_sec

        def put_nowait(self, item) -> None:
            if self.busy_until <= self.now and self.queued is None:
                self.busy_until = self.now + self.service_sec
                return
            if self.queued is None:
                self.queued = item
                return
            raise queue.Full

        def get_nowait(self):
            if self.queued is None:
                raise queue.Empty
            item = self.queued
            self.queued = None
            return item

    worker = InferenceWorker()
    # 16 * 2fps 중 service utilization 26.7%. source phase가 없으면 cycle당 burst가 난다.
    lane = worker._make_lane("yolo26n-pose.pt")
    lane.in_q = ServiceQueue(service_sec=(interval * 0.267) / len(threads))
    worker._lanes["yolo26n-pose.pt"] = lane
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    events = sorted(
        (thread._next_submit_ts + cycle * interval, thread.source_id)
        for cycle in range(4)
        for thread in threads
    )
    for at, source_id in events:
        lane.in_q.advance(at)
        worker.submit(FrameRequest(source_id, frame, at))
        worker.flush_pending()

    assert worker.get_queue_stats() == {"submitted": 64, "dropped": 0}
    deadlines = [thread._next_submit_ts for thread in threads]
    assert all(clock[0] < deadline < clock[0] + interval for deadline in deadlines)
    assert len({round(deadline, 9) for deadline in deadlines}) == len(threads)


def test_source_fps_averages_frame_intervals_without_reciprocal_bias():
    thread = VideoCaptureThread("s", "rtsp://fake")
    now = 0.0
    thread._record_frame_advance(now)
    # 평균 간격은 50ms(20fps). 순간 1/dt 평균은 55.6fps로 크게 상향 편향된다.
    for _ in range(100):
        now += 0.01
        thread._record_frame_advance(now)
        now += 0.09
        thread._record_frame_advance(now)

    assert 15.0 < thread.source_fps < 25.0
