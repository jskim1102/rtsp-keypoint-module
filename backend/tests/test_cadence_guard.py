"""런타임 cadence autotune 계약.

고정 MAX_INFER_PER_SEC/N 분배를 폐기하고 worker 실측 용량과 카메라 fps로
각 capture interval 을 정한다. 실 RTSP/GPU 없이 수학과 단일-writer 배선만 검증한다.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

import app.streaming.manager as manager_mod
from app.config import AUTOTUNE_HEADROOM, MAX_INFER_PER_SEC
from app.inference import InferenceResult
from app.streaming.manager import (
    InferenceTelemetry,
    SourceTelemetry,
    StreamManager,
    _allocate_fps,
)


class FakeCapture:
    def __init__(self, *, fps: float = 30.0, running: bool = True) -> None:
        self.is_running = running
        self.source_fps = fps
        self.interval = 0.25

    def set_inference_interval(self, interval: float) -> None:
        self.interval = interval

    def force_stop(self) -> None:
        self.is_running = False


def _ready_telemetry(infer_ms: float = 20.0) -> InferenceTelemetry:
    telemetry = InferenceTelemetry(alpha=0.5, min_samples=2)
    telemetry.observe(InferenceResult("warm", 0.0, infer_ms=infer_ms, idle_ms=infer_ms))
    telemetry.observe(InferenceResult("warm", 0.1, infer_ms=infer_ms, idle_ms=infer_ms))
    assert telemetry.ready
    return telemetry


def _mgr(caps: dict[str, FakeCapture]) -> StreamManager:
    m = object.__new__(StreamManager)
    m._lock = threading.Lock()
    m._captures = caps
    m._per_source_lock = threading.Lock()
    m._per_source_enabled = {}
    m._per_source_conf = {}
    m._per_source_models = {}
    m._tombstones = {}
    m._inference_interval = 0.033
    m._headroom = AUTOTUNE_HEADROOM
    m._target_fps_max = 60.0
    m._min_interval = 0.01
    m._max_interval = 1.0
    m._telemetry = _ready_telemetry()
    m._telemetry_lock = threading.Lock()
    m._source_telemetry = {}
    m._source_imgsz = {}
    m._imgsz_pressure_ticks = {}
    m._imgsz_relief_ticks = {}
    m._slow_interval_warned = set()
    m._latest_results = {}
    m._results_lock = threading.Lock()
    m._tune_requested = False
    m._last_drop_count = 0
    m._last_adaptive_drop_count = 0
    m._gpu_util_target = 1.0
    m._util_scale = 1.0
    m._worker = type(
        "FakeWorker",
        (),
        {
            "get_queue_stats": lambda self: {"dropped": 0},
            "get_pool_duty": lambda self: 0.5,
        },
    )()
    return m


def test_budget_allocator_preserves_total_and_relative_demand():
    allocated = _allocate_fps(
        {"fast": 30.0, "slow": 5.0},
        budget_fps=17.5,
        floor_fps=1.0,
    )

    assert sum(allocated.values()) == pytest.approx(17.5)
    assert allocated["fast"] > allocated["slow"] >= 1.0


def test_budget_allocator_keeps_capacity_when_floor_is_infeasible():
    """n*floor > capacity 면 두 계약을 동시에 만족할 수 없다. 큐 비움(총량<=용량)이 우선이다."""
    allocated = _allocate_fps({"a": 30.0, "b": 30.0}, budget_fps=1.0, floor_fps=1.0)
    assert allocated == pytest.approx({"a": 0.5, "b": 0.5})
    assert sum(allocated.values()) <= 1.0


def test_autotune_uses_measured_capacity_and_camera_fps():
    caps = {"a": FakeCapture(fps=30.0), "b": FakeCapture(fps=30.0)}
    m = _mgr(caps)

    assert m._run_autotune() is True

    budget = 50.0 * AUTOTUNE_HEADROOM
    for cap in caps.values():
        assert cap.interval == pytest.approx(1.0 / (budget / 2.0))
    assert sum(1.0 / cap.interval for cap in caps.values()) == pytest.approx(budget)


def test_gpu_ceiling_scales_final_cadence_without_triggering_imgsz_downshift():
    cap = FakeCapture(fps=30.0)
    m = _mgr({"cam": cap})
    m._gpu_util_target = 0.5
    m._worker = type(
        "BusyWorker",
        (),
        {
            "get_queue_stats": lambda self: {"dropped": 0},
            "get_pool_duty": lambda self: 0.8,
        },
    )()

    assert m._run_autotune() is True
    assert 1.0 / cap.interval == pytest.approx(30.0 * 0.625)
    assert m._source_imgsz["cam"] == 640


def test_autotune_tracks_source_fps_without_oversampling():
    slow = FakeCapture(fps=12.0)
    m = _mgr({"slow": slow})

    assert m._run_autotune() is True
    assert 1.0 / slow.interval == pytest.approx(12.0)


def test_autotune_treats_equal_fps_cameras_equally():
    first = FakeCapture(fps=30.0)
    second = FakeCapture(fps=30.0)
    m = _mgr({"first": first, "second": second})
    # capacity가 부족해도 두 카메라가 같은 source-fps 요구량을 공평하게 나눈다.
    m._telemetry = _ready_telemetry(infer_ms=100.0)

    assert m._run_autotune() is True
    assert first.interval == pytest.approx(second.interval)
    assert (
        (1.0 / first.interval) + (1.0 / second.interval)
        <= 10.0 * AUTOTUNE_HEADROOM + 1e-9
    )


def test_autotune_uses_safe_initial_cadence_until_worker_samples_are_ready():
    cap = FakeCapture()
    m = _mgr({"a": cap})
    m._telemetry = InferenceTelemetry(alpha=0.5, min_samples=2)

    assert m._run_autotune() is True
    assert cap.interval == pytest.approx(m._inference_interval)


def test_recompute_only_requests_tune_and_never_overwrites_interval():
    cap = FakeCapture()
    m = _mgr({"a": cap})
    before = cap.interval

    m._recompute_cadence()

    assert m._tune_requested is True
    assert cap.interval == before


def test_inactive_captures_are_excluded():
    active = FakeCapture(fps=15.0, running=True)
    inactive = FakeCapture(fps=30.0, running=False)
    m = _mgr({"active": active, "inactive": inactive})
    inactive_before = inactive.interval

    assert m._run_autotune() is True
    assert 1.0 / active.interval == pytest.approx(15.0)
    assert inactive.interval == inactive_before


def test_autotune_applies_closed_budget_before_worker_telemetry_is_ready():
    caps = {str(i): FakeCapture(fps=30.0) for i in range(16)}
    m = _mgr(caps)
    m._telemetry = InferenceTelemetry(alpha=0.5, min_samples=2)

    assert m._run_autotune() is True

    assigned = sum(1.0 / cap.interval for cap in caps.values())
    assert assigned <= MAX_INFER_PER_SEC * AUTOTUNE_HEADROOM + 1e-9
    assert all(cap.interval > 0.033 for cap in caps.values())


def test_new_capture_is_conservative_before_dispatch_can_apply_bootstrap(monkeypatch):
    created = []

    class StartupCapture:
        def __init__(self, source_id, source, **kwargs):
            self.source_id = source_id
            self.source = source
            self.initial_interval = kwargs["inference_interval"]
            self.is_running = False
            created.append(self)

        def acquire_viewer(self):
            return None

        def _ensure_running(self):
            self.is_running = True
            return True

    monkeypatch.setattr(manager_mod, "VideoCaptureThread", StartupCapture)
    m = _mgr({})

    assert m.start_capture("new", "rtsp://fake") is True
    assert created[0].initial_interval == pytest.approx(m._max_interval)


def test_stalled_source_does_not_block_healthy_source_autotune():
    stalled = FakeCapture(fps=0.0)
    healthy = FakeCapture(fps=30.0)
    m = _mgr({"stalled": stalled, "healthy": healthy})

    assert m._run_autotune() is True
    assert 1.0 / healthy.interval == pytest.approx(30.0)
    assert stalled.interval == pytest.approx(m._max_interval)


def test_same_model_update_preserves_converged_telemetry():
    class ModelWorker:
        def __init__(self) -> None:
            self.model = "yolo26n-pose.pt"
            self.set_calls: list[str] = []

        def get_status(self) -> dict:
            return {"model": self.model}

        def set_model(self, model: str) -> None:
            self.set_calls.append(model)
            self.model = model

    m = _mgr({})
    worker = ModelWorker()
    m._worker = worker
    telemetry = m._telemetry
    source = SourceTelemetry(infer_ms_ewma=17.0, infer_samples=3)
    m._source_telemetry["cam"] = source
    m._tune_requested = False

    m.set_inference_model("yolo26n-pose.pt")

    assert worker.set_calls == []
    assert m._telemetry is telemetry
    assert source.infer_ms_ewma == 17.0
    assert source.infer_samples == 3
    assert m._tune_requested is False


def test_changed_model_update_resets_cost_telemetry():
    class ModelWorker:
        model = "yolo26n-pose.pt"

        def __init__(self) -> None:
            self.set_calls: list[str] = []

        def get_status(self) -> dict:
            return {"model": self.model}

        def set_model(self, model: str) -> None:
            self.set_calls.append(model)
            self.model = model

    m = _mgr({})
    worker = ModelWorker()
    m._worker = worker
    telemetry = m._telemetry
    source = SourceTelemetry(infer_ms_ewma=17.0, infer_samples=3)
    m._source_telemetry["cam"] = source

    m.set_inference_model("yolo26s-pose.pt")

    assert worker.set_calls == ["yolo26s-pose.pt"]
    assert m._telemetry is not telemetry
    assert source.infer_ms_ewma == 0.0
    assert source.infer_samples == 0


def test_remove_capture_500_churn_cycles_leave_no_per_source_growth(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(manager_mod.time, "monotonic", lambda: clock[0])
    m = _mgr({})
    structures = (
        "_captures",
        "_latest_results",
        "_per_source_enabled",
        "_per_source_conf",
        "_per_source_models",
        "_source_telemetry",
        "_tombstones",
        "_slow_interval_warned",
        "_source_imgsz",
        "_imgsz_pressure_ticks",
        "_imgsz_relief_ticks",
    )
    before = {name: len(getattr(m, name)) for name in structures}

    for i in range(500):
        clock[0] = i * (StreamManager._TOMBSTONE_TTL + 1.0)
        sid = f"churn-{i}"
        m._captures[sid] = FakeCapture()
        m._latest_results[sid] = InferenceResult(sid, clock[0])
        m._per_source_enabled[sid] = False
        m._per_source_conf[sid] = 0.7
        m._per_source_models[sid] = ["yolo26n-pose.pt"]
        m._source_telemetry[sid] = SourceTelemetry(submitted=1, received=1)
        m._slow_interval_warned.add(sid)
        m._source_imgsz[sid] = 512
        m._imgsz_pressure_ticks[sid] = 1
        m._imgsz_relief_ticks[sid] = 2
        m.remove_capture(sid)

    clock[0] += StreamManager._TOMBSTONE_TTL + 1.0
    with m._lock:
        m._prune_tombstones_locked()
    after = {name: len(getattr(m, name)) for name in structures}
    assert after == before


def test_max_interval_overrun_warns_once_per_transition(caplog):
    caps = {"a": FakeCapture(), "b": FakeCapture()}
    m = _mgr(caps)
    m._telemetry = _ready_telemetry(infer_ms=2000.0)

    with caplog.at_level("WARNING"):
        assert m._run_autotune() is True
        assert m._run_autotune() is True

    overrun = [r for r in caplog.records if "MAX interval 초과" in r.message]
    assert len(overrun) == 2
    assert all("budget=" in r.message and "effective_floor=" in r.message for r in overrun)

    m._telemetry = _ready_telemetry(infer_ms=20.0)
    assert m._run_autotune() is True  # 정상 범위 복귀가 경고 latch를 해제
    m._telemetry = _ready_telemetry(infer_ms=2000.0)
    assert m._run_autotune() is True
    overrun = [r for r in caplog.records if "MAX interval 초과" in r.message]
    assert len(overrun) == 4


def test_infeasible_floor_never_reintroduces_more_load_than_budget():
    caps = {"a": FakeCapture(), "b": FakeCapture()}
    m = _mgr(caps)
    m._telemetry = _ready_telemetry(infer_ms=2000.0)

    assert m._run_autotune() is True

    assigned = sum(1.0 / cap.interval for cap in caps.values())
    assert assigned <= m._telemetry.capacity_fps * m._headroom + 1e-9
    assert all(cap.interval > m._max_interval for cap in caps.values())


def test_mixed_model_service_costs_stay_inside_worker_time_budget():
    caps = {f"light-{i}": FakeCapture() for i in range(7)}
    caps["heavy"] = FakeCapture()
    m = _mgr(caps)
    m._telemetry = _ready_telemetry(infer_ms=7.5)
    for sid in caps:
        metric = SourceTelemetry()
        metric.infer_ms_ewma = 23.1 if sid == "heavy" else 7.5
        metric.infer_samples = 2
        m._source_telemetry[sid] = metric

    assert m._run_autotune() is True

    service_ms = sum(
        (1.0 / cap.interval) * m._source_telemetry[sid].infer_ms_ewma
        for sid, cap in caps.items()
    )
    assert service_ms <= 1000.0 * m._headroom + 1e-9


def test_late_result_for_removed_source_cannot_resurrect_source_state():
    m = _mgr({})
    result = InferenceResult("removed", 1.0, infer_ms=10.0, idle_ms=10.0)

    m._record_results([result])

    assert m._telemetry.samples > 0  # worker 전역 capacity 표본은 여전히 유효
    assert "removed" not in m._source_telemetry
    assert "removed" not in m._latest_results


def test_autotune_downshifts_then_restores_imgsz_with_hysteresis():
    cap = FakeCapture(fps=30.0)
    m = _mgr({"cam": cap})
    m._telemetry = _ready_telemetry(infer_ms=100.0)

    assert m._run_autotune() is True
    assert m._run_autotune() is True
    assert m._source_imgsz["cam"] == 512

    # 5ms/item이면 30fps demand가 170fps budget의 65% 아래라 품질 복구 조건.
    m._telemetry = _ready_telemetry(infer_ms=5.0)
    for _ in range(5):
        assert m._run_autotune() is True
    assert m._source_imgsz["cam"] == 640


def test_submit_uses_source_adaptive_imgsz_and_preserves_frame_contract():
    cap = FakeCapture()
    m = _mgr({"cam": cap})
    submitted = []

    class SubmitWorker:
        @staticmethod
        def get_status():
            return {"enabled": True}

        @staticmethod
        def submit(request):
            submitted.append(request)
            return True

    m._worker = SubmitWorker()
    m._source_imgsz["cam"] = 416
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    m._submit_frame("cam", frame, captured_at=123.0)

    request = submitted[0]
    assert request.imgsz == 416
    assert request.frame.shape == (234, 416, 3)
    assert request.timestamp == 123.0


def test_worker_model_lanes_follow_active_source_model_and_on_off():
    cap = FakeCapture()
    m = _mgr({"cam": cap})

    class PoolWorker:
        def __init__(self):
            self.enabled = True
            self.configured = []

        def get_status(self):
            return {
                "enabled": self.enabled,
                "model": "yolo26n-pose.pt",
            }

        def configure_models(self, models):
            self.configured.append(set(models))

        @staticmethod
        def get_queue_stats():
            return {"dropped": 0}

    worker = PoolWorker()
    m._worker = worker
    m._per_source_models["cam"] = ["yolo26x-pose.pt"]

    m._recompute_cadence()
    assert worker.configured[-1] == {"yolo26x-pose.pt"}

    m._per_source_enabled["cam"] = False
    m._recompute_cadence()
    assert worker.configured[-1] == set()
