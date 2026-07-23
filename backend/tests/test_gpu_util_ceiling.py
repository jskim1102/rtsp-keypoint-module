"""전역 GPU duty ceiling 제어 계약."""

from __future__ import annotations

import threading

import pytest

import app.inference_api as inference_api
from app.inference import InferenceResult
from app.inference.worker import InferenceWorker
from app.streaming.manager import StreamManager, _gpu_util_scale_step


def test_worker_pool_duty_combines_model_lane_ewmas():
    worker = InferenceWorker()

    worker._observe_lane_duty(
        InferenceResult("a", 1.0, infer_ms=50.0, idle_ms=50.0, model="yolo26n-pose.pt")
    )
    worker._observe_lane_duty(
        InferenceResult("b", 1.0, infer_ms=50.0, idle_ms=50.0, model="yolo26x-pose.pt")
    )

    # 두 lane이 각각 50%면 union duty는 1 - (1 - .5)^2 = 75%.
    assert worker.get_pool_duty() == pytest.approx(0.75)


def test_worker_pool_duty_ignores_warmup_and_retires_stale_lane():
    worker = InferenceWorker()
    worker._observe_lane_duty(
        InferenceResult("a", 1.0, infer_ms=20.0, idle_ms=20.0, model="yolo26n-pose.pt")
    )
    worker._observe_lane_duty(
        InferenceResult("b", 1.0, infer_ms=0.0, idle_ms=0.0, model="yolo26x-pose.pt")
    )

    assert worker.get_pool_duty() == pytest.approx(0.5)

    worker.configure_models({"yolo26x-pose.pt"})

    assert worker.get_pool_duty() == 0.0


@pytest.mark.parametrize(
    ("current", "target", "duty", "expected"),
    [
        (0.35, 1.0, 0.95, 1.0),   # 100% 목표는 ceiling을 완전히 우회
        (1.0, 0.5, 0.8, 0.625),   # 초과 시 target/duty 비율로 즉시 감속
        (0.5, 0.8, 0.6, 0.55),    # 여유가 있으면 tick당 5%p씩 완만히 복구
        (0.5, 0.8, 0.79, 0.5),    # ±3% deadband 안에서는 유지
        (0.11, 0.1, 1.0, 0.1),    # scale 하한
    ],
)
def test_gpu_util_controller_step(
    current: float,
    target: float,
    duty: float,
    expected: float,
):
    assert _gpu_util_scale_step(current, target=target, duty=duty) == pytest.approx(
        expected
    )


def test_inference_api_roundtrips_clamped_gpu_target(monkeypatch):
    manager = StreamManager()
    monkeypatch.setattr(inference_api, "stream_manager", manager)

    response = inference_api.update_inference_config(
        inference_api.InferenceConfigUpdate(gpu_util_target=0.01)
    )

    assert response["gpu_util_target"] == pytest.approx(0.1)
    assert response["gpu_util_duty"] == 0.0


def test_manager_defaults_gpu_target_to_85_percent():
    manager = StreamManager()

    assert manager.get_inference_config()["gpu_util_target"] == pytest.approx(0.85)


def test_manager_config_exposes_realtime_worker_pool_duty():
    manager = object.__new__(StreamManager)
    manager._telemetry_lock = threading.Lock()
    manager._gpu_util_target = 0.8
    manager._util_scale = 0.6
    manager._worker = type(
        "DutyWorker",
        (),
        {
            "get_status": lambda self: {
                "enabled": True,
                "model": "yolo26n-pose.pt",
                "conf_threshold": 0.5,
                "device": "auto",
            },
            "get_pool_duty": lambda self: 0.73,
        },
    )()

    config = manager.get_inference_config()

    assert config["gpu_util_target"] == pytest.approx(0.8)
    assert config["gpu_util_duty"] == pytest.approx(0.73)
