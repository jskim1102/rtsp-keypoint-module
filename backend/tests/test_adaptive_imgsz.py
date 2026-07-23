"""부하 기반 adaptive inference resolution 계약."""

from __future__ import annotations

from app.config import INFERENCE_IMGSZ_STAGES
from app.streaming.manager import _adaptive_imgsz_step


def test_overload_downshifts_after_hysteresis():
    stage = INFERENCE_IMGSZ_STAGES[-1]
    pressure = relief = 0

    stage, pressure, relief, changed = _adaptive_imgsz_step(
        stage,
        pressure_ticks=pressure,
        relief_ticks=relief,
        allocation_ratio=0.5,
        queue_dropped=False,
    )
    assert changed is False
    assert stage == 640

    stage, pressure, relief, changed = _adaptive_imgsz_step(
        stage,
        pressure_ticks=pressure,
        relief_ticks=relief,
        allocation_ratio=0.5,
        queue_dropped=False,
    )
    assert changed is True
    assert stage == 512
    assert pressure == relief == 0


def test_sustained_capacity_restores_quality_one_stage():
    stage = 416
    pressure = relief = 0

    for _ in range(4):
        stage, pressure, relief, changed = _adaptive_imgsz_step(
            stage,
            pressure_ticks=pressure,
            relief_ticks=relief,
            allocation_ratio=1.0,
            queue_dropped=False,
            underloaded=True,
        )
        assert changed is False

    stage, pressure, relief, changed = _adaptive_imgsz_step(
        stage,
        pressure_ticks=pressure,
        relief_ticks=relief,
        allocation_ratio=1.0,
        queue_dropped=False,
        underloaded=True,
    )
    assert changed is True
    assert stage == 512
    assert pressure == relief == 0


def test_queue_drop_forces_pressure_even_when_allocation_is_high():
    stage, pressure, relief, changed = _adaptive_imgsz_step(
        640,
        pressure_ticks=1,
        relief_ticks=3,
        allocation_ratio=1.0,
        queue_dropped=True,
    )

    assert changed is True
    assert stage == 512
    assert pressure == relief == 0
