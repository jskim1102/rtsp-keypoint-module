"""fix13: _resize_for_inference — 추론 제출 전 프레임 다운스케일(IPC pickle 비용 절감).

종횡비 보존 · long side == imgsz(640) · 업스케일 금지 · dtype/채널 보존.
"""

import numpy as np
import pytest

from app.streaming.manager import _resize_for_inference


def _frame(w: int, h: int) -> np.ndarray:
    """(h, w, 3) uint8 zeros 프레임 — OpenCV/numpy 는 (rows=h, cols=w, ch)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "w,h,exp_w,exp_h",
    [
        (1920, 1080, 640, 360),   # 16:9 1080p → long side 640
        (1280, 720, 640, 360),    # 16:9 720p
        (704, 480, 640, 436),     # 다운스케일, long side 640 (round(480*640/704)=436, 종횡비 보존)
        (1000, 1000, 640, 640),   # 정사각
    ],
)
def test_downscale_long_side_640_aspect_preserved(w, h, exp_w, exp_h):
    out = _resize_for_inference(_frame(w, h))
    assert out.shape[1] == exp_w
    assert out.shape[0] == exp_h


@pytest.mark.parametrize("w,h", [(640, 480), (320, 240)])
def test_no_upscale_when_within_imgsz(w, h):
    """long side <= 640 이면 변형 없음(업스케일 금지)."""
    src = _frame(w, h)
    out = _resize_for_inference(src)
    assert out.shape == src.shape


def test_dtype_and_channels_preserved():
    out = _resize_for_inference(_frame(1920, 1080))
    assert out.dtype == np.uint8
    assert out.shape[2] == 3
