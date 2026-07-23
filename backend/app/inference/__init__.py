"""YOLO 추론 워커 패키지.

추론은 메인 FastAPI 와 다른 프로세스에서 수행한다 (별도 프로세스 원칙).
models_dir = preset(yolo26n~x) + custom .pt 가중치 관리.
"""

from app.inference.worker import (
    Detection,
    FrameRequest,
    InferenceResult,
    InferenceWorker,
)
from app.inference import models_dir

__all__ = [
    "Detection",
    "FrameRequest",
    "InferenceResult",
    "InferenceWorker",
    "models_dir",
]
