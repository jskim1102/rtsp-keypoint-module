"""영상 캡처 + 추론 통합 스트리밍 패키지 (rtsp-detection: 좌표 전용).

추론용 RTSP 캡처(VideoCaptureThread)와 단일 공유 워커 매니저(StreamManager).
manager 는 직접 `from app.streaming.manager import manager` 로 import 한다
(phase3.ckpt2 #17 에서 추가).
"""

from app.streaming.capture import VideoCaptureThread

__all__ = ["VideoCaptureThread"]
