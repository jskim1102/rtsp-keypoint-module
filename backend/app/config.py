import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# backend/.env 를 로드
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# 빈 문자열(set-but-empty)도 미설정으로 취급해 안전기본 폴백.
#   os.getenv(k, default) 는 k 가 "" 면 ""(빈값)을 그대로 돌려준다 → CORS_ORIGINS="" → [""]
#   (기본 * 아님 → CORS 차단), MAX_IPCAMS="" → int("") → ValueError(import 크래시).
#   `or` 로 빈값=거짓 → 기본값으로 폴백한다.
CORS_ORIGINS: str = os.getenv("CORS_ORIGINS") or "*"

_raw_max_ipcams = int(os.getenv("MAX_IPCAMS") or "16")
MAX_IPCAMS: int = max(1, min(64, _raw_max_ipcams))

# mediamtx API 주소 — 하드코딩 fallback 을 두지 않는다(빈 문자열 = 미설정).
# Docker compose 가 environment 블록으로 `http://mediamtx:9997` 를 주입하고,
# 로컬 실행 시에는 backend/.env 에 설정한다. 실제로 호출하는 app.mediamtx 가
# 미설정이면 명시 에러를 낸다(import 시점엔 raise 안 함 — 순수 import/테스트 허용).
MEDIAMTX_API: str = os.getenv("MEDIAMTX_API", "")

# mediamtx 인증(#100) — backend user 로 API 호출 시 Basic auth. 비번 비우면 무인증(로컬/테스트 하위호환).
MEDIAMTX_BACKEND_USER: str = os.getenv("MEDIAMTX_BACKEND_USER", "backend")
MEDIAMTX_BACKEND_PASS: str = os.getenv("MEDIAMTX_BACKEND_PASS", "")

# WebRTC 외부접속 광고 호스트(공인 IP). mediamtx.yml 의 webrtcAdditionalHosts 로
# 주입된다. 미설정이면 mediamtx 가 컨테이너 내부 주소만 광고 → 외부에서 영상 안 나옴.
MEDIAMTX_WEBRTC_HOST: str = os.getenv("MEDIAMTX_WEBRTC_HOST", "")

# ── detection (YOLO 추론 — deepeye-lite 차용) ──
# 기본 모델 + conf. 워커(InferenceWorker)도 같은 env 를 직접 읽음(worker.py).
YOLO_DEFAULT_MODEL: str = os.getenv("YOLO_DEFAULT_MODEL", "yolo26n-pose.pt")


def _env_float(name: str, default: float) -> tuple[float, str | None]:
    """env float 파싱 — 비숫자/빈값이면 default 로 폴백(bare float() import crash 방지).

    logger 가 이 시점엔 아직 미설정이라 경고를 직접 emit 하지 않고 메시지로 반환만 한다 —
    호출부가 logger 설정 후 한꺼번에 emit.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default, None
    try:
        return float(raw), None
    except ValueError:
        return default, f"{name}={raw!r} 가 숫자가 아니라 기본값 {default} 로 폴백"


YOLO_CONF_THRESHOLD, _conf_warn = _env_float("YOLO_CONF_THRESHOLD", 0.5)
# 빈값 = 워커가 torch.cuda.is_available() 자동감지 (GPU→cuda:0 / 없으면 CPU 폴백).
YOLO_DEVICE: str = os.getenv("YOLO_DEVICE", "")

# cadence autotune 내부 정책. 더 이상 .env 수동 knob가 아니다. INFERENCE_INTERVAL은
# telemetry 수렴 전 초기값, MIN/MAX는 폭주 방지 경계다. CAPTURE_INTERVAL은 inference
# cadence와 독립인 detection WS edge polling 주기라 10ms 고정으로 aliasing을 피한다.
MIN_INFERENCE_INTERVAL: float = 0.01
MAX_INFERENCE_INTERVAL: float = 1.0
INFERENCE_INTERVAL: float = 0.033
CAPTURE_INTERVAL: float = 0.01

# 실제 budget은 worker infer_ms EWMA에서 매초 계산한다. MAX_INFER_PER_SEC=52는 더 이상
# 사용자 knob가 아니라 worker 표본이 생기기 전 몇 초만 쓰는 보수적 bootstrap 상한이다.
MAX_INFER_PER_SEC: float = 52.0
# source별 pending + model micro-batch가 burst/drop을 흡수하므로 단일 size-1 queue 시절의
# 15% 여유 대신 5%만 남겨 GPU를 더 채운다.
AUTOTUNE_HEADROOM: float = 0.95
AUTOTUNE_EWMA_ALPHA: float = 0.2
AUTOTUNE_MIN_SAMPLES: int = 5
AUTOTUNE_TARGET_FPS_MAX: float = MAX_INFER_PER_SEC

# GPU worker pool 내부 정책. 사용자 튜닝 knob가 아니라 런타임 측정으로 선택되는 안전 경계다.
# 같은 모델은 한 process에서 여러 카메라를 micro-batch하고, 서로 다른 모델만 별도 process로
# 병렬화한다. imgsz는 과부하 때만 한 단계씩 낮추고 여유가 지속되면 품질을 복구한다.
INFERENCE_BATCH_MAX: int = 8
INFERENCE_BATCH_TIMEOUT_SEC: float = 0.008
INFERENCE_AGGREGATE_TIMEOUT_SEC: float = 2.0
INFERENCE_IMGSZ_STAGES: tuple[int, ...] = (320, 416, 512, 640)
ADAPTIVE_DOWNSHIFT_TICKS: int = 2
ADAPTIVE_UPSHIFT_TICKS: int = 5
ADAPTIVE_OVERLOAD_RATIO: float = 0.85
ADAPTIVE_UNDERLOAD_RATIO: float = 0.65

# custom .pt 모델 디렉토리 — 미설정 시 backend/models(네이티브 dev) = /app/models(컨테이너)
# 로 자동 결정(절대경로, cwd 무관). models_dir 가 import 시 CUSTOM_MODELS_DIR 를 읽으므로
# 반드시 그 전에(여기, config import 시점) setdefault. compose 가 명시하면 그 값이 우선.
os.environ.setdefault(
    "CUSTOM_MODELS_DIR", str(Path(__file__).resolve().parent.parent / "models")
)

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()


def setup_logging() -> logging.Logger:
    """애플리케이션 로거 설정"""
    logger = logging.getLogger("rtsp-keypoint")
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)

    return logger


logger = setup_logging()

if _raw_max_ipcams != MAX_IPCAMS:
    logger.warning("MAX_IPCAMS=%d → %d 로 보정됨 (허용 범위: 1~64)", _raw_max_ipcams, MAX_IPCAMS)
for _w in (_conf_warn,):
    if _w:
        logger.warning("%s", _w)
