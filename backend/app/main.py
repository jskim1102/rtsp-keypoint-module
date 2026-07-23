import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, MAX_IPCAMS, logger
from app.database import SessionLocal, init_db
from app.inference_api import inference_router
from app.ipcam import router as ipcam_router
from app.mediamtx import sync_streams
from app.streaming.manager import manager as stream_manager


def _bg_sync() -> None:
    """DB 카메라 재등록을 서버 시작과 분리해 백그라운드에서 수행한다."""
    db = SessionLocal()
    try:
        sync_streams(db)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """서버 시작/종료 시 리소스 관리."""
    # alembic 이 스키마 정본이지만, 테스트/로컬 첫 실행 편의를 위해 init_db() 유지.
    init_db()

    # YOLO 추론 워커 + dispatch 스레드 기동 (detection). preset-only — custom 모델 디렉토리 불필요.
    stream_manager.startup()

    # DB 카메라 재등록은 probe 지연과 무관하게 서버가 즉시 요청을 받도록 분리한다.
    threading.Thread(target=_bg_sync, daemon=True).start()

    logger.info("RTSP Detection API 서버 시작")
    yield
    logger.info("서버 종료 중 — 캡처/추론 리소스 정리")
    stream_manager.shutdown()
    logger.info("서버 종료 완료")


app = FastAPI(title="RTSP Detection API", lifespan=lifespan)

# CORS
cors_origins = ["*"] if CORS_ORIGINS == "*" else [o.strip() for o in CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ipcam_router)
app.include_router(inference_router)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict[str, int]:
    """프론트가 사용할 런타임 설정 — 등록 cap(MAX_IPCAMS). 프론트 하드코딩 제거(P2-1)."""
    return {"max_ipcams": MAX_IPCAMS}
