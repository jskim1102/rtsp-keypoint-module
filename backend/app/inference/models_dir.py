"""YOLO preset 가중치 관리 (preset-only).

custom `.pt` 업로드는 제거됐다 — `.pt`=pickle=코드실행 신뢰경계라, 인증 없는 업로드 +
메인 프로세스 로드는 RCE 였다(codex #1). 이제 ultralytics 가 자동 다운로드하는 preset 5종만
허용(allowlist). 워커·class 메타 조회 모두 이 allowlist 를 통과한 이름만 로드한다.
"""

from __future__ import annotations

# Preset 모델 — UI 토글 + worker 자동 다운로드 기본값. 신뢰경계: ultralytics 공식 가중치만.
PRESET_MODELS: tuple[str, ...] = (
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
    "yolo26l-pose.pt",
    "yolo26x-pose.pt",
)


def is_preset(name: str) -> bool:
    """name 이 허용된 preset 인지 (단일 allowlist 게이트)."""
    return name in PRESET_MODELS


def list_all_models() -> list[dict]:
    """preset 목록 (UI 드롭다운용). custom 없음."""
    return [{"name": n, "type": "preset", "size_mb": None} for n in PRESET_MODELS]


def resolve_model_path(name: str) -> str:
    """모델 이름 → ultralytics 로드용 이름. preset 만 허용, 그 외는 거부.

    경로가 아니라 preset 이름을 그대로 돌려준다(ultralytics 가 캐시/다운로드). 임의 파일
    경로 로드를 원천 차단해 pickle RCE 경로를 없앤다.
    """
    if not is_preset(name):
        raise ValueError(f"허용되지 않은 모델: {name!r} (preset 만 가능)")
    return name
