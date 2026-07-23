"""회귀 잠금 — 비숫자 YOLO_CONF_THRESHOLD 로 import 크래시하지 않는다.

이전(fix6 전)엔 worker.py 가 `float(os.getenv("YOLO_CONF_THRESHOLD"))` 를 직접 호출해
값이 숫자가 아니면 import 시점에 ValueError 로 죽었다. 지금은 config.YOLO_CONF_THRESHOLD
(guarded _env_float → 폴백)를 읽으므로 안전하다. 실제 크래시 시나리오를 서브프로세스로
재현해 잠근다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_import_manager_with_nonnumeric_conf_threshold():
    env = {
        **os.environ,
        "YOLO_CONF_THRESHOLD": "high",  # 비숫자 — fix6 전이면 float() ValueError
        "PYTHONPATH": str(BACKEND) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    r = subprocess.run(
        [sys.executable, "-c", "import app.streaming.manager; print('OK')"],
        cwd=str(BACKEND),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert "OK" in r.stdout
