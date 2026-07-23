#!/bin/sh
# 컨테이너 기동 시 스키마를 migration 으로 적용 (RULES §9 — alembic 이 정본).
# DATABASE_URL(named volume 경로)을 env.py 가 읽어 그 DB 에 upgrade.
set -e

echo "[entrypoint] alembic upgrade head ..."
alembic upgrade head

echo "[entrypoint] starting: $*"
exec "$@"
