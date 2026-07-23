"""fix wrong rtsp_streaming__ prefix → rtsp_keypoint__ (codex #3)

eb81928 가 복사잔재로 'rtsp_streaming__' 를 하드코딩해, 이 모듈 DB 의 기존 행이 남의
namespace 로 prefix 됐다(WHEP 경로 어긋남 + 크로스프로젝트 충돌). eb81928 의 문자열은
교정했지만 이미 적용된 DB 는 재실행되지 않으므로, 잘못 라벨된 행을 여기서 복구한다.

Revision ID: f7c2a9d3e1b4
Revises: eb81928ad755
Create Date: 2026-06-30 15:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f7c2a9d3e1b4'
down_revision: Union[str, Sequence[str], None] = 'eb81928ad755'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """잘못 붙은 'rtsp_streaming__' prefix 를 'rtsp_keypoint__' 로 교체.

    선두 prefix 만 한 번 교체(idempotent — rtsp_streaming__ 로 시작하는 행만 매칭).
    """
    op.execute(
        "UPDATE ip_cams "
        "SET stream_key = 'rtsp_keypoint__' || substr(stream_key, length('rtsp_streaming__') + 1) "
        "WHERE stream_key LIKE 'rtsp_streaming__%'"
    )


def downgrade() -> None:
    """롤백 — 교정했던 행을 다시 rtsp_streaming__ 로 되돌린다."""
    op.execute(
        "UPDATE ip_cams "
        "SET stream_key = 'rtsp_streaming__' || substr(stream_key, length('rtsp_keypoint__') + 1) "
        "WHERE stream_key LIKE 'rtsp_keypoint__%'"
    )
