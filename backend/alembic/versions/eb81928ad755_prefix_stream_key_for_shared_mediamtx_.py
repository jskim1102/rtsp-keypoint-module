"""prefix stream_key for shared mediamtx CEO115

Revision ID: eb81928ad755
Revises: 14d6b54de992
Create Date: 2026-06-29 18:42:12.587245

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'eb81928ad755'
down_revision: Union[str, Sequence[str], None] = '14d6b54de992'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """공유 mediamtx 이전(CEO #115) — 기존 stream_key 에 프로젝트 prefix 부착(path 네임스페이싱).

    신규 키는 app/models.py _generate_stream_key 가 MEDIAMTX_PATH_PREFIX 로 붙이고, 이미 저장된
    행은 이 마이그레이션이 부착한다. WHERE NOT LIKE 로 idempotent(재실행·이미 붙은 행 무해).
    """
    op.execute(
        "UPDATE ip_cams SET stream_key = 'rtsp_keypoint__' || stream_key "
        "WHERE stream_key NOT LIKE 'rtsp_keypoint__%'"
    )


def downgrade() -> None:
    """prefix 제거(공유→단독 mediamtx 롤백). 선두 prefix 만 제거(substr)."""
    op.execute(
        "UPDATE ip_cams SET stream_key = substr(stream_key, length('rtsp_keypoint__') + 1) "
        "WHERE stream_key LIKE 'rtsp_keypoint__%'"
    )
