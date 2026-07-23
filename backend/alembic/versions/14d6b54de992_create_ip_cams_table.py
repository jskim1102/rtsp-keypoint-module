"""create ip_cams table

Revision ID: 14d6b54de992
Revises: 
Create Date: 2026-06-13 15:47:14.732715

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '14d6b54de992'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # IpCam 스키마 — app/models.py 의 IpCam 과 정확히 일치해야 한다.
    op.create_table(
        "ip_cams",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("rtsp_url", sa.String(length=500), nullable=False),
        sa.Column("stream_key", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("stream_key"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("ip_cams")
