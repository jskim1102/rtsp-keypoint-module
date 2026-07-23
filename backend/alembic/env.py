from logging.config import fileConfig
from pathlib import Path

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- target_metadata: app.database 가 존재할 때만 autogenerate 지원 -----------
# phase1 에는 app.database 가 아직 없으므로 try-import 후 None 으로 폴백한다.
# 첫 migration 은 명시적 op.create_table 이라 app 모듈 import 가 필요 없고,
# phase3 에서 database.py 가 들어오면 자동으로 Base.metadata 를 픽업한다.
try:
    from app.database import Base, DATABASE_URL  # noqa: F401
    target_metadata = Base.metadata
    _database_url = DATABASE_URL
except ImportError:
    target_metadata = None
    # app.database 부재 시 self-contained 경로 — database.py 와 동일하게
    # backend/deepeye.db 를 가리킨다 (env.py 는 backend/alembic/ 에 위치).
    _db_path = Path(__file__).resolve().parent.parent / "deepeye.db"
    _database_url = f"sqlite:///{_db_path}"

# alembic.ini 의 placeholder URL 대신 위에서 결정한 URL 을 사용.
config.set_main_option("sqlalchemy.url", _database_url)

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
