"""AEGIS Database - SQLAlchemy async setup."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger("aegis.db")

engine = create_async_engine(
    settings.get_database_url(),
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


async def init_db():
    """Create all database tables."""
    async with engine.begin() as conn:
        # Import all models to register them with Base
        from app.models.audit_record import AuditRecord
        from app.models.drift_record import DriftRecord
        from app.models.model_record import ModelRecord

        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")


async def get_session() -> AsyncSession:
    """Get async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
