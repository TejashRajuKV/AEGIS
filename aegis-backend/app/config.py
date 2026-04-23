"""
AEGIS Configuration Module
==========================
Central configuration for the AEGIS AI Fairness Platform.
All settings are loaded from environment variables with sensible defaults.
Designed for sequential pipeline execution on a 16GB RAM gaming laptop.

Upgraded: now uses Pydantic BaseSettings for automatic .env loading,
type validation at startup, and lru_cache singleton.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings — loaded from .env file and environment variables."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── Application Identity ─────────────────────────────────────
    APP_NAME: str = Field("AEGIS")
    APP_VERSION: str = Field("6.0.0")

    # ── Project Paths ──────────────────────────────────────────────
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "aegis-shared" / "datasets"
    )
    CHECKPOINT_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "checkpoints"
    )
    UPLOAD_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "uploads"
    )
    LOG_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "logs"
    )
    CACHE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "cache"
    )

    @field_validator("BASE_DIR", "DATA_DIR", "CHECKPOINT_DIR", "UPLOAD_DIR", "LOG_DIR", "CACHE_DIR", mode="before")
    @classmethod
    def _coerce_path(cls, v: object) -> Path:
        return Path(str(v))

    # ── API Settings ───────────────────────────────────────────────
    API_HOST: str = Field("0.0.0.0", alias="AEGIS_HOST")
    API_PORT: int = Field(8000, alias="AEGIS_PORT")
    API_PREFIX: str = Field("/api", alias="AEGIS_API_PREFIX")
    DEBUG: bool = Field(True, alias="AEGIS_DEBUG")

    # ── CORS ───────────────────────────────────────────────────────
    # Stored as a comma-separated string so .env parsing never fails on bare "*".
    CORS_ORIGINS_RAW: str = Field(
        default="http://localhost:3000,http://localhost:8000,http://127.0.0.1:3000,http://127.0.0.1:8000",
        alias="CORS_ORIGINS",
    )

    @property
    def CORS_ORIGINS(self) -> List[str]:  # noqa: N802  kept UPPER for back-compat
        """Return CORS origins as a list, handling '*' and comma-separated values."""
        raw = self.CORS_ORIGINS_RAW.strip()
        if raw == "*":
            return ["*"]
        return [x.strip() for x in raw.split(",") if x.strip()]

    # ── Database ───────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./aegis.db",
        alias="AEGIS_DATABASE_URL",
    )

    def get_database_url(self) -> str:
        """Return the database URL (for backward compatibility)."""
        return self.DATABASE_URL

    # ── LLM Integration (Claude API) ──────────────────────────────
    CLAUDE_API_KEY: Optional[str] = Field(None)
    CLAUDE_MODEL: str = Field("claude-sonnet-4-20250514")
    CLAUDE_MAX_TOKENS: int = Field(4096)
    CLAUDE_TEMPERATURE: float = Field(0.2)

    # ── HuggingFace (for text bias embeddings) ─────────────────────
    HF_MODEL_NAME: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    HF_CACHE_DIR: Optional[str] = Field(None)

    # ── DAG-GNN (Causal Discovery) ─────────────────────────────────
    DAG_GNN_HIDDEN_DIM: int = Field(64)
    DAG_GNN_NUM_LAYERS: int = Field(2)
    DAG_GNN_LR: float = Field(0.003)
    DAG_GNN_EPOCHS: int = Field(300)
    DAG_GNN_TOLERANCE: float = Field(1e-3)
    DAG_GNN_DAG_THRESHOLD: float = Field(0.3)

    # ── PPO (RL Autopilot) ────────────────────────────────────────
    PPO_HIDDEN_DIM: int = Field(128)
    PPO_ACTOR_LR: float = Field(3e-4)
    PPO_CRITIC_LR: float = Field(1e-3)
    PPO_GAMMA: float = Field(0.99)
    PPO_LAMBDA_GAE: float = Field(0.95)
    PPO_CLIP_EPSILON: float = Field(0.2)
    PPO_EPOCHS: int = Field(200)
    PPO_BATCH_SIZE: int = Field(64)
    PPO_ENTROPY_COEF: float = Field(0.01)
    PPO_VALUE_COEF: float = Field(0.5)
    PPO_MAX_STEPS: int = Field(1000)

    # ── Goodhart Guard ────────────────────────────────────────────
    GOODHART_ACCURACY_FLOOR: float = Field(0.70)
    GOODHART_ALERT_THRESHOLD: float = Field(0.85)

    # ── CVAE (Counterfactual Generation) ──────────────────────────
    CVAE_INPUT_DIM: int = Field(20)
    CVAE_LATENT_DIM: int = Field(16)
    CVAE_HIDDEN_DIM: int = Field(64)
    CVAE_LR: float = Field(1e-3)
    CVAE_EPOCHS: int = Field(150)
    CVAE_BETA: float = Field(0.1)
    CVAE_KL_ANNEALING: bool = Field(True)

    # ── Drift Detection ───────────────────────────────────────────
    DRIFT_CUSUM_THRESHOLD: float = Field(5.0)
    DRIFT_CUSUM_DRIFT_PENALTY: float = Field(0.5)
    DRIFT_WASSERSTEIN_THRESHOLD: float = Field(0.1)
    DRIFT_WINDOW_SIZE: int = Field(500)
    DRIFT_MIN_REFERENCE_SAMPLES: int = Field(200)
    DRIFT_CHECK_INTERVAL_SECS: int = Field(60)

    # ── Text Bias ─────────────────────────────────────────────────
    TEXT_BIAS_SIMILARITY_THRESHOLD: float = Field(0.3)
    TEXT_BIAS_NUM_SAMPLES: int = Field(50)

    # ── Sequential Pipeline Control ───────────────────────────────
    PIPELINE_MAX_MEMORY_MB: int = Field(12288)
    PIPELINE_BATCH_SIZE: int = Field(256)
    PIPELINE_SEQUENTIAL: bool = Field(True)  # Always sequential for 16GB RAM

    # ── Task Queue ────────────────────────────────────────────────
    TASK_QUEUE_MAX_WORKERS: int = Field(1)   # Single worker for sequential execution
    TASK_TIMEOUT_SECS: int = Field(600)

    # ── Cache ─────────────────────────────────────────────────────
    CACHE_MAX_SIZE: int = Field(100)
    CACHE_TTL_SECS: int = Field(300)

    # ── Datasets ──────────────────────────────────────────────────
    DATASET_ADULT_CENSUS: str = Field("adult_census.csv")
    DATASET_COMPAS: str = Field("compas_recidivism.csv")
    DATASET_GERMAN_CREDIT: str = Field("german_credit.csv")
    DATASET_SACHS_PROTEINS: str = Field("sachs_proteins.csv")
    DATASET_ELECTRICITY: str = Field("electricity_drift.csv")
    DATASET_SEA: str = Field("sea_drift.csv")
    DATASET_STEREOSET: str = Field("stereoset.csv")
    DATASET_CROWS_PAIRS: str = Field("crows_pairs.csv")
    DATASET_IHDP: str = Field("ihdp_npci_1.csv")

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [
            self.CHECKPOINT_DIR, self.UPLOAD_DIR,
            self.LOG_DIR, self.CACHE_DIR,
            self.CHECKPOINT_DIR / "conditional_vae",
            self.CHECKPOINT_DIR / "dag_gnn",
            self.CHECKPOINT_DIR / "ppo_agent",
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    # ── snake_case aliases (forward-compat with aegis-main style) ──
    @property
    def app_name(self) -> str:
        """Alias: settings.app_name == settings.APP_NAME."""
        return self.APP_NAME

    @property
    def app_version(self) -> str:
        return self.APP_VERSION

    @property
    def app_env(self) -> str:
        return os.getenv("APP_ENV", "development")

    @property
    def host(self) -> str:
        """Alias: settings.host == settings.API_HOST."""
        return self.API_HOST

    @property
    def port(self) -> int:
        """Alias: settings.port == settings.API_PORT."""
        return self.API_PORT

    @property
    def cors_origins(self) -> List[str]:
        return self.CORS_ORIGINS

    @property
    def datasets_dir(self) -> Path:
        """Alias: settings.datasets_dir == settings.DATA_DIR."""
        return self.DATA_DIR

    @property
    def model_registry_path(self) -> Path:
        """Alias: settings.model_registry_path == settings.CHECKPOINT_DIR."""
        return self.CHECKPOINT_DIR

    @property
    def adult_csv_path(self) -> Path:
        return self.DATA_DIR / self.DATASET_ADULT_CENSUS

    @property
    def compas_csv_path(self) -> Path:
        return self.DATA_DIR / self.DATASET_COMPAS

    @property
    def supported_datasets(self) -> List[str]:
        return ["adult_census", "compas", "german_credit"]

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


# ── Singleton (lru_cache — loaded once, reused everywhere) ─────────

@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return the cached Settings singleton."""
    return Settings()


# Module-level alias — all modules do `from app.config import settings`
settings: Settings = get_settings()
