"""Central settings — loaded from the single root .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py → parents[0]=app, [1]=backend, [2]=repo root
ROOT_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Every field maps to an env var in the root .env (case-insensitive)."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── gateway ────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    data_dir: str = "data"
    max_upload_mb: int = 15

    # ── inference layer ────────────────────────────────────────────────────
    llm_provider: str = "auto"  # auto | groq | ollama | reflex
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # ── embeddings ─────────────────────────────────────────────────────────
    embed_provider: str = "auto"  # auto | ollama | hash
    embed_dim: int = 512
    ollama_embed_model: str = "nomic-embed-text"

    # ── vector store ───────────────────────────────────────────────────────
    vector_provider: str = "auto"  # auto | memory | qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "synapsecraft"

    # ── graph store / neo4j mirror ─────────────────────────────────────────
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_mirror: str = "auto"  # auto | on | off

    # ── neurodynamics ──────────────────────────────────────────────────────
    gating_threshold: float = 0.55
    recall_k: int = 24
    top_k: int = 6
    max_reflection_loops: int = 2
    lateral_inhibition: float = 0.35
    propagation_decay: float = 0.6
    critic_mode: str = "heuristic"  # heuristic | llm
    relate_threshold: float = 0.32
    chunk_max_chars: int = 1000
    chunk_overlap_sentences: int = 1
    ws_max_events: int = 240

    # ── derived helpers ────────────────────────────────────────────────────

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else (BACKEND_DIR / self.data_dir)

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
