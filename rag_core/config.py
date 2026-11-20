"""All settings come from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "qdrant"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", 6333)))
    collection_name: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "cnil_docs"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    reranker_model: str = field(default_factory=lambda: os.getenv(
        "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://host.docker.internal:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))
    default_pipeline: str = field(default_factory=lambda: os.getenv("DEFAULT_PIPELINE", "v0"))
    # Candidates each retriever returns before fusion and reranking
    candidates_per_retriever: int = field(default_factory=lambda: int(os.getenv("CANDIDATES_PER_RETRIEVER", 20)))
    # Sampling temperature of the answering model. Use 0 for benchmark runs, so
    # differences between pipelines come from the method, not from chance.
    # Tokens Ollama reads per request. Older Ollama versions default to 2048, which
    # would silently cut long contexts; 8192 fits the agent's ~7,500-character context.
    num_ctx: int = field(default_factory=lambda: int(os.getenv("OLLAMA_NUM_CTX", 8192)))
    gen_temperature: float = field(default_factory=lambda: float(os.getenv("GEN_TEMPERATURE", 0.2)))
    # v3 agent
    agent_max_rewrites: int = field(default_factory=lambda: int(os.getenv("AGENT_MAX_REWRITES", 2)))
    agent_context_chars: int = field(default_factory=lambda: int(os.getenv("AGENT_CONTEXT_CHARS", 7500)))
    agent_grounding_check: bool = field(default_factory=lambda: os.getenv("AGENT_GROUNDING_CHECK", "1") == "1")
