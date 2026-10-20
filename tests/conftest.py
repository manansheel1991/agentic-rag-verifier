"""All tests are offline, even if the workspace .env contains real keys."""
import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def blocked(*args, **kwargs):
        raise AssertionError("Network/model clients are forbidden in offline tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(config, "_client", blocked)
    monkeypatch.setattr(config, "CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(config, "CHROMA_CLIENT", None)
    monkeypatch.setattr(config, "EMBEDDING_DIMENSIONS", {
        ("gemini", config.GEMINI_EMBED_MODEL): 3,
        ("openai", config.OPENAI_EMBED_MODEL): 3,
        ("openai", "text-embedding-3-large"): 3,
    })
    monkeypatch.setenv("GEMINI_API_KEY", "offline-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-placeholder")
    monkeypatch.setattr(config.time, "sleep", lambda _: None)


@pytest.fixture
def embed(monkeypatch):
    calls = []
    def embedding(text, model_name=config.GEMINI_EMBED_MODEL):
        calls.append((text, model_name))
        return [1.0, len(text) / 100, 0.5]
    monkeypatch.setattr(config, "get_vector_embedding", embedding)
    return calls


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "document.txt"
    path.write_text("  A document about reliability and persistence.\n" * 4)
    return path
