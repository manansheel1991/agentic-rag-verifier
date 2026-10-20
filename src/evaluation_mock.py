"""Deterministic plumbing fixtures, deliberately unrelated to reference answers."""
import hashlib
import json
import re
import socket
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

from src import config

MOCK_EMBEDDING = "mock/hash-v1"
MOCK_GENERATION = "openai/mock-structured-v1"


def _embedding(model, input):
    vector = [0.0] * 32
    for word in re.findall(r"\w+", input[0].lower()):
        digest = hashlib.sha256(word.encode()).digest()
        vector[digest[0] % 32] += 1 if digest[1] % 2 else -1
    vector[0] += 0.01
    return NS(data=[NS(embedding=vector)], usage=None)


def _generate(model, messages, response_format, **kwargs):
    prompt = messages[0]["content"]
    name = response_format["json_schema"]["name"]
    if name == "BatchContextVerification":
        chunks = json.JSONDecoder().raw_decode(prompt.split("Available Document Chunks:\n", 1)[1])[0]
        data = {"evaluations": [{"chunk_id": c["chunk_id"], "is_relevant": True,
                                 "reason": "MOCK: retain every chunk"} for c in chunks]}
    elif name == "StructuredAnswer":
        chunks = json.loads(prompt.rsplit("Context: ", 1)[1])
        query = prompt.split("Query: ", 1)[1].split("\nContext:", 1)[0]
        # A deliberately simplistic fixture; never reads dataset labels or gold answers.
        text = chunks[0]["text"].split("\n\n")[-1].split(". ")[0] if chunks else ""
        data = {"requested_parts": [query], "claims": [{"claim_id": "mock-1", "part": query,
                "text": text, "cited_chunk_ids": [chunks[0]["chunk_id"]]}] if text else [],
                "unanswered_parts": [] if text else [query]}
    elif name == "ClaimAudit":
        draft = json.JSONDecoder().raw_decode(prompt.split("\nDraft: ", 1)[1])[0]
        chunks = {c["chunk_id"]: c for c in json.loads(prompt.rsplit("Context: ", 1)[1])}
        data = {"evaluations": [{"claim_id": c["claim_id"], "verdict": "insufficient_evidence",
                "evidence": [{"chunk_id": identity, "excerpt": chunks[identity]["text"]}
                             for identity in c["cited_chunk_ids"] if identity in chunks],
                "conflicting_evidence": [], "explanation": "MOCK: always withhold; no semantic judgment"}
                for c in draft["claims"]],
                "part_evaluations": [{"part": part, "is_answered": False,
                                      "explanation": "MOCK fixture"} for part in draft["requested_parts"]],
                "unanswered_parts": draft["requested_parts"]}
    else:
        raise AssertionError("Unexpected mock schema")
    return NS(choices=[NS(message=NS(content=json.dumps(data)))], usage=None)


@contextmanager
def mock_providers():
    def blocked(*args, **kwargs):
        raise AssertionError("Network is disabled in mock evaluation")
    client = NS(embeddings=NS(create=_embedding), chat=NS(completions=NS(create=_generate)))
    registry = {**config.EMBEDDING_DIMENSIONS, ("mock", "hash-v1"): 32}
    with (patch.object(config, "_client", lambda _: client),
          patch.object(config, "EMBEDDING_DIMENSIONS", registry),
          patch.object(socket.socket, "connect", blocked), patch.object(socket, "create_connection", blocked)):
        yield
