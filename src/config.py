"""Explicit provider routing. Clients are lazy; importing the app makes no API calls."""
import os
import math
import time
from contextvars import ContextVar

import httpx
from dotenv import load_dotenv
from google.genai import client as genai_client, types
from openai import OpenAI, APIConnectionError, APITimeoutError
from pydantic import BaseModel, ValidationError

load_dotenv()
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
GEMINI_EMBED_MODEL = "gemini-embedding-001"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"
OPENAI_GEN_MODEL = "gpt-4o-mini"
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "30"))
if not math.isfinite(API_TIMEOUT_SECONDS) or API_TIMEOUT_SECONDS < 0.001:
    raise ValueError("API_TIMEOUT_SECONDS must be finite and at least 0.001 seconds")
MAX_ATTEMPTS = 3
EMBEDDING_DIMENSIONS = {
    ("gemini", GEMINI_EMBED_MODEL): 3072,
    ("openai", OPENAI_EMBED_MODEL): 1536,
    ("openai", "text-embedding-3-large"): 3072,
}
CHROMA_CLIENT = None  # Optional injected client; no persistent files created on import.
gemini_client_obj = None
openai_client_obj = None
CALL_TRACE = ContextVar("provider_call_trace", default=None)
API_CALL_TRACE = ContextVar("api_attempt_trace", default=None)
GENERATION_FALLBACK_ENABLED = ContextVar("generation_fallback_enabled", default=True)


class EmbeddingSpec(BaseModel):
    provider: str
    model: str
    dimensions: int


def embedding_spec(model_name=GEMINI_EMBED_MODEL):
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
    else:
        model = model_name
        provider = "openai" if model.startswith("text-embedding-") else "gemini"
    dimensions = EMBEDDING_DIMENSIONS.get((provider, model))
    if dimensions is None:
        raise ValueError("Unsupported embedding provider/model; register its dimensions explicitly before rebuilding an index")
    return EmbeddingSpec(provider=provider, model=model, dimensions=dimensions)


class ProviderError(RuntimeError):
    def __init__(self, category, provider, causes=None):
        self.category = category
        self.provider = provider
        self.causes = causes or []
        # Never include SDK exception strings: they can contain request bodies or credentials.
        super().__init__(f"{provider}: {category}" + (f" ({', '.join(self.causes)})" if self.causes else ""))


def _classify(exc):
    if isinstance(exc, ProviderError):
        return exc.category
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status in (401, 403):
        return "authentication"
    if status == 429:
        return "rate_limit"
    if status in (408, 409) or isinstance(status, int) and status >= 500:
        return "transient"
    if isinstance(exc, (TimeoutError, ConnectionError, APIConnectionError, APITimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return "transient"
    if status is not None:
        return "invalid_request"
    if isinstance(exc, (ValidationError, ValueError, AttributeError, IndexError, TypeError)):
        return "malformed_output"
    return "provider_failure"


def _retry(provider, call):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            category = _classify(exc)
            if category in ("rate_limit", "transient") and attempt + 1 < MAX_ATTEMPTS:
                time.sleep(0.2 * 2 ** attempt)
                continue
            raise ProviderError(category, provider) from None


def _client(provider):
    global gemini_client_obj, openai_client_obj
    if provider == "gemini":
        if gemini_client_obj is None:
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                raise ProviderError("unavailable", provider)
            gemini_client_obj = genai_client.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    timeout=int(API_TIMEOUT_SECONDS * 1000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
        return gemini_client_obj
    if provider == "openai":
        if openai_client_obj is None:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise ProviderError("unavailable", provider)
            openai_client_obj = OpenAI(api_key=key, timeout=API_TIMEOUT_SECONDS, max_retries=0)
        return openai_client_obj
    raise ProviderError("invalid_request", provider)


def _record(kind, provider, model, response):
    trace = CALL_TRACE.get()
    if trace is None:
        return
    usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    tokens = getattr(usage, "total_tokens", None)
    if tokens is None:
        tokens = getattr(usage, "total_token_count", None)
    trace.append({"kind": kind, "model": f"{provider}/{model}", "tokens": tokens})


def _request(kind, provider, model, call):
    """Optional evaluation telemetry: count actual SDK attempts, including failures."""
    trace = API_CALL_TRACE.get()
    if trace is None:
        return call()
    event = {"kind": kind, "model": f"{provider}/{model}", "status": "error", "tokens": None}
    trace.append(event)
    started = time.perf_counter()
    try:
        response = call()
        usage = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
        event["tokens"] = getattr(usage, "total_tokens", None)
        if event["tokens"] is None:
            event["tokens"] = getattr(usage, "total_token_count", None)
        event["status"] = "response"
        return response
    except Exception as exc:
        event["error_category"] = _classify(exc)
        raise
    finally:
        event["latency_ms"] = (time.perf_counter() - started) * 1000


def get_vector_embedding(text, model_name=GEMINI_EMBED_MODEL):
    spec = embedding_spec(model_name)
    def call():
        client = _client(spec.provider)
        if spec.provider == "gemini":
            response = _request("embedding", spec.provider, spec.model,
                                lambda: client.models.embed_content(model=spec.model, contents=text))
            vector = response.embeddings[0].values
        else:
            response = _request("embedding", spec.provider, spec.model,
                                lambda: client.embeddings.create(model=spec.model, input=[text]))
            vector = response.data[0].embedding
        _record("embedding", spec.provider, spec.model, response)
        return vector
    # Embeddings never fail over, even to an equal-dimensional model.
    return _retry(spec.provider, call)


def _generation(prompt, temperature, model_name, schema=None):
    if "/" in model_name:
        provider, model = model_name.split("/", 1)
    else:
        model = model_name
        provider = "gemini" if model.startswith("gemini-") else "openai"
    if provider not in ("gemini", "openai"):
        raise ProviderError("invalid_request", provider)
    candidates = [(provider, model)]
    if provider == "gemini" and GENERATION_FALLBACK_ENABLED.get():
        candidates.append(("openai", OPENAI_GEN_MODEL))
    failures = []
    for current_provider, current_model in candidates:
        def call():
            client = _client(current_provider)
            if current_provider == "gemini":
                options = {"temperature": temperature}
                if schema:
                    options.update(response_mime_type="application/json", response_schema=schema)
                response = _request("generation", current_provider, current_model,
                    lambda: client.models.generate_content(
                        model=current_model, contents=prompt, config=types.GenerateContentConfig(**options)))
                content = response.text
            else:
                options = {}
                if schema:
                    options["response_format"] = {"type": "json_schema", "json_schema": {
                        "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}}
                response = _request("generation", current_provider, current_model,
                    lambda: client.chat.completions.create(
                        model=current_model, messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, **options))
                content = response.choices[0].message.content
            _record("generation", current_provider, current_model, response)
            if not isinstance(content, str) or not content.strip():
                raise ProviderError("malformed_output", current_provider)
            if schema:
                schema.model_validate_json(content)
            return content
        try:
            return _retry(current_provider, call)
        except ProviderError as exc:
            failures.append(f"{current_provider}:{exc.category}")
    raise ProviderError("exhausted_providers", "generation", failures)


def generate_structured_json(prompt, schema_class, temperature=0.0, model_name=GEMINI_GEN_MODEL):
    return _generation(prompt, temperature, model_name, schema_class)


def generate_text(prompt, temperature=0.2, model_name=GEMINI_GEN_MODEL):
    return _generation(prompt, temperature, model_name)
