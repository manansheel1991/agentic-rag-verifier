import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from src import config
from src.config import ProviderError, _client as real_client
from src.schemas import HallucinationCheck


class StatusError(Exception):
    def __init__(self, status):
        self.status_code = status
        super().__init__("sensitive SDK details must never be exposed")


@pytest.mark.parametrize("status,category,attempts", [(401, "authentication", 1), (403, "authentication", 1),
    (400, "invalid_request", 1), (404, "invalid_request", 1), (429, "rate_limit", 3),
    (500, "transient", 3), (503, "transient", 3)])
def test_failure_classification_bounded_retries_no_embedding_fallback(monkeypatch, status, category, attempts):
    client = NS(models=NS(embed_content=Mock(side_effect=StatusError(status))))
    route = Mock(return_value=client)
    monkeypatch.setattr(config, "_client", route)
    with pytest.raises(ProviderError) as caught:
        config.get_vector_embedding("text")
    assert caught.value.category == category
    assert "sensitive" not in str(caught.value)
    assert client.models.embed_content.call_count == attempts
    assert {call.args[0] for call in route.call_args_list} == {"gemini"}


@pytest.mark.parametrize("exc", [TimeoutError(), ConnectionError(), httpx.ConnectError("offline"), httpx.ReadTimeout("offline")])
def test_transient_transport_retries(monkeypatch, exc):
    request = Mock(side_effect=[exc, NS(embeddings=[NS(values=[1., 2., 3.])])])
    monkeypatch.setattr(config, "_client", lambda _: NS(models=NS(embed_content=request)))
    assert config.get_vector_embedding("text") == [1, 2, 3]
    assert request.call_count == 2


def test_exact_embedding_provider_model_and_usage(monkeypatch):
    request = Mock(return_value=NS(data=[NS(embedding=[1., 2., 3.])], usage=NS(total_tokens=7)))
    route = Mock(return_value=NS(embeddings=NS(create=request)))
    monkeypatch.setattr(config, "_client", route)
    trace = []
    token = config.CALL_TRACE.set(trace)
    try:
        assert config.get_vector_embedding("q", "openai/" + config.OPENAI_EMBED_MODEL) == [1, 2, 3]
    finally:
        config.CALL_TRACE.reset(token)
    route.assert_called_once_with("openai")
    request.assert_called_once_with(model=config.OPENAI_EMBED_MODEL, input=["q"])
    assert trace == [{"kind": "embedding", "model": "openai/" + config.OPENAI_EMBED_MODEL, "tokens": 7}]


def test_unavailable_embedding_provider_is_clear_and_no_fallback(monkeypatch):
    route = Mock(side_effect=ProviderError("unavailable", "gemini"))
    monkeypatch.setattr(config, "_client", route)
    with pytest.raises(ProviderError, match="gemini: unavailable"):
        config.get_vector_embedding("q")
    route.assert_called_once_with("gemini")


def test_generation_fallback_records_actual_model_usage(monkeypatch):
    primary = NS(models=NS(generate_content=Mock(side_effect=StatusError(429))))
    alternate_request = Mock(return_value=NS(choices=[NS(message=NS(content="answer"))], usage=NS(total_tokens=10)))
    alternate = NS(chat=NS(completions=NS(create=alternate_request)))
    monkeypatch.setattr(config, "_client", lambda p: primary if p == "gemini" else alternate)
    trace = []
    token = config.CALL_TRACE.set(trace)
    try:
        assert config.generate_text("prompt") == "answer"
    finally:
        config.CALL_TRACE.reset(token)
    assert primary.models.generate_content.call_count == 3
    assert alternate_request.call_args.kwargs["model"] == config.OPENAI_GEN_MODEL
    assert trace == [{"kind": "generation", "model": "openai/" + config.OPENAI_GEN_MODEL, "tokens": 10}]


def test_structured_openai_enforces_schema(monkeypatch):
    content = json.dumps({"is_faithful": True, "verdict": "ok"})
    request = Mock(return_value=NS(choices=[NS(message=NS(content=content))], usage=None))
    monkeypatch.setattr(config, "_client", lambda _: NS(chat=NS(completions=NS(create=request))))
    result = config.generate_structured_json("prompt", HallucinationCheck, model_name="openai/custom")
    assert HallucinationCheck.model_validate_json(result).is_faithful
    options = request.call_args.kwargs
    assert options["model"] == "custom"
    schema = options["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


@pytest.mark.parametrize("content", [None, "", "not-json", '{"is_faithful": "yes", "verdict":"x"}', '{"is_faithful":true}'])
def test_malformed_structured_output_is_categorized(monkeypatch, content):
    request = Mock(return_value=NS(choices=[NS(message=NS(content=content))], usage=None))
    monkeypatch.setattr(config, "_client", lambda _: NS(chat=NS(completions=NS(create=request))))
    with pytest.raises(ProviderError) as caught:
        config.generate_structured_json("p", HallucinationCheck, model_name="openai/test")
    assert caught.value.category == "exhausted_providers"
    assert "openai:malformed_output" in caught.value.causes
    request.assert_called_once()


def test_exhausted_generation_preserves_distinct_causes(monkeypatch):
    monkeypatch.setattr(config, "_client", Mock(side_effect=[ProviderError("authentication", "gemini"), ProviderError("unavailable", "openai")]))
    with pytest.raises(ProviderError) as caught:
        config.generate_text("q")
    assert caught.value.category == "exhausted_providers"
    assert caught.value.causes == ["gemini:authentication", "openai:unavailable"]


def test_lazy_clients_have_timeouts_disable_sdk_retries(monkeypatch):
    gemini_constructor = Mock()
    openai_constructor = Mock()
    monkeypatch.setattr(config, "gemini_client_obj", None)
    monkeypatch.setattr(config, "openai_client_obj", None)
    monkeypatch.setattr(config.genai_client, "Client", gemini_constructor)
    monkeypatch.setattr(config, "OpenAI", openai_constructor)
    real_client("gemini")
    real_client("openai")
    http_options = gemini_constructor.call_args.kwargs["http_options"]
    assert http_options.timeout == config.API_TIMEOUT_SECONDS * 1000
    assert http_options.retry_options.attempts == 1
    assert openai_constructor.call_args.kwargs["timeout"] == config.API_TIMEOUT_SECONDS
    assert openai_constructor.call_args.kwargs["max_retries"] == 0


def test_missing_keys_do_not_construct_clients(monkeypatch):
    monkeypatch.setattr(config, "gemini_client_obj", None)
    monkeypatch.setattr(config, "openai_client_obj", None)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY")
    for provider in ("gemini", "openai"):
        with pytest.raises(ProviderError, match="unavailable"):
            real_client(provider)
