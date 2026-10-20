"""Offline demo tests exercise real Chroma, pipeline gates and Streamlit widgets."""
import json
import sys
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest

from src import config, demo

APP = Path(__file__).resolve().parents[1] / "app.py"
REAL_SLEEP = time.sleep


@pytest.fixture(autouse=True)
def ui_scheduler_sleep(offline, monkeypatch):
    # The provider-test fixture removes retry sleeps globally. Streamlit's widget
    # runner needs a real yielding sleep while waiting for its background thread.
    monkeypatch.setattr(config.time, "sleep", REAL_SLEEP)
    monkeypatch.setattr(sys, "argv", list(sys.argv))


@pytest.mark.parametrize("pipeline", ["basic", "verified"])
def test_offline_supported_and_abstention_presets(pipeline):
    files = list(demo.sample_documents())
    supported = demo.run_demo(demo.SUPPORTED_QUESTION, files, pipeline)
    assert supported["outcome"] == "answered" and supported["error"] is None
    assert supported["answer"].startswith(demo.PRESET_FACT)
    assert "harbor-operations.md" in supported["answer"]
    assert supported["mock"]
    assert supported["query_cost"]["available_total_tokens"] is None
    if pipeline == "verified":
        claim = supported["arm"]["pipeline"]["claims"][0]
        assert claim["verdict"] == "supported" and claim["checks"]["passed"]
        assert claim["evidence"][0]["excerpt"] == demo.PRESET_FACT
    abstained = demo.run_demo(demo.ABSTENTION_QUESTION, files, pipeline)
    assert abstained["outcome"] == "abstained" and abstained["answer"] is None
    assert abstained["error"] is None
    assert abstained["arm"]["pipeline"]["unanswered_parts"]


def test_persistent_index_reused_and_retrieval_stays_selected():
    files = list(demo.sample_documents())
    first = demo.run_demo(demo.SUPPORTED_QUESTION, files)
    again = demo.run_demo(demo.SUPPORTED_QUESTION, files, "basic")
    assert first["indexing"]["model_calls"] == len(files)
    assert again["indexing"]["model_calls"] == 0
    assert [c["chunk_id"] for c in first["chunks"]] == [c["chunk_id"] for c in again["chunks"]]
    restricted = demo.run_demo(demo.SUPPORTED_QUESTION, ["harbor-finance.md"])
    assert restricted["outcome"] == "abstained"
    assert restricted["chunks"] and {c["file_name"] for c in restricted["chunks"]} == {"harbor-finance.md"}
    assert (Path(config.CHROMA_PERSIST_DIR) / "demo_mock" / "documents.sqlite3").exists()


@pytest.mark.parametrize("question,documents", [("", ["harbor-finance.md"]), ("q", []), ("q", ["../.env"])])
def test_invalid_input_returns_error_without_model_calls(question, documents):
    value = demo.run_demo(question, documents)
    assert value["outcome"] == "error"
    assert value["indexing"] is None
    assert value["query_cost"]["model_calls"] == 0


def test_error_details_and_credentials_are_never_displayed_or_logged(monkeypatch, capsys):
    canary = "private-key-canary-never-show"
    monkeypatch.setattr(demo, "ingest_document", Mock(side_effect=RuntimeError(canary)))
    result = demo.run_demo(demo.SUPPORTED_QUESTION, ["harbor-operations.md"])
    assert result["outcome"] == "error"
    assert result["error"].startswith("Indexing failed")
    captured = capsys.readouterr()
    assert canary not in json.dumps(result) + captured.out + captured.err


def test_live_errors_are_distinct_from_deliberate_abstention(monkeypatch):
    # The global offline fixture blocks any real client even though this tests live routing.
    route = Mock(side_effect=config.ProviderError("authentication", "openai"))
    monkeypatch.setattr(config, "_client", route)
    result = demo.run_demo(demo.SUPPORTED_QUESTION, ["harbor-operations.md"], live=True, provider="OpenAI")
    assert result["outcome"] == "error" and result["answer"] is None
    assert not result["mock"]
    assert route.called
    assert {call.args[0] for call in route.call_args_list} == {"openai"}


def test_streamlit_initial_render_is_offline_and_does_not_index(tmp_path):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "OFFLINE MOCK DEMO" in at.info[0].value
    assert at.text_area[0].value == demo.SUPPORTED_QUESTION
    assert not (tmp_path / "index").exists()
    assert not at.get("file_uploader")
    assert not at.get("text_input")  # No credential entry widgets.


def test_streamlit_supported_example_basic_mode_and_stale_result():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[2].click().run()
    assert not at.exception
    assert at.success[0].value == "Answered"
    assert any("supported" in item.value for item in at.caption)
    assert any("harbor-operations.md" in item.label for item in at.expander)
    assert any(item.value == "Unavailable" for item in at.metric)
    ids = [c["chunk_id"] for c in at.session_state["demo_result"]["chunks"]]
    at.radio[0].set_value("Basic RAG").run()
    assert not at.success  # Old verifier output cannot be mistaken for the new mode.
    at.button[2].click().run()
    assert not at.exception and at.success[0].value == "Answered"
    assert any("not_verified" in item.value for item in at.caption)
    assert ids == [c["chunk_id"] for c in at.session_state["demo_result"]["chunks"]]


def test_streamlit_abstention_preset_and_empty_selection_error():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[1].click().run()
    assert at.text_area[0].value == demo.ABSTENTION_QUESTION
    at.button[2].click().run()
    assert not at.exception
    assert any(item.value.startswith("Abstained") for item in at.info)
    assert not at.success
    at.multiselect[0].set_value([]).run()
    at.button[2].click().run()
    assert not at.exception
    assert at.error[0].value.startswith("Error:")


def test_streamlit_partial_and_withheld_claims_remain_separate():
    import copy
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button[2].click().run()
    result = at.session_state["demo_result"]
    result["outcome"] = "partial"
    result["arm"]["pipeline"]["unanswered_parts"] = ["An unanswered follow-up"]
    rejected = copy.deepcopy(result["arm"]["pipeline"]["claims"][0])
    rejected["claim"]["claim_id"] = "withheld-demo"
    rejected["claim"]["text"] = "A claim not accepted into the answer."
    rejected["verdict"] = "insufficient_evidence"
    rejected["judgment"]["verdict"] = "insufficient_evidence"
    result["arm"]["pipeline"]["claims"].append(rejected)
    at.run()
    assert not at.exception
    assert any(item.value.startswith("Partial") for item in at.warning)
    assert "A claim not accepted" not in result["answer"]
    assert any(item.label.startswith("Withheld claims") for item in at.expander)
    assert any(item.value == "An unanswered follow-up" for item in at.text)


def test_streamlit_live_mode_is_explicit_and_idle_until_ask(monkeypatch):
    route = Mock(side_effect=config.ProviderError("authentication", "openai"))
    monkeypatch.setattr(config, "_client", route)
    at = AppTest.from_string('from app import main\nimport sys\nsys.argv = ["app.py", "--live"]\nmain()',
                             default_timeout=30).run()
    assert not at.exception
    assert at.warning[0].value.startswith("LIVE API MODE")
    route.assert_not_called()
    at.selectbox[0].set_value("OpenAI").run()
    at.button[2].click().run()
    assert not at.exception
    assert at.error[0].value.startswith("Error: Indexing failed")
    assert route.called
