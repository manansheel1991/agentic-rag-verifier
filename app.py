"""Local Streamlit demo. Offline by default; live providers require -- --live."""
import sys
from pathlib import Path

# Streamlit and AppTest both execute this script outside a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from src.demo import ABSTENTION_QUESTION, SUPPORTED_QUESTION, run_demo, sample_documents


def _preset(question):
    st.session_state["question"] = question


def _passage(chunk, excerpt=None):
    page = f", page {chunk['page_number']}" if chunk.get("page_number") is not None else " (text document; no page number)"
    with st.expander(f"{chunk['file_name']}{page} · {chunk['chunk_id'][:12]}"):
        st.caption(f"Stable chunk ID: {chunk['chunk_id']}")
        if excerpt:
            st.caption("Validated excerpt")
            st.text(excerpt)
        st.caption("Source passage · treated as untrusted data")
        st.text(chunk["text"])


def _claim(item, chunks, verified):
    claim = item["claim"] if verified else item
    verdict = item["verdict"] if verified else "not_verified"
    st.caption(f"Claim {claim['claim_id']} · {verdict}")
    st.text(claim["text"])
    if verified:
        st.text(item["judgment"]["explanation"])
        if item["checks"]["errors"]:
            st.caption("Citation checks: " + "; ".join(item["checks"]["errors"]))
        for quote in item["evidence"] + item["conflicting_evidence"]:
            if quote["chunk_id"] in chunks:
                _passage(chunks[quote["chunk_id"]], quote["excerpt"])
    else:
        for identity in claim["cited_chunk_ids"]:
            if identity in chunks:
                _passage(chunks[identity])
            else:
                st.caption("Unknown cited chunk; no source passage available.")
        if not claim["cited_chunk_ids"]:
            st.caption("No citations supplied. This claim has not been verified.")


def _display(result):
    outcome = result["outcome"]
    if outcome == "error":
        st.error("Error: " + result["error"])
    elif outcome == "answered":
        st.success("Answered")
    elif outcome == "partial":
        st.warning("Partial — some requested information remains unanswered.")
    else:
        st.info("Abstained — no factual answer was accepted.")
    if result["answer"]:
        st.subheader("Answer")
        st.text(result["answer"])
    columns = st.columns(3)
    columns[0].metric("Query latency", f"{result['query_latency_ms']:.0f} ms")
    columns[1].metric("Query model calls", str(result["query_cost"]["model_calls"]))
    tokens = result["query_cost"]["available_total_tokens"]
    columns[2].metric("Available query tokens", str(tokens) if tokens is not None else "Unavailable")
    st.caption("Query metrics exclude indexing. Token totals include only reported usage; failed calls may have unknown usage.")
    if result["mock"]:
        st.caption("MOCK metrics describe local scripted fixtures, not model speed, cost, accuracy, or performance.")
    if result["indexing"]:
        cost = result["indexing"]
        with st.expander("Indexing and model details"):
            st.text(f"Indexing: {cost['latency_ms']:.0f} ms; {cost['model_calls']} embedding calls; "
                    f"tokens: {cost['available_total_tokens'] if cost['available_total_tokens'] is not None else 'unavailable'}")
            st.text(f"Embedding: {result['embedding_model']}\nGeneration: {result['generation_model']}")
            st.text(f"Query calls without token usage: {result['query_cost']['calls_without_usage']}")
    arm = result["arm"]
    if not arm or result["error"]:
        return
    pipeline = arm["pipeline"]
    if pipeline["unanswered_parts"]:
        st.subheader("Unanswered parts")
        for part in pipeline["unanswered_parts"]:
            st.text(part)
    chunks = {chunk["chunk_id"]: chunk for chunk in result["chunks"]}
    verified = result["pipeline_mode"] == "verified"
    claims = pipeline["claims"] if verified else arm["emitted_claims"]
    accepted = [item for item in claims if not verified or item["verdict"] == "supported"]
    withheld = [item for item in claims if verified and item["verdict"] != "supported"]
    if accepted:
        st.subheader("Claims and passages" if verified else "Unverified claims and cited passages")
        for item in accepted:
            _claim(item, chunks, verified)
    if withheld:
        with st.expander("Withheld claims — audit diagnostics, not the answer"):
            for item in withheld:
                _claim(item, chunks, True)
    for disagreement in pipeline["disagreements"]:
        st.warning("Source disagreement — no single source was chosen.")
        st.text(disagreement["explanation"])
        for quote in disagreement["evidence"]:
            if quote["chunk_id"] in chunks:
                _passage(chunks[quote["chunk_id"]], quote["excerpt"])
    with st.expander("Retrieved evidence (selected documents only)"):
        for chunk in result["chunks"]:
            _passage(chunk)


def main():
    st.set_page_config(page_title="RAG Verifier", layout="centered")
    st.title("RAG Verifier")
    st.caption("A personal verification pipeline over a small fictional corpus. Verification can be wrong; there is no recovery loop.")
    live = "--live" in sys.argv[1:]
    if live:
        st.warning("LIVE API MODE — submitting a question can incur provider charges. Keys are read locally, never entered here.")
    else:
        st.info("OFFLINE MOCK DEMO — scripted presets, no credentials or real model API calls. This is not model performance.")
        st.caption("The two presets are scripted. Other questions generally abstain; live models can behave differently.")
    mode = st.radio("Pipeline", ["Verifier", "Basic RAG"], horizontal=True)
    if mode == "Basic RAG":
        st.caption("Basic RAG generates cited claims without verifying their factual support.")
    selected = st.multiselect("Selected documents", list(sample_documents()), default=list(sample_documents()))
    provider = st.selectbox("Model provider", ["Gemini", "OpenAI"]) if live else "Gemini"
    st.button("Supported example", on_click=_preset, args=(SUPPORTED_QUESTION,))
    st.button("Abstention example", on_click=_preset, args=(ABSTENTION_QUESTION,))
    if "question" not in st.session_state:
        st.session_state["question"] = SUPPORTED_QUESTION
    with st.form("question_form"):
        question = st.text_area("Question", key="question")
        submitted = st.form_submit_button("Ask")
    pipeline = "verified" if mode == "Verifier" else "basic"
    request_key = (live, pipeline, tuple(selected), provider, question)
    if submitted:
        with st.spinner("Loading selected documents and running the pipeline…"):
            st.session_state["demo_result"] = run_demo(question, selected, pipeline, live=live, provider=provider)
            st.session_state["demo_request"] = request_key
    if st.session_state.get("demo_request") == request_key:
        _display(st.session_state["demo_result"])
    elif "demo_result" in st.session_state:
        st.caption("Question or settings changed. Ask again to show results for the current selection.")


if __name__ == "__main__":
    main()
