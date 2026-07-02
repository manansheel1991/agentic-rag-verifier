import numpy as np
from typing import List
from google.genai import types
from config import client, EMBEDDING_MODEL, GENERATION_MODEL
from schemas import BatchContextVerification, HallucinationCheck

# Expanded Production Knowledge Base
KNOWLEDGE_BASE = [
    "Project Alpha's Q2 revenue grew by 14% quarter-over-quarter, driven by enterprise SaaS adoptions.",
    "The cafeteria will be serving artisanal tacos on Tuesdays starting next month.",
    "We detected a memory leak in the core data pipeline when handling payloads exceeding 10GB.",
    "Project Alpha's total budget for the fiscal year is capped at $4.5 million, with 40% allocated to R&D.",
    "Our standard SLA guarantees a 99.9% uptime for core database replicas under typical load parameters."
]


def get_embedding(text: str) -> np.ndarray:
    """Invokes the Gemini Embedding engine to translate prose into a semantic vector."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    # Extract the array of vector floats and convert to a numpy array
    return np.array(response.embeddings[0].values)


def live_vector_retrieval(query: str, top_k: int = 3) -> List[str]:
    """
    Computes real-time Cosine Similarity across the local knowledge base.
    Formula: (A . B) / (||A|| * ||B||)
    """
    print(f"\n🧠 [Vector DB] Vectorizing query and scanning mathematical space...")
    query_vector = get_embedding(query)

    scored_chunks = []
    for chunk in KNOWLEDGE_BASE:
        chunk_vector = get_embedding(chunk)

        # Linear Algebra implementation of Cosine Similarity
        dot_product = np.dot(query_vector, chunk_vector)
        query_norm = np.linalg.norm(query_vector)
        chunk_norm = np.linalg.norm(chunk_vector)

        cosine_similarity = dot_product / (query_norm * chunk_norm)
        scored_chunks.append((cosine_similarity, chunk))

    # Sort descending by math score and return top 'K' fragments
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    print(f"   ↳ High-probability vector matches:")
    for score, text in scored_chunks[:top_k]:
        print(f"     [Score: {score:.4f}] -> {text[:50]}...")

    return [chunk for score, chunk in scored_chunks[:top_k]]

def verify_context(query: str, chunks: List[str]) -> List[str]:
    """CRITIC 1 (Optimized): Evaluates all context fragments concurrently in 1 API call."""
    print("\n🔍 [Critic 1] Batch-verifying retrieved context...")

    # Format all chunks together with clear array indices
    formatted_chunks = "\n".join([f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks)])

    prompt = f"""
    User Query: {query}

    Available Document Chunks:
    {formatted_chunks}

    Evaluate every document chunk above. Determine if it contains facts that directly help answer the user query.
    """

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchContextVerification,
            temperature=0.0
        ),
    )

    # Parse the multi-evaluation JSON packet
    batch_result = BatchContextVerification.model_validate_json(response.text)
    verified_chunks = []

    for evaluation in batch_result.evaluations:
        idx = evaluation.chunk_index
        print(f"   ↳ Chunk {idx}: Relevant={evaluation.is_relevant} | Reason: {evaluation.reason}")
        if evaluation.is_relevant:
            verified_chunks.append(chunks[idx])

    return verified_chunks


def generate_answer(query: str, context: List[str]) -> str:
    print("\n✍️ [Generator] Synthesizing response from verified context...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"You are a strict, factual assistant. Answer the user query using ONLY the provided context. If insufficient, say 'I cannot fulfill this request.'\n\nContext:\n{context_str}\n\nQuery: {query}"
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text


def verify_response(query: str, context: List[str], response_text: str) -> bool:
    print("\n🛡️ [Critic 2] Performing Hallucination Check...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"Context:\n{context_str}\n\nGenerated Response:\n{response_text}\n\nUser Original Query: {query}\n\nCheck if the Generated Response introduces any facts or assumptions not strictly found in the Context."
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=HallucinationCheck,
            temperature=0.0
        ),
    )
    result = HallucinationCheck.model_validate_json(response.text)
    print(f"   ↳ Faithful to Context: {result.is_faithful}\n   ↳ Verdict: {result.verdict}")
    return result.is_faithful


def run_agentic_rag(user_query: str):
    print(f"\n================ USER QUERY: '{user_query}' ================")

    # Live vector processing
    raw_chunks = live_vector_retrieval(user_query, top_k=3)
    good_context = verify_context(user_query, raw_chunks)

    if not good_context:
        print(
            "\n🚨 [Fallback Triggered] Zero relevant chunks passed verification.\nFinal Output: I'm sorry, I couldn't find any verified sources to securely answer that question.")
        return

    answer = generate_answer(user_query, good_context)
    is_safe = verify_response(user_query, good_context, answer)

    if is_safe:
        print(f"\n✅ [Pipeline Success]\nFINAL ANSWER:\n{answer}")
    else:
        print("\n❌ [Pipeline Blocked] Generator output failed the hallucination audit.")


if __name__ == "__main__":
    # Test Case 1: Semantic match requiring logical filtering of noise
    run_agentic_rag("What was Project Alpha's Q2 growth rate?")

    # Test Case 2: Triggers high semantic vector proximity (Alpha + budget/money) but fundamentally unanswerable
    run_agentic_rag("Did Project Alpha allocate any parts of its budget to buying artisanal tacos?")