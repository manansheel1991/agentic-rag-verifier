import sys
import os
import numpy as np
from typing import List
from google.genai import types
from config import client, EMBEDDING_MODEL, GENERATION_MODEL
from schemas import BatchContextVerification, HallucinationCheck
from ingestion import load_file, chunk_text

def get_embedding(text: str) -> np.ndarray:
    """Invokes Gemini Embedding engine to convert text into a vector."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return np.array(response.embeddings[0].values)

def dynamic_vector_retrieval(query: str, document_chunks: List[str], top_k: int = 3) -> List[str]:
    """Computes Cosine Similarity against dynamically loaded document chunks."""
    print(f"\n🧠 [Vector Search] Embedding query & scanning {len(document_chunks)} chunks...")
    query_vector = get_embedding(query)
    
    scored_chunks = []
    for chunk in document_chunks:
        chunk_vector = get_embedding(chunk)
        
        dot_product = np.dot(query_vector, chunk_vector)
        query_norm = np.linalg.norm(query_vector)
        chunk_norm = np.linalg.norm(chunk_vector)
        
        cosine_similarity = dot_product / (query_norm * chunk_norm)
        scored_chunks.append((cosine_similarity, chunk))
        
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    print(f"   ↳ Top Vector Matches:")
    for score, text in scored_chunks[:top_k]:
        snippet = text.replace('\n', ' ')[:60]
        print(f"     [Similarity: {score:.4f}] -> {snippet}...")
        
    return [chunk for score, chunk in scored_chunks[:top_k]]

def verify_context(query: str, chunks: List[str]) -> List[str]:
    print("\n🔍 [Critic 1] Batch-verifying retrieved context...")
    formatted_chunks = "\n".join([f"[{idx}] {chunk}" for idx, chunk in enumerate(chunks)])
    
    prompt = f"User Query: {query}\n\nAvailable Document Chunks:\n{formatted_chunks}\n\nEvaluate every chunk."
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchContextVerification,
            temperature=0.0
        ),
    )
    
    batch_result = BatchContextVerification.model_validate_json(response.text)
    verified_chunks = []
    
    for evaluation in batch_result.evaluations:
        idx = evaluation.chunk_index
        print(f"   ↳ Chunk [{idx}]: Relevant={evaluation.is_relevant} | Reason: {evaluation.reason}")
        if evaluation.is_relevant:
            verified_chunks.append(chunks[idx])
            
    return verified_chunks

def generate_answer(query: str, context: List[str]) -> str:
    print("\n✍️ [Generator] Synthesizing response from verified context...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"Answer using ONLY the provided context. If insufficient, say 'I cannot fulfill this request.'\n\nContext:\n{context_str}\n\nQuery: {query}"
    
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text

def verify_response(query: str, context: List[str], response_text: str) -> bool:
    print("\n🛡️ [Critic 2] Performing Hallucination Check...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"Context:\n{context_str}\n\nGenerated Response:\n{response_text}\n\nQuery: {query}\n\nCheck for ungrounded claims."
    
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
    print(f"   ↳ Faithful: {result.is_faithful} | Verdict: {result.verdict}")
    return result.is_faithful

def run_pipeline(file_path: str, user_query: str):
    print(f"\n📄 Loading Document: {file_path}")
    raw_text = load_file(file_path)
    chunks = chunk_text(raw_text)
    print(f"📦 Successfully parsed document into {len(chunks)} chunks.")
    
    print(f"\n================ USER QUERY: '{user_query}' ================")
    raw_chunks = dynamic_vector_retrieval(user_query, chunks, top_k=3)
    good_context = verify_context(user_query, raw_chunks)
    
    if not good_context:
        print("\n🚨 [Fallback Triggered] Zero chunks passed verification.")
        print("Final Output: I cannot fulfill this request based on the provided document.")
        return

    answer = generate_answer(user_query, good_context)
    is_safe = verify_response(user_query, good_context, answer)
    
    if is_safe:
        print(f"\n✅ [Pipeline Success]\n\nFINAL ANSWER:\n{answer}")
    else:
        print("\n❌ [Pipeline Blocked] Response failed hallucination audit.")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    sample_path = "data/sample_report.txt"
    
    if not os.path.exists(sample_path):
        with open(sample_path, "w") as f:
            f.write(
                "System Architecture Overview:\n"
                "The core streaming pipeline processes 50,000 events/sec with sub-50ms latency. "
                "Database replicas are hosted across us-east-1 and us-west-2 with automatic failover. "
                "Disaster recovery tests are conducted bi-weekly on Sundays at 02:00 UTC. "
                "The engineering budget for cloud infrastructure is capped at $120,000 per month."
            )
            
    run_pipeline(sample_path, "How often are disaster recovery tests run?")
    run_pipeline(sample_path, "What is the policy for employee expense reimbursements?")
