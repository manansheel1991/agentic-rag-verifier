from typing import List
from google.genai import types
from src.config import client
from src.schemas import ContextVerification, HallucinationCheck

KNOWLEDGE_BASE = {
    0: "Project Alpha's Q2 revenue grew by 14% quarter-over-quarter, driven by enterprise SaaS adoptions.",
    1: "The cafeteria will be serving artisanal tacos on Tuesdays starting next month.",
    2: "We detected a memory leak in the core data pipeline when handling payloads exceeding 10GB.",
    3: "Project Alpha's total budget for the fiscal year is capped at $4.5 million, with 40% allocated to R&D."
}

def mock_vector_retrieval(query: str) -> List[str]:
    if "revenue" in query.lower() or "alpha" in query.lower():
        return [KNOWLEDGE_BASE[0], KNOWLEDGE_BASE[1], KNOWLEDGE_BASE[3]]
    return [KNOWLEDGE_BASE[1], KNOWLEDGE_BASE[2]]

def verify_context(query: str, chunks: List[str]) -> List[str]:
    print("\n🔍 [Critic 1] Verifying retrieved context...")
    verified_chunks = []
    for idx, chunk in enumerate(chunks):
        prompt = f"User Query: {query}\nDocument Chunk: {chunk}\n\nEvaluate whether this document chunk contains information that directly helps answer the user query."
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ContextVerification,
                temperature=0.0
            ),
        )
        result = ContextVerification.model_validate_json(response.text)
        print(f"   ↳ Chunk {idx}: Relevant={result.is_relevant} | Reason: {result.reason}")
        if result.is_relevant:
            verified_chunks.append(chunk)
    return verified_chunks

def generate_answer(query: str, context: List[str]) -> str:
    print("\n✍️ [Generator] Synthesizing response from verified context...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"You are a strict, factual assistant. Answer the user query using ONLY the provided context. If insufficient, say 'I cannot fulfill this request based on the available data.'\n\nContext:\n{context_str}\n\nQuery: {query}"
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2)
    )
    return response.text

def verify_response(query: str, context: List[str], response_text: str) -> bool:
    print("\n🛡️ [Critic 2] Performing Hallucination Check...")
    context_str = "\n".join([f"- {c}" for c in context])
    prompt = f"Context:\n{context_str}\n\nGenerated Response:\n{response_text}\n\nUser Original Query: {query}\n\nCheck if the Generated Response introduces any facts or assumptions not strictly found in the Context."
    response = client.models.generate_content(
        model='gemini-2.5-flash',
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
    raw_chunks = mock_vector_retrieval(user_query)
    good_context = verify_context(user_query, raw_chunks)
    
    if not good_context:
        print("\n🚨 [Fallback Triggered] Zero relevant chunks passed verification.\nFinal Output: I'm sorry, I couldn't find any verified sources to securely answer that question.")
        return

    answer = generate_answer(user_query, good_context)
    is_safe = verify_response(user_query, good_context, answer)
    
    if is_safe:
        print(f"\n✅ [Pipeline Success]\nFINAL ANSWER:\n{answer}")
    else:
        print("\n❌ [Pipeline Blocked] Generator output failed the hallucination audit.")

if __name__ == "__main__":
    run_agentic_rag("How much did Project Alpha grow in Q2?")
    run_agentic_rag("Did Project Alpha design a marketing plan for the taco initiative?")

