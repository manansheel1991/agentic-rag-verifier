import os
from google.genai import client as genai_client
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Automatically loads variables from .env into os.environ

# Detect available API keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# Primary Gemini Config
GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_GEN_MODEL = "gemini-2.5-flash-lite"

# Fallback OpenAI Config
OPENAI_EMBED_MODEL = "text-embedding-3-small"
OPENAI_GEN_MODEL = "gpt-4o-mini"

# Initialize clients
gemini_client_obj = genai_client.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
openai_client_obj = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


def get_vector_embedding(text: str):
    """Retrieves text embeddings, falling back to OpenAI if Gemini fails or hits quota."""
    if gemini_client_obj:
        try:
            resp = gemini_client_obj.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=text
            )
            return resp.embeddings[0].values
        except Exception as e:
            print(f"⚠️ [Gemini Embedding Quota/Error]: {e}")
            print("🔄 Switching to OpenAI embedding fallback...")

    if openai_client_obj:
        resp = openai_client_obj.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=[text]
        )
        return resp.data[0].embedding

    raise RuntimeError("No working API keys or quota available for embeddings.")


def generate_structured_json(prompt: str, schema_class, temperature: float = 0.0):
    """Generates structured JSON responses using available models."""
    if gemini_client_obj:
        try:
            resp = gemini_client_obj.models.generate_content(
                model=GEMINI_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_class,
                    temperature=temperature
                )
            )
            return resp.text
        except Exception as e:
            print(f"⚠️ [Gemini Generation Quota/Error]: {e}")
            print("🔄 Switching to OpenAI structured output fallback...")

    if openai_client_obj:
        resp = openai_client_obj.beta.chat.completions.parse(
            model=OPENAI_GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise data auditor."},
                {"role": "user", "content": prompt}
            ],
            response_format=schema_class,
            temperature=temperature
        )
        return resp.choices[0].message.content

    raise RuntimeError("No working API keys or quota available for generation.")

def generate_text(prompt: str, temperature: float = 0.2) -> str:
    """Generates standard unconstrained text response using Gemini, falling back to OpenAI on error/quota exhaustion."""
    if gemini_client_obj:
        try:
            resp = gemini_client_obj.models.generate_content(
                model=GEMINI_GEN_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=temperature)
            )
            return resp.text
        except Exception as e:
            print(f"⚠️ [Gemini Generation Quota/Error]: {e}")
            print("🔄 Switching to OpenAI text generation fallback...")

    if openai_client_obj:
        resp = openai_client_obj.chat.completions.create(
            model=OPENAI_GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return resp.choices[0].message.content

    raise RuntimeError("No working API keys or quota available for generation.")