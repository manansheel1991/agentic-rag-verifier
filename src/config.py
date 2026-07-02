import os
from google import genai

# Validate environment setup defensively before booting up
if "GEMINI_API_KEY" not in os.environ:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable. Please run 'source .env'")

client = genai.Client()

# Core model definitions for the pipeline
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATION_MODEL = "gemini-2.5-flash-lite"