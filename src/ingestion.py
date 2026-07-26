import os
from typing import List
from pypdf import PdfReader

def load_file(file_path: str) -> str:
    """Reads raw text from .txt, .md, or .pdf files."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md"]:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Splits document text into overlapping chunks to preserve context at boundaries."""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        if end < text_length:
            last_space = chunk.rfind(" ")
            if last_space != -1:
                end = start + last_space
                chunk = text[start:end]
                
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
        
    return [c for c in chunks if len(c) > 20]
