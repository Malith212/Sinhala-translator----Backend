from typing import List, Optional
from pydantic import BaseModel


class TranslateTextRequest(BaseModel):
    text: str


class TranslateTextResponse(BaseModel):
    original_text: str
    translated_text: str


class ChunkTranslation(BaseModel):
    id: int
    original_text: str
    translated_text: str


class TranslateDocumentResponse(BaseModel):
    filename: str
    chunk_count: int
    chunks: List[ChunkTranslation]
    translated_text: str  # full reconstructed Sinhala document


class HealthResponse(BaseModel):
    status: str
    indices_loaded: bool
    gemini_configured: bool
