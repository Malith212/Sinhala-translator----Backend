from typing import List, Optional
from pydantic import BaseModel


class TranslateTextRequest(BaseModel):
    text: str


class TranslateTextResponse(BaseModel):
    original_text: str
    translated_text: str


class RequiredTerm(BaseModel):
    english: str
    sinhala: str


class EnforcementSummary(BaseModel):
    required_count: int
    satisfied_count_first_pass: int
    satisfied_count_final: int
    retried: bool
    required_terms: List[RequiredTerm]
    missing_after_first_pass: List[RequiredTerm]
    missing_after_retry: List[RequiredTerm]


class TranslateEnforcedResponse(BaseModel):
    original_text: str
    translated_text: str
    enforcement: EnforcementSummary


class TranslateCompareResponse(BaseModel):
    original_text: str
    baseline_translation: str
    enforced_translation: str
    enforcement: EnforcementSummary


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