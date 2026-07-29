import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import config
from .chunking import chunk_text
from .models import (
    ChunkTranslation,
    EnforcementSummary,
    HealthResponse,
    RequiredTerm,
    TranslateCompareResponse,
    TranslateDocumentResponse,
    TranslateEnforcedResponse,
    TranslateTextRequest,
    TranslateTextResponse,
)
from .pdf_extract import extract_text_from_pdf_bytes
from .translator import translate, translate_with_enforcement
from .vocab_index import get_indices


def _to_enforcement_summary(report) -> EnforcementSummary:
    return EnforcementSummary(
        required_count=report.required_count,
        satisfied_count_first_pass=report.satisfied_count_first_pass,
        satisfied_count_final=report.satisfied_count_final,
        retried=report.retried,
        required_terms=[RequiredTerm(**t) for t in report.required_terms],
        missing_after_first_pass=[RequiredTerm(**t) for t in report.missing_after_first_pass],
        missing_after_retry=[RequiredTerm(**t) for t in report.missing_after_retry],
    )


app = FastAPI(
    title="PDPA Sinhala Translator API",
    description="Translates English privacy-policy text into Sinhala using a "
    "retrieval-augmented (word/phrase/sentence vocabulary) + LLM pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "PDPA Sinhala Translator API is running."}


@app.get("/health", response_model=HealthResponse)
def health():
    indices_loaded = True
    try:
        get_indices()
    except Exception:
        indices_loaded = False
    return HealthResponse(
        status="ok",
        indices_loaded=indices_loaded,
        gemini_configured=bool(config.GEMINI_API_KEY),
    )


@app.post("/translate", response_model=TranslateTextResponse)
def translate_text(request: TranslateTextRequest) -> TranslateTextResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        translated = translate(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return TranslateTextResponse(original_text=request.text, translated_text=translated)


@app.post("/translate/enforced", response_model=TranslateEnforcedResponse)
def translate_text_enforced(request: TranslateTextRequest) -> TranslateEnforcedResponse:
    """Same as /translate, but with deterministic terminology enforcement:
    detects required glossary terms in the source, forces them into the
    prompt, verifies they appear in the output, and retries once with
    corrective feedback if any are missing. See app/enforcement.py."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        translated, report = translate_with_enforcement(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return TranslateEnforcedResponse(
        original_text=request.text,
        translated_text=translated,
        enforcement=_to_enforcement_summary(report),
    )


@app.post("/translate/compare", response_model=TranslateCompareResponse)
def translate_text_compare(request: TranslateTextRequest) -> TranslateCompareResponse:
    """Runs BOTH the plain (unenforced) and enforced translation paths on
    the same input and returns both outputs side by side, along with the
    enforcement report. Built for ablation testing -- to demonstrate
    whether term enforcement measurably improves terminology consistency
    over baseline retrieval-augmented translation."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty.")
    try:
        baseline = translate(request.text)
        enforced, report = translate_with_enforcement(request.text)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    return TranslateCompareResponse(
        original_text=request.text,
        baseline_translation=baseline,
        enforced_translation=enforced,
        enforcement=_to_enforcement_summary(report),
    )


@app.post("/translate-document", response_model=TranslateDocumentResponse)
async def translate_document(file: UploadFile = File(...)) -> TranslateDocumentResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    english_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not english_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in this PDF (it may be a scanned image).",
        )

    chunks = chunk_text(english_text)

    translated_chunks: list[ChunkTranslation] = []
    try:
        for i, chunk in enumerate(chunks):
            translated_chunks.append(
                ChunkTranslation(id=i, original_text=chunk, translated_text=translate(chunk))
            )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    full_translation = "\n\n".join(c.translated_text for c in translated_chunks)

    return TranslateDocumentResponse(
        filename=file.filename,
        chunk_count=len(chunks),
        chunks=translated_chunks,
        translated_text=full_translation,
    )


@app.post("/translate-document/pdf")
async def translate_document_as_pdf(file: UploadFile = File(...)):
    """Same as /translate-document, but returns a rendered Sinhala PDF file
    instead of JSON. Requires weasyprint + a bundled Sinhala font -- see
    app/pdf_render.py and the README for setup."""
    from .pdf_render import render_sinhala_pdf  # lazy import, see pdf_render.py

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes = await file.read()
    english_text = extract_text_from_pdf_bytes(pdf_bytes)
    if not english_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in this PDF.")

    chunks = chunk_text(english_text)
    try:
        translated_paragraphs = [translate(chunk) for chunk in chunks]
        output_pdf_bytes = render_sinhala_pdf(translated_paragraphs)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        io.BytesIO(output_pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=translated_sinhala.pdf"},
    )