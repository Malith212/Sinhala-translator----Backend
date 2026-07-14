import fitz  # pymupdf


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts plain text from a PDF file's raw bytes (no temp file needed)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + "\n"
        return full_text
    finally:
        doc.close()
