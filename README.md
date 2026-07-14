# PDPA Sinhala Translator — Backend (Part 1)

A retrieval-augmented English→Sinhala translator for privacy-policy text,
built around the pipeline from your notebook: FAISS similarity search over
three vocabulary tiers (word / phrase / sentence) feeds relevant
translation-memory examples into a Gemini prompt, so the model translates
using your curated legal terminology instead of a generic word-for-word
translation.

This is a **separate backend** from the Part 2 PDPA Compliance Mapper —
run both at once, on different ports, and the frontend talks to each.

```
sinhala-translator-backend/
├── app/
│   ├── main.py          FastAPI app: /translate, /translate-document, /translate-document/pdf
│   ├── config.py        paths, model names, reads GEMINI_API_KEY from .env
│   ├── vocab_index.py   builds/caches FAISS indices for the 3 vocab tiers
│   ├── translator.py    retrieval + prompt building + Gemini call
│   ├── chunking.py      splits long documents (matches notebook's settings)
│   ├── pdf_extract.py   PDF → text (pymupdf)
│   ├── pdf_render.py    text → Sinhala PDF (optional, needs weasyprint + font)
│   └── models.py        request/response schemas
├── vocabulary/           your word/phrase/sentence CSVs
├── fonts/                 put NotoSansSinhala-Regular.ttf here (optional)
└── requirements.txt
```

## ⚠️ About the API key

Your uploaded notebook had a live Gemini API key hardcoded in a cell. Since
that notebook has now been shared, **rotate that key** at
https://aistudio.google.com/apikey before using it further — treat it as
compromised. This backend reads the key from environment variables only;
it's never in source code.

## Setup

This project has been tested and confirmed working on **Python 3.9** (the
version your Mac has installed) as well as newer versions — the type-hint
syntax throughout is 3.9-compatible.

```bash
cd sinhala-translator-backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and paste in your (rotated) Gemini API key
```

### First run will be slow — that's expected

The first time the server handles a request, it embeds all ~3,340
vocabulary entries (word + phrase + sentence tiers) and builds FAISS
indices, then **caches them to disk** in `cache/`. Every run after that
loads instantly from cache. If you edit the CSVs, delete the `cache/`
folder to force a rebuild.

### Run the server

```bash
uvicorn app.main:app --reload --port 8001
```

Note the port: **8001**, not 8000 — that's the Part 2 compliance-mapper
backend. Run both at the same time in separate terminals if you want both
features working together in the frontend.

Visit `http://localhost:8001/docs` to see the interactive API docs.

## Endpoints

- `POST /translate` — `{ "text": "..." }` → translates a single sentence/paragraph.
- `POST /translate-document` — upload a PDF → extracts text, chunks it, translates each chunk, returns the reconstructed Sinhala text as JSON.
- `POST /translate-document/pdf` — same as above, but returns a downloadable Sinhala PDF. **Optional** — requires `weasyprint` (see below) and a Sinhala font file.

## Optional: PDF-rendering endpoint setup

`/translate-document/pdf` needs two extra things the JSON endpoints don't:

1. **weasyprint's system dependencies.** On macOS: `brew install pango cairo gdk-pixbuf libffi`. On Ubuntu/Debian: `apt install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0 libffi-dev` (same packages your notebook installed via `apt`).
2. **A Sinhala font file.** Download "Noto Sans Sinhala" from https://fonts.google.com/noto/specimen/Noto+Sans+Sinhala, and place `NotoSansSinhala-Regular.ttf` in the `fonts/` folder.

If either is missing, that one endpoint returns a clear error — the rest of the API still works fine.

## Testing without spending API calls

`app/translator.py`'s retrieval logic (FAISS search + prompt building) can
be tested independently of Gemini by mocking `get_embedding_model()` — see
this pattern if you want to write tests:

```python
from unittest.mock import patch
with patch('app.vocab_index.get_embedding_model', return_value=your_fake_model):
    ...
```
