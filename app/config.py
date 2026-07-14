"""
Central configuration. The Gemini API key is read from the environment
(or a .env file, see .env.example) -- it is never hardcoded in source,
unlike the original notebook. Get a key at https://aistudio.google.com/apikey
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_DIR = BASE_DIR / "vocabulary"
WORD_VOCAB_PATH = VOCAB_DIR / "word_vocab.csv"
PHRASE_VOCAB_PATH = VOCAB_DIR / "phrase_vocab.csv"
SENTENCE_VOCAB_PATH = VOCAB_DIR / "sentence_vocab.csv"

# Cached FAISS indices + embedding model are written here after first build,
# so subsequent server restarts skip re-embedding ~3,300 vocab entries.
CACHE_DIR = BASE_DIR / "cache"

FONTS_DIR = BASE_DIR / "fonts"
SINHALA_FONT_PATH = FONTS_DIR / "NotoSansSinhala-Regular.ttf"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # same model used by Part 2

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

# How many retrieved examples from each vocabulary tier are fed into the
# translation prompt -- matches the values used in the original notebook.
TOP_K_SENTENCES = 3
TOP_K_PHRASES = 5
TOP_K_WORDS = 5

# Matches the chunking values used in the original notebook.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
