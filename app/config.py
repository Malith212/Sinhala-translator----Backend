
import os
from pathlib import Path
from dotenv import load_dotenv

# Loads variables from a .env file (like the Gemini API key) into the
# environment, so we never hardcode secrets directly in the code.
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to the 3 glossary CSV files (word / phrase / sentence level)
VOCAB_DIR = BASE_DIR / "vocabulary"
WORD_VOCAB_PATH = VOCAB_DIR / "word_vocab.csv"
PHRASE_VOCAB_PATH = VOCAB_DIR / "phrase_vocab.csv"
SENTENCE_VOCAB_PATH = VOCAB_DIR / "sentence_vocab.csv"

# Folder where the FAISS indices are saved after the first build, so we
# don't have to re-embed ~3,300 glossary entries every time the server
# restarts (that would be slow).
CACHE_DIR = BASE_DIR / "cache"

FONTS_DIR = BASE_DIR / "fonts"
SINHALA_FONT_PATH = FONTS_DIR / "NotoSansSinhala-Regular.ttf"

# The AI model used to turn text into embeddings (meaning-vectors)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")


SIMILARITY_THRESHOLD = 0.7

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100