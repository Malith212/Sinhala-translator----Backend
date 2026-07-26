"""
Vocabulary Index Module

Loads the word / phrase / sentence vocabulary CSVs and builds a FAISS
similarity index for each tier, exactly mirroring the notebook's approach:
each English entry is embedded with sentence-transformers, and FAISS does
nearest-neighbour search at translation time to retrieve relevant
translation-memory examples for the prompt.

Indices are cached to disk after the first build (embedding ~3,300 entries
takes a while) so subsequent server restarts are instant.
"""

import json
import pickle
from functools import lru_cache
from typing import Optional, TypedDict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config

#Just defines the shape of one glossary entry: {english, sinhala}
class VocabEntry(TypedDict):
    english: str
    sinhala: str


class VocabTier:
    """One vocabulary tier (word / phrase / sentence) with its FAISS index."""

    def __init__(self, name: str, index: faiss.Index, meta: list[VocabEntry]):
        self.name = name
        self.index = index
        self.meta = meta

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[VocabEntry]:
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.meta[i] for i in indices[0] if i != -1]
    

#Loads the sentence-transformer model (all-MiniLM-L6-v2)
@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

#Reads a CSV file into a list of {english, sinhala} dictionaries
def _load_vocab_csv(path) -> list[VocabEntry]:
    df = pd.read_csv(path)
    return [
        {"english": str(row["English"]).strip(), "sinhala": str(row["Sinhala"]).strip()}
        for _, row in df.iterrows()
    ]

# Loads a CSV
# Converts every English entry into an embedding (vector)
# Builds a FAISS index from those embeddings
# Saves the index + entries to disk (cache) so it doesn't need to rebuild next time
def _build_and_cache_tier(name: str, csv_path) -> VocabTier:
    entries = _load_vocab_csv(csv_path)
    model = get_embedding_model()
    embeddings = model.encode(
        [e["english"] for e in entries], convert_to_numpy=True, show_progress_bar=False
    )

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.CACHE_DIR / f"{name}.index"))
    with open(config.CACHE_DIR / f"{name}_meta.pkl", "wb") as f:
        pickle.dump(entries, f)

    return VocabTier(name, index, entries)

# Checks if a cached index already exists on disk
# If yes, loads it directly (fast) instead of rebuilding
def _load_cached_tier(name: str) -> Optional[VocabTier]:
    index_path = config.CACHE_DIR / f"{name}.index"
    meta_path = config.CACHE_DIR / f"{name}_meta.pkl"
    if not (index_path.exists() and meta_path.exists()):
        return None
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return VocabTier(name, index, meta)

# Tries to load from cache first
# If no cache exists, builds it fresh and caches it
def load_or_build_tier(name: str, csv_path) -> VocabTier:
    cached = _load_cached_tier(name)
    if cached is not None:
        return cached
    return _build_and_cache_tier(name, csv_path)

# Loads all 3 tiers (word, phrase, sentence) together when the app starts
# Each tier is either loaded from cache or built fresh
class VocabIndices:
    """Holds all three loaded tiers, ready for retrieval."""

    def __init__(self):
        self.word = load_or_build_tier("word", config.WORD_VOCAB_PATH)
        self.phrase = load_or_build_tier("phrase", config.PHRASE_VOCAB_PATH)
        self.sentence = load_or_build_tier("sentence", config.SENTENCE_VOCAB_PATH)


_indices: Optional[VocabIndices] = None

# Returns the loaded VocabIndices object
# Only builds/loads it once (_indices stored globally), reused for every future call
def get_indices() -> VocabIndices:
    """Lazily builds (or loads cached) indices on first use, then reuses them."""
    global _indices
    if _indices is None:
        _indices = VocabIndices()
    return _indices