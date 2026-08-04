import pickle
from functools import lru_cache
from typing import Optional, TypedDict

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config


# Defines the shape of one glossary entry: an English phrase + its
# Sinhala translation.
class VocabEntry(TypedDict):
    english: str
    sinhala: str


class VocabTier:
    """One vocabulary tier (word / phrase / sentence) with its own FAISS
    index. The embeddings stored here are NORMALIZED (scaled to the same
    length), which means we can use a simple inner-product search to get
    cosine similarity scores directly -- no extra maths needed.
    """

    def __init__(self, name: str, index: faiss.Index, meta: list[VocabEntry]):
        self.name = name
        self.index = index
        self.meta = meta

#     Finds the single closest glossary entry to a given query
#     Checks its similarity score against a threshold
#     If score is too low (or nothing found) → returns None
#     If score passes → returns that one glossary entry
    def search_best(
        self, query_embedding: np.ndarray, threshold: float
    ) -> Optional[VocabEntry]:

        scores, indices = self.index.search(query_embedding, 1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        # -1 means FAISS found no result at all; below threshold means
        # the best result it did find still isn't good enough.
        if best_idx == -1 or best_score < threshold:
            return None

        return self.meta[best_idx]


# Loads the AI model that turns text into embeddings.
# @lru_cache means: load it ONCE, reuse it for every future call (loading
# it fresh every time would be slow).
@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)

# Converts text into embeddings
# Normalizes them (scales all vectors to the same length)
# This normalization is what makes a simple inner-product search equal to cosine similarity
def _encode_normalized(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# Reads a glossary CSV file into a list of {english, sinhala} dictionaries.
def _load_vocab_csv(path) -> list[VocabEntry]:
    df = pd.read_csv(path)
    return [
        {"english": str(row["English"]).strip(), "sinhala": str(row["Sinhala"]).strip()}
        for _, row in df.iterrows()
    ]


def _build_and_cache_tier(name: str, csv_path) -> VocabTier:
    """Builds a fresh FAISS index for one glossary tier:
    1. Load the CSV
    2. Convert every English entry into a normalized embedding
    3. Build a FAISS index using inner-product search (= cosine similarity,
       since the embeddings are normalized)
    4. Save everything to disk so we don't have to rebuild it next time
    """
    entries = _load_vocab_csv(csv_path)
    model = get_embedding_model()
    embeddings = _encode_normalized(model, [e["english"] for e in entries])

    # IndexFlatIP = "inner product" search. On normalized vectors, this
    # gives the exact same result as cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(config.CACHE_DIR / f"{name}.index"))
    with open(config.CACHE_DIR / f"{name}_meta.pkl", "wb") as f:
        pickle.dump(entries, f)

    return VocabTier(name, index, entries)


# Checks if a cached index already exists on disk.
# If yes, loads it directly (fast) instead of rebuilding from scratch.
def _load_cached_tier(name: str) -> Optional[VocabTier]:
    index_path = config.CACHE_DIR / f"{name}.index"
    meta_path = config.CACHE_DIR / f"{name}_meta.pkl"
    if not (index_path.exists() and meta_path.exists()):
        return None
    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return VocabTier(name, index, meta)


# Tries to load from cache first. If no cache exists yet, builds it fresh.
def load_or_build_tier(name: str, csv_path) -> VocabTier:
    cached = _load_cached_tier(name)
    if cached is not None:
        return cached
    return _build_and_cache_tier(name, csv_path)

# Loads all 3 tiers (word, phrase, sentence) together when the app starts
# Each tier is either loaded from cache or built fresh
class VocabIndices:

    def __init__(self):
        self.word = load_or_build_tier("word", config.WORD_VOCAB_PATH)
        self.phrase = load_or_build_tier("phrase", config.PHRASE_VOCAB_PATH)
        self.sentence = load_or_build_tier("sentence", config.SENTENCE_VOCAB_PATH)


_indices: Optional[VocabIndices] = None

# Returns the loaded VocabIndices object
# Only builds/loads it once (stored globally), reused for every future call
def get_indices() -> VocabIndices:
    global _indices
    if _indices is None:
        _indices = VocabIndices()
    return _indices