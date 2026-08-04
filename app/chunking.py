import re

from . import config


def chunk_text(text: str) -> list[str]:
    """Splits a long document into chunks that end on a sentence boundary.

    Instead of cutting at a hard character limit (which can slice a
    sentence in half, forcing us to use overlap as a safety net), we:
      1. Walk through the text sentence by sentence.
      2. Keep adding sentences to the current chunk.
      3. Once the chunk passes config.CHUNK_SIZE, close it off at that
         sentence boundary and start a new chunk.

    This means chunks are variable-length (usually a bit over
    CHUNK_SIZE), but no sentence -- and therefore no content -- ever
    appears in two chunks. No overlap needed, no de-duplication needed
    at merge time.
    """
    if not text or not text.strip():
        return []

    sentences = _split_into_sentences(text)

    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        # Edge case: a single sentence longer than CHUNK_SIZE by itself
        # (rare, but legal text can have long run-on sentences). Flush
        # whatever we have, then let this sentence be its own chunk
        # rather than waiting forever for a shorter one.
        if len(sentence) > config.CHUNK_SIZE:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(sentence.strip())
            continue

        if current and len(current) + len(sentence) > config.CHUNK_SIZE:
            # Adding this sentence would push us over the limit --
            # close the current chunk off here (clean sentence boundary)
            # and start the next chunk with this sentence.
            chunks.append(current.strip())
            current = sentence
        else:
            current = f"{current} {sentence}".strip()

    if current:
        chunks.append(current.strip())

    return chunks


def _split_into_sentences(text: str) -> list[str]:
    """Splits raw text into sentences/clauses. Treats newlines and bullet
    points as boundaries too, so a bullet list doesn't get glued into one
    giant 'sentence'."""
    normalised = re.sub(r"[•●▪\-\*]\s*", "\n", text)
    normalised = re.sub(r"\s*\n\s*", "\n", normalised.strip())

    raw_pieces = re.split(r"(?<=[.!?])\s+|\n", normalised)
    return [p.strip() for p in raw_pieces if p.strip()]