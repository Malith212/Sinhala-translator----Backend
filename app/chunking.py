import re

from . import config


def chunk_text(text: str) -> list[str]:
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