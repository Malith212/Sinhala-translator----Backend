from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def chunk_text(text: str) -> list[str]:
    """Splits a long document into overlapping chunks so each chunk stays
    within the LLM's effective context window while preserving some
    surrounding context across chunk boundaries (via the overlap)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)
