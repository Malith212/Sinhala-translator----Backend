import re

from google import genai

from . import config
from .enforcement import (
    EnforcementReport,
    build_enforcement_block,
    build_retry_correction_block,
    detect_required_terms,
    find_missing_terms,
)
from .vocab_index import get_indices, get_embedding_model, VocabEntry

_gemini_client = None


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to a .env file or export it "
                "as an environment variable before starting the server."
            )
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def _split_into_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter for a single chunk of text.

    Kept deliberately simple (regex-based), same approach as the
    compliance app's extraction.py -- splits on '.', '!', '?' followed by
    whitespace, plus line breaks. Good enough for well-formed policy text,
    and easy to explain in a viva.
    """
    if not text or not text.strip():
        return []

    normalised = re.sub(r"\s*\n\s*", "\n", text.strip())
    raw_pieces = re.split(r"(?<=[.!?])\s+|\n", normalised)

    sentences = [p.strip() for p in raw_pieces if len(p.strip()) >= 10]
    return sentences


def _search_tier(text: str, tier_name: str, top_k: int) -> list[VocabEntry]:
    indices = get_indices()
    tier = getattr(indices, tier_name)
    model = get_embedding_model()
    query_embedding = model.encode([text], convert_to_numpy=True)
    return tier.search(query_embedding, top_k)


def _dedupe_entries(entries: list[VocabEntry]) -> list[VocabEntry]:
    """Removes duplicate glossary entries (same English+Sinhala pair)
    while preserving the order they were first seen in."""
    seen = set()
    unique: list[VocabEntry] = []
    for entry in entries:
        key = (entry["english"], entry["sinhala"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def _search_tier_by_sentences(
    chunk_text: str, tier_name: str, top_k: int
) -> list[VocabEntry]:
    """Splits the chunk into individual sentences and searches a glossary
    tier using EACH sentence separately, then merges and de-duplicates the
    results. This avoids chunk-level embedding dilution -- a specific
    detail in one sentence gets its own focused embedding, instead of
    being blended in with the rest of the chunk's content."""

    sentence_level_matches: list[VocabEntry] = []
    for sentence in _split_into_sentences(chunk_text):
        sentence_level_matches.extend(_search_tier(sentence, tier_name, top_k))

    combined = _dedupe_entries(sentence_level_matches)

    # Cap the total so the prompt doesn't grow unbounded for chunks with
    # many sentences -- keep a generous margin over top_k for coverage.
    return combined[: top_k * 2]


def build_prompt(
    text: str,
    sentence_matches: list[VocabEntry],
    phrase_matches: list[VocabEntry],
    word_matches: list[VocabEntry],
    enforcement_block: str = "",
) -> str:
    prompt = "You are a professional Sinhala translator for privacy policies.\n\n"

    prompt += "Sentence Matches:\n"
    for s in sentence_matches:
        prompt += f'EN: {s["english"]}\nSI: {s["sinhala"]}\n'

    prompt += "\nPhrase Matches:\n"
    for p in phrase_matches:
        prompt += f'EN: {p["english"]}\nSI: {p["sinhala"]}\n'

    prompt += "\nWord Matches:\n"
    for w in word_matches:
        prompt += f'EN: {w["english"]}\nSI: {w["sinhala"]}\n'

    if enforcement_block:
        prompt += enforcement_block

    prompt += (
        "\nUsing the terminology and phrasing shown above as reference where "
        "relevant, translate the following English privacy-policy text into "
        "natural, legally accurate Sinhala. Preserve the meaning precisely -- "
        "this is a legal document. Respond with ONLY the Sinhala translation, "
        "no preamble, no explanation.\n\n"
        f"Text to translate:\n{text}"
    )
    return prompt


def _generate(prompt: str) -> str:
    client = _get_gemini_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL_NAME, contents=prompt
    )
    return response.text.strip()


def _get_all_tier_matches(text: str):
    """Runs sentence-level retrieval across all 3 glossary tiers for the
    given chunk. Shared by both translate() and translate_with_enforcement()
    so the retrieval logic only lives in one place."""
    sentence_matches = _search_tier_by_sentences(text, "sentence", config.TOP_K_SENTENCES)
    phrase_matches = _search_tier_by_sentences(text, "phrase", config.TOP_K_PHRASES)
    word_matches = _search_tier_by_sentences(text, "word", config.TOP_K_WORDS)
    return sentence_matches, phrase_matches, word_matches


def translate(text: str) -> str:
    """Translates a single piece of English text (sentence, paragraph, or
    chunk) into Sinhala, using retrieved vocabulary matches as
    translation-memory context. This is the plain (unenforced) path --
    glossary matches are suggestions only. Use translate_with_enforcement()
    for the verified/enforced version.
    """
    sentence_matches, phrase_matches, word_matches = _get_all_tier_matches(text)
    prompt = build_prompt(text, sentence_matches, phrase_matches, word_matches)
    return _generate(prompt)


def translate_with_enforcement(
    text: str, allow_retry: bool = True
) -> tuple[str, EnforcementReport]:
    """Translates text with deterministic terminology enforcement:

    1. Detects glossary terms EXACTLY present in the source text (DETECT).
    2. Injects them into the prompt as non-negotiable requirements (LOCK),
       alongside the usual semantic-search reference matches.
    3. Generates the translation, then checks every required term actually
       appears in the output (VERIFY).
    4. If any are missing and allow_retry is True, regenerates once with an
       explicit correction listing exactly which terms were missed (RETRY).

    Returns the final translation text and an EnforcementReport describing
    what was required, what was satisfied, and whether a retry happened --
    this is the data an ablation study needs.
    """
    sentence_matches, phrase_matches, word_matches = _get_all_tier_matches(text)

    required_terms = detect_required_terms(text)
    enforcement_block = build_enforcement_block(required_terms)

    prompt = build_prompt(
        text, sentence_matches, phrase_matches, word_matches, enforcement_block
    )
    translated = _generate(prompt)

    report = EnforcementReport(required_terms=required_terms)

    if not required_terms:
        return translated, report

    missing = find_missing_terms(translated, required_terms)
    report.missing_after_first_pass = missing

    if missing and allow_retry:
        correction = build_retry_correction_block(missing)
        retry_prompt = prompt + "\n" + correction
        translated = _generate(retry_prompt)
        report.retried = True
        report.missing_after_retry = find_missing_terms(translated, required_terms)

    return translated, report