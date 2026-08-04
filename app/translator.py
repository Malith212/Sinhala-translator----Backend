import re
from typing import Optional

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

#LLM Configuration
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
    """Splits one chunk of text into individual sentences.

    Kept deliberately simple (just regex rules) -- splits on '.', '!',
    '?' followed by whitespace, plus line breaks. Good enough for
    well-formed policy text, and easy to explain in a viva.
    """
    if not text or not text.strip():
        return []

    normalised = re.sub(r"\s*\n\s*", "\n", text.strip())
    raw_pieces = re.split(r"(?<=[.!?])\s+|\n", normalised)

    # Drop tiny fragments (like stray punctuation) that are too short to
    # be a real sentence.
    sentences = [p.strip() for p in raw_pieces if len(p.strip()) >= 10]
    return sentences


def _search_tier_best(text: str, tier_name: str) -> Optional[VocabEntry]:
    """Searches ONE glossary tier (word / phrase / sentence) for the
    single closest match to this piece of text, using cosine similarity.
    """

# get_indices() → loads all 3 glossary tiers (only builds once, then reuses)
# getattr(indices, "phrase") → picks out just the phrase tier object (a VocabTier)
# model.encode([text], ...) → converts our sentence into a normalized embedding (a list of numbers representing meaning)
# tier.search_best(query_embedding, 0.5) → hands off to VocabTier.search_best() (defined in vocab_index.py), passing our embedding + the threshold (0.5)
    indices = get_indices()
    tier = getattr(indices, tier_name)
    model = get_embedding_model()

    # Turn the input text into a normalized embedding, matching how the
    # glossary itself was embedded (see vocab_index.py).
    query_embedding = model.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True
    )
    return tier.search_best(query_embedding, config.SIMILARITY_THRESHOLD)


def _dedupe_entries(entries: list[VocabEntry]) -> list[VocabEntry]:
    """Removes duplicate glossary entries (same English+Sinhala pair)
    while keeping the order they were first found in."""
    seen = set()
    unique: list[VocabEntry] = []
    for entry in entries:
        key = (entry["english"], entry["sinhala"])
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def _search_tier_by_sentences(chunk_text: str, tier_name: str) -> list[VocabEntry]:
    """Splits the chunk into individual sentences, then for EACH sentence
    finds at most ONE verified glossary match (only if it clears the
    similarity threshold). This avoids "chunk-level dilution" -- where a
    specific detail in one sentence could get blended/lost if the whole
    chunk were embedded as a single vector instead of sentence-by-sentence.

    Sentences with no genuinely relevant match simply contribute nothing.
    Results across all sentences are then de-duplicated.
    """
    matches: list[VocabEntry] = []
    for sentence in _split_into_sentences(chunk_text):
        best = _search_tier_best(sentence, tier_name)
        if best is not None:
            matches.append(best)

    return _dedupe_entries(matches)


def build_prompt(
    text: str,
    sentence_matches: list[VocabEntry],
    phrase_matches: list[VocabEntry],
    word_matches: list[VocabEntry],
    enforcement_block: str = "",
) -> str:
    """Builds the full instruction text sent to Gemini: shows the
    retrieved glossary matches as reference examples, then asks it to
    translate the given text using them where relevant. If an
    enforcement_block is supplied (see enforcement.py), it's appended as
    a non-negotiable requirement rather than a mere suggestion."""
    prompt = "You are a professional Sinhala translator for privacy policies.\n\n"

    # Only include a section if there are actually matches to show --
    # no point showing an empty "Matches:" heading with nothing under it.
    if sentence_matches:
        prompt += "Sentence Matches:\n"
        for s in sentence_matches:
            prompt += f'EN: {s["english"]}\nSI: {s["sinhala"]}\n'

    if phrase_matches:
        prompt += "\nPhrase Matches:\n"
        for p in phrase_matches:
            prompt += f'EN: {p["english"]}\nSI: {p["sinhala"]}\n'

    if word_matches:
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


def translate(text: str) -> str:
    """Translates one chunk of English text into Sinhala. Plain
    (unenforced) path -- glossary matches are suggestions only. Use
    translate_with_enforcement() for the verified/enforced version.

    Steps:
    1. For each of the 3 glossary tiers, find the best sentence-level
       matches (only genuinely relevant ones, thanks to the similarity
       threshold).
    2. Build a prompt showing those matches as reference.
    3. Send the prompt to Gemini and return its Sinhala translation.
    """
    sentence_matches = _search_tier_by_sentences(text, "sentence")
    phrase_matches = _search_tier_by_sentences(text, "phrase")
    word_matches = _search_tier_by_sentences(text, "word")

    prompt = build_prompt(text, sentence_matches, phrase_matches, word_matches)
    return _generate(prompt)


def translate_with_enforcement(text: str, allow_retry: bool = True) -> tuple[str, EnforcementReport]:
    """Translates text with deterministic terminology enforcement:

    1. Detects glossary terms present in the source text (DETECT).
    2. Injects them into the prompt as non-negotiable requirements (LOCK).
    3. Generates the translation, then checks every required term actually
       appears in the output (VERIFY).
    4. If any are missing and allow_retry is True, regenerates once with an
       explicit correction listing exactly which terms were missed (RETRY).

    Returns the final translation text and an EnforcementReport describing
    what was required, what was satisfied, and whether a retry happened.
    """
    sentence_matches = _search_tier_by_sentences(text, "sentence")
    phrase_matches = _search_tier_by_sentences(text, "phrase")
    word_matches = _search_tier_by_sentences(text, "word")

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