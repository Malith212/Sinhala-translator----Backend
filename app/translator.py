"""
Translator Module

For a given English text, retrieves the closest matching examples from
each vocabulary tier (sentences, phrases, words) and feeds them to Gemini
as translation-memory context, so the model translates using the project's
curated legal terminology rather than a generic word-for-word translation.
"""

from google import genai

from . import config
from .vocab_index import get_indices, get_embedding_model, VocabEntry

_gemini_client = None

# Creates the Gemini API client (only once, reused after)
# Fails with a clear error if no API key is set
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

# Converts input text into an embedding
# Searches one glossary tier (word/phrase/sentence) for the closest matches
# Returns the top-k results
def _search_tier(text: str, tier_name: str, top_k: int) -> list[VocabEntry]:
    indices = get_indices()
    tier = getattr(indices, tier_name)
    model = get_embedding_model()
    query_embedding = model.encode([text], convert_to_numpy=True)
    return tier.search(query_embedding, top_k)

# Builds the full instruction text sent to Gemini
# Lists retrieved sentence, phrase, and word matches as reference examples
# Adds a final instruction: translate the text using these as guidance, output Sinhala only
def build_prompt(
    text: str,
    sentence_matches: list[VocabEntry],
    phrase_matches: list[VocabEntry],
    word_matches: list[VocabEntry],
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

    prompt += (
        "\nUsing the terminology and phrasing shown above as reference where "
        "relevant, translate the following English privacy-policy text into "
        "natural, legally accurate Sinhala. Preserve the meaning precisely -- "
        "this is a legal document. Respond with ONLY the Sinhala translation, "
        "no preamble, no explanation.\n\n"
        f"Text to translate:\n{text}"
    )
    return prompt

# Searches all 3 tiers (sentence, phrase, word) for the input text
# Builds the prompt using those matches
# Sends the prompt to Gemini
# Returns the generated Sinhala translation
def translate(text: str) -> str:
    """Translates a single piece of English text (sentence, paragraph, or chunk)
    into Sinhala, using retrieved vocabulary matches as translation-memory context."""
    sentence_matches = _search_tier(text, "sentence", config.TOP_K_SENTENCES)
    phrase_matches = _search_tier(text, "phrase", config.TOP_K_PHRASES)
    word_matches = _search_tier(text, "word", config.TOP_K_WORDS)

    prompt = build_prompt(text, sentence_matches, phrase_matches, word_matches)

    client = _get_gemini_client()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL_NAME, contents=prompt
    )
    return response.text.strip()
