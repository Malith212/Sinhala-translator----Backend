import re
from dataclasses import dataclass, field
from typing import List, Optional

from .vocab_index import VocabEntry, get_indices


@dataclass
class TermMatch:
    """Result of checking ONE required term against the generated output."""
    english: str
    required_sinhala: str
    found: bool
    position: Optional[int] = None   # character index where it was found
    context: Optional[str] = None    # a snippet of surrounding text, for manual review


@dataclass
class EnforcementReport:
    required_terms: List[VocabEntry] = field(default_factory=list)
    missing_after_first_pass: List[VocabEntry] = field(default_factory=list)
    missing_after_retry: List[VocabEntry] = field(default_factory=list)
    retried: bool = False

    @property
    def required_count(self) -> int:
        return len(self.required_terms)

    @property
    def satisfied_count_first_pass(self) -> int:
        return self.required_count - len(self.missing_after_first_pass)

    @property
    def satisfied_count_final(self) -> int:
        final_missing = self.missing_after_retry if self.retried else self.missing_after_first_pass
        return self.required_count - len(final_missing)


def _word_boundary_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)", re.IGNORECASE)


def detect_required_terms(source_text: str) -> List[VocabEntry]:
    """Finds every glossary term EXACTLY present in the source text.

    Longer (phrase-level) matches are checked first and mask out the text
    they cover, so a shorter word contained within an already-matched
    phrase isn't also flagged as a separate requirement (e.g. "data"
    inside an already-matched "personal data").
    """
    indices = get_indices()

    # Combine phrase + word tiers, longest English term first, so phrases
    # are matched before their component words.
    candidates: List[VocabEntry] = list(indices.phrase.meta) + list(indices.word.meta)
    candidates.sort(key=lambda e: len(e["english"]), reverse=True)

    working_text = source_text
    required: List[VocabEntry] = []
    seen_sinhala = set()

    for entry in candidates:
        term = entry["english"].strip()
        if not term:
            continue
        pattern = _word_boundary_pattern(term)
        match = pattern.search(working_text)
        if match:
            if entry["sinhala"] not in seen_sinhala:
                required.append(entry)
                seen_sinhala.add(entry["sinhala"])
            # Mask the matched span so shorter subsumed terms (e.g. "data"
            # inside an already-matched "personal data") aren't also flagged.
            working_text = (
                working_text[: match.start()]
                + (" " * (match.end() - match.start()))
                + working_text[match.end():]
            )

    return required


def verify_terms_detailed(
    translated_text: str, required_terms: List[VocabEntry]
) -> List[TermMatch]:
    """Checks each required term against the generated Sinhala output.

    This is a plain exact substring check on the SINHALA side only -- not
    English, not similarity-based. For each term it also records the exact
    character position and a snippet of surrounding text, so the match (or
    the absence of one) can be manually verified by a human reviewer
    rather than just trusted as a pass/fail flag.
    """
    results: List[TermMatch] = []
    for entry in required_terms:
        term = entry["sinhala"].strip()
        position = translated_text.find(term)

        if position == -1:
            results.append(
                TermMatch(
                    english=entry["english"],
                    required_sinhala=term,
                    found=False,
                )
            )
        else:
            start = max(0, position - 15)
            end = min(len(translated_text), position + len(term) + 15)
            context = translated_text[start:end]
            results.append(
                TermMatch(
                    english=entry["english"],
                    required_sinhala=term,
                    found=True,
                    position=position,
                    context=f"...{context}...",
                )
            )
    return results


def find_missing_terms(translated_text: str, required_terms: List[VocabEntry]) -> List[VocabEntry]:
    """Checks which required Sinhala terms are absent from the translated
    output. Thin wrapper around verify_terms_detailed() that returns just
    the missing glossary entries, in the shape EnforcementReport expects."""
    matches = verify_terms_detailed(translated_text, required_terms)
    missing_sinhala = {m.required_sinhala for m in matches if not m.found}
    return [entry for entry in required_terms if entry["sinhala"].strip() in missing_sinhala]


def build_enforcement_block(required_terms: List[VocabEntry]) -> str:
    if not required_terms:
        return ""
    lines = "\n".join(
        f'- "{e["english"]}" MUST be translated as "{e["sinhala"]}"' for e in required_terms
    )
    return (
        "\nREQUIRED TERMINOLOGY (non-negotiable -- these exact Sinhala terms "
        "MUST appear in your translation, not a paraphrase or synonym):\n"
        f"{lines}\n"
    )


def build_retry_correction_block(missing_terms: List[VocabEntry]) -> str:
    lines = "\n".join(
        f'- "{e["english"]}" was NOT translated as required. It MUST be "{e["sinhala"]}".'
        for e in missing_terms
    )
    return (
        "\nYour previous translation did not use the following required terms "
        "correctly. Regenerate the FULL translation, this time ensuring every "
        "required term below appears exactly as specified:\n"
        f"{lines}\n"
    )