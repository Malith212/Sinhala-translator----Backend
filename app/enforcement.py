"""
Term Enforcement Module

Addresses a specific weakness of plain retrieval-augmented translation: the
retrieved glossary matches are only *suggestions* in the prompt, so the LLM
is free to ignore them. There is no guarantee the correct terminology
actually appears in the output.

This module makes glossary usage a verified, enforced constraint instead
of a hope:

1. DETECT   -- scan the *source* English text for exact glossary terms
               (phrase-level entries take priority over word-level ones,
               so "personal data" is required as a unit rather than
               separately requiring "personal" and "data").
2. LOCK     -- inject those terms into the prompt as non-negotiable
               requirements, not just similar examples.
3. VERIFY   -- after generation, check whether each required Sinhala term
               actually appears in the output.
4. RETRY    -- if any required term is missing, regenerate once with an
               explicit corrective note naming exactly which terms were
               missed and what they must be, then verify again.

The result of every translation includes an enforcement report (terms
required, terms satisfied, retry count) -- this is the data an ablation
study needs to demonstrate the glossary is doing real work, rather than
just being retrieved and ignored.
"""

import re
from dataclasses import dataclass, field
from typing import List

from .vocab_index import VocabEntry, get_indices


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
    """Finds every glossary term present in the source text. Longer
    (phrase-level) matches are found first and mask out the text they
    cover, so a shorter word contained within an already-matched phrase
    isn't also flagged as a separate requirement."""
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
            working_text = working_text[: match.start()] + (" " * (match.end() - match.start())) + working_text[match.end() :]

    return required


def find_missing_terms(translated_text: str, required_terms: List[VocabEntry]) -> List[VocabEntry]:
    """Checks which required Sinhala terms are absent from the translated output."""
    missing = []
    for entry in required_terms:
        if entry["sinhala"].strip() not in translated_text:
            missing.append(entry)
    return missing


def build_enforcement_block(required_terms: List[VocabEntry]) -> str:
    if not required_terms:
        return ""
    lines = "\n".join(f'- "{e["english"]}" MUST be translated as "{e["sinhala"]}"' for e in required_terms)
    return (
        "\nREQUIRED TERMINOLOGY (non-negotiable -- these exact Sinhala terms "
        "MUST appear in your translation, not a paraphrase or synonym):\n"
        f"{lines}\n"
    )


def build_retry_correction_block(missing_terms: List[VocabEntry]) -> str:
    lines = "\n".join(f'- "{e["english"]}" was NOT translated as required. It MUST be "{e["sinhala"]}".' for e in missing_terms)
    return (
        "\nYour previous translation did not use the following required terms "
        "correctly. Regenerate the FULL translation, this time ensuring every "
        "required term below appears exactly as specified:\n"
        f"{lines}\n"
    )