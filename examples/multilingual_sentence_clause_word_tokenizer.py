from __future__ import annotations

from dataclasses import dataclass
import re

from langdetect import DetectorFactory, LangDetectException, detect
import spacy


DetectorFactory.seed = 42


DUTCH_MARKERS = {
    "aan",
    "als",
    "bij",
    "dat",
    "de",
    "een",
    "en",
    "factuur",
    "geen",
    "het",
    "ik",
    "inloggen",
    "kan",
    "klantenservice",
    "klopt",
    "mijn",
    "moet",
    "niet",
    "op",
    "reageerde",
    "te",
    "van",
    "voor",
    "wachten",
    "werkt",
    "wil",
    "drinken",
    "eten",
    "zegt",
}

ENGLISH_MARKERS = {
    "and",
    "app",
    "at",
    "checkout",
    "crashes",
    "crashing",
    "delivery",
    "failed",
    "keeps",
    "late",
    "missing",
    "not",
    "order",
    "payment",
    "processed",
    "product",
    "refund",
    "slow",
    "support",
    "the",
    "is",
    "was",
}

SUBJECT_LIKE = {
    "de",
    "een",
    "het",
    "i",
    "ik",
    "mijn",
    "my",
    "payment",
    "refund",
    "the",
}

VERB_OR_ISSUE_LIKE = {
    "crashes",
    "crashing",
    "failed",
    "is",
    "kan",
    "keeps",
    "klopt",
    "late",
    "missing",
    "moet",
    "not",
    "niet",
    "processed",
    "reageerde",
    "slow",
    "was",
    "werkt",
    "zegt",
}

PROTECTED_PHRASES = {
    "black and decker",
    "sales en marketing",
    "terms and conditions",
}


@dataclass(frozen=True)
class WordResult:
    text: str
    language: str


@dataclass(frozen=True)
class ClauseResult:
    text: str
    language: str
    words: list[WordResult]


@dataclass(frozen=True)
class SentenceResult:
    text: str
    language: str
    clauses: list[ClauseResult]


def clean_token(text: str) -> str:
    return text.lower().strip(".,!?;:\"'()[]{}")


def simple_words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def detect_language(text: str, fallback: str = "unknown") -> str:
    words = {clean_token(word) for word in simple_words(text)}
    dutch_score = len(words & DUTCH_MARKERS)
    english_score = len(words & ENGLISH_MARKERS)

    if dutch_score > english_score:
        return "nl"
    if english_score > dutch_score:
        return "en"

    try:
        language = detect(text).split("-")[0]
    except LangDetectException:
        return fallback

    return language if language in {"nl", "en"} else fallback


def detect_word_language(word: str) -> str:
    cleaned = clean_token(word)

    if not cleaned:
        return "punct"
    if cleaned in DUTCH_MARKERS:
        return "nl"
    if cleaned in ENGLISH_MARKERS:
        return "en"

    # Single-word language detection is noisy. Only use it for longer words.
    if len(cleaned) >= 5:
        return detect_language(cleaned, fallback="unknown")

    return "unknown"


def looks_like_independent_clause(words: list[str]) -> bool:
    normalized = {clean_token(word) for word in words if clean_token(word)}

    if len(normalized) < 2:
        return False

    has_subject = bool(normalized & SUBJECT_LIKE)
    has_verb_or_issue = bool(normalized & VERB_OR_ISSUE_LIKE)

    return has_subject and has_verb_or_issue


def is_protected_connector(words: list[str], connector_index: int) -> bool:
    left = max(0, connector_index - 3)
    right = min(len(words), connector_index + 4)
    window = " ".join(clean_token(word) for word in words[left:right])

    return any(phrase in window for phrase in PROTECTED_PHRASES)


def split_clauses(sentence: str) -> list[str]:
    words = simple_words(sentence)
    split_indexes: list[int] = []

    for index, word in enumerate(words):
        if clean_token(word) not in {"and", "en"}:
            continue
        if is_protected_connector(words, index):
            continue

        left_words = words[:index]
        right_words = words[index + 1 :]

        if looks_like_independent_clause(left_words) and looks_like_independent_clause(
            right_words
        ):
            split_indexes.append(index)

    if not split_indexes:
        return [sentence.strip()]

    clauses: list[str] = []
    start = 0
    for split_index in split_indexes:
        clauses.append(" ".join(words[start:split_index]).strip())
        start = split_index + 1
    clauses.append(" ".join(words[start:]).strip())

    return [clause for clause in clauses if clause]


def build_pipeline() -> spacy.Language:
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")
    return nlp


def tokenize_multilingual(text: str) -> list[SentenceResult]:
    nlp = build_pipeline()
    doc = nlp(text)

    sentence_results: list[SentenceResult] = []
    for sentence in doc.sents:
        sentence_text = sentence.text.strip()
        clauses: list[ClauseResult] = []

        for clause_text in split_clauses(sentence_text):
            clause_doc = nlp.make_doc(clause_text)
            words = [
                WordResult(token.text, detect_word_language(token.text))
                for token in clause_doc
                if not token.is_space
            ]
            clauses.append(
                ClauseResult(
                    text=clause_text,
                    language=detect_language(clause_text),
                    words=words,
                )
            )

        sentence_results.append(
            SentenceResult(
                text=sentence_text,
                language=detect_language(sentence_text),
                clauses=clauses,
            )
        )

    return sentence_results


def print_results(text: str) -> None:
    print(f"\nTEXT: {text}")
    for sentence_index, sentence in enumerate(tokenize_multilingual(text), start=1):
        print(f"  S{sentence_index} [{sentence.language}]: {sentence.text}")
        for clause_index, clause in enumerate(sentence.clauses, start=1):
            print(f"    C{clause_index} [{clause.language}]: {clause.text}")
            word_output = ", ".join(
                f"{word.text}/{word.language}" for word in clause.words
            )
            print(f"      W: {word_output}")


def main() -> None:
    examples = [
        "Ik kan niet inloggen en checkout keeps crashing at payment.",
        "Refund was not processed en de factuur klopt niet.",
        "Ik wil drinken en eten.",
        "Black and Decker product is missing.",
        "The app is slow and payment failed.",
        "Sales en marketing dashboard werkt niet.",
    ]

    for text in examples:
        print_results(text)


if __name__ == "__main__":
    main()
