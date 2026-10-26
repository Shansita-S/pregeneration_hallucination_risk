import re
from .schemas import QueryFeatures

UNCERTAINTY_WORDS = {
    "maybe", "possibly", "uncertain", "guess", "speculate", "unknown",
    "prove", "exactly", "always", "never", "guarantee", "why", "predict",
}
BROAD_HINTS = {
    "best", "latest", "current", "everyone", "all", "always", "never",
    "guaranteed", "predict", "future", "exactly",
}


def extract_features(question: str) -> QueryFeatures:
    text = question.strip()
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    # These are deliberately simple, explainable question-only features.
    # They are not a semantic detector and need validation on real labelled data.
    capitalized = re.findall(r"\b[A-Z][a-z]{2,}\b", text)
    uncertainty_count = sum(word in UNCERTAINTY_WORDS for word in words)
    broad_count = sum(word in BROAD_HINTS for word in words)
    return QueryFeatures(
        question_length=len(text),
        word_count=len(words),
        question_mark_count=text.count("?"),
        has_named_entity_hint=int(bool(capitalized)),
        uncertainty_word_count=uncertainty_count,
        broad_question_hint=int(broad_count > 0),
    )


def feature_vector(question: str) -> list[float]:
    f = extract_features(question)
    return [
        f.question_length, f.word_count, f.question_mark_count,
        f.has_named_entity_hint, f.uncertainty_word_count, f.broad_question_hint,
    ]
