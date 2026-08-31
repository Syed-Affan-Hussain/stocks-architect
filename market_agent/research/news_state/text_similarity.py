"""A disclosed, NON-semantic paraphrase-similarity proxy.

No sentence-embedding model or embeddings API is available in this
environment (no sentence-transformers/torch installed, no embeddings
client configured - see the feasibility discussion this design started
from). TF-IDF cosine similarity (scikit-learn, already a project
dependency) is a real, honest, bag-of-words distance: it will correctly
recognize that two articles sharing most of their vocabulary describe the
same story, but it has NO notion of meaning - "revenue grew" and "sales
increased" share almost no vocabulary and will NOT score as similar here,
even though they are near-paraphrases economically. This is exactly why
the structured EventVector (event_vector.py) is the primary
representation and this module is a supplementary, shallow signal, never
a substitute for it - see the validation report's Experiment D for where
this module succeeds and where it visibly fails.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_pairwise_similarity(texts: list[str]) -> list[list[float]]:
    """Symmetric matrix of TF-IDF cosine similarities, texts[i] vs
    texts[j] - values in [0, 1]. Requires at least 2 non-empty texts;
    returns an all-1.0-diagonal, all-0.0-elsewhere matrix for a
    degenerate (all-identical-vocabulary or single-text) input rather
    than raising."""
    if len(texts) < 2:
        return [[1.0] * len(texts)] if texts else []
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        # every text was pure stopwords/empty after vectorization - no vocabulary to compare
        n = len(texts)
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    sim = cosine_similarity(matrix)
    return sim.tolist()


def tfidf_similarity(text_a: str, text_b: str) -> float:
    matrix = tfidf_pairwise_similarity([text_a, text_b])
    return float(matrix[0][1])
