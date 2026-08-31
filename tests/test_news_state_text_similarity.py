from market_agent.research.news_state.text_similarity import tfidf_similarity


def test_similar_wording_scores_high():
    a = "NVIDIA reports stronger-than-expected data-center revenue growth this quarter."
    b = "NVIDIA reported stronger than expected data center revenue growth this quarter."
    assert tfidf_similarity(a, b) > 0.6


def test_unrelated_text_scores_low():
    a = "NVIDIA reports stronger-than-expected data-center revenue growth this quarter."
    b = "The mayor announced a new public transit initiative downtown."
    assert tfidf_similarity(a, b) < 0.2


def test_shared_vocabulary_paraphrase_still_scores_meaningfully():
    a = "Revenue increased significantly this quarter for the company."
    b = "The company's revenue increased significantly this quarter."
    assert tfidf_similarity(a, b) > 0.8


def test_disclosed_limitation_different_words_same_meaning_scores_low():
    """This is the honest failure mode this module is disclosed to have -
    no semantic understanding, so a true paraphrase with different
    vocabulary does NOT score as similar."""
    a = "Revenue grew substantially."
    b = "Sales climbed considerably."
    assert tfidf_similarity(a, b) < 0.3


def test_single_text_input_does_not_crash():
    from market_agent.research.news_state.text_similarity import tfidf_pairwise_similarity
    assert tfidf_pairwise_similarity(["only one"]) == [[1.0]]


def test_empty_input_does_not_crash():
    from market_agent.research.news_state.text_similarity import tfidf_pairwise_similarity
    assert tfidf_pairwise_similarity([]) == []
