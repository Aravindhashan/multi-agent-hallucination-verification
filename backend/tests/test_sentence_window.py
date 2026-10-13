"""Unit and integration tests for claim-aware sentence window extraction."""

import pytest
import torch

from app.services.nli_config import NLISettings
from app.services.nli_engine import NLIEngine
from app.services.nli_models import NLILabel
from app.services.sentence_window import (
    SentenceWindow,
    score_sentence,
    select_best_sentence_window,
    split_into_sentences,
)
from tests.test_nli_engine import MockModel, MockTokenizer


# ---------------------------------------------------------------------------
# 1. Sentence Boundary Splitting Tests
# ---------------------------------------------------------------------------

def test_split_into_sentences_empty_and_whitespace():
    assert split_into_sentences("") == []
    assert split_into_sentences("   \n\t  ") == []
    assert split_into_sentences(None) == []  # type: ignore


def test_split_into_sentences_abbreviations_and_numbers():
    raw_text = (
        "Dr. Jane Smith visited the U.S. in Jan. 1969. "
        "The project cost approx. $3.14 million. "
        "Prof. Adams from Univ. of Oxford confirmed the data."
    )
    sents = split_into_sentences(raw_text)
    assert len(sents) == 3
    assert sents[0] == "Dr. Jane Smith visited the U.S. in Jan. 1969."
    assert sents[1] == "The project cost approx. $3.14 million."
    assert sents[2] == "Prof. Adams from Univ. of Oxford confirmed the data."


def test_split_into_sentences_punctuation_and_quotes():
    raw_text = (
        'Apollo 11 reached orbit. '
        'Armstrong announced, "The Eagle has landed!" '
        'Was the mission televised live? '
        'Yes, it was broadcast globally.'
    )
    sents = split_into_sentences(raw_text)
    assert len(sents) == 4
    assert sents[0] == "Apollo 11 reached orbit."
    assert 'Armstrong announced, "The Eagle has landed!"' in sents[1]
    assert sents[2] == "Was the mission televised live?"
    assert sents[3] == "Yes, it was broadcast globally."


def test_split_into_sentences_single_sentence():
    raw = "Angela Merkel served as Chancellor of Germany from 2005 to 2021."
    sents = split_into_sentences(raw)
    assert len(sents) == 1
    assert sents[0] == raw


# ---------------------------------------------------------------------------
# 2. Claim-Sentence Relevance Scoring Tests
# ---------------------------------------------------------------------------

def test_score_sentence_empty():
    assert score_sentence("", "Valid sentence.") == 0.0
    assert score_sentence("Valid claim.", "") == 0.0


def test_score_sentence_feature_weighting():
    claim = "Alexander Fleming discovered penicillin in 1928."

    sent_exact = "In 1928, Alexander Fleming discovered penicillin at St Mary Hospital."
    sent_partial = "Alexander Fleming was a Scottish biologist."
    sent_unrelated = "The weather in London was unusually foggy that autumn."

    score_exact = score_sentence(claim, sent_exact)
    score_partial = score_sentence(claim, sent_partial)
    score_unrelated = score_sentence(claim, sent_unrelated)

    assert score_exact > score_partial
    assert score_partial > score_unrelated
    assert score_unrelated == 0.0


def test_score_sentence_numbers_and_entities_boost():
    claim = "Apollo 11 landed on the Moon in July 1969."
    sent_with_year = "In July 1969, three astronauts launched towards the Moon."
    sent_without_year = "Three astronauts launched towards the Moon."

    score_year = score_sentence(claim, sent_with_year)
    score_no_year = score_sentence(claim, sent_without_year)

    assert score_year > score_no_year


# ---------------------------------------------------------------------------
# 3. Sentence Window Selection Tests
# ---------------------------------------------------------------------------

def test_select_window_short_passage_returns_all():
    short_passage = "Sentence one. Sentence two."
    claim = "Any claim."
    window = select_best_sentence_window(claim, short_passage, max_sentences=3)

    assert window.sentence_count == 2
    assert window.total_passage_sentences == 2
    assert window.sentence_indices == [0, 1]
    assert not window.is_truncated_window


def test_select_window_contiguous_strategy():
    passage = (
        "The Apollo program was conceived during the Eisenhower administration. "
        "NASA initiated plans for manned circumlunar flights. "
        "Apollo 11 was the spaceflight that landed humans on the Moon on July 20, 1969. "
        "Commander Neil Armstrong and Buzz Aldrin landed the lunar module Eagle. "
        "Six hours later, Armstrong became the first person to step on the Moon. "
        "Aldrin joined him 19 minutes later. "
        "The astronauts returned safely to Earth on July 24."
    )
    claim = "Neil Armstrong stepped onto the lunar surface during Apollo 11 in 1969."

    window = select_best_sentence_window(claim, passage, max_sentences=3, strategy="contiguous")

    assert 1 <= window.sentence_count <= 3
    assert window.total_passage_sentences == 7
    assert window.is_truncated_window
    assert "Armstrong" in window.text
    assert "Apollo 11" in window.text or "first person" in window.text


def test_select_window_top_k_non_contiguous():
    passage = (
        "Marie Curie won a Nobel Prize in Physics in 1903. "
        "She was born in Warsaw, Poland. "
        "She studied at the University of Paris. "
        "She later won a second Nobel Prize in Chemistry in 1911. "
        "She died in 1934."
    )
    claim = "Marie Curie won Nobel Prizes in both 1903 and 1911."

    window = select_best_sentence_window(claim, passage, max_sentences=2, strategy="top_k")

    assert window.sentence_indices == [0, 3]
    assert "1903" in window.text
    assert "1911" in window.text


def test_select_window_hybrid_picks_best_candidate():
    passage = (
        "Polymer chemistry developed rapidly in the twentieth century. "
        "Laboratories across Europe experimented with cellulose. "
        "In 1907, Leo Baekeland synthesized Bakelite in New York. "
        "Bakelite was the world's first completely synthetic plastic. "
        "It was widely used in electrical insulators and radio casings. "
        "Later plastics replaced it in modern manufacturing."
    )
    claim = "Leo Baekeland synthesized Bakelite in 1907 as the first synthetic plastic."

    window = select_best_sentence_window(claim, passage, max_sentences=3, strategy="hybrid")

    assert 1 <= window.sentence_count <= 3
    assert "Leo Baekeland" in window.text
    assert "1907" in window.text
    assert "Bakelite" in window.text


# ---------------------------------------------------------------------------
# 4. NLIEngine Integration with Sentence Windows
# ---------------------------------------------------------------------------

def test_nli_engine_uses_sentence_windows_when_enabled():
    mock_engine = NLIEngine(
        config=NLISettings(
            enable_sentence_windows=True,
            max_window_sentences=2,
            truncation_policy="truncate",
        ),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )

    long_passage = (
        "Paragraph introductory sentence with irrelevant context. "
        "Second sentence discussing background history. "
        "Alexander Fleming discovered penicillin in 1928 at St Mary Hospital. "
        "Penicillin became the first mass-produced antibiotic. "
        "Fleming was awarded the Nobel Prize in 1945."
    )
    claim = "Alexander Fleming discovered penicillin in 1928."

    result = mock_engine.predict(claim, long_passage)

    assert result.selected_sentence_indices is not None
    assert len(result.selected_sentence_indices) <= 2
    assert "Fleming discovered penicillin in 1928" in result.premise_text
    assert result.evidence == long_passage
    assert result.raw_token_length is not None
    assert result.token_length is not None
    assert result.token_length <= result.raw_token_length


def test_nli_engine_disables_sentence_windows_when_configured():
    mock_engine = NLIEngine(
        config=NLISettings(
            enable_sentence_windows=False,
            truncation_policy="truncate",
        ),
        model=MockModel(),
        tokenizer=MockTokenizer(),
    )

    passage = (
        "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5."
    )
    claim = "Claim"

    result = mock_engine.predict(claim, passage)

    assert result.selected_sentence_indices is None
    assert result.premise_text == passage
