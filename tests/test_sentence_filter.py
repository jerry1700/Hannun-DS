from hannun.stance.pipeline import preprocess_article
from hannun.stance.sentence_filter import (
    filter_opinion_sentences,
    split_sentences,
)


def test_split_sentences():
    text = "정부가 정책을 발표했다. 야당은 반발했다! 추가 논의가 필요할까?"

    result = split_sentences(text)

    assert result == [
        "정부가 정책을 발표했다.",
        "야당은 반발했다!",
        "추가 논의가 필요할까?",
    ]


def test_filter_opinion_sentences():
    sentences = [
        "정부는 오늘 정책을 발표했다.",
        "추가 지원이 필요하다.",
        "관련 예산은 100억 원이다.",
    ]

    result = filter_opinion_sentences(sentences)

    assert result == [
        "정부는 오늘 정책을 발표했다.",
        "관련 예산은 100억 원이다.",
    ]


def test_preprocess_article():
    text = (
        "정부는 오늘 정책을 발표했다. "
        "추가 지원이 필요하다. "
        "관련 예산은 100억 원이다."
    )

    result = preprocess_article(text)

    assert result == [
        "정부는 오늘 정책을 발표했다.",
        "관련 예산은 100억 원이다.",
    ]


def test_empty_content():
    assert preprocess_article("") == []

def test_reported_speech_is_preserved():
    sentences = [
        "추가 지원이 필요하다.",
        '야당은 "추가 지원이 필요하다"고 밝혔다.',
        "관련 예산은 100억 원이다.",
    ]

    result = filter_opinion_sentences(sentences)

    assert result == [
        '야당은 "추가 지원이 필요하다"고 밝혔다.',
        "관련 예산은 100억 원이다.",
    ]

def test_split_sentences_without_space_after_period():
    text = (
        "말라리아 환자가 급증하고 있다."
        "질병청은 경보를 발령했다."
        "주민들에게 주의를 당부했다."
    )

    result = split_sentences(text)

    assert result == [
        "말라리아 환자가 급증하고 있다.",
        "질병청은 경보를 발령했다.",
        "주민들에게 주의를 당부했다.",
    ]


def test_split_sentences_keeps_decimal_number():
    text = (
        "환자 수는 지난해의 3.3배로 늘었다."
        "질병청은 상황을 점검하고 있다."
    )

    result = split_sentences(text)

    assert result == [
        "환자 수는 지난해의 3.3배로 늘었다.",
        "질병청은 상황을 점검하고 있다.",
    ]
