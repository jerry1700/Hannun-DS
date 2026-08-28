from .sentence_filter import (
    filter_opinion_sentences,
    split_sentences,
)


def preprocess_article(content: str) -> list[str]:
    """기사 본문을 문장 단위로 정리해 반환한다."""

    sentences = split_sentences(content)
    return filter_opinion_sentences(sentences)