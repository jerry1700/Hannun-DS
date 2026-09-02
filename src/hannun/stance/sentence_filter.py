import re


OPINION_MARKERS = (
    "것으로 보인다",
    "필요하다",
    "해야 한다",
    "바람직하다",
    "문제다",
    "우려된다",
    "아쉽다",
)

REPORTING_MARKERS = (
    "라고 밝혔다",
    "고 밝혔다",
    "라고 말했다",
    "고 말했다",
    "라고 전했다",
    "고 전했다",
    "라고 설명했다",
    "고 설명했다",
    "라고 주장했다",
    "고 주장했다",
    "라고 강조했다",
    "고 강조했다",
)


def split_sentences(text: str) -> list[str]:
    """기사 본문을 문장 단위로 분리한다."""

    if not text or not text.strip():
        return []

    sentences = re.split(
        r"(?<=[.!?。！？])(?:\s+|(?=[가-힣]))|\n+",
        text.strip(),
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def is_reported_speech(sentence: str) -> bool:
    """인용 또는 발언 전달 문장인지 판단한다."""

    has_quote = '"' in sentence or "'" in sentence

    has_reporting_marker = any(
        marker in sentence
        for marker in REPORTING_MARKERS
    )

    return has_quote or has_reporting_marker


def is_opinion_sentence(sentence: str) -> bool:
    """제거 대상 의견성 문장인지 판단한다."""

    if is_reported_speech(sentence):
        return False

    return any(
        marker in sentence
        for marker in OPINION_MARKERS
    )


def filter_opinion_sentences(sentences: list[str]) -> list[str]:
    """의견성 문장을 제외한다."""

    return [
        sentence
        for sentence in sentences
        if not is_opinion_sentence(sentence)
    ]
