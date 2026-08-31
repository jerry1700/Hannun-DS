import re

from .sentence_filter import REPORTING_MARKERS, split_sentences


def extract_quotes(text: str) -> list[str]:
    """본문에서 큰따옴표로 감싼 인용구를 추출한다."""

    if not text:
        return []

    matches = re.finditer(
        r'"([^"]+)"|“([^”]+)”',
        text,
    )

    return [
        (match.group(1) or match.group(2)).strip()
        for match in matches
        if (match.group(1) or match.group(2)).strip()
    ]


def parse_quote_sentences(text: str) -> list[dict]:
    """본문에서 인용·발언 문장을 찾아 후속 분석용으로 구조화한다."""

    results = []

    for sentence in split_sentences(text):
        quotes = extract_quotes(sentence)

        if quotes:
            speech_type = "direct_quote"
        elif any(marker in sentence for marker in REPORTING_MARKERS):
            speech_type = "reported_speech"
        else:
            continue

        results.append(
            {
                "sentence": sentence,
                "quotes": quotes,
                "speech_type": speech_type,
            }
        )

    return results
