import re
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


NEWS_PREFIX_PATTERN = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*)+"
)

EDITORIAL_MODIFIER_PATTERN = re.compile(
    r"(?<!\w)(?:전격|잔혹|엽기)(?!\w)"
)


def _clean_title(title: str) -> str:
    """기사 제목의 첫 유효 줄을 사용하고 뉴스 머리표현을 제거한다."""

    first_line = next(
        (
            line.strip()
            for line in title.splitlines()
            if line.strip()
            and line.strip().upper().rstrip(":").strip() != "LIVE"
        ),
        "",
    )

    cleaned = NEWS_PREFIX_PATTERN.sub("", first_line).strip()
    cleaned = EDITORIAL_MODIFIER_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned


def _get_titles(issue_data: dict[str, Any]) -> list[str]:
    """이슈에 포함된 유효한 기사 제목을 수집한다."""

    titles = []

    for article in issue_data.get("articles", []):
        title = article.get("title")

        if isinstance(title, str) and title.strip():
            cleaned = _clean_title(title)

            if cleaned:
                titles.append(cleaned)

    return titles


def _select_central_title(titles: list[str]) -> str:
    """TF-IDF 제목 유사도 기준으로 가장 중앙적인 제목을 선택한다."""

    if len(titles) == 1:
        return titles[0]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
    )

    matrix = vectorizer.fit_transform(titles)
    similarities = cosine_similarity(matrix)

    centrality = similarities.mean(axis=1)
    best_index = int(centrality.argmax())

    return titles[best_index]


def generate_event_name(issue_data: dict[str, Any]) -> str:
    """DS1 이슈 데이터를 기반으로 중립적인 event_name을 생성한다."""

    titles = _get_titles(issue_data)

    if titles:
        return _select_central_title(titles)

    representative_title = issue_data.get("representative_title", "")

    if isinstance(representative_title, str) and representative_title.strip():
        return _clean_title(representative_title)

    return ""
