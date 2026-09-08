import re
from collections import Counter
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hannun.stance.cross_validation import (
    select_common_facts,
    sentence_similarity,
)
from hannun.stance.evidence import (
    build_issue_evidence,
    group_evidence_by_sentence,
)


UNCERTAIN_PATTERNS = (
    re.compile(r"(?:것|중)으로\s+(?:알려|전해)"),
    re.compile(r"것으로\s+(?:보인다|예상된다|전망된다)"),
    re.compile(r"가능성(?:이|도)?\s+(?:거론|제기)"),
    re.compile(r"관측(?:이|도)?\s+(?:나오|제기)"),
)

BOILERPLATE_PATTERNS = (
    re.compile(r"저작권자"),
    re.compile(r"무단\s*전재"),
    re.compile(r"재배포\s*금지"),
    re.compile(r"^(?:사진|영상|자료)\s*[=:]"),
    re.compile(r"^[가-힣A-Za-z·\s]{2,20}\s+기자$"),
    re.compile(r"\((?:왼쪽|오른쪽|가운데)\)"),
)

TOPIC_STOPWORDS = {
    "대통령",
    "정부",
    "관련",
    "논의",
    "발표",
    "대한",
    "대해",
    "이번",
    "이날",
    "했다",
    "한다",
    "밝혔다",
    "말했다",
    "전했다",
    "추진",
    "검토",
}


def _clean_display_token(text: str) -> str:
    return text.strip(
        " \t\r\n.,·:;()[]{}<>\"'“”‘’"
    )


def _has_unbalanced_quotes(sentence: str) -> bool:
    """열고 닫는 따옴표가 맞지 않는 불완전 문장을 판별한다."""

    quote_pairs = (
        ("“", "”"),
        ("‘", "’"),
    )

    return any(
        sentence.count(opening) != sentence.count(closing)
        for opening, closing in quote_pairs
    )


def _is_boilerplate_sentence(
    sentence: str,
    publisher_names: set[str] | None = None,
    min_chars: int = 6,
) -> bool:
    """출처명·기자명·저작권 문구처럼 사실이 아닌 문장을 제거한다."""

    cleaned = _clean_display_token(sentence)

    if len(re.sub(r"\s+", "", cleaned)) < min_chars:
        return True

    if publisher_names and cleaned in publisher_names:
        return True

    # 기사 본문 안에 다른 언론사 출처명이 단독으로 남은 경우도 제거한다.
    if re.fullmatch(
        r"[가-힣A-Za-z0-9·]{2,20}(?:뉴스|신문|일보|방송|통신)",
        cleaned,
    ):
        return True

    return any(
        pattern.search(cleaned)
        for pattern in BOILERPLATE_PATTERNS
    )


def _extract_issue_keywords(
    articles: list[dict[str, Any]],
) -> set[str]:
    """기사 제목에서 이슈를 반복적으로 설명하는 핵심어를 추출한다."""

    counts: Counter[str] = Counter()

    for article in articles:
        title = str(article.get("title") or "")

        tokens = {
            token.lower()
            for token in re.findall(
                r"[0-9A-Za-z가-힣]{2,}",
                title,
            )
            if token.lower() not in TOPIC_STOPWORDS
        }

        counts.update(tokens)

    if not counts:
        return set()

    repeated = {
        token
        for token, count in counts.items()
        if count >= 2
    }

    if repeated:
        return repeated

    # 제목이 매우 적어 반복 핵심어가 없으면 전체 제목 핵심어를 사용한다.
    return set(counts)


def _normalize_topic_text(text: str) -> str:
    return re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "",
        text,
    ).lower()


def _is_issue_relevant(
    sentence: str,
    issue_keywords: set[str],
) -> bool:
    """후보 문장이 현재 이슈의 핵심어를 충분히 포함하는지 확인한다."""

    if not issue_keywords:
        return True

    normalized_sentence = _normalize_topic_text(sentence)

    matched_keywords = {
        keyword
        for keyword in issue_keywords
        if _normalize_topic_text(keyword)
        in normalized_sentence
    }

    required_matches = min(
        2,
        len(issue_keywords),
    )

    return len(matched_keywords) >= required_matches

def _is_uncertain_sentence(sentence: str) -> bool:
    """확정 사실이 아닌 전망·추측성 문장인지 판단한다."""

    return any(
        pattern.search(sentence)
        for pattern in UNCERTAIN_PATTERNS
    )


def _is_redundant_sentence(
    sentence: str,
    selected_sentences: list[str],
    threshold: float = 0.65,
) -> bool:
    """이미 선택한 문장과 같은 사실을 반복하는지 판단한다."""

    return any(
        sentence_similarity(sentence, selected) >= threshold
        for selected in selected_sentences
    )


def _lexrank_scores(
    sentences: list[str],
    threshold: float = 0.1,
    damping: float = 0.85,
    max_iter: int = 100,
    tolerance: float = 1e-6,
) -> list[float]:
    """문장 간 TF-IDF 코사인 유사도로 LexRank 점수를 계산한다."""

    sentence_count = len(sentences)

    if sentence_count == 0:
        return []

    if sentence_count == 1:
        return [1.0]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
    )

    try:
        matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return [1.0 / sentence_count] * sentence_count

    similarities = cosine_similarity(matrix)

    graph = [
        [0.0] * sentence_count
        for _ in range(sentence_count)
    ]

    for source_index in range(sentence_count):
        for target_index in range(sentence_count):
            if source_index == target_index:
                continue

            similarity = float(
                similarities[source_index, target_index]
            )

            if similarity >= threshold:
                graph[source_index][target_index] = similarity

    scores = [1.0 / sentence_count] * sentence_count

    for _ in range(max_iter):
        next_scores = [
            (1.0 - damping) / sentence_count
            for _ in range(sentence_count)
        ]

        for source_index in range(sentence_count):
            edge_sum = sum(graph[source_index])

            if edge_sum == 0.0:
                share = (
                    damping
                    * scores[source_index]
                    / sentence_count
                )

                for target_index in range(sentence_count):
                    next_scores[target_index] += share

                continue

            for target_index in range(sentence_count):
                weight = graph[source_index][target_index]

                if weight == 0.0:
                    continue

                next_scores[target_index] += (
                    damping
                    * scores[source_index]
                    * weight
                    / edge_sum
                )

        difference = sum(
            abs(next_scores[index] - scores[index])
            for index in range(sentence_count)
        )

        scores = next_scores

        if difference < tolerance:
            break

    return scores


def generate_fact_summary(
    issue_data: dict[str, Any],
    limit: int = 3,
) -> list[str]:
    """이슈 기사에서 교차 검증된 공통 사실 브리핑을 생성한다."""

    if limit <= 0:
        return []

    articles = issue_data.get("articles", [])

    if not articles:
        return []

    sentence_evidence = build_issue_evidence(articles)

    if not sentence_evidence:
        return []

    grouped_evidence = group_evidence_by_sentence(
        sentence_evidence
    )

    common_facts = select_common_facts(
        grouped_evidence,
        limit=len(grouped_evidence),
    )

    publisher_names = {
        str(article.get("publisher_name") or "").strip()
        for article in articles
        if article.get("publisher_name")
    }

    issue_keywords = _extract_issue_keywords(
        articles
    )

    sentences = [
        evidence["sentence"]
        for evidence in common_facts
        if not _is_uncertain_sentence(
            evidence["sentence"]
        )
        and not _has_unbalanced_quotes(
            evidence["sentence"]
        )
        and not _is_boilerplate_sentence(
            evidence["sentence"],
            publisher_names,
        )
        and _is_issue_relevant(
            evidence["sentence"],
            issue_keywords,
        )
    ]

    if not sentences:
        return []

    scores = _lexrank_scores(sentences)

    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = []

    for index in ranked_indices:
        sentence = sentences[index]

        if _is_redundant_sentence(
            sentence,
            selected,
        ):
            continue

        selected.append(sentence)

        if len(selected) >= limit:
            break

    return selected
