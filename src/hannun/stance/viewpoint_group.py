"""같은 stance 안에서 기사별 세부 견해 그룹을 생성한다."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


VALID_STANCES = {
    "positive",
    "neutral",
    "negative",
}

GENERIC_WORDS = {
    "비판",
    "우려",
    "평가",
    "지적",
    "분석",
    "주장",
    "입장",
    "관련",
    "대해",
    "대한",
    "있다",
    "없다",
    "하다",
    "된다",
    "됐다",
    "나왔다",
    "제기됐다",
    "따른",
    "가능성",
    "평가다",
    "입장이다",
    "지적이다",
    "우려다",
    "비판이다",
}

PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "께서",
    "까지",
    "부터",
    "처럼",
    "보다",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
    "에",
    "로",
    "와",
    "과",
    "도",
    "만",
)

MAX_SUBCLUSTERS = 3
MAX_LABEL_CHARS = 60
PAIR_SIMILARITY_THRESHOLD = 0.25


def _first_sentence(text: str) -> str:
    """여러 evidence가 이어진 경우 첫 번째 문장을 사용한다."""

    cleaned = " ".join(str(text or "").split())

    if not cleaned:
        return ""

    return re.split(
        r"(?<=[.!?])\s+",
        cleaned,
        maxsplit=1,
    )[0]


def _strip_reporting_tail(text: str) -> str:
    """라벨에 필요 없는 보도 표현을 제거한다."""

    cleaned = " ".join(str(text or "").split())

    patterns = (
        r"(?:라는|다는|이라고|라고)\s*"
        r"(?:비판|우려|평가|지적|분석|주장|입장)"
        r"(?:이|가)?"
        r"(?:\s*(?:나왔다|제기됐다|이어졌다|있다|이다|전해졌다))?"
        r"[.!?]*$",
        r"\s*(?:비판|우려|평가|지적|분석|주장|입장)"
        r"(?:이|가)?\s*(?:나왔다|제기됐다|있다|이다)?[.!?]*$",
    )

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
        )

    return cleaned.strip(" .!?")


def _normalize_for_similarity(
    text: str,
    target: str = "",
) -> str:
    """서브클러스터링용 텍스트를 토큰 단위로 정규화한다."""

    cleaned = _strip_reporting_tail(text)

    target_tokens = {
        _normalize_label_token(token)
        for token in re.findall(
            r"[0-9A-Za-z가-힣]+",
            target,
        )
    }

    normalized_tokens = []

    for raw_token in re.findall(
        r"[0-9A-Za-z가-힣]+",
        cleaned,
    ):
        token = _normalize_label_token(raw_token)

        if len(token) < 2:
            continue

        if token in GENERIC_WORDS:
            continue

        if token in target_tokens:
            continue

        normalized_tokens.append(token)

    return " ".join(normalized_tokens)


def _strip_particle(token: str) -> str:
    """간단한 조사 제거로 키워드 표현을 정리한다."""

    for suffix in PARTICLE_SUFFIXES:
        if (
            token.endswith(suffix)
            and len(token) - len(suffix) >= 2
        ):
            return token[:-len(suffix)]

    return token


def _normalize_label_token(token: str) -> str:
    """라벨용 토큰의 조사와 간단한 서술형 어미를 정리한다."""

    if token.endswith("이므로") and len(token) > 3:
        token = token[:-3]

    token = _strip_particle(token)

    if token.endswith("하다") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("하는") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("할") and len(token) > 1:
        token = token[:-1]
    elif token.endswith("하") and len(token) > 1:
        token = token[:-1]

    return token


def _label_terms(
    text: str,
    target: str = "",
) -> list[str]:
    """라벨 후보용 unigram/bigram을 만든다."""

    raw_tokens = re.findall(
        r"[0-9A-Za-z가-힣]+",
        _strip_reporting_tail(text),
    )

    target_tokens = {
        _normalize_label_token(token)
        for token in re.findall(
            r"[0-9A-Za-z가-힣]+",
            target,
        )
    }

    tokens = []

    for raw_token in raw_tokens:
        token = _normalize_label_token(raw_token)

        if len(token) < 2:
            continue

        if token in GENERIC_WORDS:
            continue

        if token in target_tokens:
            continue

        tokens.append(token)

    terms = list(tokens)

    terms.extend(
        f"{tokens[index]} {tokens[index + 1]}"
        for index in range(len(tokens) - 1)
    )

    return terms


def _tfidf_matrix(texts: list[str]):
    """char TF-IDF 행렬을 생성한다."""

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
    )

    try:
        return vectorizer.fit_transform(texts)
    except ValueError:
        return None


def _cluster_two_articles(
    texts: list[str],
) -> list[list[int]]:
    """기사 2건은 유사도에 따라 한 그룹 또는 두 그룹으로 나눈다."""

    matrix = _tfidf_matrix(texts)

    if matrix is None:
        return [[0], [1]]

    similarity = float(
        cosine_similarity(matrix)[0, 1]
    )

    if similarity >= PAIR_SIMILARITY_THRESHOLD:
        return [[0, 1]]

    return [[0], [1]]


def _cluster_many_articles(
    texts: list[str],
) -> list[list[int]]:
    """기사 3건 이상은 2~3개 후보 중 가장 적합한 k를 선택한다."""

    matrix = _tfidf_matrix(texts)

    if matrix is None:
        return [
            [index]
            for index in range(len(texts))
        ]

    unique_count = len(set(texts))

    if unique_count == 1:
        return [
            list(range(len(texts)))
        ]

    max_k = min(
        MAX_SUBCLUSTERS,
        len(texts) - 1,
        unique_count,
    )

    candidate_ks = [
        k
        for k in (2, 3)
        if k <= max_k
    ]

    if not candidate_ks:
        return [
            list(range(len(texts)))
        ]

    best_labels = None
    best_score = None

    for k in candidate_ks:
        model = KMeans(
            n_clusters=k,
            random_state=0,
            n_init=10,
        )

        labels = model.fit_predict(matrix)

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            matrix,
            labels,
            metric="cosine",
        )

        if (
            best_score is None
            or score > best_score
        ):
            best_score = score
            best_labels = labels

    if best_labels is None:
        return [
            list(range(len(texts)))
        ]

    groups: dict[int, list[int]] = {}

    for index, label in enumerate(best_labels):
        groups.setdefault(
            int(label),
            [],
        ).append(index)

    return list(groups.values())


def _cluster_texts(
    texts: list[str],
) -> list[list[int]]:
    """한 stance 안의 evidence를 세부 견해 그룹으로 나눈다."""

    if not texts:
        return []

    if len(texts) == 1:
        return [[0]]

    if len(texts) == 2:
        return _cluster_two_articles(texts)

    return _cluster_many_articles(texts)


def _fallback_label(
    articles: list[dict[str, Any]],
) -> str:
    """키워드 라벨 생성 실패 시 대표 evidence를 짧게 사용한다."""

    claim = _strip_reporting_tail(
        _first_sentence(
            articles[0]["content"]
        )
    )

    if len(claim) <= MAX_LABEL_CHARS:
        return claim

    return (
        claim[:MAX_LABEL_CHARS]
        .rstrip()
        + "…"
    )


def _build_group_labels(
    groups: list[list[dict[str, Any]]],
    target: str,
) -> list[str]:
    """그룹 공통 키워드와 TF-IDF 점수로 짧은 라벨을 만든다."""

    documents = [
        " ".join(
            _strip_reporting_tail(
                _first_sentence(article["content"])
            )
            for article in group
        )
        for group in groups
    ]

    article_term_sets = [
        [
            set(
                _label_terms(
                    article["content"],
                    target=target,
                )
            )
            for article in group
        ]
        for group in groups
    ]

    vectorizer = TfidfVectorizer(
        tokenizer=lambda text: _label_terms(
            text,
            target=target,
        ),
        token_pattern=None,
        lowercase=False,
    )

    try:
        matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return [
            _fallback_label(group)
            for group in groups
        ]

    terms = vectorizer.get_feature_names_out()
    used_labels: set[str] = set()
    labels = []

    for group_index, group in enumerate(groups):
        scores = matrix[group_index].toarray()[0]

        coverage = {
            str(term): sum(
                term in term_set
                for term_set
                in article_term_sets[group_index]
            )
            for term in terms
        }

        valid = [
            index
            for index in range(len(terms))
            if scores[index] > 0
        ]

        label = None

        if valid:
            max_coverage = max(
                coverage[str(terms[index])]
                for index in valid
            )

            common = [
                index
                for index in valid
                if coverage[str(terms[index])]
                == max_coverage
            ]

            common.sort(
                key=lambda index: (
                    scores[index],
                    " " in str(terms[index]),
                    len(str(terms[index])),
                ),
                reverse=True,
            )

            bigrams = [
                str(terms[index]).strip()
                for index in common
                if " " in str(terms[index])
            ]

            if bigrams:
                label = bigrams[0]
            else:
                unigrams = [
                    str(terms[index]).strip()
                    for index in common
                    if " " not in str(terms[index])
                ]

                unigrams = list(dict.fromkeys(unigrams))

                if len(unigrams) >= 2:
                    label = " · ".join(unigrams[:2])
                elif unigrams:
                    label = unigrams[0]

        if label is None or label in used_labels:
            ranked = sorted(
                valid,
                key=lambda index: scores[index],
                reverse=True,
            )

            for index in ranked:
                candidate = str(terms[index]).strip()

                if candidate and candidate not in used_labels:
                    label = candidate
                    break

        if label is None:
            label = _fallback_label(group)

        if len(label) > MAX_LABEL_CHARS:
            label = label[:MAX_LABEL_CHARS].rstrip() + "…"

        used_labels.add(label)
        labels.append(label)

    return labels


def generate_viewpoint_group_labels(
    articles: list[dict[str, Any]],
    target: str = "",
) -> dict[str, str | None]:
    """
    같은 stance 안에서 2~3개 세부 견해 그룹을 만들고,
    형제 그룹 TF-IDF 대조로 article별 라벨을 반환한다.
    """

    by_stance: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    result: dict[str, str | None] = {}
    seen_ids: set[str] = set()

    for article in articles:
        article_id = str(article["article_id"])
        stance = article["stance"]
        content = str(
            article.get("content") or ""
        ).strip()

        if article_id in seen_ids:
            raise ValueError(
                f"duplicate article_id: {article_id}"
            )

        if stance not in VALID_STANCES:
            raise ValueError(
                f"invalid stance: {stance}"
            )

        seen_ids.add(article_id)
        result[article_id] = None

        if content:
            by_stance[stance].append(
                {
                    "article_id": article_id,
                    "stance": stance,
                    "content": content,
                }
            )

    for stance_articles in by_stance.values():
        normalized_texts = [
            _normalize_for_similarity(
                article["content"],
                target=target,
            )
            for article in stance_articles
        ]

        cluster_indices = _cluster_texts(
            normalized_texts
        )

        groups = [
            [
                stance_articles[index]
                for index in indices
            ]
            for indices in cluster_indices
        ]

        group_labels = _build_group_labels(
            groups,
            target=target,
        )

        for group, label in zip(
            groups,
            group_labels,
        ):
            for article in group:
                result[
                    article["article_id"]
                ] = label

    return result
