import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


STANCES = (
    "positive",
    "neutral",
    "negative",
)

SUPPORT_POWER = 2.0

STOPWORDS = {
    # 뉴스 기사에서 반복되는 일반 표현
    "기자",
    "관련",
    "대한",
    "대해",
    "통해",
    "위해",
    "위한",
    "이번",
    "현재",
    "지난",
    "오는",
    "이라고",
    "라고",
    "하며",
    "했다",
    "한다",
    "밝혔다",
    "말했다",
    "전했다",
    "설명했다",
    "것으로",
    "것이다",
    "것",
    "등",
    "씨",
    "있다",
    "없다",
    "있는",
    "없는",
    "있도록",
    "있다고",
    "됩니다",
    "된다",
    "됐다",
    "한다는",

    # 일반적인 연결·서술 잡음
    "면서",
    "이라며",
}

PARTICLE_SUFFIXES = (
    # 길이가 긴 조사
    "으로부터",
    "에게서",
    "에서는",
    "에서도",
    "으로",
    "에서",
    "에게",
    "까지",
    "부터",
    "처럼",
    "에는",
    "에도",

    # 서술·활용 어미
    "했습니다",
    "이었다",
    "입니다",
    "하면서",
    "으면서",
    "하도록",
    "이라고",
    "이라며",
    "한다는",
    "된다는",
    "된다고",
    "한다",
    "하는",
    "했다",
    "된다",
    "라고",
    "다고",
    "이다",

    # 비교적 안전한 조사
    "은",
    "는",
    "을",
    "를",
    "과",
    "와",
    "에",
    "의",
)

URL_PATTERN = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)

DOMAIN_PATTERN = re.compile(
    r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}\b",
    re.IGNORECASE,
)


NUMBER_UNIT_PATTERN = re.compile(
    r"\d+(?:년|월|일|시|분|초|명|개|건|차|%)?"
)


def _normalize_token(token: str) -> str:
    """키워드 후보에서 비교적 안전한 조사만 제거한다."""

    for suffix in PARTICLE_SUFFIXES:
        if not token.endswith(suffix):
            continue

        candidate = token[:-len(suffix)]

        # 어근이 지나치게 짧아지는 경우 제거하지 않는다.
        if len(candidate) >= 2:
            return candidate

    return token


def _tokenize(text: str) -> list[str]:
    """기사 본문에서 키워드 후보 토큰을 추출한다."""

    # URL과 도메인은 토큰화 전에 통째로 제거한다.
    text = URL_PATTERN.sub(" ", text)
    text = DOMAIN_PATTERN.sub(" ", text)

    raw_tokens = re.findall(
        r"[가-힣A-Za-z0-9]+",
        text,
    )

    tokens = []

    for raw_token in raw_tokens:
        token = _normalize_token(raw_token)

        if len(token) < 2:
            continue

        if token in STOPWORDS:
            continue

        if (
            token.isdigit()
            or NUMBER_UNIT_PATTERN.fullmatch(token)
        ):
            continue

        tokens.append(token)

    return tokens


def _stance_profile(
    matrix: np.ndarray,
    article_indices: list[int],
) -> np.ndarray:
    """
    한 stance의 대표 단어 점수를 만든다.

    단순 빈도뿐 아니라 같은 stance의 여러 기사에서
    반복적으로 등장했는지도 함께 반영한다.
    """

    stance_matrix = matrix[article_indices]

    # stance 내 기사들의 평균 TF-IDF
    mean_scores = stance_matrix.mean(axis=0)

    # 해당 단어가 stance 기사 중 몇 %에 등장하는지
    support = (
        (stance_matrix > 0)
        .mean(axis=0)
    )

    # 한 기사에서만 반복된 단어의 과대평가를 줄인다.
    return mean_scores * (
        support ** SUPPORT_POWER
    )


def extract_stance_keywords(
    articles: list[dict],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """
    기사별 TF-IDF를 기반으로 stance 대표 키워드를 추출한다.

    1. 각 기사를 독립된 문서로 TF-IDF 계산
    2. 같은 stance 여러 기사에서 반복되는 단어를 우선
    3. 다른 stance에서도 강한 공통 주제어는 감점
    4. 해당 stance에서 상대적으로 특징적인 키워드를 반환
    """

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    grouped_indices: dict[str, list[int]] = defaultdict(list)

    documents = []
    document_stances = []

    for article in articles:
        stance = article.get("stance")

        if stance not in STANCES:
            raise ValueError(
                f"invalid stance: {stance}"
            )

        content = str(
            article.get("content", "")
        ).strip()

        if not content:
            continue

        document_index = len(documents)

        documents.append(content)
        document_stances.append(stance)
        grouped_indices[stance].append(
            document_index
        )

    result = {
        stance: []
        for stance in STANCES
    }

    if not documents:
        return result

    if not any(
        _tokenize(document)
        for document in documents
    ):
        return result

    vectorizer = TfidfVectorizer(
        analyzer=_tokenize,
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        binary=True,
        sublinear_tf=False,
    )

    matrix = (
        vectorizer
        .fit_transform(documents)
        .toarray()
    )

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    # ========================================================
    # stance별 대표 프로필 생성
    # ========================================================

    profiles = {}

    for stance in STANCES:
        indices = grouped_indices.get(
            stance,
            [],
        )

        if not indices:
            profiles[stance] = np.zeros(
                len(feature_names)
            )
            continue

        profiles[stance] = _stance_profile(
            matrix,
            indices,
        )

    # ========================================================
    # stance별 contrastive keyword
    # ========================================================

    for stance in STANCES:
        if not grouped_indices.get(stance):
            continue

        own_profile = profiles[stance]

        other_stances = [
            other
            for other in STANCES
            if other != stance
            and grouped_indices.get(other)
        ]

        ranked = []

        for column_index, keyword in enumerate(
            feature_names
        ):
            own_score = float(
                own_profile[column_index]
            )

            if own_score <= 0:
                continue

            if other_stances:
                other_max = max(
                    float(
                        profiles[other][
                            column_index
                        ]
                    )
                    for other in other_stances
                )
            else:
                other_max = 0.0

            contrastive_score = (
                own_score - other_max
            )

            if contrastive_score <= 0:
                continue

            ranked.append(
                (
                    keyword,
                    contrastive_score,
                    own_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[0],
            )
        )

        result[stance] = [
            keyword
            for keyword, _, _ in ranked[:top_k]
        ]

    return result


def extract_stance_phrases(
    articles: list[dict],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """관점별 대표 2단어 표현을 추출한다."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    grouped_indices: dict[str, list[int]] = defaultdict(list)

    documents = []

    for article in articles:
        stance = article.get("stance")

        if stance not in STANCES:
            raise ValueError(
                f"invalid stance: {stance}"
            )

        content = str(
            article.get("content", "")
        ).strip()

        if not content:
            continue

        index = len(documents)

        documents.append(content)

        grouped_indices[
            stance
        ].append(index)

    result = {
        stance: []
        for stance in STANCES
    }

    if not documents:
        return result

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        preprocessor=None,
        token_pattern=None,
        analyzer="word",
        ngram_range=(2, 2),
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        binary=True,
        sublinear_tf=False,
    )

    try:
        matrix = (
            vectorizer
            .fit_transform(documents)
            .toarray()
        )
    except ValueError:
        return result

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    profiles = {}

    for stance in STANCES:
        indices = grouped_indices.get(
            stance,
            [],
        )

        if not indices:
            profiles[stance] = np.zeros(
                len(feature_names)
            )
            continue

        profiles[stance] = _stance_profile(
            matrix,
            indices,
        )

    for stance in STANCES:
        if not grouped_indices.get(stance):
            continue

        own_profile = profiles[stance]

        other_stances = [
            other
            for other in STANCES
            if (
                other != stance
                and grouped_indices.get(other)
            )
        ]

        ranked = []

        for column_index, phrase in enumerate(
            feature_names
        ):
            own_score = float(
                own_profile[column_index]
            )

            if own_score <= 0:
                continue

            if other_stances:
                other_max = max(
                    float(
                        profiles[other][
                            column_index
                        ]
                    )
                    for other in other_stances
                )
            else:
                other_max = 0.0

            contrastive_score = (
                own_score - other_max
            )

            if contrastive_score <= 0:
                continue

            ranked.append(
                (
                    phrase,
                    contrastive_score,
                    own_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[0],
            )
        )

        result[stance] = [
            phrase
            for phrase, _, _ in ranked[:top_k]
        ]

    return result


def extract_stance_analysis(
    articles: list[dict],
    keyword_top_k: int = 5,
    phrase_top_k: int = 5,
) -> dict[str, dict[str, list[str]]]:
    """관점별 단일 키워드와 대표 표현을 함께 반환한다."""

    keywords = extract_stance_keywords(
        articles,
        top_k=keyword_top_k,
    )

    phrases = extract_stance_phrases(
        articles,
        top_k=phrase_top_k,
    )

    return {
        stance: {
            "keywords": keywords[stance],
            "phrases": phrases[stance],
        }
        for stance in STANCES
    }


def extract_stance_phrases(
    articles: list[dict],
    top_k: int = 5,
) -> dict[str, list[str]]:
    """관점별 대표 2단어 표현을 추출한다."""

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    grouped_indices: dict[str, list[int]] = defaultdict(list)

    documents = []

    for article in articles:
        stance = article.get("stance")

        if stance not in STANCES:
            raise ValueError(
                f"invalid stance: {stance}"
            )

        content = str(
            article.get("content", "")
        ).strip()

        if not content:
            continue

        index = len(documents)

        documents.append(content)

        grouped_indices[
            stance
        ].append(index)

    result = {
        stance: []
        for stance in STANCES
    }

    if not documents:
        return result

    vectorizer = TfidfVectorizer(
        tokenizer=_tokenize,
        preprocessor=None,
        token_pattern=None,
        analyzer="word",
        ngram_range=(2, 2),
        lowercase=False,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        binary=True,
        sublinear_tf=False,
    )

    try:
        matrix = (
            vectorizer
            .fit_transform(documents)
            .toarray()
        )
    except ValueError:
        return result

    feature_names = (
        vectorizer
        .get_feature_names_out()
    )

    profiles = {}

    for stance in STANCES:
        indices = grouped_indices.get(
            stance,
            [],
        )

        if not indices:
            profiles[stance] = np.zeros(
                len(feature_names)
            )
            continue

        profiles[stance] = _stance_profile(
            matrix,
            indices,
        )

    for stance in STANCES:
        if not grouped_indices.get(stance):
            continue

        own_profile = profiles[stance]

        other_stances = [
            other
            for other in STANCES
            if (
                other != stance
                and grouped_indices.get(other)
            )
        ]

        ranked = []

        for column_index, phrase in enumerate(
            feature_names
        ):
            own_score = float(
                own_profile[column_index]
            )

            if own_score <= 0:
                continue

            if other_stances:
                other_max = max(
                    float(
                        profiles[other][
                            column_index
                        ]
                    )
                    for other in other_stances
                )
            else:
                other_max = 0.0

            contrastive_score = (
                own_score - other_max
            )

            if contrastive_score <= 0:
                continue

            ranked.append(
                (
                    phrase,
                    contrastive_score,
                    own_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                -item[1],
                -item[2],
                item[0],
            )
        )

        result[stance] = [
            phrase
            for phrase, _, _ in ranked[:top_k]
        ]

    return result


def extract_stance_analysis(
    articles: list[dict],
    keyword_top_k: int = 5,
    phrase_top_k: int = 5,
) -> dict[str, dict[str, list[str]]]:
    """관점별 단일 키워드와 대표 표현을 함께 반환한다."""

    keywords = extract_stance_keywords(
        articles,
        top_k=keyword_top_k,
    )

    phrases = extract_stance_phrases(
        articles,
        top_k=phrase_top_k,
    )

    return {
        stance: {
            "keywords": keywords[stance],
            "phrases": phrases[stance],
        }
        for stance in STANCES
    }
