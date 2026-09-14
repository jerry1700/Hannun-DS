"""같은 stance 안에서 기사별 세부 견해 그룹을 생성한다."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Any

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from hannun.embedding.encoder import EncoderConfig


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
SINGLE_GROUP_SIMILARITY_THRESHOLD = 0.55
REASON_SIMILARITY_THRESHOLD = 0.75


# 확정 라벨 가이드:
# 최종 주장과 핵심 근거가 모두 실질적으로 같아야
# 같은 세부 견해로 판단한다.
# 주장 또는 핵심 근거가 다르면 별도 세부 견해로 본다.
CONCLUSION_ACTION_PATTERNS = (
    ("폐기", r"폐기"),
    ("수정", r"수정"),
    ("철회", r"철회"),
    ("중단", r"중단"),
    ("취소", r"취소"),
    ("사퇴", r"사퇴|퇴진|물러나"),
    ("탄핵", r"탄핵"),
    ("축소", r"축소|줄이|줄여"),
    ("확대", r"확대|늘리|늘려"),
    ("유지", r"유지"),
    ("금지", r"금지"),
    ("허용", r"허용"),
    ("개정", r"개정"),
    ("도입", r"도입"),
    ("시행", r"시행"),
    ("추진", r"추진"),
    ("공급", r"공급"),
    ("조성", r"조성"),
)


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
    """주장 내용은 보존하고 뒤의 보도 표현만 제거한다."""

    cleaned = " ".join(str(text or "").split())

    replacements = (
        (
            r"다는\s*"
            r"(?:비판|우려|평가|지적|분석|주장|입장)"
            r"(?:이|가)?"
            r"(?:\s*(?:나왔다|제기됐다|이어졌다|있다|이다|전해졌다))?"
            r"[.!?]*$",
            "다",
        ),
        (
            r"다고\s*"
            r"(?:말했다|밝혔다|설명했다|강조했다|주장했다|"
            r"평가했다|지적했다|비판했다|전했다|했다)"
            r"[.!?]*$",
            "다",
        ),
        (
            r"(?:라는|이라고|라고)\s*"
            r"(?:비판|우려|평가|지적|분석|주장|입장)"
            r"(?:이|가)?"
            r"(?:\s*(?:나왔다|제기됐다|이어졌다|있다|이다|전해졌다))?"
            r"[.!?]*$",
            "",
        ),
        (
            r"\s*(?:비판|우려|평가|지적|분석|주장|입장)"
            r"(?:이|가)?\s*"
            r"(?:나왔다|제기됐다|있다|이다)?"
            r"[.!?]*$",
            "",
        ),
    )

    for pattern, replacement in replacements:
        cleaned = re.sub(
            pattern,
            replacement,
            cleaned,
        )

    return cleaned.strip(" .!?")


def _final_claim_sentence(text: str) -> str:
    """evidence 중 마지막 주장 문장을 결론 후보로 사용한다."""

    cleaned = " ".join(str(text or "").split())

    if not cleaned:
        return ""

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )
        if sentence.strip()
    ]

    if not sentences:
        return ""

    return _strip_reporting_tail(
        sentences[-1]
    )


def _concise_claim_label(text: str) -> str:
    """최종 결론을 보존하면서 짧은 주장형 라벨을 만든다."""

    claim = _final_claim_sentence(text)

    if not claim:
        return ""

    words = claim.split()

    while (
        len(" ".join(words)) > MAX_LABEL_CHARS
        and len(words) > 1
    ):
        words.pop(0)

    label = " ".join(words).strip()

    if len(label) > MAX_LABEL_CHARS:
        label = (
            label[:MAX_LABEL_CHARS]
            .rstrip()
            + "…"
        )

    return label


def _normalize_for_similarity(
    text: str,
    target: str = "",
) -> str:
    """
    최종 주장 문장의 술어·결론은 유지하고,
    공통 Target 표현만 줄여 임베딩 비교에 사용한다.
    """

    claim = _final_claim_sentence(text)

    if not claim:
        return ""

    target_tokens = {
        _normalize_label_token(token)
        for token in re.findall(
            r"[0-9A-Za-z가-힣]+",
            target,
        )
    }

    kept = []

    for raw_token in re.findall(
        r"[0-9A-Za-z가-힣]+",
        claim,
    ):
        normalized = _normalize_label_token(
            raw_token
        )

        if (
            normalized
            and normalized in target_tokens
        ):
            continue

        # 임베딩에서는 주장 술어를 살리기 위해
        # 원래 형태를 그대로 유지한다.
        kept.append(raw_token)

    normalized_claim = " ".join(kept).strip()

    return normalized_claim or claim


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


@lru_cache(maxsize=1)
def _get_embedding_model() -> Any:
    """세부 견해 군집용 문장 임베딩 모델을 한 번만 로드한다."""

    from sentence_transformers import SentenceTransformer

    config = EncoderConfig()

    return SentenceTransformer(
        config.model_name,
        device="cpu",
        local_files_only=True,
    )


def _embed_texts(
    texts: list[str],
):
    """정규화한 evidence를 E5 문장 임베딩으로 변환한다."""

    config = EncoderConfig()
    model = _get_embedding_model()

    return model.encode(
        [
            config.passage_prefix + text
            for text in texts
        ],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def _extract_reason_clause(
    text: str,
) -> str:
    """명시적인 인과 표현 앞의 핵심 근거를 추출한다."""

    claim = _final_claim_sentence(text)

    if not claim:
        return ""

    match = re.search(
        r"(.+?)(?:기 때문에|때문에|이므로|라서|해서|"
        r"로 인해|탓에|까닭에)",
        claim,
    )

    if not match:
        return ""

    return match.group(1).strip(" ,")


def _cluster_two_articles(
    texts: list[str],
) -> list[list[int]]:
    """기사 2건을 임베딩 유사도로 한두 세부 견해로 나눈다."""

    vectors = _embed_texts(texts)

    similarity = float(
        cosine_similarity(vectors)[0, 1]
    )

    if (
        similarity
        < SINGLE_GROUP_SIMILARITY_THRESHOLD
    ):
        return [[0], [1]]

    reasons = [
        _extract_reason_clause(text)
        for text in texts
    ]

    if all(reasons):
        reason_vectors = _embed_texts(reasons)

        reason_similarity = float(
            cosine_similarity(
                reason_vectors
            )[0, 1]
        )

        if (
            reason_similarity
            < REASON_SIMILARITY_THRESHOLD
        ):
            return [[0], [1]]

    return [[0, 1]]


def _cluster_many_articles(
    texts: list[str],
    max_subclusters: int = MAX_SUBCLUSTERS,
) -> list[list[int]]:
    """기사 3건 이상을 임베딩 기준 세부 견해로 나눈다."""

    unique_count = len(set(texts))

    if unique_count == 1:
        return [
            list(range(len(texts)))
        ]

    vectors = _embed_texts(texts)

    similarities = cosine_similarity(
        vectors
    )

    pair_values = [
        float(similarities[i, j])
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
    ]

    # 모든 evidence가 충분히 비슷하면 하나의 세부 견해로 유지한다.
    if (
        pair_values
        and min(pair_values)
        >= SINGLE_GROUP_SIMILARITY_THRESHOLD
    ):
        return [
            list(range(len(texts)))
        ]

    max_k = min(
        max_subclusters,
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
            n_init=20,
        )

        labels = model.fit_predict(
            vectors
        )

        if len(set(labels)) < 2:
            continue

        score = silhouette_score(
            vectors,
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

    for index, label in enumerate(
        best_labels
    ):
        groups.setdefault(
            int(label),
            [],
        ).append(index)

    return list(groups.values())


def _extract_conclusion_action(
    text: str,
) -> str | None:
    """evidence의 최종 주장에 명확한 결론 동작이 있는지 찾는다."""

    claim = _final_claim_sentence(text)

    if not claim:
        return None

    for action, pattern in CONCLUSION_ACTION_PATTERNS:
        if re.search(pattern, claim):
            return action

    return None


def _cluster_with_conclusion_constraints(
    articles: list[dict[str, Any]],
    texts: list[str],
    max_subclusters: int,
) -> list[list[int]]:
    """
    명확히 다른 최종 결론은 E5 유사도가 높아도 분리한다.

    동일한 결론이라도 핵심 근거가 다르면 분리할 수 있으며,
    결론 동작을 명확히 잡지 못한 기사에는 E5 군집을 사용한다.
    """

    conclusion_keys = [
        _extract_conclusion_action(
            article["content"]
        )
        for article in articles
    ]

    explicit_keys = list(
        dict.fromkeys(
            key
            for key in conclusion_keys
            if key is not None
        )
    )

    # 명확히 서로 다른 결론이 2개 이상 잡힌 경우에만
    # 결론 기준 hard constraint를 적용한다.
    if (
        len(explicit_keys) < 2
        or len(explicit_keys) > max_subclusters
    ):
        return _cluster_texts(
            texts,
            max_subclusters=max_subclusters,
        )

    groups = [
        [
            index
            for index, key in enumerate(
                conclusion_keys
            )
            if key == explicit_key
        ]
        for explicit_key in explicit_keys
    ]

    unknown_indices = [
        index
        for index, key in enumerate(
            conclusion_keys
        )
        if key is None
    ]

    if not unknown_indices:
        return groups

    remaining_slots = (
        max_subclusters - len(groups)
    )

    if remaining_slots > 0:
        unknown_texts = [
            texts[index]
            for index in unknown_indices
        ]

        unknown_groups = _cluster_texts(
            unknown_texts,
            max_subclusters=remaining_slots,
        )

        groups.extend(
            [
                [
                    unknown_indices[index]
                    for index in unknown_group
                ]
                for unknown_group in unknown_groups
            ]
        )

        return groups

    # 이미 최대 그룹 수를 모두 사용한 경우,
    # 결론이 명확하지 않은 기사는 가장 유사한 명시 그룹에 붙인다.
    vectors = _embed_texts(texts)
    similarities = cosine_similarity(vectors)

    for unknown_index in unknown_indices:
        best_group_index = max(
            range(len(groups)),
            key=lambda group_index: max(
                float(
                    similarities[
                        unknown_index,
                        member_index,
                    ]
                )
                for member_index
                in groups[group_index]
            ),
        )

        groups[best_group_index].append(
            unknown_index
        )

    return groups


def _cluster_texts(
    texts: list[str],
    max_subclusters: int = MAX_SUBCLUSTERS,
) -> list[list[int]]:
    """한 stance 안의 evidence를 임베딩 기반 세부 견해로 나눈다."""

    if not texts:
        return []

    if len(texts) == 1:
        return [[0]]

    if len(texts) == 2:
        return _cluster_two_articles(texts)

    return _cluster_many_articles(
        texts,
        max_subclusters=max_subclusters,
    )


def _fallback_label(
    articles: list[dict[str, Any]],
) -> str:
    """대표 evidence의 최종 주장을 라벨로 사용한다."""

    return _concise_claim_label(
        articles[0]["content"]
    )


def _build_group_labels(
    groups: list[list[dict[str, Any]]],
    target: str,
) -> list[str]:
    """
    각 세부 견해 그룹에서 실제 주장을 담은
    evidence 문장을 대표 라벨로 사용한다.
    """

    labels = []

    for group in groups:
        claims = []

        for article in group:
            claim = _final_claim_sentence(
                article["content"]
            )

            if (
                claim
                and claim not in claims
            ):
                claims.append(claim)

        if not claims:
            label = _fallback_label(group)
        else:
            # '없다고 했다' 같은 짧은 표현보다
            # Target 밖의 실질 정보가 많은 주장 문장을 우선한다.
            def claim_score(claim: str):
                normalized = (
                    _normalize_for_similarity(
                        claim,
                        target=target,
                    )
                )

                content_tokens = {
                    token
                    for token in normalized.split()
                    if len(token) >= 2
                }

                return (
                    len(content_tokens),
                    min(
                        len(claim),
                        MAX_LABEL_CHARS,
                    ),
                )

            label = max(
                claims,
                key=claim_score,
            )

        label = _concise_claim_label(label)

        labels.append(label)

    return labels


def generate_viewpoint_group_labels(
    articles: list[dict[str, Any]],
    target: str = "",
) -> dict[str, str | None]:
    """
    같은 stance 안에서 최대 3개 세부 견해 그룹을 만들고,
    최종 결론과 문장 임베딩을 기준으로 article별 라벨을 반환한다.
    """

    by_stance: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    result: dict[str, str | None] = {}
    seen_ids: set[str] = set()
    stances_with_empty_evidence: set[str] = set()

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
        else:
            stances_with_empty_evidence.add(
                stance
            )

    for stance, stance_articles in by_stance.items():
        normalized_texts = [
            _normalize_for_similarity(
                article["content"],
                target=target,
            )
            for article in stance_articles
        ]

        max_subclusters = (
            MAX_SUBCLUSTERS - 1
            if stance in stances_with_empty_evidence
            else MAX_SUBCLUSTERS
        )

        cluster_indices = _cluster_with_conclusion_constraints(
            stance_articles,
            normalized_texts,
            max_subclusters=max_subclusters,
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
