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
MAX_VIEWPOINT_LABEL_CHARS = 8
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


SHORT_LABEL_DIRECTION_WORDS = {
    "부족",
    "훼손",
    "침해",
    "문제",
    "정당",
    "부당",
    "강화",
    "약화",
    "확대",
    "축소",
    "폐기",
    "수정",
    "철회",
    "중단",
    "취소",
    "사퇴",
    "탄핵",
    "유지",
    "금지",
    "허용",
    "개정",
    "도입",
    "시행",
    "추진",
    "공급",
    "조성",
    "필요",
    "반대",
    "찬성",
    "부담",
    "개입",
    "효과",
}

SHORT_LABEL_NOISE_WORDS = {
    "한다",
    "한다는",
    "해야",
    "해야한다",
    "때문",
    "때문에",
    "방향",
    "이번",
    "해당",
    "것",
}


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
    """라벨용 토큰의 조사와 서술형 어미를 정리한다."""

    token = str(token or "").strip()

    if not token:
        return ""

    if token.endswith("이므로") and len(token) > 3:
        token = token[:-3]

    token = _strip_particle(token)

    # 세부 견해에서 자주 쓰는 행동·평가 표현은
    # 활용형을 짧은 명사형으로 통일한다.
    canonical_stems = {
        "부족",
        "훼손",
        "침해",
        "강화",
        "약화",
        "확대",
        "축소",
        "폐기",
        "수정",
        "철회",
        "중단",
        "취소",
        "사퇴",
        "탄핵",
        "유지",
        "금지",
        "허용",
        "개정",
        "도입",
        "시행",
        "추진",
        "공급",
        "조성",
        "필요",
        "반대",
        "찬성",
        "개입",
        "청구",
        "판결",
        "요구",
        "거부",
        "반발",
        "제안",
        "채택",
        "소멸",
        "확인",
        "대비",
        "점검",
        "관리",
        "분할",
        "지급",
        "해제",
        "발효",
        "비판",
        "우려",
    }

    verbal_suffixes = {
        "하다",
        "한다",
        "한다고",
        "한다는",
        "하며",
        "하면서",
        "했다",
        "했다고",
        "했다는",
        "하다며",
        "한다며",
        "해",
        "해서",
        "했고",
        "해도",
        "해왔다",
        "해왔다고",
        "해야",
        "해야한다",
        "해야하는",
        "해야할",
        "하기",
        "하기도",
        "한",
        "할",
    }

    for stem in sorted(
        canonical_stems,
        key=len,
        reverse=True,
    ):
        if token == stem:
            return stem

        if token.startswith(stem):
            suffix = token[len(stem):]

            if suffix in verbal_suffixes:
                return stem

    # 기존의 단순 서술형 정리도 유지한다.
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



def _directional_viewpoint_label(
    text: str,
) -> str | None:
    """
    명확한 '대상 + 의견 방향' 관계를 짧은 라벨로 보존한다.

    단순 행동 단어보다 우려·규탄·요구·거부처럼
    문장의 실제 방향을 먼저 해석한다.
    """

    cleaned = " ".join(
        str(text or "").split()
    )

    if not cleaned:
        return None

    # 명시적인 긍정 평가만 우선 보존한다.
    # 애매한 positive 문장은 기존 로직으로 넘겨
    # 억지 라벨 생성을 피한다.
    if (
        "개최" in cleaned
        and (
            (
                "기쁘" in cleaned
                and "기쁘지 않" not in cleaned
            )
            or (
                "영광" in cleaned
                and "영광스럽지 않" not in cleaned
            )
        )
    ):
        return "개최 환영"

    if (
        "기대" in cleaned
        and "기대하지 않" not in cleaned
    ):
        matches = list(
            re.finditer(
                r"([0-9A-Za-z가-힣]{2,8})"
                r"(?:을|를)\s*기대",
                cleaned,
            )
        )

        if matches:
            anchor = _normalize_label_token(
                matches[-1].group(1)
            )
            label = f"{anchor} 기대"

            if len(label) <= MAX_VIEWPOINT_LABEL_CHARS:
                return label

    if (
        "발전" in cleaned
        and "기여" in cleaned
        and "기여하지 않" not in cleaned
    ):
        return "발전 기여"

    patterns = (
        (
            "거부",
            r"(?:^|\s)"
            r"([0-9A-Za-z가-힣]{2,8})"
            r"(?:에|에는)\s*"
            r"선을\s*(?:긋|그)",
        ),
        (
            "우려",
            r"(?:^|\s)"
            r"([0-9A-Za-z가-힣]{2,8})"
            r"(?:에|에는|을|를)\s*"
            r"(?:대해\s*)?"
            r"우려",
        ),
        (
            "규탄",
            r"(?:^|\s)"
            r"([0-9A-Za-z가-힣]{2,8})"
            r"(?:을|를)\s*"
            r"규탄",
        ),
        (
            "요구",
            r"(?:^|\s)"
            r"([0-9A-Za-z가-힣]{2,8})"
            r"(?:을|를)\s*"
            r"요구",
        ),
    )

    for direction, pattern in patterns:
        matches = list(
            re.finditer(
                pattern,
                cleaned,
            )
        )

        if not matches:
            continue

        # 결론부에 가까운 마지막 관계를 우선한다.
        anchor = _normalize_label_token(
            matches[-1].group(1)
        )

        if len(anchor) < 2:
            continue

        label = f"{anchor} {direction}"

        if (
            len(label)
            <= MAX_VIEWPOINT_LABEL_CHARS
        ):
            return label

    return None

def _short_viewpoint_label(
    text: str,
    target: str = "",
) -> str:
    """세부 견해를 최대 8자의 짧은 명사형 라벨로 압축한다."""

    claim = _final_claim_sentence(text)

    if not claim:
        return "기타 관점"

    directional_label = (
        _directional_viewpoint_label(text)
    )

    if directional_label:
        return directional_label

    target_parts = [
        part.strip(".,!?()[]{}\"'")
        for part in target.split()
        if part.strip(".,!?()[]{}\"'")
    ]
    target_anchor = (
        target_parts[-1]
        if target_parts
        else ""
    )

    # "문제가 없고 정당하다"처럼 긍정 의미가 명확한 경우
    # "절차 문제"로 잘못 축약되지 않도록 먼저 처리한다.
    if (
        "정당하" in claim
        and "정당하지" not in claim
    ):
        anchor = (
            "절차"
            if "절차" in claim
            else target_anchor
        )

        if anchor:
            label = f"{anchor} 정당"

            if len(label) <= MAX_VIEWPOINT_LABEL_CHARS:
                return label

        return "정당"

    if (
        "문제가 없" in claim
        or "문제 없" in claim
    ):
        return "문제 없음"

    # 폐기/수정처럼 문장의 결론부에 가까운 명확한 행동은
    # 공통 근거보다 우선해 세부 견해로 보존한다.
    action = _extract_conclusion_action(claim)

    strong_actions = {
        "폐기",
        "수정",
        "철회",
        "중단",
        "취소",
        "사퇴",
        "탄핵",
        "유지",
        "금지",
        "허용",
        "개정",
        "도입",
        "시행",
        "추진",
        "확대",
        "축소",
        "강화",
        "약화",
        "반대",
        "찬성",
    }

    if action in strong_actions:
        action_pos = claim.rfind(action)

        # 문장 앞부분의 부수적 행동을 결론으로 오인하지 않는다.
        if (
            action_pos >= 0
            and len(claim) - action_pos <= 24
        ):
            # 같은 결론이라도 이유가 다르면 세부 견해를
            # 구분할 수 있도록 결론 앞의 이유 표현을 함께 본다.
            reason_text = claim[:action_pos]

            reason_words = {
                "부족",
                "훼손",
                "침해",
                "문제",
                "부당",
                "부담",
                "개입",
                "우려",
                "비판",
                "필요",
                "효과",
            }

            reason_candidates = []

            for term in _label_terms(
                reason_text,
                target=target,
            ):
                term = " ".join(term.split()).strip()

                if not (
                    2 <= len(term)
                    <= MAX_VIEWPOINT_LABEL_CHARS
                ):
                    continue

                if any(
                    word in reason_words
                    for word in term.split()
                ):
                    reason_candidates.append(term)

            if reason_candidates:
                reason = max(
                    reason_candidates,
                    key=lambda term: (
                        len(term.split()),
                        reason_text.rfind(
                            term.split()[-1]
                        ),
                        len(term),
                    ),
                )

                # "재정 부담 폐기"처럼 8자 안이면
                # 이유와 결론을 모두 보존한다.
                combined = f"{reason} {action}"

                if (
                    len(combined)
                    <= MAX_VIEWPOINT_LABEL_CHARS
                ):
                    return combined

                # "정당성 부족 폐기"처럼 너무 길면
                # 방향어를 떼고 핵심 이유 + 결론을 시도한다.
                reason_parts = reason.split()

                if (
                    len(reason_parts) >= 2
                    and reason_parts[-1] in reason_words
                ):
                    compact_reason = " ".join(
                        reason_parts[:-1]
                    )
                    combined = (
                        f"{compact_reason} {action}"
                    )

                    if (
                        len(combined)
                        <= MAX_VIEWPOINT_LABEL_CHARS
                    ):
                        return combined

                # 그래도 8자를 넘으면 최소한 이유 자체를
                # 보존해 서로 다른 세부 견해를 구분한다.
                return reason

            if target_anchor:
                combined = f"{target_anchor} {action}"

                if (
                    len(combined)
                    <= MAX_VIEWPOINT_LABEL_CHARS
                ):
                    return combined

            if len(action) <= MAX_VIEWPOINT_LABEL_CHARS:
                return action

    reporting_words = {
        "했다",
        "됐다",
        "한다",
        "된다",
        "이다",
        "있다",
        "없다",
        "냈다",
        "말했다",
        "밝혔다",
        "설명했다",
        "전했다",
        "알려졌다",
        "보인다",
        "전망이다",
        "판단했다",
        "지냈다",
        "맞이했다",
        "파악됐다",
        "예상된다",
        "것이다",
        "셈이다",
        "중이다",
        "나왔다",
    }

    generic_words = {
        "이번",
        "해당",
        "이날",
        "현재",
        "결국",
        "재차",
        "대폭",
        "위해",
        "것을",
        "것은",
        "것이",
        "것으",
        "라고",
        "라며",
        "면서",
    }

    priority_words = set(SHORT_LABEL_DIRECTION_WORDS) | {
        "청구",
        "소송",
        "판결",
        "제기",
        "요구",
        "거부",
        "반발",
        "제안",
        "채택",
        "소멸",
        "확인",
        "대비",
        "점검",
        "관리",
        "분할",
        "지급",
        "해제",
        "발효",
        "불신임",
        "투표",
        "사퇴",
        "우려",
        "비판",
        "책임",
        "피해",
        "안전",
        "개헌",
        "협력",
        "통합",
        "독립",
        "정당성",
    }

    conclusion_words = {
        "부족",
        "훼손",
        "침해",
        "문제",
        "부당",
        "강화",
        "약화",
        "확대",
        "축소",
        "폐기",
        "수정",
        "철회",
        "중단",
        "취소",
        "사퇴",
        "탄핵",
        "유지",
        "금지",
        "허용",
        "개정",
        "도입",
        "시행",
        "추진",
        "공급",
        "조성",
        "필요",
        "반대",
        "찬성",
        "부담",
        "개입",
        "효과",
        "청구",
        "소송",
        "판결",
        "요구",
        "거부",
        "반발",
        "제안",
        "채택",
        "소멸",
        "확인",
        "대비",
        "점검",
        "관리",
        "분할",
        "지급",
        "해제",
        "발효",
        "불신임",
        "투표",
        "우려",
        "비판",
    }

    candidates = []

    for term in _label_terms(
        claim,
        target=target,
    ):
        term = " ".join(term.split()).strip()

        if not (
            2 <= len(term) <= MAX_VIEWPOINT_LABEL_CHARS
        ):
            continue

        words = term.split()

        if any(
            word in SHORT_LABEL_NOISE_WORDS
            or word in reporting_words
            or word in generic_words
            for word in words
        ):
            continue

        digit_count = sum(
            char.isdigit()
            for char in term
        )

        has_priority = any(
            word in priority_words
            for word in words
        )

        if digit_count >= 2 and not has_priority:
            continue

        if term not in candidates:
            candidates.append(term)

    if not candidates:
        return "기타 관점"

    # 의미 있는 행동·평가 표현이 없는 단순 기사 조각은
    # 억지로 세부 견해 라벨로 만들지 않는다.
    meaningful_candidates = [
        term
        for term in candidates
        if any(
            word in priority_words
            for word in term.split()
        )
    ]

    if not meaningful_candidates:
        return "기타 관점"

    def term_score(term: str):
        words = term.split()
        last_word = words[-1]

        ends_with_conclusion = (
            last_word in conclusion_words
        )

        contains_conclusion = any(
            word in conclusion_words
            for word in words
        )

        has_priority = any(
            word in priority_words
            for word in words
        )

        token_score = min(len(words), 2)
        content_length = len(term.replace(" ", ""))
        position = claim.rfind(last_word)

        return (
            ends_with_conclusion,
            contains_conclusion,
            token_score,
            has_priority,
            content_length,
            position,
        )

    return max(
        meaningful_candidates,
        key=term_score,
    )


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

    directional_keys = [
        _directional_viewpoint_label(
            article["content"]
        )
        for article in articles
    ]

    distinct_directional_keys = list(
        dict.fromkeys(
            key
            for key in directional_keys
            if key is not None
        )
    )

    # 같은 stance 안에서도 "동결 우려"와 "동결 규탄"처럼
    # 방향이 명확히 다른 주장은 동일 세부견해로 합치지 않는다.
    # 단, 명확한 방향이 2종 이상일 때만 적용해
    # 기존 결론 기반 군집에 미치는 영향을 최소화한다.
    if len(distinct_directional_keys) >= 2:
        conclusion_keys = directional_keys
    else:
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

        label = _short_viewpoint_label(
            label,
            target=target,
        )

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
