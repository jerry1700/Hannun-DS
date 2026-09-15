import re

from .sentence_filter import split_sentences


TARGET_STOPWORDS = {
    "것",
    "대한",
    "관련",
    "정책",
    "행위",
    "이슈",
    "바람직",
}

TOPIC_SWITCH_MARKERS = (
    "정책",
    "산업",
    "사업",
    "법안",
    "대책",
    "계획",
)

# 특정 이슈가 아니라 여러 뉴스에서 공통적으로 나타나는 행위 표현.
ACTION_KEYWORDS = (
    "제정",
    "개정",
    "통과",
    "시행",
    "도입",
    "추진",
    "재추진",
    "유지",
    "확대",
    "강화",
    "축소",
    "폐지",
    "징수",
    "방류",
    "공급",
    "제한",
    "규제",
    "지원",
    "해임",
    "사퇴",
    "기소",
    "압수수색",
    "부결",
    "재표결",
    "재투표",
    "공포",
    "거부권",
    "재의요구",
)

# target을 막거나 뒤집는 데 사용되는 일반적인 행위.
BLOCKING_TERMS = (
    "재의요구권",
    "재의요구",
    "거부권",
    "거부",
    "반대",
    "저지",
    "폐기",
    "철회",
    "중단",
    "취소",
    "금지",
    "부결",
)

# Target 식별용으로 쓰기에는 너무 범용적인 표현.
# 이런 단어 하나만 겹친다고 같은 이슈 문장으로 보지 않는다.
TARGET_GENERIC_TERMS = {
    *ACTION_KEYWORDS,
    *BLOCKING_TERMS,
    "검토",
    "논의",
    "방안",
    "가능성",
    "문제",
    "필요",
    "요구",
    "촉구",
    "찬성",
    "비판",
    "우려",
    "지지",
    "최악",
    "물러나라",
}


STANCE_SIGNAL_PATTERN = re.compile(
    r"찬성(?:한다고|한다|했다|하며|하고| 입장)|"
    r"지지(?:한다고|한다|했다|하며|하고| 입장)|"
    r"환영(?:한다고|한다|했다|하며)|"
    r"긍정적|바람직하|필요하|정당하|타당하|"
    r"반대|우려|비판|규탄|반발|저지|폐기|철회|중단|취소|"
    r"거부|재의요구|부결|촉구|요구|건의|투쟁|"
    r"악법|폭거|최악|물러나라|사퇴|퇴진|무책임|선동|몽니|어깃장|외면|"
    r"실망스(?:럽|러)|참담|수치스럽|진일보|이정표|진전"
)

POSITIVE_PATTERNS = (
    r"찬성(?:한다고|한다|했다|하며| 입장)",
    r"지지(?:한다고|한다|했다|하며| 입장)",
    r"환영(?:한다고|한다|했다|하며)",
    r"긍정적(?:이라고|으로)\s*(?:평가|전망)",
    r"(?:정당|타당|바람직|필요)하(?:다고|다|며)",
    r"(?:논의|검토)(?:하자|하자는|해야|할\\s*필요)",
    r"경의(?:를)?\s*표",
    r"(?:재표결|재투표).{0,20}(?:추진|촉구|요구)",
    r"공포(?:해\s*달라|를\s*(?:촉구|요구)|해야)",
    r"(?:것이|게)\s*맞(?:다고|다|으며|고)",
    r"(?:제정|통과|처리|공포|시행|도입|추진|성사|확정|마무리)"
    r".{0,20}다행(?:이라고|이다|이라|스럽|하게)?",
    r"필수불가결",
    r"급선무",
    r"(?:대표적인\s*)?(?:성공|우수)\s*사례",
    r"(?:매우\s*)?기쁘",
    r"(?:성과|협력|시너지|발전|성장).{0,20}기대",
    r"(?:완벽한\s*)?신임",
    r"많은\s*이점",
    r"상대적\s*우위",
    r"뜨거운\s*반응",
    r"긍정적인?\s*반응",
    r"호응",
    r"세계(?:적|\s*최고|\s*1위)\s*수준",
    r"버팀목",
    r"진일보",
    r"(?:중요한|의미\s*있는)\s*이정표",
    r"(?:상당한\s*수준의\s*)?진전",
)

NEGATIVE_PATTERNS = (
    r"반대(?:한다고|한다|했다|하며|하고|해| 입장)",
    r"우려(?:를)?\s*(?:나타|표|제기|했다|한다)",
    r"우려(?:된다고|된다)",
    r"우려스럽",
    r"비판(?:한다고|한다|했다|하며|하고|해)",
    r"규탄(?:한다고|한다|했다|하며| 대회| 집회)",
    r"반발(?:해|하여|하고|하며|했다고|했다|한다| 입장| 집회)",
    r"저지(?:해야|하자|를\s*(?:촉구|요구)| 투쟁| 집회| 파업)",
    r"폐기(?:하|를|해야| 요구)",
    r"철회(?:하|를|해야| 요구)",
    r"부당하(?:다고|다|며)",
    r"악법",
    r"폭거",
    r"최악",
    r"물러나라",
    r"사퇴",
    r"퇴진",
    r"몽니",
    r"사퇴(?:하라|해야|를\\s*(?:촉구|요구))",
    r"퇴진(?:하라|해야|을\\s*(?:촉구|요구))",
    r"터무니없는\s*거짓",
    r"실망스(?:럽|러)",
    r"불안하",
    r"자해행위",
    r"국익\s*침해",
)


def _normalize_word(word: str) -> str:
    """간단한 조사·어미를 제거해 Target 비교용 단어로 만든다."""

    for suffix in (
        "으로",
        "에서",
        "에게",
        "하고",
        "하며",
        "해서",
        "한다",
        "하는",
        "하다",
        "해",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "과",
        "와",
        "에",
        "도",
    ):
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]

    return word


def _extract_terms(text: str) -> list[str]:
    words = re.findall(r"[가-힣A-Za-z0-9]+", text)

    return [
        normalized
        for word in words
        if (
            (normalized := _normalize_word(word))
            and len(normalized) >= 2
            and normalized not in TARGET_STOPWORDS
        )
    ]


def _is_target_related(target: str, content: str) -> bool:
    """Target의 이슈 식별 단서가 실제로 겹치는지 확인한다."""

    target_terms = _extract_terms(target)
    content_terms = _extract_terms(content)

    if not target_terms or not content_terms:
        return False

    anchor_terms = [
        term
        for term in target_terms
        if term not in TARGET_GENERIC_TERMS
    ]

    terms_to_match = (
        anchor_terms
        if anchor_terms
        else target_terms
    )

    return any(
        target_term in content_term
        or content_term in target_term
        for target_term in terms_to_match
        for content_term in content_terms
    )


def _introduces_other_topic(target: str, sentence: str) -> bool:
    if _is_target_related(target, sentence):
        return False

    return any(marker in sentence for marker in TOPIC_SWITCH_MARKERS)


def _select_target_sentences(target: str, content: str) -> list[str]:
    """
    Target 직접 언급 문장을 우선 선택한다.

    Target 문장이 나온 뒤 모든 문장을 계속 포함하지 않고,
    바로 다음 문장이 stance 표현을 포함하는 경우에만 문맥을 이어받는다.
    """

    sentences = split_sentences(content)

    if not sentences:
        return []

    target_related = [
        _is_target_related(target, sentence)
        for sentence in sentences
    ]

    selected = []

    for index, sentence in enumerate(sentences):
        if target_related[index]:
            selected.append(sentence)
            continue

        previous_is_target = (
            index > 0
            and target_related[index - 1]
        )

        if (
            previous_is_target
            and STANCE_SIGNAL_PATTERN.search(sentence)
            and not _introduces_other_topic(target, sentence)
        ):
            selected.append(sentence)

    return selected


def _target_actions(target: str) -> list[str]:
    """Target 자체를 구성하는 핵심 행위 표현을 찾는다."""

    actions = [
        action
        for action in ACTION_KEYWORDS
        if action in target
    ]

    if actions:
        return actions

    terms = _extract_terms(target)

    return terms[-1:] if terms else []


def _blocking_terms_outside_target(
    target: str,
    sentence: str,
) -> list[str]:
    """
    Target 자체에는 없지만 문장에 새롭게 등장한 방해/철회 행위를 찾는다.

    다른 법안명이나 복합명사 안에 포함된 단어는
    독립적인 blocking action으로 보지 않는다.
    """

    matched = []
    occupied_spans = []

    # 긴 표현부터 잡아 '거부권' 안의 '거부'처럼
    # 같은 위치가 중복 매칭되는 것을 줄인다.
    for term in sorted(
        BLOCKING_TERMS,
        key=len,
        reverse=True,
    ):
        if term in target:
            continue

        for match in re.finditer(
            re.escape(term),
            sentence,
        ):
            start, end = match.span()

            # '면허취소법'처럼 다른 명칭의 일부인 경우 제외.
            if (
                end < len(sentence)
                and sentence[end] == "법"
            ):
                continue

            overlaps = any(
                start < used_end
                and end > used_start
                for used_start, used_end
                in occupied_spans
            )

            if overlaps:
                continue

            matched.append(term)
            occupied_spans.append(
                (start, end)
            )

    return matched


def _blocking_action_direction(
    target: str,
    sentence: str,
) -> str | None:
    """
    Target을 막는 행위를 다시 지지/비판하는 관계를 해석한다.

    막는 행위를 비판 -> 원래 Target 지지
    막는 행위를 지지 -> 원래 Target 반대
    """

    blocking_terms = _blocking_terms_outside_target(
        target,
        sentence,
    )

    if not blocking_terms:
        return None

    criticism = (
        r"규탄|비판|비난|반발|부당|문제|무책임|선동|"
        r"어깃장|몽니|외면|왜곡|무모|도전|반대"
    )

    approval = (
        r"환영|지지|찬성|촉구|요구|건의|"
        r"행사해야|필요"
    )

    for term in blocking_terms:
        escaped = re.escape(term)

        if re.search(
            rf"{escaped}.{{0,40}}(?:{criticism})",
            sentence,
        ):
            return "positive"

        if re.search(
            rf"(?:{criticism}).{{0,20}}{escaped}",
            sentence,
        ):
            return "positive"

        if re.search(
            rf"{escaped}.{{0,40}}(?:{approval})",
            sentence,
        ):
            return "negative"

        if re.search(
            rf"(?:{approval}).{{0,20}}{escaped}",
            sentence,
        ):
            return "negative"

    return None


def _target_action_direction(
    target: str,
    sentence: str,
) -> str | None:
    """Target의 핵심 행위를 직접 지지/반대하는 표현을 찾는다."""

    positive_cues = (
        r"지지(?:한다고|한다|했다|하며| 입장)|"
        r"찬성(?:한다고|한다|했다|하며| 입장)|"
        r"환영(?:한다고|한다|했다|하며)|"
        r"필요하|정당하|타당하|바람직하"
    )

    negative_cues = (
        r"반대|저지|폐기|철회|중단|취소|"
        r"부당|문제|우려|비판|규탄|반발"
    )

    target_actions = _target_actions(target)

    if any(
        action in {"검토", "논의"}
        for action in target_actions
    ):
        if re.search(
            r"(?:검토|논의).{0,20}"
            r"(?:해야|하자|하자는|필요하|바람직하)",
            sentence,
        ):
            return "positive"

    for action in target_actions:
        escaped = re.escape(action)

        # "제정을 요구", "시행을 촉구"처럼
        # Target 행위 자체를 요구하는 경우.
        if re.search(
            rf"{escaped}(?:안)?(?:을|를)?\s*(?:요구|촉구)",
            sentence,
        ):
            return "positive"

        if re.search(
            rf"{escaped}.{{0,30}}(?:{positive_cues})",
            sentence,
        ):
            return "positive"

        if re.search(
            rf"(?:{positive_cues}).{{0,20}}{escaped}",
            sentence,
        ):
            return "positive"

        if re.search(
            rf"{escaped}.{{0,30}}(?:{negative_cues})",
            sentence,
        ):
            return "negative"

        if re.search(
            rf"(?:{negative_cues}).{{0,20}}{escaped}",
            sentence,
        ):
            return "negative"

    return None


def _sentence_direction(
    target: str,
    sentence: str,
) -> str:
    """Target과의 관계를 기준으로 문장의 stance를 판정한다."""

    # 찬성/반대가 투표 수치로 제시된 표결 결과는
    # 기사나 발언자의 입장이 아니라 발생 사실이다.
    has_vote_counts = (
        re.search(r"찬성\s*\d+\s*표", sentence)
        and re.search(r"반대\s*\d+\s*표", sentence)
    )

    if has_vote_counts and re.search(
        r"표결|투표|가결|부결|재의결",
        sentence,
    ):
        return "neutral"

    blocking_terms = _blocking_terms_outside_target(
        target,
        sentence,
    )

    # Target을 막는 사건이 단순히 발생했다는 보도는
    # 그 사건에 대한 지지·반대를 의미하지 않는다.
    # 방향 해석보다 먼저 처리해야 한다.
    if blocking_terms and re.search(
        r"(?:행사|부결|폐기|철회|중단|취소)"
        r"(?:됐|되었|했다|하였다|됐습니다|되었습니다|했습니다)",
        sentence,
    ):
        return "neutral"

    # Target을 막는 행위를 지지하거나 비판하는 경우.
    blocking_direction = _blocking_action_direction(
        target,
        sentence,
    )

    if blocking_direction is not None:
        return blocking_direction

    # 2. Target의 핵심 행위 자체를 평가하는 경우.
    action_direction = _target_action_direction(
        target,
        sentence,
    )

    if action_direction is not None:
        return action_direction

    # 3. 직접적인 stance 표현.
    has_positive = any(
        re.search(pattern, sentence)
        for pattern in POSITIVE_PATTERNS
    )

    has_negative = any(
        re.search(pattern, sentence)
        for pattern in NEGATIVE_PATTERNS
    )

    if has_positive and not has_negative:
        return "positive"

    if has_negative and not has_positive:
        return "negative"

    return "neutral"


def classify_stance(target: str, content: str) -> dict:
    """이슈 Target을 기준으로 기사 전체의 Stance를 분류한다."""

    if not target or not content:
        return {
            "stance": "neutral",
            "stance_confidence": 0.0,
        }

    relevant_sentences = _select_target_sentences(
        target,
        content,
    )

    if not relevant_sentences:
        return {
            "stance": "neutral",
            "stance_confidence": 0.0,
        }

    directions = [
        _sentence_direction(target, sentence)
        for sentence in relevant_sentences
    ]

    positive_count = directions.count("positive")
    negative_count = directions.count("negative")

    directional_count = (
        positive_count
        + negative_count
    )

    if directional_count == 0:
        return {
            "stance": "neutral",
            "stance_confidence": 0.5,
        }

    if positive_count == negative_count:
        return {
            "stance": "neutral",
            "stance_confidence": 0.5,
        }

    if positive_count > negative_count:
        stance = "positive"
        winning_count = positive_count
    else:
        stance = "negative"
        winning_count = negative_count

    confidence = (
        winning_count
        / directional_count
    )

    return {
        "stance": stance,
        "stance_confidence": round(
            confidence,
            3,
        ),
    }


def select_stance_evidence_sentences(
    target: str,
    content: str,
    stance: str,
) -> list[str]:
    """최종 stance와 같은 방향의 Target 관련 근거 문장을 반환한다."""

    if stance not in {
        "positive",
        "neutral",
        "negative",
    }:
        raise ValueError(
            f"invalid stance: {stance}"
        )

    selected = _select_target_sentences(
        target,
        content,
    )

    return [
        sentence
        for sentence in selected
        if _sentence_direction(
            target,
            sentence,
        ) == stance
    ]
