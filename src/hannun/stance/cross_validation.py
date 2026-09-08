import re
import unicodedata
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STRICT_SEQUENCE_THRESHOLD = 0.8
RELAXED_SEQUENCE_THRESHOLD = 0.45
TFIDF_SIMILARITY_THRESHOLD = 0.33


def is_cross_validated(evidence: dict) -> bool:
    """서로 다른 언론사 2곳 이상에서 확인된 근거인지 판단한다."""

    publishers = {
        source["publisher_name"]
        for source in evidence["sources"]
    }

    return len(publishers) >= 2


def normalize_text(sentence: str) -> str:
    """문장 비교를 위해 공백·따옴표·유니코드를 정규화한다."""

    normalized = unicodedata.normalize("NFKC", sentence)
    normalized = normalized.lower()
    normalized = re.sub(r"[\"'“”‘’]", "", normalized)
    normalized = re.sub(r"(?<!\S)총(?!\S)", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip()

    return normalized


def normalize_numbers(sentence: str) -> str:
    """문장의 숫자를 공통 토큰으로 치환한다."""

    return re.sub(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        "<NUM>",
        sentence,
    )


def _extract_fact_numbers(sentence: str) -> list[str]:
    """사실 비교에 필요한 수치만 추출한다.

    인물 소개에 붙는 '(58·24기)' 같은
    나이·기수 메타데이터는 사실 수치 충돌에서 제외한다.
    """

    cleaned = re.sub(
        r"\(\s*\d{1,3}\s*[·,/]\s*\d{1,3}\s*기?\s*\)",
        "",
        sentence,
    )

    return re.findall(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        cleaned,
    )


def sentence_similarity(
    sentence_a: str,
    sentence_b: str,
) -> float:
    """두 문장의 표현 유사도를 0~1 사이 값으로 반환한다."""

    normalized_a = normalize_numbers(normalize_text(sentence_a))
    normalized_b = normalize_numbers(normalize_text(sentence_b))

    return SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    ).ratio()


def _pair_tfidf_similarity(
    sentence_a: str,
    sentence_b: str,
) -> float:
    """두 문장의 문자 n-gram TF-IDF 코사인 유사도를 계산한다."""

    normalized = [
        normalize_numbers(normalize_text(sentence_a)),
        normalize_numbers(normalize_text(sentence_b)),
    ]

    try:
        matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
        ).fit_transform(normalized)
    except ValueError:
        return 0.0

    return float(cosine_similarity(matrix)[0, 1])


def is_same_fact_candidate(
    sentence_a: str,
    sentence_b: str,
    threshold: float = STRICT_SEQUENCE_THRESHOLD,
) -> bool:
    """문장 형태 또는 문자 n-gram 유사도로 같은 사실 후보인지 판단한다."""

    sequence_score = sentence_similarity(
        sentence_a,
        sentence_b,
    )

    if sequence_score >= threshold:
        return True

    if sequence_score < RELAXED_SEQUENCE_THRESHOLD:
        return False

    return (
        _pair_tfidf_similarity(
            sentence_a,
            sentence_b,
        )
        >= TFIDF_SIMILARITY_THRESHOLD
    )

def has_numeric_conflict(sentence_a: str, sentence_b: str) -> bool:
    """같은 사실 후보에서 수치가 서로 다른지 판단한다."""

    numbers_a = _extract_fact_numbers(sentence_a)
    numbers_b = _extract_fact_numbers(sentence_b)

    if not numbers_a or not numbers_b:
        return False

    if numbers_a == numbers_b:
        return False

    return is_same_fact_candidate(
        sentence_a,
        sentence_b,
    )


def has_conflict_with_others(
    evidence: dict,
    all_evidence: list[dict],
) -> bool:
    """다른 근거 문장과 수치 충돌이 있는지 확인한다."""

    sentence = evidence["sentence"]

    for other in all_evidence:
        if other is evidence:
            continue

        if has_numeric_conflict(sentence, other["sentence"]):
            return True

    return False


def _normalized_similarity_at_least(
    normalized_a: str,
    normalized_b: str,
    threshold: float = 0.8,
) -> bool:
    """정확한 유사도 계산 전 상한 검사를 사용해 빠르게 탈락시킨다."""

    matcher = SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    )

    if matcher.real_quick_ratio() < threshold:
        return False

    if matcher.quick_ratio() < threshold:
        return False

    return matcher.ratio() >= threshold


def _build_tfidf_similarity_matrix(
    evidence_list: list[dict],
):
    """전체 후보 문장의 문자 n-gram TF-IDF 유사도 행렬을 만든다."""

    count = len(evidence_list)

    if count == 0:
        return []

    sentences = [
        normalize_numbers(
            normalize_text(evidence["sentence"])
        )
        for evidence in evidence_list
    ]

    try:
        matrix = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
        ).fit_transform(sentences)
    except ValueError:
        return [
            [0.0] * count
            for _ in range(count)
        ]

    return cosine_similarity(matrix)


def _same_fact_by_scores(
    normalized_a: str,
    normalized_b: str,
    tfidf_score: float,
) -> bool:
    """엄격 문자열 유사도 또는 완화된 문자열+TF-IDF 조건을 적용한다."""

    matcher = SequenceMatcher(
        None,
        normalized_a,
        normalized_b,
    )

    if (
        matcher.real_quick_ratio()
        < RELAXED_SEQUENCE_THRESHOLD
    ):
        return False

    if (
        matcher.quick_ratio()
        < RELAXED_SEQUENCE_THRESHOLD
    ):
        return False

    sequence_score = matcher.ratio()

    if sequence_score >= STRICT_SEQUENCE_THRESHOLD:
        return True

    return (
        sequence_score >= RELAXED_SEQUENCE_THRESHOLD
        and tfidf_score >= TFIDF_SIMILARITY_THRESHOLD
    )


def merge_similar_evidence(
    evidence_list: list[dict],
) -> list[dict]:
    """표현이 다른 동일 사실 후보의 출처를 하나로 합친다."""

    normalized = [
        normalize_numbers(
            normalize_text(evidence["sentence"])
        )
        for evidence in evidence_list
    ]

    numbers = [
        _extract_fact_numbers(
            evidence["sentence"]
        )
        for evidence in evidence_list
    ]

    tfidf_similarities = (
        _build_tfidf_similarity_matrix(
            evidence_list
        )
    )

    merged = []
    representative_indices = []

    for index, evidence in enumerate(evidence_list):
        matched = False

        for group_index, group in enumerate(merged):
            representative_index = (
                representative_indices[group_index]
            )

            tfidf_score = float(
                tfidf_similarities[
                    index,
                    representative_index,
                ]
            )

            if not _same_fact_by_scores(
                normalized[index],
                normalized[representative_index],
                tfidf_score,
            ):
                continue

            numbers_a = numbers[index]
            numbers_b = numbers[representative_index]

            if (
                numbers_a
                and numbers_b
                and numbers_a != numbers_b
            ):
                continue

            existing_article_ids = {
                source["article_id"]
                for source in group["sources"]
            }

            for source in evidence["sources"]:
                if (
                    source["article_id"]
                    not in existing_article_ids
                ):
                    group["sources"].append(source)

            matched = True
            break

        if not matched:
            merged.append(
                {
                    "sentence": evidence["sentence"],
                    "sources": list(evidence["sources"]),
                }
            )
            representative_indices.append(index)

    return merged

def _find_conflicting_indices(
    evidence_list: list[dict],
) -> set[int]:
    """같은 사실인데 수치가 다른 후보를 한 번의 쌍 비교로 찾는다."""

    numbers = [
        _extract_fact_numbers(
            evidence["sentence"]
        )
        for evidence in evidence_list
    ]

    normalized = [
        normalize_numbers(
            normalize_text(evidence["sentence"])
        )
        for evidence in evidence_list
    ]

    tfidf_similarities = (
        _build_tfidf_similarity_matrix(
            evidence_list
        )
    )

    conflicting = set()

    for index_a in range(len(evidence_list)):
        for index_b in range(
            index_a + 1,
            len(evidence_list),
        ):
            numbers_a = numbers[index_a]
            numbers_b = numbers[index_b]

            if not numbers_a or not numbers_b:
                continue

            if numbers_a == numbers_b:
                continue

            tfidf_score = float(
                tfidf_similarities[
                    index_a,
                    index_b,
                ]
            )

            if _same_fact_by_scores(
                normalized[index_a],
                normalized[index_b],
                tfidf_score,
            ):
                conflicting.add(index_a)
                conflicting.add(index_b)

    return conflicting

def select_common_facts(
    evidence_list: list[dict],
    limit: int = 3,
) -> list[dict]:
    """교차 검증된 근거 중 수치 충돌이 없는 문장을 선택한다."""

    conflicting_indices = _find_conflicting_indices(
        evidence_list
    )

    conflict_free = [
        evidence
        for index, evidence in enumerate(evidence_list)
        if index not in conflicting_indices
    ]

    merged_evidence = merge_similar_evidence(conflict_free)

    candidates = [
        evidence
        for evidence in merged_evidence
        if is_cross_validated(evidence)
    ]

    candidates.sort(
        key=lambda evidence: len(
            {
                source["publisher_name"]
                for source in evidence["sources"]
            }
        ),
        reverse=True,
    )

    return candidates[:limit]
