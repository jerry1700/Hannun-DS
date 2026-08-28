import re
import unicodedata
from difflib import SequenceMatcher


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


def is_same_fact_candidate(
    sentence_a: str,
    sentence_b: str,
    threshold: float = 0.8,
) -> bool:
    """표현이 조금 달라도 같은 사실을 설명하는 후보인지 판단한다."""

    return sentence_similarity(sentence_a, sentence_b) >= threshold


def has_numeric_conflict(sentence_a: str, sentence_b: str) -> bool:
    """같은 사실 후보에서 수치가 서로 다른지 판단한다."""

    numbers_a = re.findall(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        sentence_a,
    )
    numbers_b = re.findall(
        r"\d+(?:,\d{3})*(?:\.\d+)?",
        sentence_b,
    )

    same_fact = is_same_fact_candidate(
        sentence_a,
        sentence_b,
    )

    return (
        same_fact
        and numbers_a != numbers_b
        and bool(numbers_a or numbers_b)
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


def merge_similar_evidence(
    evidence_list: list[dict],
) -> list[dict]:
    """표현이 유사한 동일 사실 후보의 출처를 하나로 합친다."""

    merged = []

    for evidence in evidence_list:
        matched = False

        for group in merged:
            if (
                is_same_fact_candidate(
                    evidence["sentence"],
                    group["sentence"],
                )
                and not has_numeric_conflict(
                    evidence["sentence"],
                    group["sentence"],
                )
            ):
                existing_article_ids = {
                    source["article_id"]
                    for source in group["sources"]
                }

                for source in evidence["sources"]:
                    if source["article_id"] not in existing_article_ids:
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

    return merged


def select_common_facts(
    evidence_list: list[dict],
    limit: int = 3,
) -> list[dict]:
    """교차 검증된 근거 중 수치 충돌이 없는 문장을 선택한다."""

    conflict_free = [
        evidence
        for evidence in evidence_list
        if not has_conflict_with_others(
            evidence,
            evidence_list,
        )
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
