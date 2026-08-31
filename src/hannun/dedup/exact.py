"""완전 중복 — 원문이 글자까지 같은 기사를 묶는다. STEP 1 의 첫 관문."""

import collections
import hashlib
from dataclasses import dataclass, field


@dataclass
class ExactDuplicates:
    """duplicate_of 는 중복 기사 → 대표 기사. 대표 기사 자신은 여기 없다."""

    duplicate_of: dict[str, str] = field(default_factory=dict)
    groups: int = 0


def find_exact_duplicates(rows: list[dict]):
    """원문 content 의 SHA-256 이 같은 기사들을 묶고 대표를 정한다.

    해시는 정제본이 아니라 **원문**으로 한다. 통신사 전재는 글자까지 같아서 여기서 걸리고,
    공백 하나라도 다르면 완전 중복이 아니므로 다음 단계(MinHash)로 넘긴다.
    rows 의 각 항목은 article_id, content, published_at, publisher_id 를 가진 딕셔너리다.
    """
    by_hash = collections.defaultdict(list)
    for row in rows:
        digest = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        by_hash[digest].append(row)

    result = ExactDuplicates()
    for group in by_hash.values():
        if len(group) < 2:
            continue
        representative = min(group, key=_representative_key)
        result.groups += 1
        for row in group:
            if row["article_id"] != representative["article_id"]:
                result.duplicate_of[row["article_id"]] = representative["article_id"]
    return result


def _representative_key(row):
    # 최초 발행이 대표. 발행 시각까지 같으면 article_id 로 고정해 실행마다 같은 대표가 나오게 한다.
    return (row["published_at"], row["article_id"])
