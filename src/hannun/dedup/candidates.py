"""근사 중복 후보 찾기 — 정제본의 char 4-gram MinHash 를 LSH 두 개로 인덱싱한다.

전수 비교는 5만 건이면 25억 쌍이라 불가능하다. 여기서는 "비슷할 가능성이 있는 쌍"만
추리고, 진짜 중복인지는 다음 단계(TF-IDF 검증)가 정한다. 그래서 문턱은 느슨하게 잡는다
— 여기서 놓친 쌍은 영영 못 찾지만, 잘못 잡은 쌍은 다음 단계가 걸러 준다.

인덱스가 둘인 이유: 자카드 LSH 는 짧은 속보가 긴 종합기사에 통째로 포함된 경우를
놓친다(합집합이 커서 자카드가 낮다). 이전 프로토타입에서 실측으로 확인한 것이라
containment 용 MinHashLSHEnsemble 을 처음부터 같이 쓴다.
"""

from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble


@dataclass
class CandidateConfig:
    """두 threshold 모두 후보 단계라 느슨하다. containment 를 0.9 로 두면 Ensemble 의 밴딩이
    엄격해져 실제 containment 1.0 인 쌍(긴 기사 앞 100자만 뗀 속보)도 놓치는 것을 실측했다.
    설계의 containment ≥0.90 은 검증 단계(TF-IDF) 기준이고 여기가 아니다."""

    shingle_size: int = 4
    num_perm: int = 128
    jaccard_threshold: float = 0.7
    containment_threshold: float = 0.7
    ensemble_partitions: int = 16


@dataclass
class CandidateResult:
    """pairs 의 각 원소는 (article_id, article_id), 사전순으로 고정."""

    pairs: set[tuple[str, str]] = field(default_factory=set)
    from_jaccard: int = 0
    from_containment: int = 0
    skipped_short: int = 0


def shingle_set(text: str, size: int = 4):
    return {text[i:i + size] for i in range(len(text) - size + 1)}


def build_minhash(shingles: set[str], num_perm: int = 128):
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def find_candidate_pairs(rows: list[dict], config: CandidateConfig | None = None):
    """rows 의 각 항목은 article_id 와 text(정제본)를 가진 딕셔너리다."""
    config = config or CandidateConfig()
    result = CandidateResult()

    entries = []
    for row in rows:
        shingles = shingle_set(row["text"], config.shingle_size)
        if not shingles:
            result.skipped_short += 1
            continue
        entries.append((row["article_id"], build_minhash(shingles, config.num_perm), len(shingles)))
    if len(entries) < 2:
        return result

    jaccard_index = MinHashLSH(threshold=config.jaccard_threshold, num_perm=config.num_perm)
    for article_id, minhash, _ in entries:
        jaccard_index.insert(article_id, minhash)
    containment_index = MinHashLSHEnsemble(
        threshold=config.containment_threshold, num_perm=config.num_perm,
        num_part=config.ensemble_partitions,
    )
    containment_index.index(entries)

    for article_id, minhash, size in entries:
        for other in jaccard_index.query(minhash):
            if _add_pair(result.pairs, article_id, other):
                result.from_jaccard += 1
        # containment 는 비대칭이라 짧은 쪽으로 질의해야 "짧은 기사가 긴 기사에 포함"이 잡힌다.
        # 모든 기사를 한 번씩 질의하므로 짧은 쪽 차례에 걸린다.
        for other in containment_index.query(minhash, size):
            if _add_pair(result.pairs, article_id, other):
                result.from_containment += 1
    return result


def _add_pair(pairs, a, b):
    if a == b:
        return False
    pair = (a, b) if a < b else (b, a)
    if pair in pairs:
        return False
    pairs.add(pair)
    return True
