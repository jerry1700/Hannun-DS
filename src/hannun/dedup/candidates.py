"""근사 중복 후보 찾기 — 정제본의 char 4-gram MinHash 를 LSH 두 개로 인덱싱한다.

전수 비교는 5만 건이면 25억 쌍이라 불가능하다. 여기서는 "비슷할 가능성이 있는 쌍"만
추리고, 진짜 중복인지는 다음 단계(TF-IDF 검증)가 정한다. 그래서 문턱은 느슨하게 잡는다
— 여기서 놓친 쌍은 영영 못 찾지만, 잘못 잡은 쌍은 다음 단계가 걸러 준다.

인덱스가 둘인 이유: 자카드 LSH 는 짧은 속보가 긴 종합기사에 통째로 포함된 경우를
놓친다(합집합이 커서 자카드가 낮다). 이전 프로토타입에서 실측으로 확인한 것이라
containment 용 MinHashLSHEnsemble 을 처음부터 같이 쓴다.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
from datasketch import MinHash, MinHashLSH, MinHashLSHEnsemble


@dataclass
class CandidateConfig:
    """두 문턱 모두 후보 단계라 검증 단계(0.95/0.90)보다 느슨하다.

    containment 를 검증 기준까지 올리면 Ensemble 의 밴딩이 엄격해져 실제로는 통째로 포함된
    쌍(긴 기사 앞부분만 뗀 속보)도 놓친다. 값의 근거는 티켓 10. seed 는 MinHash 순열을
    정하므로 서명 캐시(signatures)의 유효 조건에 들어간다.
    """

    shingle_size: int = 4
    num_perm: int = 128
    seed: int = 1
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


def text_sha1(text: str):
    """서명 캐시의 키. 정제본이 한 글자라도 바뀌면 서명을 다시 만들어야 한다."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_minhash(shingles: set[str], num_perm: int = 128, seed: int = 1):
    minhash = MinHash(num_perm=num_perm, seed=seed)
    for shingle in shingles:
        minhash.update(shingle.encode("utf-8"))
    return minhash


def minhash_scheme():
    """이 datasketch 가 새 MinHash 에 쓰는 해시 방식 이름.

    서명 캐시의 유효 조건에 들어간다 — 라이브러리가 바뀌어 방식이 달라지면 옛 서명은
    조용히 틀린 후보를 만들기 때문이다.
    """
    return getattr(MinHash(num_perm=1), "scheme", "legacy")


def minhash_from(hashvalues: np.ndarray, num_perm: int = 128, seed: int = 1):
    """저장된 hashvalues 로 MinHash 를 되살린다. 순열은 같은 seed 로 다시 만들어지므로 동일하다."""
    values = np.asarray(hashvalues, dtype=np.uint64)
    scheme = minhash_scheme()
    if scheme == "legacy":   # datasketch 1.x 는 scheme 인자가 없다
        return MinHash(num_perm=num_perm, seed=seed, hashvalues=values)
    return MinHash(num_perm=num_perm, seed=seed, hashvalues=values, scheme=scheme)


def build_entries(rows: list[dict], config: CandidateConfig | None = None,
                  cached: dict | None = None):
    """rows(article_id·text)를 LSH 입력 (article_id, MinHash, shingle_count) 목록으로 만든다.

    cached 는 article_id → (text_sha1, hashvalues, shingle_count). 정제본 해시가 같으면 서명을
    다시 계산하지 않는다. entries 와 함께 새로 계산한 서명 fresh(같은 형식, 저장용)와
    skipped_short 를 돌려준다.
    """
    config = config or CandidateConfig()
    cached = cached or {}
    entries, fresh, skipped_short = [], {}, 0
    for row in rows:
        sha1 = text_sha1(row["text"])
        hit = cached.get(row["article_id"])
        if hit is not None and hit[0] == sha1:
            entries.append((row["article_id"], minhash_from(hit[1], config.num_perm, config.seed), hit[2]))
            continue
        shingles = shingle_set(row["text"], config.shingle_size)
        if not shingles:
            skipped_short += 1
            continue
        minhash = build_minhash(shingles, config.num_perm, config.seed)
        entries.append((row["article_id"], minhash, len(shingles)))
        fresh[row["article_id"]] = (sha1, minhash.hashvalues, len(shingles))
    return entries, fresh, skipped_short


def pairs_from_entries(entries: list[tuple], config: CandidateConfig | None = None):
    """(article_id, MinHash, shingle_count) 목록을 두 LSH 에 넣고 후보 쌍을 뽑는다."""
    config = config or CandidateConfig()
    found = CandidateResult()
    if len(entries) < 2:
        return found

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
            if _add_pair(found.pairs, article_id, other):
                found.from_jaccard += 1
        # containment 는 비대칭이라 짧은 쪽으로 질의해야 "짧은 기사가 긴 기사에 포함"이 잡힌다.
        # 모든 기사를 한 번씩 질의하므로 짧은 쪽 차례에 걸린다.
        for other in containment_index.query(minhash, size):
            if _add_pair(found.pairs, article_id, other):
                found.from_containment += 1
    return found


def find_candidate_pairs(rows: list[dict], config: CandidateConfig | None = None):
    """rows 의 각 항목은 article_id 와 text(정제본)를 가진 딕셔너리다."""
    config = config or CandidateConfig()
    entries, _, skipped_short = build_entries(rows, config)
    found = pairs_from_entries(entries, config)
    found.skipped_short = skipped_short
    return found


def _add_pair(pairs, a, b):
    if a == b:
        return False
    pair = (a, b) if a < b else (b, a)
    if pair in pairs:
        return False
    pairs.add(pair)
    return True
