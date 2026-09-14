"""후보 쌍이 진짜 중복인지 판정한다 — TF-IDF 코사인 또는 containment.

후보 단계(LSH)는 느슨해서 오탐이 섞여 있다. 여기서 두 조건 중 하나를 만족해야 중복이다.
코사인 ≥ cosine_threshold 는 전재·미세 수정 기사를, containment ≥ containment_threshold 는
짧은 속보가 긴 종합기사에 포함된 경우를 잡는다. containment 쪽은 짧은 글일수록 우연히
높아지므로 짧은 쪽이 containment_min_len 이상일 때만 인정한다.

재작성 기사(같은 사건을 다른 문장으로)는 코사인이 문턱에 못 미쳐 여기서 떨어진다.
의도된 동작이다 — 재작성은 중복이 아니라 같은 이슈이고, 묶는 것은 STEP 3 의 몫이다.
"""

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .candidates import shingle_set

# 쌍별 코사인을 한 번에 계산할 덩어리. 정형 템플릿이 많은 창은 후보가 수십만 쌍이라 통째로 곱하면 메모리가 튄다
COSINE_CHUNK = 20_000


@dataclass
class VerifyConfig:
    cosine_threshold: float = 0.95
    containment_threshold: float = 0.90
    containment_min_len: int = 300
    shingle_size: int = 4


@dataclass
class VerifyResult:
    """confirmed 의 각 원소는 (article_id, article_id), 사전순. method 는 쌍별 판정 근거."""

    confirmed: set[tuple[str, str]] = field(default_factory=set)
    method: dict[tuple[str, str], str] = field(default_factory=dict)
    rejected: int = 0


def verify_pairs(texts: dict[str, str], pairs: set[tuple[str, str]], config: VerifyConfig | None = None):
    """texts 는 article_id → 정제본. pairs 는 후보 단계가 추린 (id, id) 집합."""
    config = config or VerifyConfig()
    verified = VerifyResult()
    if not pairs:
        return verified

    involved = sorted({article_id for pair in pairs for article_id in pair})
    # 벡터화는 후보에 걸린 기사만. TF-IDF 행은 L2 정규화돼 있어 쌍별 코사인 = 두 행의 내적 —
    # 쌍마다 함수를 부르지 않고 덩어리로 곱해 한 번에 뽑는다(티켓 123)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    matrix = vectorizer.fit_transform(texts[article_id] for article_id in involved)
    row_of = {article_id: i for i, article_id in enumerate(involved)}

    ordered = sorted(pairs)
    cosines = np.empty(len(ordered))
    for start in range(0, len(ordered), COSINE_CHUNK):
        chunk = ordered[start:start + COSINE_CHUNK]
        left = matrix[[row_of[a] for a, _ in chunk]]
        right = matrix[[row_of[b] for _, b in chunk]]
        cosines[start:start + len(chunk)] = np.asarray(left.multiply(right).sum(axis=1)).ravel()

    # containment 는 코사인에서 떨어진 쌍만 본다. 기사 하나가 여러 쌍에 걸리므로 4-gram 집합은
    # 기사당 한 번만 만든다 — 쌍마다 다시 만들면 기사 수의 몇 배를 재계산한다(티켓 123)
    shingles = {}

    def shingles_of(article_id):
        if article_id not in shingles:
            shingles[article_id] = shingle_set(texts[article_id], config.shingle_size)
        return shingles[article_id]

    for (a, b), cosine in zip(ordered, cosines):
        if cosine >= config.cosine_threshold:
            verified.confirmed.add((a, b))
            verified.method[(a, b)] = "cosine"
            continue
        if _contained(a, b, texts, shingles_of, config):
            verified.confirmed.add((a, b))
            verified.method[(a, b)] = "containment"
            continue
        verified.rejected += 1
    return verified


def _contained(a, b, texts, shingles_of, config):
    short, long = (a, b) if len(texts[a]) <= len(texts[b]) else (b, a)
    if len(texts[short]) < config.containment_min_len:
        return False
    short_shingles = shingles_of(short)
    if not short_shingles:
        return False
    overlap = len(short_shingles & shingles_of(long)) / len(short_shingles)
    return overlap >= config.containment_threshold
