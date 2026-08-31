"""후보 쌍이 진짜 중복인지 판정한다 — TF-IDF 코사인 또는 containment.

후보 단계(LSH)는 느슨해서 오탐이 섞여 있다. 여기서 두 조건 중 하나를 만족해야 중복이다.
코사인 ≥ cosine_threshold 는 전재·미세 수정 기사를, containment ≥ containment_threshold 는
짧은 속보가 긴 종합기사에 포함된 경우를 잡는다. containment 쪽은 짧은 글일수록 우연히
높아지므로 짧은 쪽이 containment_min_len 이상일 때만 인정한다.

재작성 기사(같은 사건을 다른 문장으로)는 코사인이 문턱에 못 미쳐 여기서 떨어진다.
의도된 동작이다 — 재작성은 중복이 아니라 같은 이슈이고, 묶는 것은 STEP 3 의 몫이다.
"""

from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .candidates import shingle_set


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
    result = VerifyResult()
    if not pairs:
        return result

    involved = sorted({article_id for pair in pairs for article_id in pair})
    # 벡터화는 후보에 걸린 기사만. 코사인은 쌍마다 두 벡터 내적이면 되고 전체 행렬이 필요 없다.
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4))
    matrix = vectorizer.fit_transform(texts[article_id] for article_id in involved)
    row_of = {article_id: i for i, article_id in enumerate(involved)}

    for a, b in sorted(pairs):
        cosine = cosine_similarity(matrix[row_of[a]], matrix[row_of[b]])[0, 0]
        if cosine >= config.cosine_threshold:
            result.confirmed.add((a, b))
            result.method[(a, b)] = "cosine"
            continue
        if _contained(texts[a], texts[b], config):
            result.confirmed.add((a, b))
            result.method[(a, b)] = "containment"
            continue
        result.rejected += 1
    return result


def _contained(text_a, text_b, config):
    short, long = (text_a, text_b) if len(text_a) <= len(text_b) else (text_b, text_a)
    if len(short) < config.containment_min_len:
        return False
    short_shingles = shingle_set(short, config.shingle_size)
    if not short_shingles:
        return False
    long_shingles = shingle_set(long, config.shingle_size)
    return len(short_shingles & long_shingles) / len(short_shingles) >= config.containment_threshold
