"""품질 규칙 — 정형(템플릿) 이슈 판별과 노이즈 구제. STEP 4 의 판단 기준."""

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityConfig:
    # 정형 이슈 판별 — 적은 언론사가 많은 기사를 내는 시황·인사·포토 류 템플릿을 잡는다.
    # 더 느슨하면 지역·단독 보도가 걸린다. 값의 근거는 티켓 93
    structured_max_publishers: int = 3
    structured_min_size: int = 10
    # 노이즈 구제 — 이 아래 구간에서 정탐률이 급락한다(티켓 93 검수). 벡터가 정규화돼 있어
    # 내적이 곧 코사인
    rescue_min_sim: float = 0.85
    # 증분 배정(티켓 102) — 낮추면 오배정이 압도한다(시뮬레이션). 구제와 같은 값이라 규칙이 하나다
    assign_min_sim: float = 0.85


def structured_issue_ids(meta: dict, config: QualityConfig):
    """이슈별 (규모, 언론사 수) 로 정형 이슈 번호 집합을 고른다. meta 는 issue_local → (size, publishers)."""
    return {
        issue_id for issue_id, (size, publishers) in meta.items()
        if publishers <= config.structured_max_publishers and size >= config.structured_min_size
    }


def issue_centroids(vectors_by_issue: dict):
    """이슈별 중심 벡터(정규화 평균)를 (행렬, 번호 리스트) 로. vectors_by_issue 는 issue_local → 벡터 목록."""
    ids, rows = [], []
    for issue_id, vectors in vectors_by_issue.items():
        mean = np.mean(np.asarray(vectors, dtype="float32"), axis=0)
        norm = np.linalg.norm(mean)
        if norm == 0:
            continue
        ids.append(issue_id)
        rows.append(mean / norm)
    if not rows:
        return np.empty((0, 0), dtype="float32"), []
    return np.stack(rows), ids


def nearest_issue_assignments(vectors: dict, centroid_matrix, centroid_ids: list, min_sim: float):
    """벡터를 가장 가까운 이슈 중심에 붙인다. 문턱 미달은 제외.

    구제(티켓 93)와 증분 배정(티켓 102)이 같은 기계를 쓴다. vectors 는 article_id → 벡터,
    결과는 article_id → (issue_local, 유사도).
    """
    if not len(centroid_ids) or not vectors:
        return {}
    placements = {}
    for article_id, vector in vectors.items():
        v = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        sims = centroid_matrix @ (v / norm)
        best = int(np.argmax(sims))
        if sims[best] >= min_sim:
            placements[article_id] = (centroid_ids[best], float(sims[best]))
    return placements
