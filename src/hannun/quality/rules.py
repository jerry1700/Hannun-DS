"""품질 규칙 — 정형(템플릿) 이슈 판별과 노이즈 구제. STEP 4 의 판단 기준."""

from dataclasses import dataclass

import numpy as np


@dataclass
class QualityConfig:
    # 정형 이슈 판별 — 1월 2일 실측(티켓 93): 이 문턱에서 걸린 이슈 7개·기사 671건이
    # 전부 시황·인사·포토 류 템플릿이고 오폭 0. 더 느슨하면 지역·단독 보도가 걸린다
    structured_max_publishers: int = 3
    structured_min_size: int = 10
    # 노이즈 구제 — 후보 146건 전수 검수 실측: 0.85 이상 정탐률 97.4%,
    # 0.80~0.85 구간은 82.9%로 급락. 벡터가 정규화돼 있어 내적이 곧 코사인
    rescue_min_sim: float = 0.85
    # 증분 배정(티켓 102) — 시뮬레이션 실측: 0.85에서 재군집과 일치 0.789,
    # 0.80 이하는 0.35로 오배정이 압도. 구제와 같은 값이라 규칙이 하나로 통일된다
    assign_min_sim: float = 0.85


def structured_issue_ids(meta, config: QualityConfig):
    """이슈별 (규모, 언론사 수) 로 정형 이슈 번호 집합을 고른다.

    meta: {issue_local: (size, publishers)}
    """
    return {
        issue_id for issue_id, (size, publishers) in meta.items()
        if publishers <= config.structured_max_publishers and size >= config.structured_min_size
    }


def issue_centroids(vectors_by_issue):
    """이슈별 중심 벡터(정규화 평균). {issue_local: [vector, ...]} → (행렬, 번호 리스트)."""
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


def nearest_issue_assignments(vectors, centroid_matrix, centroid_ids, min_sim: float):
    """벡터를 가장 가까운 이슈 중심에 붙인다. 문턱 미달은 제외.

    구제(93)와 증분 배정(102)이 같은 기계를 쓴다.
    vectors: {article_id: vector} → {article_id: (issue_local, sim)}
    """
    if not len(centroid_ids) or not vectors:
        return {}
    result = {}
    for article_id, vector in vectors.items():
        v = np.asarray(vector, dtype="float32")
        norm = np.linalg.norm(v)
        if norm == 0:
            continue
        sims = centroid_matrix @ (v / norm)
        best = int(np.argmax(sims))
        if sims[best] >= min_sim:
            result[article_id] = (centroid_ids[best], float(sims[best]))
    return result
