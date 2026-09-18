"""이슈 중심 벡터와 최근접 배정 — 노이즈 구제(티켓 93)·증분 배정(티켓 102)·새 이슈 탐지(티켓 128)가 같은 기계를 쓴다."""

import numpy as np


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

    vectors 는 article_id → 벡터, 결과는 article_id → (issue_local, 유사도). 벡터가 정규화돼
    있어 내적이 곧 코사인이다.
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
