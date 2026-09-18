"""벡터를 이슈로 묶는 축소·군집 로직 — STEP 3 의 심장."""

import collections
from dataclasses import dataclass

import numpy as np

from .centroids import issue_centroids, nearest_issue_assignments


@dataclass
class ClusterConfig:
    # 384차원 그대로는 HDBSCAN 이 밀도를 못 잰다(차원의 저주). 5차원은 설계 때 정한 값 —
    # 국소 구조 보존과 밀도 추정 사이의 절충. 0 이면 축소를 생략한다(저차원 테스트용).
    umap_dims: int = 5
    # 이웃을 넓게 봐야 같은 이슈의 조각들이 붙는다 — 골드셋 스윕(티켓 104)에서 정밀도·과분리·
    # 재현율이 함께 좋아졌다(과분리의 원인은 min_cluster_size 가 아니었음).
    # 표본 수보다 크면 umap 이 알아서 줄인다 — 초소형 창(테스트)도 안전
    umap_neighbors: int = 50
    # 몇 건부터 이슈로 인정할지. 키우면 작은 군집이 노이즈로 밀려나기만 해 과분리가
    # 오히려 나빠진다(같은 스윕). 값의 근거는 티켓 89·104
    min_cluster_size: int = 3
    # UMAP 은 seed 를 주면 단일 스레드로 돌지만, 실행마다 같은 이슈가 나오는 재현성이
    # 속도보다 우선이다 — 48h 창 수만 건은 어차피 수십 초 수준
    seed: int = 42
    # 고정 지도(티켓 128)가 노이즈로 남긴 점 중 기존 이슈 중심과 이 이상 가까운 것은 새 이슈
    # 후보에서 뺀다 — 뒤의 품질 구제가 붙일 몫이라 값도 구제 문턱과 같다
    near_issue_sim: float = 0.85
    # 남은 노이즈를 원래 벡터 공간의 코사인 평균 연결로 묶어 새 이슈로 인정할 문턱. 값의 근거는 티켓 128
    new_issue_sim: float = 0.70


def fit_map(matrix, config: ClusterConfig):
    """UMAP 을 학습해 reducer 를 돌려준다. 학습 점 좌표는 embedding_, 새 점은 transform 으로 얹는다.

    입력 벡터는 정규화돼 있으므로 cosine 거리로 이웃을 잰다. umap 임포트를 함수 안에 두는
    이유: numba 가 무거워서, clustering 의존성 그룹을 설치하지 않은 환경에서도 패키지의
    다른 모듈은 그대로 임포트되어야 한다.
    """
    import umap

    return umap.UMAP(n_components=config.umap_dims, n_neighbors=config.umap_neighbors,
                     metric="cosine", random_state=config.seed).fit(matrix)


def reduce_vectors(matrix, config: ClusterConfig):
    """UMAP 을 새로 학습해 좌표만 돌려준다 — 지도를 고정하지 않는 옛 경로와 스윕용."""
    return fit_map(matrix, config).embedding_


def cluster_vectors(points, config: ClusterConfig):
    """HDBSCAN 으로 군집 라벨을 매긴다. -1 은 노이즈 — 지우지 않고 STEP 4 가 구제한다."""
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=config.min_cluster_size, copy=True).fit_predict(points)


def attach_new_issues(matrix, labels, config: ClusterConfig):
    """고정 지도가 노이즈로 남긴 점에서 새 이슈를 찾아 labels 에 새 번호로 덧붙인다. (labels, 새 이슈 수).

    transform 으로 얹힌 기사는 학습 점 이웃 옆에 놓이므로, 지도 학습 뒤 터진 사건의 기사들은
    서로 가까워야 할 학습 이웃이 없어 흩어지고 어떤 transform 으로도 밀도를 못 채운다(티켓 128
    시뮬레이션). 그래서 노이즈를 원래 벡터 공간에서 코사인 평균 연결로 묶는다 — UMAP 이 없어
    점이 늘어도 기존 병합이 뒤집히지 않는다. 기존 이슈 중심 곁의 점은 구제 몫으로 남긴다.
    """
    from sklearn.cluster import AgglomerativeClustering

    labels = np.array(labels, dtype=int)
    noise = [int(i) for i in np.flatnonzero(labels < 0)]
    if len(noise) < config.min_cluster_size:
        return labels, 0

    vectors_by_issue = collections.defaultdict(list)
    for i in np.flatnonzero(labels >= 0):
        vectors_by_issue[int(labels[i])].append(matrix[i])
    centroid_matrix, centroid_ids = issue_centroids(vectors_by_issue)
    near = nearest_issue_assignments({i: matrix[i] for i in noise}, centroid_matrix, centroid_ids,
                                     config.near_issue_sim)
    rest = [i for i in noise if i not in near]
    if len(rest) < config.min_cluster_size:
        return labels, 0

    groups = AgglomerativeClustering(n_clusters=None, distance_threshold=1 - config.new_issue_sim,
                                     metric="cosine", linkage="average").fit_predict(matrix[rest])
    sizes = collections.Counter(int(group) for group in groups)
    next_label = int(labels.max()) + 1
    label_of_group = {}
    for i, group in zip(rest, groups):
        group = int(group)
        if sizes[group] < config.min_cluster_size:
            continue
        if group not in label_of_group:
            label_of_group[group] = next_label
            next_label += 1
        labels[i] = label_of_group[group]
    return labels, len(label_of_group)
