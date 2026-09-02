"""벡터를 이슈로 묶는 축소·군집 로직 — STEP 3 의 심장."""

from dataclasses import dataclass


@dataclass
class ClusterConfig:
    # 384차원 그대로는 HDBSCAN 이 밀도를 못 잰다(차원의 저주). 5차원은 설계 때 정한 값 —
    # 국소 구조 보존과 밀도 추정 사이의 절충. 0 이면 축소를 생략한다(저차원 테스트용).
    umap_dims: int = 5
    umap_neighbors: int = 15
    # 몇 건부터 이슈로 인정할지. 파일럿의 최소 이슈가 3건(북미 정상회담)이라 3 에서 시작
    min_cluster_size: int = 3
    # UMAP 은 seed 를 주면 단일 스레드로 돌지만, 실행마다 같은 이슈가 나오는 재현성이
    # 속도보다 우선이다 — 48h 창 수만 건은 어차피 수십 초 수준
    seed: int = 42


def reduce_vectors(matrix, config: ClusterConfig):
    """UMAP 으로 차원을 줄인다. 입력 벡터는 정규화돼 있으므로 cosine 거리로 이웃을 잰다.

    umap 임포트를 함수 안에 두는 이유: numba 가 무거워서, clustering 의존성 그룹을
    설치하지 않은 환경에서도 패키지의 다른 모듈은 그대로 임포트되어야 한다.
    """
    import umap

    reducer = umap.UMAP(n_components=config.umap_dims, n_neighbors=config.umap_neighbors,
                        metric="cosine", random_state=config.seed)
    return reducer.fit_transform(matrix)


def cluster_vectors(points, config: ClusterConfig):
    """HDBSCAN 으로 군집 라벨을 매긴다. -1 은 노이즈 — 지우지 않고 STEP 4 가 구제한다."""
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=config.min_cluster_size, copy=True).fit_predict(points)
