from datetime import datetime, timezone

import numpy as np
import pytest
from conftest import FakeReducer

from hannun.clustering import ClusterConfig, IssueStore, MapStore, attach_new_issues, cluster
from hannun.embedding import EmbeddingStore

# 2차원 합성 벡터 — UMAP 없이(umap_dims=0) HDBSCAN 만으로 군집이 갈리는 배치.
# 뭉치 A(원점 근처) 4개, 뭉치 B(멀리) 3개, 외톨이 1개.
CLUSTER_A = [[0.00, 0.00], [0.01, 0.00], [0.00, 0.01], [0.01, 0.01]]
CLUSTER_B = [[5.00, 5.00], [5.01, 5.00], [5.00, 5.01]]
OUTLIER = [[100.0, -100.0]]


def build_embeddings(tmp_path, model="test-model"):
    store = EmbeddingStore(tmp_path / "gold")
    encoded_at = datetime(2026, 8, 20, tzinfo=timezone.utc)
    vectors = CLUSTER_A + CLUSTER_B + OUTLIER
    dates = ["2026-08-20"] * 5 + ["2026-08-21"] * 3  # 뭉치 B 의 일부가 다음 날로 넘어간다
    rows_by_date = {}
    for i, (vector, date_str) in enumerate(zip(vectors, dates)):
        rows_by_date.setdefault(date_str, []).append({
            "article_id": f"art{i}",
            "publisher_id": "pub",
            "published_date": date_str,
            "vector": vector,
            "dim": 2,
            "model": model,
            "encoded_at": encoded_at,
        })
    for date_str, rows in rows_by_date.items():
        store.write_partition(date_str, rows)
    return store


def config():
    return ClusterConfig(umap_dims=0, min_cluster_size=3)


def test_two_clusters_and_noise(tmp_path):
    embeddings = build_embeddings(tmp_path)
    store = IssueStore(tmp_path / "gold")
    stats = cluster(embeddings, store, config())
    df = store.read().set_index("article_id")

    assert stats.rows == 8 and stats.issues == 2 and stats.noise == 1
    # 같은 뭉치는 같은 이슈 번호, 다른 뭉치는 다른 번호
    a_labels = {df.loc[f"art{i}"].issue_local for i in range(4)}
    b_labels = {df.loc[f"art{i}"].issue_local for i in range(4, 7)}
    assert len(a_labels) == 1 and len(b_labels) == 1 and a_labels != b_labels
    assert df.loc["art7"].issue_local == -1  # 외톨이는 노이즈 — 지우지 않고 -1 로 남긴다
    assert df.loc["art0"].issue_size == 4 and df.loc["art4"].issue_size == 3


def test_window_crosses_partition_boundary(tmp_path):
    # 뭉치 B 는 8/20 두 건 + 8/21 세 건... 이 아니라 날짜가 갈려 있어도 한 창이면 한 이슈다
    embeddings = build_embeddings(tmp_path)
    store = IssueStore(tmp_path / "gold")
    cluster(embeddings, store, config())
    df = store.read()

    assert store.partition_dates() == ["2026-08-20", "2026-08-21"]
    b = df[df.article_id.isin(["art4", "art5", "art6"])]
    assert b.published_date.nunique() == 2 and b.issue_local.nunique() == 1


def test_rerun_is_deterministic(tmp_path):
    embeddings = build_embeddings(tmp_path)
    store = IssueStore(tmp_path / "gold")
    cluster(embeddings, store, config())
    first = store.read(columns=["article_id", "issue_local"]).values.tolist()
    cluster(embeddings, store, config())
    second = store.read(columns=["article_id", "issue_local"]).values.tolist()

    assert first == second


def test_mixed_models_raise(tmp_path):
    embeddings = build_embeddings(tmp_path)
    extra = [{
        "article_id": "art9", "publisher_id": "pub", "published_date": "2026-08-21",
        "vector": [1.0, 1.0], "dim": 2, "model": "other-model",
        "encoded_at": datetime(2026, 8, 21, tzinfo=timezone.utc),
    }]
    existing = embeddings.read_table("2026-08-21", "2026-08-21").to_pylist()
    embeddings.write_partition("2026-08-21", existing + extra)

    with pytest.raises(ValueError):
        cluster(embeddings, IssueStore(tmp_path / "gold"), config())


def test_reduce_fn_injection_is_used(tmp_path):
    # 축소 함수를 주입하면 그 결과로 군집한다 — 외톨이(art7)를 뭉치 B 자리로 옮기면
    # 원본 좌표에서는 노이즈였던 것이 군집에 들어간다
    embeddings = build_embeddings(tmp_path)
    store = IssueStore(tmp_path / "gold")
    stats = cluster(embeddings, store, ClusterConfig(umap_dims=5, min_cluster_size=3),
                    reduce_fn=lambda m: [[0.0, 0.0]] * 4 + [[9.0, 9.0]] * 4)

    assert stats.issues == 2 and stats.noise == 0
    df = store.read().set_index("article_id")
    assert df.loc["art7"].issue_local >= 0 and df.loc["art7"].issue_size == 4


def unit(*components):
    vector = np.array(components, dtype="float32")
    return (vector / np.linalg.norm(vector)).tolist()


# 정규화 벡터 — 이슈 0 은 x 축 근처, 노이즈 셋(n1~n3)은 y 축 근처에서 서로 가깝고, n4 는 이슈 0 곁, n5 는 외톨이
NEW_ISSUE_MATRIX = np.array([
    unit(1.0, 0.0, 0.0), unit(1.0, 0.05, 0.0), unit(1.0, 0.0, 0.05),
    unit(0.0, 1.0, 0.0), unit(0.05, 1.0, 0.0), unit(0.0, 1.0, 0.05),
    unit(1.0, 0.2, 0.0),
    unit(0.0, 0.0, 1.0),
], dtype="float32")
NEW_ISSUE_LABELS = [0, 0, 0, -1, -1, -1, -1, -1]


def test_attach_new_issues_groups_similar_noise_into_a_new_issue():
    labels, created = attach_new_issues(NEW_ISSUE_MATRIX, NEW_ISSUE_LABELS, ClusterConfig())

    assert created == 1
    assert labels[3] == labels[4] == labels[5] == 1


def test_attach_new_issues_leaves_points_near_existing_issue_to_rescue():
    labels, _ = attach_new_issues(NEW_ISSUE_MATRIX, NEW_ISSUE_LABELS, ClusterConfig())
    assert labels[6] == -1   # 이슈 0 중심과 코사인 0.98 — 구제가 붙일 몫


def test_attach_new_issues_leaves_lonely_noise_as_noise():
    labels, _ = attach_new_issues(NEW_ISSUE_MATRIX, NEW_ISSUE_LABELS, ClusterConfig())
    assert labels[7] == -1


def test_attach_new_issues_respects_similarity_threshold():
    strict = ClusterConfig(new_issue_sim=0.9999)
    labels, created = attach_new_issues(NEW_ISSUE_MATRIX, NEW_ISSUE_LABELS, strict)
    assert created == 0 and list(labels) == NEW_ISSUE_LABELS


def test_cluster_with_map_store_fits_once_then_transforms_new_articles(tmp_path):
    embeddings = build_embeddings(tmp_path)
    store = IssueStore(tmp_path / "gold")
    map_store = MapStore(tmp_path / "gold")
    # umap_dims=2 라 고정 지도 경로를 타되 FakeReducer 가 좌표를 그대로 돌려줘 HDBSCAN 결과는 종전과 같다
    settings = dict(config=ClusterConfig(umap_dims=2, min_cluster_size=3), map_store=map_store,
                    fit_fn=FakeReducer)

    first = cluster(embeddings, store, **settings)
    second = cluster(embeddings, store, **settings)
    assert first.map_fitted and not second.map_fitted and second.map_transformed == 0
    assert first.issues == second.issues == 2

    late = {"article_id": "art9", "publisher_id": "pub", "published_date": "2026-08-21",
            "vector": [5.02, 5.01], "dim": 2, "model": "test-model",
            "encoded_at": datetime(2026, 8, 21, tzinfo=timezone.utc)}
    existing = embeddings.read_table("2026-08-21", "2026-08-21").to_pylist()
    embeddings.write_partition("2026-08-21", existing + [late])
    third = cluster(embeddings, store, **settings)

    assert not third.map_fitted and third.map_transformed == 1
    # transform 점(+100)이 뭉치 B 에서 떨어져 노이즈가 되고, 2단계는 혼자라 새 이슈를 만들지 않는다
    assert store.read().set_index("article_id").loc["art9"].issue_local == -1 and third.new_issues == 0
