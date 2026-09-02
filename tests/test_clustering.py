from datetime import datetime, timezone

import pytest

from hannun.clustering import ClusterConfig, IssueStore, cluster
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
