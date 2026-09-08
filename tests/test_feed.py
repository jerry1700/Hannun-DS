from datetime import datetime, timezone

import pyarrow as pa

from hannun.clustering import RegistryStore
from hannun.embedding import EmbeddingStore
from hannun.feed import FeedConfig, SummaryStore, hot_score, summarize
from hannun.ingest import GoldStore
from hannun.ingest.gold import write_parquet_atomic
from hannun.quality import QualityStore

# 이슈 1: m1(중심 최근접)·m2(최초 발행)·m3 — 대표는 m1 이어야 한다
# 이슈 0: 정형 2건 / n1: 노이즈(요약에서 제외)
D1 = "2026-08-20"
T = {
    "m1": datetime(2026, 8, 20, 9, 0, tzinfo=timezone.utc),
    "m2": datetime(2026, 8, 20, 8, 0, tzinfo=timezone.utc),
    "m3": datetime(2026, 8, 20, 10, 0, tzinfo=timezone.utc),
    "s1": datetime(2026, 8, 20, 7, 0, tzinfo=timezone.utc),
    "s2": datetime(2026, 8, 20, 7, 30, tzinfo=timezone.utc),
    "n1": datetime(2026, 8, 20, 11, 0, tzinfo=timezone.utc),
}
VECTORS = {
    "m1": [0.0, 1.0], "m2": [0.2, 0.98], "m3": [-0.2, 0.98],
    "s1": [1.0, 0.0], "s2": [1.0, 0.01], "n1": [0.7, -0.7],
}
MEMBERS = [
    ("m1", "p1", 1, False), ("m2", "p2", 1, False), ("m3", "p3", 1, False),
    ("s1", "bot", 0, True), ("s2", "bot", 0, True), ("n1", "p4", -1, False),
]


def build_stores(tmp_path, with_registry=True):
    root = tmp_path / "gold"
    qualified_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
    QualityStore(root).write_partition(D1, [{
        "article_id": a, "publisher_id": pub, "published_date": D1,
        "issue_local": label, "issue_size": 3 if label == 1 else 2,
        "structured": structured, "rescued": False, "rescue_sim": None,
        "window_start": D1, "window_end": D1, "qualified_at": qualified_at,
    } for a, pub, label, structured in MEMBERS])
    EmbeddingStore(root).write_partition(D1, [{
        "article_id": a, "publisher_id": pub, "published_date": D1,
        "vector": VECTORS[a], "dim": 2, "model": "test-model", "encoded_at": qualified_at,
    } for a, pub, _, _ in MEMBERS])
    gold_table = pa.Table.from_pylist(
        [{"article_id": a, "published_at": T[a]} for a, _, _, _ in MEMBERS],
        schema=pa.schema([("article_id", pa.string()),
                          ("published_at", pa.timestamp("us", tz="UTC"))]))
    write_parquet_atomic(gold_table, GoldStore(root).partition_path(D1))
    if with_registry:
        RegistryStore(root).write_window(D1, [{
            "window_start": D1, "window_end": D1, "article_id": a, "published_date": D1,
            "issue_local": label, "issue_id": 100 + label, "status": "new",
            "succeeded_at": qualified_at,
        } for a, _, label, _ in MEMBERS if label >= 0])
    return root


def run(root):
    stats = summarize(QualityStore(root), EmbeddingStore(root), RegistryStore(root),
                      GoldStore(root), SummaryStore(root), start_date=D1, end_date=D1)
    return stats, SummaryStore(root).read(D1).set_index("issue_local")


def test_representative_is_centroid_nearest_not_earliest(tmp_path):
    _, df = run(build_stores(tmp_path))
    row = df.loc[1]

    assert row.representative == "m1"  # 최초 발행(m2)이 아니라 중심 최근접
    assert row.first_published_at == T["m2"] and row.last_published_at == T["m3"]
    assert row.issue_size == 3 and row.publishers == 3


def test_issue_id_and_structured_flag(tmp_path):
    stats, df = run(build_stores(tmp_path))

    assert stats.issues == 2 and stats.structured_issues == 1 and stats.with_issue_id == 2
    assert df.loc[1].issue_id == 101 and df.loc[0].issue_id == 100
    assert bool(df.loc[0].structured) and not bool(df.loc[1].structured)


def test_noise_is_excluded(tmp_path):
    _, df = run(build_stores(tmp_path))

    assert len(df) == 2 and -1 not in df.index


def test_missing_registry_yields_minus_one(tmp_path):
    stats, df = run(build_stores(tmp_path, with_registry=False))

    assert stats.with_issue_id == 0
    assert set(df.issue_id) == {-1}


def test_hot_score_prefers_publisher_diversity():
    config = FeedConfig()
    # 같은 규모·신선도면 언론사가 많은 쪽이 화제성이 높다
    assert hot_score(10, 15, 0.0, config) > hot_score(10, 3, 0.0, config)
    # 다양성 가중 때문에 규모 열세도 다양성이 크게 앞서면 뒤집힌다
    assert hot_score(10, 15, 0.0, config) > hot_score(30, 2, 0.0, config)


def test_hot_score_decays_with_staleness():
    config = FeedConfig(recency_tau_hours=24.0)
    fresh = hot_score(10, 10, 0.0, config)
    day_old = hot_score(10, 10, 24.0, config)

    assert day_old < fresh
    assert abs(day_old / fresh - 0.3679) < 1e-3  # τ시간 경과 = 1/e
    assert hot_score(10, 10, -5.0, config) == fresh  # 미래 시각은 0으로 클램프


def test_pipeline_writes_hot_score(tmp_path):
    _, df = run(build_stores(tmp_path))

    # 이슈 1(3건/3곳, 최신)이 이슈 0(2건/1곳, 오래됨)보다 화제성 높다
    assert df.loc[1].hot_score > df.loc[0].hot_score > 0
