from datetime import datetime, timezone

import pyarrow as pa
import pytest

from hannun.clustering import RegistryStore
from hannun.embedding import EmbeddingStore
from hannun.feed import (CATEGORIES, FeedConfig, SummaryStore, hot_score, majority_category,
                         normalize_category, summarize)
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
# 이슈 1: 대표 m1 은 라벨이 없지만 m2·m3 이 정치 → 다수결로 정치. 이슈 0: 전부 OTHER
CATEGORY = {"m1": "OTHER", "m2": "정치", "m3": "정치", "s1": "OTHER", "s2": "OTHER", "n1": "경제"}


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
        [{"article_id": a, "published_at": T[a], "category": CATEGORY[a]}
         for a, _, _, _ in MEMBERS],
        schema=pa.schema([("article_id", pa.string()),
                          ("published_at", pa.timestamp("us", tz="UTC")),
                          ("category", pa.string())]))
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
    assert df.loc[1].representative == "m1"  # 최초 발행(m2)이 아니라 중심 최근접


def test_first_and_last_published_span_the_members(tmp_path):
    _, df = run(build_stores(tmp_path))
    row = df.loc[1]
    assert row.first_published_at == T["m2"] and row.last_published_at == T["m3"]


def test_issue_size_and_publishers_are_counted(tmp_path):
    _, df = run(build_stores(tmp_path))
    row = df.loc[1]
    assert row.issue_size == 3 and row.publishers == 3


def test_issue_id_comes_from_registry(tmp_path):
    stats, df = run(build_stores(tmp_path))

    assert stats.with_issue_id == 2
    assert df.loc[1].issue_id == 101 and df.loc[0].issue_id == 100


def test_structured_flag_follows_quality_table(tmp_path):
    stats, df = run(build_stores(tmp_path))

    assert stats.issues == 2 and stats.structured_issues == 1
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


def test_hot_score_decays_to_one_over_e_after_tau():
    config = FeedConfig(recency_tau_hours=24.0)
    fresh = hot_score(10, 10, 0.0, config)
    day_old = hot_score(10, 10, 24.0, config)

    assert day_old < fresh
    assert abs(day_old / fresh - 0.3679) < 1e-3


def test_hot_score_clamps_future_time_to_now():
    config = FeedConfig()
    assert hot_score(10, 10, -5.0, config) == hot_score(10, 10, 0.0, config)


def test_issue_category_is_majority_of_labeled_members(tmp_path):
    _, df = run(build_stores(tmp_path))

    assert df.loc[1].category == "정치"   # 대표 m1 은 OTHER 지만 구성원 다수결로 채워진다
    assert df.loc[0].category == "OTHER"  # 라벨 기사가 없으면 대표 값(OTHER) 유지


@pytest.mark.parametrize("categories, fallback, expected", [
    (["정치", "정치", "경제"], "경제", "정치"),   # 다수결
    (["정치", "경제"], "경제", "경제"),          # 동률에 대표 값이 있으면 대표
    (["정치", "경제"], "사회", "정치"),          # 동률에 대표 값이 없으면 최다 첫째
    (["OTHER", None, ""], "사회", "사회"),       # 라벨 없음 → 대표 값
    (["OTHER", None], None, "OTHER"),           # 대표 값도 없으면 OTHER
    (["산업", "경제", "사회"], "사회", "경제"),   # 별칭은 합쳐서 센다 — 산업+경제 2표
    (["IT과학"], "산업", "IT/과학"),            # 대표 값도 어휘로 맞춘다
])
def test_majority_category_rules(categories, fallback, expected):
    assert majority_category(categories, fallback) == expected


@pytest.mark.parametrize("raw, expected", [
    ("산업", "경제"), ("IT과학", "IT/과학"), ("정치", "정치"), ("", "OTHER"), (None, "OTHER"),
])
def test_normalize_category_maps_to_team_vocabulary(raw, expected):
    assert normalize_category(raw) == expected


def test_normalize_category_passes_unknown_label_through():
    assert normalize_category("최신기사") == "최신기사"


def test_aliases_map_into_the_vocabulary():
    assert normalize_category("산업") in CATEGORIES and normalize_category("IT과학") in CATEGORIES


def test_pipeline_writes_hot_score(tmp_path):
    _, df = run(build_stores(tmp_path))

    # 이슈 1(3건/3곳, 최신)이 이슈 0(2건/1곳, 오래됨)보다 화제성 높다
    assert df.loc[1].hot_score > df.loc[0].hot_score > 0
