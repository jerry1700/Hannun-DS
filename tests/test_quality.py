from datetime import datetime, timezone

import pandas as pd

from hannun.clustering import IssueStore
from hannun.embedding import EmbeddingStore
from hannun.quality import QualityConfig, QualityStore, qualify

# 2차원 합성 벡터 — 코사인이라 방향이 곧 의미다.
# 이슈 0: 한 언론사가 10건 (정형 후보), [1, 0] 방향
# 이슈 1: 네 언론사가 4건 (진짜 이슈), [0, 1] 방향
# 노이즈: 이슈 1 옆(n_close), 어디에도 안 가까움(n_far), 정형 이슈 옆(n_bot)
WINDOW = ("2026-08-20", "2026-08-21")


def build_stores(tmp_path):
    issues = IssueStore(tmp_path / "gold")
    embeddings = EmbeddingStore(tmp_path / "gold")
    clustered_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
    encoded_at = datetime(2026, 8, 20, tzinfo=timezone.utc)

    articles = []
    for i in range(10):
        articles.append((f"bot{i}", "bot_pub", "2026-08-20", 0, 10, [1.0, 0.01 * i]))
    for i, pub in enumerate(["p1", "p2", "p3", "p4"]):
        date_str = "2026-08-21" if i == 3 else "2026-08-20"
        articles.append((f"real{i}", pub, date_str, 1, 4, [0.01 * i - 0.01, 1.0]))
    articles.append(("n_close", "p5", "2026-08-21", -1, 0, [0.05, 1.0]))
    articles.append(("n_far", "p6", "2026-08-20", -1, 0, [0.7, -0.7]))
    articles.append(("n_bot", "p7", "2026-08-20", -1, 0, [1.0, 0.02]))

    issue_rows, emb_rows = {}, {}
    for article_id, pub, date_str, label, size, vector in articles:
        issue_rows.setdefault(date_str, []).append({
            "article_id": article_id, "publisher_id": pub, "published_date": date_str,
            "issue_local": label, "issue_size": size,
            "window_start": WINDOW[0], "window_end": WINDOW[1], "clustered_at": clustered_at,
        })
        emb_rows.setdefault(date_str, []).append({
            "article_id": article_id, "publisher_id": pub, "published_date": date_str,
            "vector": vector, "dim": 2, "model": "test-model", "encoded_at": encoded_at,
        })
    for date_str in issue_rows:
        issues.write_partition(date_str, issue_rows[date_str])
        embeddings.write_partition(date_str, emb_rows[date_str])
    return issues, embeddings, QualityStore(tmp_path / "gold")


def test_structured_issue_is_flagged_not_deleted(tmp_path):
    issues, embeddings, store = build_stores(tmp_path)
    stats = qualify(issues, embeddings, store)
    df = store.read().set_index("article_id")

    assert stats.issues == 2 and stats.structured_issues == 1 and stats.structured_articles == 10
    assert bool(df.loc["bot0"].structured) and df.loc["bot0"].issue_local == 0
    assert not bool(df.loc["real0"].structured)
    assert df.loc["bot0"].issue_size == 10  # 표시만 하고 배정은 그대로


def test_close_noise_is_rescued(tmp_path):
    issues, embeddings, store = build_stores(tmp_path)
    stats = qualify(issues, embeddings, store)
    df = store.read().set_index("article_id")

    assert stats.noise_before == 3 and stats.rescued == 1 and stats.noise_after == 2
    row = df.loc["n_close"]
    assert row.issue_local == 1 and bool(row.rescued) and row.rescue_sim >= 0.85
    # 구제 반영 후 이슈 규모가 커진다 — 기존 구성원 행에도 반영
    assert row.issue_size == 5 and df.loc["real0"].issue_size == 5


def test_far_noise_stays_noise(tmp_path):
    issues, embeddings, store = build_stores(tmp_path)
    qualify(issues, embeddings, store)
    row = store.read().set_index("article_id").loc["n_far"]

    assert row.issue_local == -1 and not bool(row.rescued) and pd.isna(row.rescue_sim)


def test_structured_issue_does_not_absorb_noise(tmp_path):
    # n_bot 은 정형 이슈(0) 바로 옆이지만, 정형은 구제 대상에서 빠지므로 노이즈로 남는다
    issues, embeddings, store = build_stores(tmp_path)
    qualify(issues, embeddings, store)
    row = store.read().set_index("article_id").loc["n_bot"]

    assert row.issue_local == -1 and not bool(row.rescued)


def test_threshold_blocks_rescue(tmp_path):
    issues, embeddings, store = build_stores(tmp_path)
    stats = qualify(issues, embeddings, store, QualityConfig(rescue_min_sim=0.9999))

    assert stats.rescued == 0 and stats.noise_after == 3


def test_rerun_is_deterministic(tmp_path):
    issues, embeddings, store = build_stores(tmp_path)
    columns = ["article_id", "issue_local", "issue_size", "structured", "rescued"]
    qualify(issues, embeddings, store)
    first = store.read(columns=columns).values.tolist()
    qualify(issues, embeddings, store)
    second = store.read(columns=columns).values.tolist()

    assert store.partition_dates() == ["2026-08-20", "2026-08-21"]
    assert first == second
