from datetime import datetime, timezone

from hannun.clustering import IssueStore
from hannun.embedding import EmbeddingStore
from hannun.quality import QualityConfig, assign

# 2차원 합성 벡터 — 방향이 의미.
# 이슈 0: 한 언론사 10건 (정형), [1, 0] 방향 / 이슈 1: 네 언론사 4건 (진짜), [0, 1] 방향
# 새 기사: 이슈 1 옆(new_close), 어디에도 안 가까움(new_far), 정형 옆(new_bot)
WINDOW = ("2026-08-20", "2026-08-21")


def build_stores(tmp_path):
    issues = IssueStore(tmp_path / "gold")
    embeddings = EmbeddingStore(tmp_path / "gold")
    clustered_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
    encoded_at = datetime(2026, 8, 20, tzinfo=timezone.utc)

    members = []
    for i in range(10):
        members.append((f"bot{i}", "bot_pub", "2026-08-20", 0, 10, [1.0, 0.01 * i]))
    for i, pub in enumerate(["p1", "p2", "p3", "p4"]):
        members.append((f"real{i}", pub, "2026-08-20", 1, 4, [0.01 * i - 0.01, 1.0]))
    incoming = [
        ("new_close", "p5", "2026-08-21", [0.03, 1.0]),
        ("new_far", "p6", "2026-08-21", [0.7, -0.7]),
        ("new_bot", "p7", "2026-08-21", [1.0, 0.02]),
    ]

    issue_rows, emb_rows = {}, {}
    for article_id, pub, date_str, label, size, vector in members:
        issue_rows.setdefault(date_str, []).append({
            "article_id": article_id, "publisher_id": pub, "published_date": date_str,
            "issue_local": label, "issue_size": size,
            "window_start": WINDOW[0], "window_end": WINDOW[1], "clustered_at": clustered_at,
        })
        emb_rows.setdefault(date_str, []).append({
            "article_id": article_id, "publisher_id": pub, "published_date": date_str,
            "vector": vector, "dim": 2, "model": "test-model", "encoded_at": encoded_at,
        })
    for article_id, pub, date_str, vector in incoming:
        emb_rows.setdefault(date_str, []).append({
            "article_id": article_id, "publisher_id": pub, "published_date": date_str,
            "vector": vector, "dim": 2, "model": "test-model", "encoded_at": encoded_at,
        })
    for date_str in issue_rows:
        issues.write_partition(date_str, issue_rows[date_str])
    for date_str in emb_rows:
        embeddings.write_partition(date_str, emb_rows[date_str])
    return issues, embeddings


def test_close_new_article_is_assigned(tmp_path):
    issues, embeddings = build_stores(tmp_path)
    stats = assign(issues, embeddings)
    row = issues.read().set_index("article_id").loc["new_close"]

    assert stats.new_articles == 3 and stats.assigned == 1 and stats.unassigned == 2
    assert row.issue_local == 1 and row.published_date == "2026-08-21"


def test_assignment_updates_issue_size_of_existing_members(tmp_path):
    issues, embeddings = build_stores(tmp_path)
    assign(issues, embeddings)
    df = issues.read().set_index("article_id")

    assert df.loc["new_close"].issue_size == 5 and df.loc["real0"].issue_size == 5


def test_far_new_article_stays_noise(tmp_path):
    issues, embeddings = build_stores(tmp_path)
    assign(issues, embeddings)

    assert issues.read().set_index("article_id").loc["new_far"].issue_local == -1


def test_structured_issue_does_not_take_new_articles(tmp_path):
    # 정형 이슈(0) 바로 옆이라도 정형은 배정 대상이 아니다
    issues, embeddings = build_stores(tmp_path)
    assign(issues, embeddings)
    df = issues.read().set_index("article_id")

    assert df.loc["new_bot"].issue_local == -1
    assert df.loc["bot0"].issue_size == 10


def test_rerun_assigns_nothing(tmp_path):
    issues, embeddings = build_stores(tmp_path)
    assign(issues, embeddings)
    first = issues.read(columns=["article_id", "issue_local", "issue_size"]).values.tolist()
    stats = assign(issues, embeddings)
    second = issues.read(columns=["article_id", "issue_local", "issue_size"]).values.tolist()

    assert stats.new_articles == 0 and stats.assigned == 0
    assert first == second


def test_threshold_blocks_assignment(tmp_path):
    issues, embeddings = build_stores(tmp_path)
    stats = assign(issues, embeddings, QualityConfig(assign_min_sim=0.9999))

    assert stats.assigned == 0 and stats.unassigned == 3


def test_empty_issue_table_is_noop(tmp_path):
    issues = IssueStore(tmp_path / "gold")
    embeddings = EmbeddingStore(tmp_path / "gold")
    stats = assign(issues, embeddings)

    assert stats.existing == 0 and stats.new_articles == 0
    assert issues.partition_dates() == []
