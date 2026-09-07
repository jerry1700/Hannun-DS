from datetime import datetime, timezone

from hannun.clustering import IssueStore, RegistryStore, SuccessionConfig, succeed

# 창 1(d1~d2): 이슈 0 = {a1(d1), a2(d2), a3(d2)}, 이슈 1 = {b1..b4(d2)}
# 창 2(d2~d3): 5 = {a2, a3, c3} → 이슈 0 승계 / 7 = {b1, b2, e1} 과 8 = {b3, b4, f1}
#   → 둘 다 이슈 1 을 주장(분열, 한쪽만 승계) / 9 = {c1, c2} → 신규
D1, D2, D3 = "2026-08-20", "2026-08-21", "2026-08-22"
CLUSTERED_AT = datetime(2026, 8, 22, tzinfo=timezone.utc)


def write_window(store, window, clusters):
    by_date = {}
    for label, articles in clusters.items():
        for article_id, date_str in articles:
            by_date.setdefault(date_str, []).append({
                "article_id": article_id, "publisher_id": "pub", "published_date": date_str,
                "issue_local": label, "issue_size": len(articles),
                "window_start": window[0], "window_end": window[1], "clustered_at": CLUSTERED_AT,
            })
    for date_str, rows in by_date.items():
        store.write_partition(date_str, rows)


def build_w1(tmp_path):
    issues = IssueStore(tmp_path / "gold")
    registry = RegistryStore(tmp_path / "gold")
    write_window(issues, (D1, D2), {
        0: [("a1", D1), ("a2", D2), ("a3", D2)],
        1: [("b1", D2), ("b2", D2), ("b3", D2), ("b4", D2)],
    })
    return issues, registry


def advance_to_w2(issues):
    # 창이 하루 밀리면 겹치는 날(d2) 파티션은 새 군집으로 덮인다
    write_window(issues, (D2, D3), {
        5: [("a2", D2), ("a3", D2), ("c3", D3)],
        7: [("b1", D2), ("b2", D2), ("e1", D3)],
        8: [("b3", D2), ("b4", D2), ("f1", D3)],
        9: [("c1", D3), ("c2", D3)],
    })


def test_first_window_creates_ids(tmp_path):
    issues, registry = build_w1(tmp_path)
    stats = succeed(issues, registry, start_date=D1, end_date=D2)

    assert stats.issues == 2 and stats.created == 2 and stats.inherited == 0
    table = registry.read_window(D1).to_pydict()
    id_of = dict(zip(table["article_id"], table["issue_id"]))
    assert id_of["a1"] == id_of["a2"] == id_of["a3"]
    assert id_of["b1"] == id_of["b4"] and id_of["a1"] != id_of["b1"]


def test_inheritance_split_and_new(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    w1 = registry.read_window(D1).to_pydict()
    id_a = dict(zip(w1["article_id"], w1["issue_id"]))["a1"]
    id_b = dict(zip(w1["article_id"], w1["issue_id"]))["b1"]

    advance_to_w2(issues)
    stats = succeed(issues, registry, start_date=D2, end_date=D3)
    w2 = registry.read_window(D2).to_pydict()
    id_of = dict(zip(w2["article_id"], w2["issue_id"]))
    status_of = dict(zip(w2["article_id"], w2["status"]))

    assert stats.issues == 4 and stats.inherited == 2 and stats.created == 2 and stats.splits == 1
    # 겹침 과반인 군집이 ID 를 승계한다
    assert id_of["a2"] == id_a and status_of["a2"] == "inherited"
    # 분열: 이슈 1 을 주장한 두 군집 중 하나만 승계, 다른 하나는 새 ID
    b_ids = {id_of["b1"], id_of["b3"]}
    assert id_b in b_ids and len(b_ids) == 2
    # 겹침이 없는 군집은 새 ID
    assert status_of["c1"] == "new" and id_of["c1"] not in {id_a, id_b}


def test_rerun_is_idempotent(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    advance_to_w2(issues)
    succeed(issues, registry, start_date=D2, end_date=D3)
    first = registry.read_window(D2).to_pydict()
    succeed(issues, registry, start_date=D2, end_date=D3)
    second = registry.read_window(D2).to_pydict()

    assert first["article_id"] == second["article_id"]
    assert first["issue_id"] == second["issue_id"]
    assert first["status"] == second["status"]


def test_min_shared_blocks_single_article_overlap(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    # 겹침이 한 건뿐인 군집은 승계하지 못한다
    write_window(issues, (D2, D3), {5: [("b1", D2), ("g1", D3), ("g2", D3)]})
    stats = succeed(issues, registry, SuccessionConfig(min_shared=2),
                    start_date=D2, end_date=D3)

    assert stats.inherited == 0 and stats.created == 1


def test_empty_window_is_noop(tmp_path):
    issues = IssueStore(tmp_path / "gold")
    registry = RegistryStore(tmp_path / "gold")
    stats = succeed(issues, registry, start_date=D1, end_date=D2)

    assert stats.issues == 0 and registry.window_starts() == []
