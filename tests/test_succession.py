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


def regrow_w1(issues):
    # 같은 창 재군집: 라벨은 리셋되고(0,1 → 3,4), 기사가 늘고, 새 이슈(6)도 탄생
    write_window(issues, (D1, D2), {
        3: [("a1", D1), ("a2", D2), ("a3", D2), ("a4", D2)],
        4: [("b1", D2), ("b2", D2), ("b3", D2), ("b4", D2)],
        6: [("h1", D2), ("h2", D2), ("h3", D2)],
    })


def columns_of(registry, window_start, column):
    table = registry.read_window(window_start).to_pydict()
    return dict(zip(table["article_id"], table[column]))


def succeed_w1_then_w2(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    first_ids = columns_of(registry, D1, "issue_id")
    advance_to_w2(issues)
    stats = succeed(issues, registry, start_date=D2, end_date=D3)
    return stats, first_ids, columns_of(registry, D2, "issue_id"), columns_of(registry, D2, "status")


def test_first_window_creates_ids(tmp_path):
    issues, registry = build_w1(tmp_path)
    stats = succeed(issues, registry, start_date=D1, end_date=D2)
    id_of = columns_of(registry, D1, "issue_id")

    assert stats.issues == 2 and stats.created == 2 and stats.inherited == 0
    assert id_of["a1"] == id_of["a2"] == id_of["a3"]
    assert id_of["b1"] == id_of["b4"] and id_of["a1"] != id_of["b1"]


def test_majority_overlap_inherits_previous_id(tmp_path):
    stats, first_ids, id_of, status_of = succeed_w1_then_w2(tmp_path)

    assert stats.inherited == 2
    assert id_of["a2"] == first_ids["a1"] and status_of["a2"] == "inherited"


def test_split_gives_previous_id_to_one_claimant_only(tmp_path):
    stats, first_ids, id_of, _ = succeed_w1_then_w2(tmp_path)

    assert stats.splits == 1
    b_ids = {id_of["b1"], id_of["b3"]}
    assert first_ids["b1"] in b_ids and len(b_ids) == 2


def test_cluster_without_overlap_gets_new_id(tmp_path):
    stats, first_ids, id_of, status_of = succeed_w1_then_w2(tmp_path)

    assert stats.created == 2
    assert status_of["c1"] == "new" and id_of["c1"] not in set(first_ids.values())


def test_nothing_retires_when_every_previous_issue_is_inherited(tmp_path):
    stats, _, _, _ = succeed_w1_then_w2(tmp_path)
    assert stats.retired == 0


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


def test_same_window_rerun_with_grown_data_keeps_ids(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    first_ids = columns_of(registry, D1, "issue_id")

    regrow_w1(issues)
    stats = succeed(issues, registry, start_date=D1, end_date=D2)
    id_of = columns_of(registry, D1, "issue_id")

    assert stats.self_window and stats.inherited == 2 and stats.created == 1 and stats.retired == 0
    assert id_of["a1"] == first_ids["a1"] and id_of["a4"] == first_ids["a1"]
    assert id_of["b1"] == first_ids["b1"]


def test_same_window_rerun_keeps_status_lineage(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)

    regrow_w1(issues)
    succeed(issues, registry, start_date=D1, end_date=D2)
    status_of = columns_of(registry, D1, "status")

    # status 는 직전 창 대비 계보 — 이 창에서 태어난 이슈는 재실행 뒤에도 new
    assert status_of["a1"] == "new" and status_of["h1"] == "new"


def test_new_ids_continue_from_global_max(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)
    first_ids = columns_of(registry, D1, "issue_id")

    regrow_w1(issues)
    succeed(issues, registry, start_date=D1, end_date=D2)
    id_of = columns_of(registry, D1, "issue_id")

    # 채번은 전 창 최대 다음부터 — 죽은 ID 재사용 없음
    assert id_of["h1"] > max(first_ids.values())


def test_retired_counts_issues_nobody_inherits(tmp_path):
    issues, registry = build_w1(tmp_path)
    succeed(issues, registry, start_date=D1, end_date=D2)

    # 같은 창 재실행에서 이슈 b 의 기사들이 노이즈로 흩어짐 — 이슈 a 만 살아남는다
    write_window(issues, (D1, D2), {
        3: [("a1", D1), ("a2", D2), ("a3", D2)],
        -1: [("b1", D2), ("b2", D2), ("b3", D2), ("b4", D2)],
    })
    stats = succeed(issues, registry, start_date=D1, end_date=D2)

    assert stats.inherited == 1 and stats.created == 0 and stats.retired == 1


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
