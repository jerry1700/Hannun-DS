import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_ds2_jsonl.py"
spec = importlib.util.spec_from_file_location("export_ds2_jsonl", SCRIPT)
export_ds2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_ds2)


def write_previous(path, messages):
    with path.open("w", encoding="utf-8") as out:
        for message in messages:
            out.write(json.dumps(message, ensure_ascii=False) + "\n")


def test_load_previous_indexes_by_cluster_and_links(tmp_path):
    target = tmp_path / "2026-09-14.jsonl"
    write_previous(target, [
        {"clusterId": 7, "category": "정치", "factSummary": "제목", "commonFactsBriefing": "요약",
         "articles": [{"link": "u1", "stance": "neutral"}, {"link": "u2", "stance": "positive"}]},
        {"clusterId": 8, "category": "경제", "factSummary": "다른 제목", "commonFactsBriefing": "",
         "articles": [{"link": "u3", "stance": "neutral"}]},
    ])

    previous = export_ds2.load_previous(target)

    assert set(previous) == {7, 8}
    assert previous[7][1] == frozenset({"u1", "u2"}) and previous[7][4] is False
    assert previous[8][4] is True   # 빈 요약은 재사용 시 empty_briefings 로 집계된다
    assert json.loads(previous[7][0])["clusterId"] == 7   # 원문 줄 그대로 보존


def test_unchanged_requires_same_members_title_and_category(tmp_path):
    target = tmp_path / "day.jsonl"
    write_previous(target, [{
        "clusterId": 1, "category": "사회", "factSummary": "제목", "commonFactsBriefing": "x",
        "articles": [{"link": "a"}, {"link": "b"}],
    }])
    entry = export_ds2.load_previous(target)[1]

    assert export_ds2.unchanged(entry, frozenset({"a", "b"}), "제목", "사회")
    # 기사가 하나 붙었으면 다시 분석해야 한다
    assert not export_ds2.unchanged(entry, frozenset({"a", "b", "c"}), "제목", "사회")
    # 대표 제목·카테고리가 바뀐 경우도 재분석
    assert not export_ds2.unchanged(entry, frozenset({"a", "b"}), "새 제목", "사회")
    assert not export_ds2.unchanged(entry, frozenset({"a", "b"}), "제목", "정치")
    assert not export_ds2.unchanged(None, frozenset({"a", "b"}), "제목", "사회")


BODY = ("정부는 오늘 새로운 청년 지원 정책을 발표했다. 이번 정책은 청년 주거 안정과 취업 지원을 핵심으로 한다. "
        "야당은 재원 대책이 빠졌다며 국회 심의 과정에서 따져 보겠다고 밝혔다. ")


def issue_data(issue_id, n_articles):
    return {
        "issue_cluster_id": str(issue_id),
        "representative_title": f"이슈 {issue_id} 대표 제목",
        "articles": [{"article_id": f"{issue_id}-{i}", "title": f"기사 {i}", "content": BODY + f"추가 문장 {i}.",
                      "publisher_name": f"p{i}"} for i in range(n_articles)],
    }


def test_analyze_issues_parallel_matches_sequential_in_order():
    issues = [issue_data(i, 2 + i % 3) for i in range(5)]
    sequential = export_ds2.analyze_issues(issues, workers=1)
    parallel = export_ds2.analyze_issues(issues, workers=2)

    assert [e["issue_cluster_id"] for e in parallel] == [str(i) for i in range(5)]
    assert parallel == sequential


def test_analyze_issues_single_issue_stays_sequential():
    assert export_ds2.analyze_issues([issue_data(9, 2)], workers=3)[0]["issue_cluster_id"] == "9"
    assert export_ds2.analyze_issues([], workers=3) == []


def test_load_previous_returns_empty_for_missing_file(tmp_path):
    assert export_ds2.load_previous(tmp_path / "none.jsonl") == {}


def test_load_previous_skips_broken_lines(tmp_path):
    target = tmp_path / "day.jsonl"
    target.write_text('{"clusterId": 1, "articles": []}\nnot json\n\n', encoding="utf-8")
    assert set(export_ds2.load_previous(target)) == {1}
