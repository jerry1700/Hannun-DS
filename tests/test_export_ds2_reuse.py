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


def test_load_previous_returns_empty_for_missing_file(tmp_path):
    assert export_ds2.load_previous(tmp_path / "none.jsonl") == {}


def test_load_previous_skips_broken_lines(tmp_path):
    target = tmp_path / "day.jsonl"
    target.write_text('{"clusterId": 1, "articles": []}\nnot json\n\n', encoding="utf-8")
    assert set(export_ds2.load_previous(target)) == {1}
