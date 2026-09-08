import copy
import json
from pathlib import Path

import pytest

from hannun.enrichment import pipeline as enrichment_pipeline
from hannun.enrichment.pipeline import enrich_issue


SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "ds1_issue_output_v1.example.json"
)


@pytest.fixture
def issue():
    assert SAMPLE.exists(), f"샘플 파일이 없습니다: {SAMPLE}"

    with SAMPLE.open("r", encoding="utf-8") as file:
        issues = json.load(file)

    return issues[0]


def test_enrich_issue_returns_real_ds2_analysis(issue):
    result = enrich_issue(issue)

    assert result["issue_cluster_id"] == issue["issue_cluster_id"]

    assert isinstance(result["event_name"], str)
    assert result["event_name"].strip()
    assert "[DUMMY]" not in result["event_name"]

    assert isinstance(result["fact_summary"], list)
    assert all(
        "[DUMMY]" not in sentence
        for sentence in result["fact_summary"]
    )

    assert isinstance(result["viewpoint_analysis"], dict)
    assert set(result["viewpoint_analysis"]) == {
        "positive",
        "neutral",
        "negative",
    }

    for stance_analysis in result["viewpoint_analysis"].values():
        assert "keywords" in stance_analysis
        assert "phrases" in stance_analysis
        assert isinstance(stance_analysis["keywords"], list)
        assert isinstance(stance_analysis["phrases"], list)


def test_enrich_issue_returns_stance_for_every_article(issue):
    result = enrich_issue(issue)

    assert len(result["articles"]) == len(issue["articles"])

    expected_ids = [
        article["article_id"]
        for article in issue["articles"]
    ]
    actual_ids = [
        article["article_id"]
        for article in result["articles"]
    ]

    assert actual_ids == expected_ids

    for article in result["articles"]:
        assert article["stance"] in {
            "positive",
            "neutral",
            "negative",
        }

        confidence = article["stance_confidence"]

        assert isinstance(confidence, (int, float))
        assert 0.0 <= confidence <= 1.0


def test_enrich_issue_preserves_ds1_keywords(issue):
    issue = copy.deepcopy(issue)
    issue["keywords"] = [
        "반도체",
        "투자",
    ]

    result = enrich_issue(issue)

    assert result["keywords"] == [
        "반도체",
        "투자",
    ]


def test_enrich_dummy_delegates_to_enrich_issue(monkeypatch):
    expected = {
        "result": "real-enrichment",
    }

    def fake_enrich_issue(issue_data):
        assert issue_data == {
            "issue_cluster_id": "issue-test",
        }
        return expected

    monkeypatch.setattr(
        enrichment_pipeline,
        "enrich_issue",
        fake_enrich_issue,
    )

    result = enrichment_pipeline.enrich_dummy(
        {
            "issue_cluster_id": "issue-test",
        }
    )

    assert result is expected


def test_viewpoint_analysis_uses_only_stance_evidence(monkeypatch):
    issue = {
        "issue_cluster_id": "issue-test",
        "representative_title": "테스트 이슈",
        "keywords": [],
        "articles": [
            {
                "article_id": "article-1",
                "title": "테스트 기사",
                "content": (
                    "타깃과 관련 없는 민주당 기사 내용이다. "
                    "타깃과 관련된 핵심 근거 문장이다."
                ),
                "publisher_name": "테스트일보",
            }
        ],
    }

    monkeypatch.setattr(
        enrichment_pipeline,
        "generate_event_name",
        lambda issue_data: "테스트 타깃",
    )

    monkeypatch.setattr(
        enrichment_pipeline,
        "generate_fact_summary",
        lambda issue_data: [],
    )

    monkeypatch.setattr(
        enrichment_pipeline,
        "classify_stance",
        lambda target, content: {
            "stance": "positive",
            "stance_confidence": 1.0,
        },
    )

    monkeypatch.setattr(
        enrichment_pipeline,
        "select_stance_evidence_sentences",
        lambda target, content, stance: [
            "타깃과 관련된 핵심 근거 문장이다."
        ],
    )

    captured = {}

    def fake_extract_stance_analysis(articles):
        captured["articles"] = articles

        return {
            "positive": {
                "keywords": [],
                "phrases": [],
            },
            "neutral": {
                "keywords": [],
                "phrases": [],
            },
            "negative": {
                "keywords": [],
                "phrases": [],
            },
        }

    monkeypatch.setattr(
        enrichment_pipeline,
        "extract_stance_analysis",
        fake_extract_stance_analysis,
    )

    enrich_issue(issue)

    assert captured["articles"] == [
        {
            "content": "타깃과 관련된 핵심 근거 문장이다.",
            "stance": "positive",
        }
    ]

    assert (
        "민주당"
        not in captured["articles"][0]["content"]
    )
