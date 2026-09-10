import copy
import json
from pathlib import Path

import pytest

from hannun.enrichment.event_name import generate_event_name
from hannun.enrichment.pipeline import enrich_dummy
from hannun.enrichment.schema import validate_ds2_output


SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "ds1_issue_output_v1.example.json"
)


@pytest.fixture
def issues():
    assert SAMPLE.exists(), f"샘플 파일이 없습니다: {SAMPLE}"

    with SAMPLE.open("r", encoding="utf-8") as file:
        return json.load(file)


def test_valid_input_returns_ds2_output(issues):
    issue = issues[0]

    result = enrich_dummy(issue)

    assert result["issue_cluster_id"] == issue["issue_cluster_id"]
    assert len(result["articles"]) == len(issue["articles"])


def test_missing_issue_cluster_id_raises_error(issues):
    issue = copy.deepcopy(issues[0])
    issue.pop("issue_cluster_id")

    with pytest.raises(ValueError):
        enrich_dummy(issue)


def test_invalid_stance_confidence_raises_error(issues):
    result = enrich_dummy(issues[0])

    result["articles"][0]["stance_confidence"] = 1.5

    with pytest.raises(ValueError):
        validate_ds2_output(result)


def test_event_name_is_generated_by_event_name_module(issues):
    issue = issues[0]

    result = enrich_dummy(issue)

    assert result["event_name"] == generate_event_name(issue)
    assert result["event_name"].strip()
    assert "[DUMMY]" not in result["event_name"]


def test_missing_publisher_name_raises_error():
    from hannun.enrichment.schema import validate_ds1_issue

    issue = {
        "issue_cluster_id": "issue-test",
        "representative_title": "테스트 이슈",
        "articles": [
            {
                "article_id": "article-1",
                "title": "테스트 기사",
                "content": "테스트 본문",
            }
        ],
    }

    with pytest.raises(
        ValueError,
        match="publisher_name",
    ):
        validate_ds1_issue(issue)


def test_null_viewpoint_group_label_raises_error(issues):
    result = enrich_dummy(issues[0])

    result["articles"][0]["viewpoint_group_label"] = None

    with pytest.raises(
        ValueError,
        match="viewpoint_group_label",
    ):
        validate_ds2_output(result)


def test_viewpoint_group_label_over_200_chars_raises_error(issues):
    result = enrich_dummy(issues[0])

    result["articles"][0]["viewpoint_group_label"] = "가" * 201

    with pytest.raises(
        ValueError,
        match="viewpoint_group_label",
    ):
        validate_ds2_output(result)
