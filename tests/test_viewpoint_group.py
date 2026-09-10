import pytest

from hannun.stance.viewpoint_group import (
    generate_viewpoint_group_labels,
)


def test_known_viewpoints_are_grouped_and_labeled():
    articles = [
        {
            "article_id": "neg-1",
            "stance": "negative",
            "content": (
                "대법원장 제청 절차가 충분한 논의 없이 진행돼 "
                "절차적 정당성이 부족하다는 비판이 나왔다."
            ),
        },
        {
            "article_id": "neg-2",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정에서 협의가 부족해 "
                "절차적 정당성에 문제가 있다는 우려가 제기됐다."
            ),
        },
        {
            "article_id": "neg-3",
            "stance": "negative",
            "content": (
                "대법원장 제청이 사법부 독립을 훼손할 수 있다는 "
                "우려가 나왔다."
            ),
        },
        {
            "article_id": "pos-1",
            "stance": "positive",
            "content": (
                "현행법에 따른 대법원장 제청 절차이므로 "
                "정당하다는 평가가 나왔다."
            ),
        },
        {
            "article_id": "pos-2",
            "stance": "positive",
            "content": (
                "법에 정해진 절차를 따른 제청으로 "
                "문제가 없고 정당하다는 입장이다."
            ),
        },
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="대법원장 제청",
    )

    assert result["neg-1"] == "절차적 정당성"
    assert result["neg-2"] == "절차적 정당성"

    assert result["neg-3"] == "사법부 독립"
    assert result["neg-3"] != result["neg-1"]

    assert result["pos-1"] == "절차 · 정당"
    assert result["pos-2"] == "절차 · 정당"


def test_subclusters_do_not_exceed_three():
    articles = [
        {
            "article_id": f"a{index}",
            "stance": "negative",
            "content": content,
        }
        for index, content in enumerate(
            [
                "절차적 정당성이 부족하다는 비판이다.",
                "충분한 협의가 없어 절차에 문제가 있다는 지적이다.",
                "사법부 독립 훼손 가능성이 있다는 우려다.",
                "법원 독립성을 침해할 수 있다는 비판이다.",
                "정치적 개입 가능성이 있다는 우려다.",
                "정치권 영향력이 커질 수 있다는 지적이다.",
            ],
            start=1,
        )
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="대법원장 제청",
    )

    labels = {
        label
        for label in result.values()
        if label is not None
    }

    assert 1 <= len(labels) <= 3


def test_article_without_evidence_keeps_null_before_pipeline_fallback():
    articles = [
        {
            "article_id": "a1",
            "stance": "neutral",
            "content": "",
        }
    ]

    result = generate_viewpoint_group_labels(
        articles
    )

    assert result == {
        "a1": None,
    }


def test_duplicate_article_id_raises_error():
    articles = [
        {
            "article_id": "a1",
            "stance": "neutral",
            "content": "첫 번째 근거",
        },
        {
            "article_id": "a1",
            "stance": "neutral",
            "content": "두 번째 근거",
        },
    ]

    with pytest.raises(
        ValueError,
        match="duplicate article_id",
    ):
        generate_viewpoint_group_labels(
            articles
        )


def test_invalid_stance_raises_error():
    articles = [
        {
            "article_id": "a1",
            "stance": "unknown",
            "content": "근거 문장",
        }
    ]

    with pytest.raises(
        ValueError,
        match="invalid stance",
    ):
        generate_viewpoint_group_labels(
            articles
        )
