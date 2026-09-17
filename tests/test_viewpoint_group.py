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

    assert result["neg-1"] == result["neg-2"]
    assert "정당성" in result["neg-1"]
    assert "부족" in result["neg-1"]

    assert result["neg-3"] != result["neg-1"]
    assert "독립" in result["neg-3"]
    assert "훼손" in result["neg-3"]

    assert result["pos-1"] == result["pos-2"]
    assert "정당" in result["pos-1"]

    for label in result.values():
        assert label is not None
        assert len(label) <= 8


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


def test_semantic_paraphrases_stay_in_one_group():
    articles = [
        {
            "article_id": "same-1",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정에서 충분한 협의가 없어 "
                "절차적 정당성이 부족하다는 비판이 나왔다."
            ),
        },
        {
            "article_id": "same-2",
            "stance": "negative",
            "content": (
                "대법원장 제청 절차에 논의가 부족해 "
                "절차적 정당성 문제가 있다는 지적이 제기됐다."
            ),
        },
        {
            "article_id": "same-3",
            "stance": "negative",
            "content": (
                "충분한 논의 없이 대법원장 제청이 진행돼 "
                "절차의 정당성이 부족하다는 우려가 나왔다."
            ),
        },
        {
            "article_id": "same-4",
            "stance": "negative",
            "content": (
                "대법원장 제청 전에 협의가 충분하지 않아 "
                "절차적으로 정당하지 않다는 비판이다."
            ),
        },
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

    assert len(labels) == 1


def test_three_semantic_viewpoints_form_three_groups():
    articles = [
        {
            "article_id": "procedure-1",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정의 협의 부족으로 "
                "절차적 정당성이 훼손됐다는 비판이다."
            ),
        },
        {
            "article_id": "procedure-2",
            "stance": "negative",
            "content": (
                "충분한 논의 없이 대법원장 제청이 진행돼 "
                "절차적 정당성이 부족하다는 지적이다."
            ),
        },
        {
            "article_id": "independence-1",
            "stance": "negative",
            "content": (
                "대법원장 제청이 사법부 독립을 "
                "훼손할 수 있다는 우려가 나왔다."
            ),
        },
        {
            "article_id": "independence-2",
            "stance": "negative",
            "content": (
                "이번 대법원장 제청으로 법원의 독립성이 "
                "침해될 가능성이 있다는 비판이다."
            ),
        },
        {
            "article_id": "politics-1",
            "stance": "negative",
            "content": (
                "대법원장 제청에 정치권의 영향력이 "
                "커질 수 있다는 우려가 제기됐다."
            ),
        },
        {
            "article_id": "politics-2",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정에서 정치적 개입 "
                "가능성이 높아졌다는 지적이다."
            ),
        },
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="대법원장 제청",
    )

    procedure = result["procedure-1"]
    independence = result["independence-1"]
    politics = result["politics-1"]

    assert result["procedure-2"] == procedure
    assert result["independence-2"] == independence
    assert result["politics-2"] == politics

    assert len({
        procedure,
        independence,
        politics,
    }) == 3


def test_empty_evidence_reserves_fallback_group():
    articles = [
        {
            "article_id": "procedure-1",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정의 협의 부족으로 "
                "절차적 정당성이 훼손됐다는 비판이다."
            ),
        },
        {
            "article_id": "procedure-2",
            "stance": "negative",
            "content": (
                "충분한 논의 없이 대법원장 제청이 진행돼 "
                "절차적 정당성이 부족하다는 지적이다."
            ),
        },
        {
            "article_id": "independence-1",
            "stance": "negative",
            "content": (
                "대법원장 제청이 사법부 독립을 "
                "훼손할 수 있다는 우려가 나왔다."
            ),
        },
        {
            "article_id": "independence-2",
            "stance": "negative",
            "content": (
                "이번 대법원장 제청으로 법원의 독립성이 "
                "침해될 가능성이 있다는 비판이다."
            ),
        },
        {
            "article_id": "politics-1",
            "stance": "negative",
            "content": (
                "대법원장 제청에 정치권의 영향력이 "
                "커질 수 있다는 우려가 제기됐다."
            ),
        },
        {
            "article_id": "politics-2",
            "stance": "negative",
            "content": (
                "대법원장 제청 과정에서 정치적 개입 "
                "가능성이 높아졌다는 지적이다."
            ),
        },
        {
            "article_id": "no-evidence",
            "stance": "negative",
            "content": "",
        },
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="대법원장 제청",
    )

    assert result["no-evidence"] is None

    evidence_labels = {
        label
        for article_id, label in result.items()
        if article_id != "no-evidence"
        and label is not None
    }

    # 빈 evidence는 pipeline에서 "기타 부정 관점"이 되므로
    # 실제 E5 세부 그룹은 최대 2개만 사용해야 최종 3개 이하가 된다.
    assert len(evidence_labels) <= 2


def test_same_conclusion_with_different_reasons_forms_different_groups():
    articles = [
        {
            "article_id": "same-1",
            "stance": "negative",
            "content": (
                "재정 부담이 지나치게 크기 때문에 "
                "청년주택 정책을 폐기해야 한다는 주장이다."
            ),
        },
        {
            "article_id": "same-2",
            "stance": "negative",
            "content": (
                "절차적 정당성이 부족하기 때문에 "
                "청년주택 정책을 폐기해야 한다는 입장이다."
            ),
        },
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="청년주택 정책",
    )

    assert result["same-1"] != result["same-2"]


def test_different_conclusions_form_different_groups():
    articles = [
        {
            "article_id": "discard",
            "stance": "negative",
            "content": (
                "재정 부담이 커서 "
                "청년주택 정책을 폐기해야 한다는 주장이다."
            ),
        },
        {
            "article_id": "revise",
            "stance": "negative",
            "content": (
                "재정 부담을 줄이는 방향으로 "
                "청년주택 정책을 수정해야 한다는 주장이다."
            ),
        },
    ]

    result = generate_viewpoint_group_labels(
        articles,
        target="청년주택 정책",
    )

    assert result["discard"] != result["revise"]

def test_short_label_preserves_opinion_direction_regressions():
    from hannun.stance.viewpoint_group import _short_viewpoint_label

    cases = [
        (
            "건강보험료율 동결에 우려를 표하는 입장이다.",
            "2027년 건강보험료율 동결",
            "동결 우려",
        ),
        (
            "용 후보자의 자진 사퇴에 선을 긋는 상황이다.",
            "김승원 용혜인 인사청문회",
            "사퇴 거부",
        ),
        (
            "정부의 건강보험료율 동결을 규탄했다.",
            "2027년 건강보험료율 동결",
            "동결 규탄",
        ),
        (
            "야권은 후보자 지명 철회를 요구했다.",
            "후보자 인사청문회",
            "철회 요구",
        ),
    ]

    for text, target, expected in cases:
        actual = _short_viewpoint_label(
            text,
            target=target,
        )

        assert actual == expected, (
            f"{text!r}: expected={expected!r}, "
            f"actual={actual!r}"
        )

        assert len(actual) <= 8

def test_different_directional_claims_are_not_forced_into_same_viewpoint_group():
    from hannun.stance.viewpoint_group import generate_viewpoint_group_labels

    articles = [
        {
            "article_id": "article-concern",
            "stance": "negative",
            "content": "대한의사협회는 건강보험료 동결에 우려를 표하는 입장이다.",
        },
        {
            "article_id": "article-condemn",
            "stance": "negative",
            "content": "보건의료계 노조는 정부의 건강보험료 동결을 규탄했다.",
        },
    ]

    labels = generate_viewpoint_group_labels(
        articles,
        target="2027년 건강보험료율 동결",
    )

    assert labels["article-concern"] == "동결 우려"
    assert labels["article-condemn"] == "동결 규탄"
    assert labels["article-concern"] != labels["article-condemn"]

def test_short_label_preserves_explicit_positive_direction():
    from hannun.stance.viewpoint_group import _short_viewpoint_label

    cases = [
        (
            "올해 다시 한번 KLPGA 정규투어를 개최하게 되어 기쁘고 영광스럽게 생각한다.",
            "KLPGA 오픈 개최",
            "개최 환영",
        ),
        (
            "이번 대회의 성공을 기대한다.",
            "KLPGA 오픈 개최",
            "성공 기대",
        ),
        (
            "한국 골프 발전과 골프 대중화에 지속적으로 기여하겠다.",
            "KLPGA 오픈 개최",
            "발전 기여",
        ),
    ]

    for text, target, expected in cases:
        actual = _short_viewpoint_label(
            text,
            target=target,
        )

        assert actual == expected, (
            f"{text!r}: expected={expected!r}, "
            f"actual={actual!r}"
        )

        assert len(actual) <= 8
