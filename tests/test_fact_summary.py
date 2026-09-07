from hannun.enrichment.fact_summary import (
    _is_redundant_sentence,
    _is_uncertain_sentence,
    _lexrank_scores,
    generate_fact_summary,
)


def test_lexrank_empty_sentences():
    assert _lexrank_scores([]) == []


def test_lexrank_single_sentence():
    assert _lexrank_scores(["정부는 정책을 발표했다."]) == [1.0]


def test_uncertain_sentence_is_filtered():
    assert _is_uncertain_sentence(
        "정부가 추가 대책을 발표할 것으로 알려졌다."
    ) is True


def test_confirmed_sentence_is_not_uncertain():
    assert _is_uncertain_sentence(
        "정부는 오늘 추가 대책을 발표했다."
    ) is False


def test_redundant_sentence_detection():
    selected = [
        "KDI는 내년 성장률 전망치를 1.7%에서 2.2%로 올렸다."
    ]

    candidate = (
        "내년 성장률 전망치도 1.7%에서 "
        "2.2%로 상향 조정했다."
    )

    assert _is_redundant_sentence(
        candidate,
        selected,
    ) is True


def test_generate_fact_summary_returns_cross_validated_facts():
    issue = {
        "articles": [
            {
                "article_id": "a1",
                "publisher_name": "연합뉴스",
                "content": (
                    "정부는 청년 지원 정책을 발표했다. "
                    "추가 지원이 필요하다."
                ),
            },
            {
                "article_id": "a2",
                "publisher_name": "한겨레",
                "content": "정부가 청년 지원 정책을 발표했다.",
            },
            {
                "article_id": "a3",
                "publisher_name": "KBS",
                "content": "신청은 다음 달 시작된다.",
            },
            {
                "article_id": "a4",
                "publisher_name": "SBS",
                "content": "신청은 다음 달 시작된다.",
            },
        ],
    }

    result = generate_fact_summary(issue)

    assert len(result) == 2
    assert any(
        "청년 지원 정책을 발표했다" in sentence
        for sentence in result
    )
    assert "신청은 다음 달 시작된다." in result


def test_generate_fact_summary_limits_to_three():
    facts = [
        "정부는 정책을 발표했다.",
        "신청은 다음 달 시작된다.",
        "지원 대상은 청년이다.",
        "지원 기간은 3년이다.",
    ]

    articles = []

    for index, fact in enumerate(facts):
        articles.extend(
            [
                {
                    "article_id": f"a{index}-1",
                    "publisher_name": f"언론사{index}-A",
                    "content": fact,
                },
                {
                    "article_id": f"a{index}-2",
                    "publisher_name": f"언론사{index}-B",
                    "content": fact,
                },
            ]
        )

    issue = {
        "articles": articles,
    }

    result = generate_fact_summary(
        issue,
        limit=3,
    )

    assert len(result) == 3


def test_generate_fact_summary_non_positive_limit():
    issue = {
        "articles": [
            {
                "article_id": "a1",
                "publisher_name": "연합뉴스",
                "content": "정부는 정책을 발표했다.",
            }
        ],
    }

    assert generate_fact_summary(issue, limit=0) == []
