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


def test_boilerplate_outlet_name_is_filtered():
    from hannun.enrichment.fact_summary import (
        _is_boilerplate_sentence,
    )

    assert _is_boilerplate_sentence(
        "연합뉴스",
    ) is True


def test_boilerplate_copyright_is_filtered():
    from hannun.enrichment.fact_summary import (
        _is_boilerplate_sentence,
    )

    assert _is_boilerplate_sentence(
        "저작권자 연합뉴스 무단 전재 및 재배포 금지",
    ) is True


def test_issue_relevance_uses_repeated_title_keywords():
    from hannun.enrichment.fact_summary import (
        _extract_issue_keywords,
        _is_issue_relevant,
    )

    articles = [
        {
            "title": "한미훈련 축소 결정에 트럼프 입장",
        },
        {
            "title": "한미훈련 축소 두고 엇갈린 반응",
        },
    ]

    keywords = _extract_issue_keywords(articles)

    assert _is_issue_relevant(
        "한미훈련은 일부 축소됐다.",
        keywords,
    ) is True

    assert _is_issue_relevant(
        "호르무즈 해협 통항 대책도 논의됐다.",
        keywords,
    ) is False


def test_generate_fact_summary_filters_unrelated_common_sentence():
    issue = {
        "articles": [
            {
                "article_id": "a1",
                "publisher_name": "언론사A",
                "title": "한미훈련 축소 결정에 트럼프 입장",
                "content": (
                    "한미훈련은 일부 축소됐다. "
                    "호르무즈 해협 통항 대책도 논의됐다."
                ),
            },
            {
                "article_id": "a2",
                "publisher_name": "언론사B",
                "title": "한미훈련 축소 두고 엇갈린 반응",
                "content": (
                    "한미훈련은 일부 축소됐다. "
                    "호르무즈 해협 통항 대책도 논의됐다."
                ),
            },
        ],
    }

    result = generate_fact_summary(issue)

    assert "한미훈련은 일부 축소됐다." in result
    assert (
        "호르무즈 해협 통항 대책도 논의됐다."
        not in result
    )


def test_unbalanced_quote_is_filtered():
    from hannun.enrichment.fact_summary import (
        _has_unbalanced_quotes,
    )

    assert _has_unbalanced_quotes(
        "송 의원은 “대법관은 대통령이 임명한다."
    ) is True

    assert _has_unbalanced_quotes(
        "그는 “정책을 시행한다”고 밝혔다."
    ) is False


def test_photo_caption_is_boilerplate():
    from hannun.enrichment.fact_summary import (
        _is_boilerplate_sentence,
    )

    assert _is_boilerplate_sentence(
        "김윤덕(왼쪽) 국토교통부 장관이 국회에서 의원 질의에 답변하고 있다."
    ) is True


def test_issue_relevance_requires_multiple_keywords():
    from hannun.enrichment.fact_summary import (
        _is_issue_relevant,
    )

    keywords = {
        "조희대",
        "대법원장",
        "대법관",
        "제청",
    }

    assert _is_issue_relevant(
        "조희대 대법원장은 대법관 후보를 임명 제청했다.",
        keywords,
    ) is True

    assert _is_issue_relevant(
        "대법관 후보 재추천 방안을 논의했다.",
        keywords,
    ) is False
