from hannun.stance.cross_validation import (
    has_numeric_conflict,
    is_cross_validated,
    select_common_facts,
)
from hannun.stance.evidence import (
    build_issue_evidence,
    group_evidence_by_sentence,
)


def test_is_cross_validated_with_two_publishers():
    evidence = {
        "sentence": "정부는 정책을 발표했다.",
        "sources": [
            {
                "article_id": "sha256:article1",
                "publisher_name": "연합뉴스",
            },
            {
                "article_id": "sha256:article2",
                "publisher_name": "한겨레",
            },
        ],
    }

    assert is_cross_validated(evidence) is True


def test_is_not_cross_validated_with_same_publisher():
    evidence = {
        "sentence": "정부는 정책을 발표했다.",
        "sources": [
            {
                "article_id": "sha256:article1",
                "publisher_name": "연합뉴스",
            },
            {
                "article_id": "sha256:article2",
                "publisher_name": "연합뉴스",
            },
        ],
    }

    assert is_cross_validated(evidence) is False


def test_numeric_conflict():
    sentence_a = "지원 예산은 100억 원이다."
    sentence_b = "지원 예산은 120억 원이다."

    assert has_numeric_conflict(sentence_a, sentence_b) is True


def test_same_numeric_value_is_not_conflict():
    sentence_a = "지원 예산은 100억 원이다."
    sentence_b = "지원 예산은 100억 원이다."

    assert has_numeric_conflict(sentence_a, sentence_b) is False


def test_select_common_facts_limits_to_three_and_excludes_conflict():
    evidence_list = [
        {
            "sentence": "정부는 정책을 발표했다.",
            "sources": [
                {"article_id": "a1", "publisher_name": "연합뉴스"},
                {"article_id": "a2", "publisher_name": "한겨레"},
                {"article_id": "a3", "publisher_name": "KBS"},
            ],
        },
        {
            "sentence": "지원 대상은 청년이다.",
            "sources": [
                {"article_id": "b1", "publisher_name": "연합뉴스"},
                {"article_id": "b2", "publisher_name": "한겨레"},
            ],
        },
        {
            "sentence": "신청은 다음 달 시작된다.",
            "sources": [
                {"article_id": "c1", "publisher_name": "연합뉴스"},
                {"article_id": "c2", "publisher_name": "SBS"},
            ],
        },
        {
            "sentence": "지원 기간은 3년이다.",
            "sources": [
                {"article_id": "d1", "publisher_name": "KBS"},
                {"article_id": "d2", "publisher_name": "SBS"},
            ],
        },
        {
            "sentence": "지원 예산은 100억 원이다.",
            "sources": [
                {"article_id": "e1", "publisher_name": "연합뉴스"},
                {"article_id": "e2", "publisher_name": "한겨레"},
            ],
        },
        {
            "sentence": "지원 예산은 120억 원이다.",
            "sources": [
                {"article_id": "f1", "publisher_name": "KBS"},
                {"article_id": "f2", "publisher_name": "SBS"},
            ],
        },
    ]

    result = select_common_facts(evidence_list)

    assert len(result) == 3
    assert result[0]["sentence"] == "정부는 정책을 발표했다."
    assert all(
        "지원 예산은" not in evidence["sentence"]
        for evidence in result
    )


def test_numeric_conflict_with_similar_wording():
    sentence_a = "지원 예산은 100억 원이다."
    sentence_b = "지원 예산은 총 120억 원이다."

    assert has_numeric_conflict(sentence_a, sentence_b) is True


def test_similar_sentences_are_merged_as_common_fact():
    evidence_list = [
        {
            "sentence": "정부는 청년 지원 정책을 발표했다.",
            "sources": [
                {"article_id": "a1", "publisher_name": "연합뉴스"},
            ],
        },
        {
            "sentence": "정부가 청년 지원 정책을 발표했다.",
            "sources": [
                {"article_id": "a2", "publisher_name": "한겨레"},
            ],
        },
    ]

    result = select_common_facts(evidence_list)

    assert len(result) == 1
    assert len(result[0]["sources"]) == 2
    assert {
        source["publisher_name"]
        for source in result[0]["sources"]
    } == {"연합뉴스", "한겨레"}


def test_numeric_conflict_with_real_news_formats():
    cases = [
        (
            "사업비는 1,000억 원이다.",
            "사업비는 1,200억 원이다.",
        ),
        (
            "증가율은 3.5%다.",
            "증가율은 4.2%다.",
        ),
        (
            "지원 규모는 약 100억 원이다.",
            "지원 규모는 약 120억 원이다.",
        ),
    ]

    for sentence_a, sentence_b in cases:
        assert has_numeric_conflict(sentence_a, sentence_b) is True


def test_ds2_sentence_to_common_fact_integration():
    articles = [
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
        {
            "article_id": "a5",
            "publisher_name": "MBC",
            "content": "지원 예산은 100억 원이다.",
        },
        {
            "article_id": "a6",
            "publisher_name": "JTBC",
            "content": "지원 예산은 120억 원이다.",
        },
    ]

    sentence_evidence = build_issue_evidence(articles)
    grouped_evidence = group_evidence_by_sentence(sentence_evidence)
    result = select_common_facts(grouped_evidence)

    sentences = [
        evidence["sentence"]
        for evidence in result
    ]

    assert len(result) == 2
    assert "정부는 청년 지원 정책을 발표했다." in sentences
    assert "신청은 다음 달 시작된다." in sentences
    assert all(
        "지원 예산은" not in sentence
        for sentence in sentences
    )



def test_select_common_facts_merges_different_wording():
    evidence_list = [
        {
            "sentence": (
                "청와대는 “금싸라기 땅이 잘 이용되는 것이 중요하다”며 "
                "용산공원 청년주택 건설 구상을 공식화했다."
            ),
            "sources": [
                {
                    "article_id": "a1",
                    "publisher_name": "동아일보",
                }
            ],
        },
        {
            "sentence": (
                "하준경 청와대 경제성장수석이 서울 용산공원을 "
                "주택 공급 부지로 활용하는 방안을 두고 "
                "“이 금싸라기 땅이 잘 이용되는 게 중요하다”며 "
                "일부는 청년주택으로 공급할 방침이라고 밝혔다."
            ),
            "sources": [
                {
                    "article_id": "a2",
                    "publisher_name": "경향신문",
                }
            ],
        },
    ]

    result = select_common_facts(evidence_list)

    assert len(result) == 1
    assert {
        source["publisher_name"]
        for source in result[0]["sources"]
    } == {
        "동아일보",
        "경향신문",
    }


def test_select_common_facts_does_not_merge_topic_only():
    evidence_list = [
        {
            "sentence": (
                "청와대는 용산공원 청년주택 공급 구상을 공식화했다."
            ),
            "sources": [
                {
                    "article_id": "a1",
                    "publisher_name": "동아일보",
                }
            ],
        },
        {
            "sentence": (
                "오세훈 시장은 용산공원 주택 공급 정책에 반대했다."
            ),
            "sources": [
                {
                    "article_id": "a2",
                    "publisher_name": "경향신문",
                }
            ],
        },
    ]

    assert select_common_facts(evidence_list) == []


def test_numeric_conflict_with_different_wording():
    sentence_a = (
        "정부는 용산공원에 청년주택 "
        "1000가구를 공급한다고 밝혔다."
    )
    sentence_b = (
        "청와대는 용산공원 청년주택 "
        "2000가구 공급 방침을 밝혔다."
    )

    assert has_numeric_conflict(
        sentence_a,
        sentence_b,
    ) is True


def test_number_missing_on_one_side_is_not_conflict():
    sentence_a = (
        "조 대법원장은 지난 18일 손봉기 부장판사와 "
        "김성수 부장판사를 임명 제청했다."
    )
    sentence_b = (
        "조 대법원장은 전날 손봉기 부장판사와 "
        "김성수 부장판사를 임명 제청했다."
    )

    assert has_numeric_conflict(
        sentence_a,
        sentence_b,
    ) is False


def test_common_fact_merges_when_one_sentence_omits_date_number():
    evidence_list = [
        {
            "sentence": (
                "조 대법원장은 지난 18일 손봉기 부장판사와 "
                "김성수 부장판사를 대통령에게 임명 제청했다."
            ),
            "sources": [
                {
                    "article_id": "a1",
                    "publisher_name": "오마이뉴스",
                }
            ],
        },
        {
            "sentence": (
                "조 대법원장은 전날 손봉기 부장판사와 "
                "김성수 부장판사를 대통령에게 임명 제청했다."
            ),
            "sources": [
                {
                    "article_id": "a2",
                    "publisher_name": "경향신문",
                }
            ],
        },
    ]

    result = select_common_facts(evidence_list)

    assert len(result) == 1
    assert {
        source["publisher_name"]
        for source in result[0]["sources"]
    } == {
        "오마이뉴스",
        "경향신문",
    }


def test_biographical_numbers_do_not_create_numeric_conflict():
    sentence_a = (
        "조 대법원장은 새 대법관 후보자로 김성수 "
        "서울고법 부장판사(58·24기)를 전날(18일) "
        "이재명 대통령에게 임명 제청했다."
    )
    sentence_b = (
        "조 대법원장은 지난 18일 김성수 "
        "서울고등법원 부장판사를 이재명 대통령에게 "
        "임명 제청했다."
    )

    assert has_numeric_conflict(
        sentence_a,
        sentence_b,
    ) is False


def test_common_fact_merges_despite_age_and_class_metadata():
    evidence_list = [
        {
            "sentence": (
                "조 대법원장은 새 대법관 후보자로 손 부장판사와 "
                "김성수 서울고법 부장판사(58·24기)를 "
                "전날(18일) 이재명 대통령에게 임명 제청했다."
            ),
            "sources": [
                {
                    "article_id": "a1",
                    "publisher_name": "동아일보",
                }
            ],
        },
        {
            "sentence": (
                "조 대법원장은 지난 18일 손봉기 대구지방법원 "
                "부장판사와 김성수 서울고등법원 부장판사를 "
                "이재명 대통령에게 임명 제청했다."
            ),
            "sources": [
                {
                    "article_id": "a2",
                    "publisher_name": "오마이뉴스",
                }
            ],
        },
    ]

    result = select_common_facts(evidence_list)

    assert len(result) == 1
    assert {
        source["publisher_name"]
        for source in result[0]["sources"]
    } == {
        "동아일보",
        "오마이뉴스",
    }
